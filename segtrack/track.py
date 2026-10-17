"""SAM 2 video tracking over one chunk of frames.

Objects seeded on the same frame share one SAM 2 session, which propagates
  forward  from the seed frame to the end of the chunk, and
  backward from the seed frame as far as the caller asks (to fill frames before the object was
           detected; the pipeline is offline, so this is allowed).
SAM 2's video model cannot start a forward pass before an object's first prompt, so objects
seeded on different frames get separate sessions; that only repeats the cheap per-frame image
encoding, since the dominant cost (memory attention) is per object anyway. Callers should only
pass objects that are currently being tracked, because that cost grows with the object count.
"""

from __future__ import annotations

from collections import defaultdict
from typing import Callable

import numpy as np
import torch
from transformers import Sam2VideoModel, Sam2VideoProcessor

from .device import get_device

DEFAULT_SAM2_VIDEO = "facebook/sam2.1-hiera-small"

FrameMasks = dict[int, np.ndarray]      # track id → (H, W) bool mask, only for objects present
FrameScores = dict[int, float]          # track id → SAM 2 object-presence probability


class VideoTracker:
    def __init__(
        self,
        model_id: str = DEFAULT_SAM2_VIDEO,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
        presence_threshold: float = 0.5,
    ):
        self.device = device or get_device()
        # fp16 gives the same masks as fp32 here (mean IoU 0.999) at ~3x the speed on GPU/MPS.
        self.dtype = dtype or (torch.float16 if self.device.type in ("cuda", "mps") else torch.float32)
        self.presence_threshold = presence_threshold
        self.processor = Sam2VideoProcessor.from_pretrained(model_id)
        self.model = Sam2VideoModel.from_pretrained(model_id).to(self.device, dtype=self.dtype).eval()

    @torch.no_grad()
    def track(
        self,
        frames: np.ndarray,
        prompts: dict[int, tuple],
        backward: bool = True,
    ) -> tuple[list[FrameMasks], list[FrameScores]]:
        """Track objects through `frames` (N, H, W, 3).

        prompts: track id → (seed frame, seed mask) or (seed frame, seed mask, back_to), where
            back_to is the earliest frame the backward pass must reach (None: no backward pass).
            Without back_to, the backward pass runs to the chunk start (if `backward`).
        Returns per-frame masks and presence scores, indexed like `frames`.
        """
        masks: list[FrameMasks] = [{} for _ in range(len(frames))]
        scores: list[FrameScores] = [{} for _ in range(len(frames))]
        if not prompts:
            return masks, scores

        by_frame: dict[int, list[tuple[int, np.ndarray]]] = defaultdict(list)
        back_to: dict[int, list[int]] = defaultdict(list)
        for tid, prompt in prompts.items():
            f, mask = prompt[0], prompt[1]
            stop = prompt[2] if len(prompt) > 2 else (0 if backward else None)
            by_frame[f].append((tid, mask))
            if stop is not None and stop < f:
                back_to[f].append(stop)

        for seed, items in sorted(by_frame.items()):
            session = self.processor.init_video_session(
                video=list(frames), inference_device=self.device, video_storage_device=self.device,
                processing_device=self.device, dtype=self.dtype,
            )
            self.processor.add_inputs_to_inference_session(
                inference_session=session, frame_idx=seed,
                obj_ids=[tid for tid, _ in items], input_masks=[m for _, m in items],
            )
            self._propagate(session, seed, False, None, lambda f: True, masks, scores)
            if back_to[seed]:
                # Backward only as far as some object in this group needs filling.
                stop = min(back_to[seed])
                self._propagate(session, seed, True, seed - stop, lambda f, s=seed: f < s, masks, scores)
        return masks, scores

    def _propagate(
        self, session, start: int, reverse: bool, max_frames: int | None, keep: Callable[[int], bool],
        masks: list[FrameMasks], scores: list[FrameScores],
    ) -> None:
        size = [[session.video_height, session.video_width]]
        for out in self.model.propagate_in_video_iterator(session, start_frame_idx=start, reverse=reverse,
                                                          max_frame_num_to_track=max_frames):
            f = out.frame_idx
            if not keep(f):
                continue
            frame_masks = self.processor.post_process_masks([out.pred_masks.float()], original_sizes=size)[0][:, 0]
            frame_masks = frame_masks.cpu().numpy()
            presence = torch.sigmoid(out.object_score_logits.float())[:, 0].cpu().numpy()
            for k, tid in enumerate(out.object_ids):
                if presence[k] >= self.presence_threshold and frame_masks[k].any():
                    masks[f][tid] = frame_masks[k]
                    scores[f][tid] = float(presence[k])
                else:
                    masks[f].pop(tid, None)
                    scores[f].pop(tid, None)
