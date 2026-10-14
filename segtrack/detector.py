"""Grounding DINO: find candidate boxes for target nouns in a frame.

Returns every surviving box with a score for *each* noun (not just its best label), so the
assignment step can resolve boxes that plausibly match two nouns (e.g. a spoon in a cup).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from PIL import Image
from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor

from .boxes import nms
from .device import get_device

DEFAULT_DETECTOR = "IDEA-Research/grounding-dino-tiny"


@dataclass
class Detections:
    boxes: np.ndarray       # (N, 4) xyxy pixels
    scores: np.ndarray      # (N, len(nouns)) per-noun score in [0, 1]
    nouns: list[str]

    def __len__(self) -> int:
        return len(self.boxes)


class GroundingDetector:
    def __init__(self, model_id: str = DEFAULT_DETECTOR, device: torch.device | None = None):
        self.device = device or get_device()
        self.processor = AutoProcessor.from_pretrained(model_id)
        self.model = AutoModelForZeroShotObjectDetection.from_pretrained(model_id).to(self.device).eval()

    @torch.no_grad()
    def __call__(
        self,
        image: np.ndarray | Image.Image,
        nouns: list[str],
        threshold: float = 0.2,
        nms_iou: float = 0.5,
    ) -> Detections:
        """Boxes whose best noun score ≥ `threshold`, after class-agnostic NMS."""
        if isinstance(image, np.ndarray):
            image = Image.fromarray(image)
        text = " . ".join(nouns) + " ."
        inputs = self.processor(images=image, text=text, return_tensors="pt").to(self.device)
        outputs = self.model(**inputs)

        token_logits = outputs.logits[0].sigmoid().cpu().numpy()           # (900, max_text_len)
        noun_scores = np.stack(
            [token_logits[:, idx].max(axis=1) for idx in self._noun_token_indices(text, nouns)],
            axis=1,
        )                                                                   # (900, n_nouns)

        w, h = image.size
        cx, cy, bw, bh = outputs.pred_boxes[0].cpu().numpy().T              # normalized
        boxes = np.stack([(cx - bw / 2) * w, (cy - bh / 2) * h, (cx + bw / 2) * w, (cy + bh / 2) * h], axis=1)
        boxes = np.clip(boxes, 0, [w, h, w, h])

        best = noun_scores.max(axis=1)
        keep = np.nonzero(best >= threshold)[0]
        keep = keep[nms(boxes[keep], best[keep], nms_iou)]
        return Detections(boxes=boxes[keep], scores=noun_scores[keep], nouns=list(nouns))

    def _noun_token_indices(self, text: str, nouns: list[str]) -> list[list[int]]:
        """Token positions covering each noun's characters in `text`."""
        enc = self.processor.tokenizer(text, return_offsets_mapping=True)
        spans, pos = [], 0
        for noun in nouns:
            start = text.index(noun, pos)
            spans.append((start, start + len(noun)))
            pos = start + len(noun)
        indices = []
        for start, end in spans:
            idx = [k for k, (s, e) in enumerate(enc["offset_mapping"]) if e > s and s >= start and e <= end]
            if not idx:
                raise RuntimeError(f"could not locate tokens for {text[start:end]!r}")
            indices.append(idx)
        return indices
