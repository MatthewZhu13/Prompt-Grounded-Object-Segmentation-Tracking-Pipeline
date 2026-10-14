"""Ground all targets in one frame: Grounding DINO → SAM 2 masks → SigLIP → assignment.

Used to seed tracks and, later, on every refresh frame to re-find lost objects and check
tracked ones.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np
import torch

from .assign import AssignConfig, AssignResult, assign
from .attributes import AttributeScorer, build_negatives, masked_crop
from .boxes import containment
from .detector import GroundingDetector
from .device import get_device, synchronize
from .parser import ParsedTask
from .segmenter import BoxSegmenter


@dataclass
class FrameGrounding:
    boxes: np.ndarray           # (C, 4) candidate boxes
    masks: np.ndarray           # (C, H, W) candidate masks
    mask_quality: np.ndarray    # (C,) SAM 2's predicted IoU for each mask
    dino: np.ndarray            # (C, T) Grounding DINO score for each target's noun
    siglip: np.ndarray          # (C, T) SigLIP score for each target's phrase
    siglip_neg: np.ndarray      # (C, K) SigLIP score for each negative phrase
    negatives: list[str]
    result: AssignResult
    timings: dict[str, float] = field(default_factory=dict)
    destination_masks: np.ndarray = field(default_factory=lambda: np.zeros((0, 0, 0), bool))  # (D, H, W)

    def mask_for(self, target_id: int) -> np.ndarray | None:
        m = next(m for m in self.result.matches if m.target_id == target_id)
        return None if m.candidate is None else self.masks[m.candidate]


class FrameGrounder:
    def __init__(
        self,
        parsed: ParsedTask,
        cfg: AssignConfig = AssignConfig(),
        device: torch.device | None = None,
        detector: GroundingDetector | None = None,
        segmenter: BoxSegmenter | None = None,
        scorer: AttributeScorer | None = None,
        containment_threshold: float = 0.8,
        destination_iou: float = 0.5,
    ):
        self.parsed = parsed
        self.cfg = cfg
        self.device = device or get_device()
        self.detector = detector or GroundingDetector(device=self.device)
        self.segmenter = segmenter or BoxSegmenter(device=self.device)
        self.scorer = scorer or AttributeScorer(device=self.device)
        self.negatives = build_negatives(parsed)
        self.containment_threshold = containment_threshold
        self.destination_iou = destination_iou
        self._phrases = list(dict.fromkeys(t.phrase for t in parsed.targets))
        # Destinations ("the container") are queried too: they are in the scene by definition and
        # are the main thing target masks get confused with (cups seen inside a clear container).
        self._destinations = [d for d in parsed.destinations if d not in parsed.nouns]

    def __call__(self, frame: np.ndarray) -> FrameGrounding:
        targets, nouns = self.parsed.targets, self.parsed.nouns
        timings: dict[str, float] = {}

        t0 = time.perf_counter()
        det = self.detector(frame, nouns + self._destinations, threshold=self.cfg.dino_threshold)
        synchronize(self.device)
        timings["detector"] = time.perf_counter() - t0
        

        t0 = time.perf_counter()
        masks, quality = self.segmenter(frame, det.boxes)
        synchronize(self.device)
        timings["segmenter"] = time.perf_counter() - t0

        # Split off destination candidates (best match is a destination phrase), and drop target
        # candidates whose mask is essentially a destination's mask.
        is_dest = det.scores.argmax(1) >= len(nouns) if len(det) else np.zeros(0, bool)
        dest_masks = masks[is_dest]
        not_dest = ~is_dest
        for i in np.nonzero(not_dest)[0]:
            if any(_iou(masks[i], d) > self.destination_iou for d in dest_masks):
                not_dest[i] = False
        masks, quality, boxes = masks[not_dest], quality[not_dest], det.boxes[not_dest]
        target_scores = det.scores[not_dest][:, :len(nouns)]

        keep = self._drop_parts(masks, target_scores)
        boxes, masks, quality, noun_scores = boxes[keep], masks[keep], quality[keep], target_scores[keep]

        t0 = time.perf_counter()
        crops = [masked_crop(frame, m) for m in masks]
        scores = self.scorer(crops, self._phrases + self.negatives)
        synchronize(self.device)
        timings["attributes"] = time.perf_counter() - t0

        phrase_scores, neg_scores = scores[:, :len(self._phrases)], scores[:, len(self._phrases):]
        siglip = phrase_scores[:, [self._phrases.index(t.phrase) for t in targets]]
        dino = noun_scores[:, [nouns.index(t.noun) for t in targets]]

        t0 = time.perf_counter()
        result = assign(targets, dino, siglip, neg_scores, self.cfg)
        timings["assign"] = time.perf_counter() - t0

        return FrameGrounding(boxes, masks, quality, dino, siglip, neg_scores, self.negatives, result, timings,
                              destination_masks=dest_masks)

    def _drop_parts(self, masks: np.ndarray, noun_scores: np.ndarray) -> np.ndarray:
        """Indices of candidates to keep: drops empty masks, and masks lying mostly inside a
        larger mask with the same best noun (a cup's rim inside the cup). Different nouns are
        kept, so a spoon resting in a bowl survives."""
        if len(masks) == 0:
            return np.zeros(0, dtype=int)
        areas = masks.reshape(len(masks), -1).sum(1)
        keep = areas > 0
        if len(masks) > 1:
            inside = containment(masks)
            best_noun = noun_scores.argmax(1)
            for i in range(len(masks)):
                for j in range(len(masks)):
                    if (i != j and keep[j] and areas[i] < areas[j] and best_noun[i] == best_noun[j]
                            and inside[i, j] > self.containment_threshold):
                        keep[i] = False
        return np.nonzero(keep)[0]


def _iou(a: np.ndarray, b: np.ndarray) -> float:
    union = np.logical_or(a, b).sum()
    return float(np.logical_and(a, b).sum() / union) if union else 0.0
