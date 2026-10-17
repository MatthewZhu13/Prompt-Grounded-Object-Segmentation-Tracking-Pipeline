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
from .attributes import AttributeScorer, build_negatives, context_crop, masked_crop
from .boxes import containment, mask_iou, mask_to_box
from .detector import GroundingDetector
from .device import get_device, synchronize
from .parser import ParsedTask, Target
from .segmenter import BoxSegmenter

OCCLUDERS = ["robot gripper"]
# Lightness-only color words. Glare, blur and shadow move a crop along this axis (a gray or
# yellow cup on a wrist camera reads "white cup") but never add hue, so reading a track as
# another instance whose distinguishing color is one of these is not evidence against it.
ACHROMATIC = {"white", "gray", "grey", "black", "silver", "clear", "transparent"}


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
    # (D, H, W) destination masks, and (C, E) SigLIP image embeddings of the candidates
    destination_masks: np.ndarray = field(default_factory=lambda: np.zeros((0, 0, 0), bool))
    embeddings: np.ndarray = field(default_factory=lambda: np.zeros((0, 0), np.float32))

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
        min_after_occluder: float = 0.25,
        split_min_piece: float = 0.01,
        split_max_cover: float = 0.75,
        split_max_overlap: float = 0.2,
        context_min_frac: float = 0.15,
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
        # Per target noun: its own phrases, and the negatives that share the noun ("gray bowl"
        # for a task with white and yellow bowls), see build_negatives.
        self._own_phrases = {
            n: np.array([i for i, p in enumerate(self._phrases) if p == n or p.endswith(" " + n)])
            for n in parsed.nouns
        }
        self._same_noun_neg = {
            n: np.array([i for i, p in enumerate(self.negatives) if p.endswith(" " + n)], int)
            for n in parsed.nouns
        }
        # Destinations ("the container") are queried too: they are in the scene by definition and
        # are the main thing target masks get confused with (cups seen inside a clear container).
        self._destinations = [d for d in parsed.destinations if d not in parsed.nouns]
        # The robot's own gripper is in every wrist-camera frame and touches what it grasps.
        self._occluders = [o for o in OCCLUDERS if o not in parsed.nouns]
        self.min_after_occluder = min_after_occluder
        # Splitting merged candidates (see _split_merged): pieces are at least split_min_piece of
        # the frame; try a split when the first object covers less than split_max_cover of the
        # mask, keep it when the second overlaps the first less than split_max_overlap.
        self.split_min_piece = split_min_piece
        self.split_max_cover = split_max_cover
        self.split_max_overlap = split_max_overlap
        # Masks at least this fraction of the frame (an object held up to a wrist camera) get
        # the "is it something else, e.g. a bowl?" test on an unblanked crop instead of the
        # masked one. Only for such close-ups: on smaller objects the unblanked crop shows the
        # neighbours too (cups packed in the container read as "plate" / "mug").
        self.context_min_frac = context_min_frac

    def verify(self, frame: np.ndarray, masks: dict[int, np.ndarray]
               ) -> dict[int, tuple[bool, np.ndarray, tuple[int, ...] | None]]:
        """Does each tracked mask still look like its target? (SigLIP only, no detection.)

        A track fails if SigLIP's best target phrase for the masked crop is another target: a
        different noun (a "spoon" mask that looks like a cup) or another attributed instance of
        the same noun (a "white cup" mask that reads "yellow cup" — a track that slid onto the
        yellow cup in the container). For close-ups only, it also fails if a negative phrase
        beats every target phrase on the unblanked crop (a held "white cup" that is a white
        bowl). Cheap: one or two crops per tracked object.
        Returns {track id: (passed, the crop's SigLIP image embedding, the ids of the targets
        SigLIP read the crop as if *only* that test failed, else None)}. The caller may excuse
        the failure when all of those targets are tracked elsewhere on the frame.
        """
        if not masks:
            return {}
        tids = list(masks)
        emb = self.scorer.embed([masked_crop(frame, masks[t]) for t in tids])
        scores = self.scorer.score(emb, self._phrases + self.negatives)
        ctx = self._context_scores(frame, np.stack([masks[t] for t in tids]))
        n = len(self._phrases)
        by_id = {t.id: t for t in self.parsed.targets}
        out = {}
        for k, tid in enumerate(tids):
            target, negative = scores[k, :n], scores[k, n:]
            if k in ctx:
                # Close-up: judged on the unblanked crop, also at the frame edge.
                target, negative = ctx[k][:n], ctx[k][n:]
                negative_ok = not (len(negative) and negative.max() > target.max())
            else:
                # Not a close-up: whether a small masked crop is a cup or a bowl is a coin flip
                # for SigLIP (it cut correct cups as often as wrong ones across episodes), so
                # cross-noun negatives are left to detection and the close-up test. Color
                # survives small crops, so a same-noun negative ("gray bowl" against a "white
                # bowl" track) still counts, judged against the track's own noun phrases.
                noun = by_id[tid].noun
                same, own = self._same_noun_neg[noun], self._own_phrases[noun]
                negative_ok = not (len(same) and negative[same].max() > target[own].max())
            read_as = self._read_as(by_id[tid], self._phrases[int(target.argmax())], close_up=k in ctx)
            out[tid] = (not read_as and negative_ok, emb[k], read_as if negative_ok and read_as else None)
        return out

    def _read_as(self, own: Target, best_phrase: str, close_up: bool) -> tuple[int, ...]:
        """IDs of the other targets a crop of `own` was read as, given SigLIP's best target
        phrase for it; () if the reading is consistent with `own`."""
        targets = self.parsed.targets
        best_noun = next(t.noun for t in targets if t.phrase == best_phrase)
        if best_noun != own.noun:
            return tuple(t.id for t in targets if t.noun == best_noun)
        if close_up or not own.attributes or best_phrase in (own.phrase, own.noun):
            return ()
        # Same noun, another instance's hue: a "white cup" track that slid onto the yellow cup
        # in the container reads "yellow cup". Only a reading that adds a hue counts (see
        # ACHROMATIC), and never on close-ups, where wrist-camera glare decides the colour.
        # A bare-noun reading ("cup") claims nothing about colour and passes.
        their_attributes = {a for t in targets if t.phrase == best_phrase for a in t.attributes}
        if their_attributes - set(own.attributes) - ACHROMATIC:
            return tuple(t.id for t in targets if t.phrase == best_phrase)
        return ()

    def __call__(self, frame: np.ndarray) -> FrameGrounding:
        targets, nouns = self.parsed.targets, self.parsed.nouns
        timings: dict[str, float] = {}

        t0 = time.perf_counter()
        queries = nouns + self._destinations + self._occluders
        det = self.detector(frame, queries, threshold=self.cfg.dino_threshold)
        synchronize(self.device)
        timings["detector"] = time.perf_counter() - t0

        t0 = time.perf_counter()
        masks, quality = self.segmenter(frame, det.boxes)
        synchronize(self.device)
        timings["segmenter"] = time.perf_counter() - t0

        # Each box's role is its best phrase: a target noun, a destination, or the robot gripper.
        n_t, n_d = len(nouns), len(self._destinations)
        role = det.scores.argmax(1) if len(det) else np.zeros(0, dtype=int)
        is_grip = role >= n_t + n_d
        is_dest = (role >= n_t) & ~is_grip

        # Destination and gripper boxes must also look like that to SigLIP. Up close, Grounding
        # DINO can score a large cup higher for "container" than for "cup", and its "robot
        # gripper" box often lands on the cup the gripper is holding (whose mask is the cup).
        # Unconfirmed boxes go back to being ordinary target candidates.
        special = np.nonzero(is_dest | is_grip)[0]
        relabeled = np.zeros(len(det), dtype=bool)
        if len(special):
            n_p = len(self._phrases)
            labels = self._phrases + self._destinations + self._occluders
            sc = self.scorer([masked_crop(frame, masks[i]) for i in special], labels)
            for k, i in enumerate(special):
                own = sc[k, n_p:n_p + n_d] if is_dest[i] else sc[k, n_p + n_d:]
                if sc[k, :n_p].max() > own.max():
                    is_dest[i] = is_grip[i] = False
                    relabeled[i] = True
        dest_masks = masks[is_dest]

        # Remove gripper pixels from target candidates: SAM 2 often merges an object with the
        # finger touching it, and a seed mask that includes the finger gets the gripper tracked.
        cand = ~is_dest & ~is_grip
        gripper = masks[is_grip].any(axis=0) if is_grip.any() else None
        for i in np.nonzero(cand)[0]:
            if gripper is not None:
                before = masks[i].sum()
                masks[i] &= ~gripper
                if masks[i].sum() < self.min_after_occluder * before:
                    cand[i] = False         # the candidate was mostly gripper
                    continue
            if any(mask_iou(masks[i], d) > self.destination_iou for d in dest_masks):
                cand[i] = False             # the candidate is the destination itself
        # Relabeled boxes: Grounding DINO found an object there but named it wrongly, so its
        # score for the target nouns is low by construction. Use its overall confidence that an
        # object is there for every noun, and let SigLIP (noun agreement + attributes) decide.
        noun_scores_all = det.scores[:, :n_t].copy()
        noun_scores_all[relabeled] = det.scores[relabeled].max(axis=1, keepdims=True)
        masks, quality, boxes = masks[cand], quality[cand], det.boxes[cand]
        target_scores = noun_scores_all[cand]

        t0 = time.perf_counter()
        masks, src = self._split_merged(frame, masks)
        quality, target_scores = quality[src], target_scores[src]
        boxes = np.array([mask_to_box(m) for m in masks]).reshape(-1, 4)
        synchronize(self.device)
        timings["split"] = time.perf_counter() - t0

        keep = self._drop_parts(masks, target_scores)
        boxes, masks, quality, noun_scores = boxes[keep], masks[keep], quality[keep], target_scores[keep]

        t0 = time.perf_counter()
        crops = [masked_crop(frame, m) for m in masks]
        emb = self.scorer.embed(crops)
        scores = self.scorer.score(emb, self._phrases + self.negatives)
        n_p = len(self._phrases)
        # Close-ups are judged on the unblanked crop instead: up close the masked outline alone
        # does not tell a cup from a bowl (see context_crop), and the object fills the crop
        # anyway, so the surroundings barely touch the color reading.
        for k, row in self._context_scores(frame, masks).items():
            scores[k] = row
        synchronize(self.device)
        timings["attributes"] = time.perf_counter() - t0

        cols = [self._phrases.index(t.phrase) for t in targets]
        phrase_scores, neg_scores = scores[:, :n_p], scores[:, n_p:]
        siglip = phrase_scores[:, cols]
        dino = noun_scores[:, [nouns.index(t.noun) for t in targets]]

        t0 = time.perf_counter()
        result = assign(targets, dino, siglip, neg_scores, self.cfg)
        timings["assign"] = time.perf_counter() - t0

        return FrameGrounding(boxes, masks, quality, dino, siglip, neg_scores, self.negatives, result,
                              timings, destination_masks=dest_masks, embeddings=emb)

    def _context_scores(self, frame: np.ndarray, masks: np.ndarray) -> dict[int, np.ndarray]:
        """{index: SigLIP scores (phrases + negatives) on the unblanked crop} for close-up masks."""
        if not len(masks):
            return {}
        big = np.nonzero(masks.reshape(len(masks), -1).mean(axis=1) >= self.context_min_frac)[0]
        if not len(big):
            return {}
        emb = self.scorer.embed([context_crop(frame, masks[k]) for k in big])
        sc = self.scorer.score(emb, self._phrases + self.negatives)
        return {int(k): sc[j] for j, k in enumerate(big)}

    def _split_merged(self, frame: np.ndarray, masks: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Split candidate masks that cover two touching objects. Returns (masks, source index).

        Grounding DINO sometimes puts one box around two touching objects (a yellow cup resting
        on a white cup), and the box's mask then covers both; tracked, that mask carries one ID
        across both objects. Test: prompt SAM 2 with a point at the mask's most interior pixel
        and, if the object found there covers only part of the mask, with a second point in the
        remainder. The split is kept only if each point's object leaves out the other's
        (`split_max_overlap`): a point on a cup's rim usually returns the whole cup, so one
        object's parts stay one candidate. Which piece is which target is left to assignment,
        which compares pieces against each other (SigLIP's absolute color reading of a dim white
        cup next to a yellow one can itself be "yellow").
        """
        h, w = frame.shape[:2]
        min_area = self.split_min_piece * h * w
        firsts = [(i, _interior_point(m)) for i, m in enumerate(masks) if m.sum() >= 2 * min_area]
        pairs = []
        if firsts:
            pieces = self.segmenter.points(frame, [p for _, p in firsts])
            seconds = []
            for (i, _), piece in zip(firsts, pieces):
                a = piece & masks[i]
                rest = masks[i] & ~a
                big_enough = a.sum() >= min_area and rest.sum() >= min_area
                if big_enough and a.sum() < self.split_max_cover * masks[i].sum():
                    seconds.append((i, a, _interior_point(rest)))
            if seconds:
                pieces = self.segmenter.points(frame, [p for _, _, p in seconds])
                for (i, a, _), piece in zip(seconds, pieces):
                    b = piece & masks[i] & ~a
                    if b.sum() >= min_area and (piece & a).sum() < self.split_max_overlap * a.sum():
                        pairs.append((i, a, b))
        if not pairs:
            return masks, np.arange(len(masks))
        split = {i: (a, b) for i, a, b in pairs}
        out, src = [], []
        for i, m in enumerate(masks):
            for piece in split.get(i, (m,)):
                out.append(piece)
                src.append(i)
        return np.stack(out), np.array(src)

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


def _interior_point(mask: np.ndarray) -> tuple[int, int]:
    """(x, y) of the mask pixel farthest from its boundary (the frame edge counts as boundary)."""
    from scipy import ndimage

    d = ndimage.distance_transform_edt(np.pad(mask, 1))[1:-1, 1:-1]
    y, x = np.unravel_index(int(d.argmax()), d.shape)
    return int(x), int(y)

