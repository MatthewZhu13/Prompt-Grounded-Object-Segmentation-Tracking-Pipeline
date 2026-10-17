"""End-to-end pipeline: task + episode → per-frame masks with persistent track IDs.

Per camera, the episode is processed in chunks (bounded memory). For each chunk:

  1. Track    SAM 2 follows the IDs that were visible at the end of the previous chunk.
  2. Detect   ground all targets on "refresh" frames (every N frames; denser for wrist cameras),
              skipping refresh frames where every target is already tracked.
  3. Correct  per ID, the first refresh frame where a detection should replace the tracker:
                seed /      ID not tracked there (never found yet, or lost) and the detection is
                re-acquire  strong (confident, composite ≥ seed_strong_score), or plausible
                            (≥ seed_min_score) on two consecutive refreshes in the same place;
                drift       ID tracked there but a strong, confident detection is elsewhere on
                            two consecutive refreshes.
              Detections sitting on another ID's tracked object are ignored (no ID stealing).
              An identity check on every refresh frame cuts tracks that stop looking like
              their target.
  4. Re-track SAM 2 from each correction frame, forward to the chunk end and backward only
              across the gap before it was re-detected (offline, so allowed).
  5. Resolve  if two IDs cover the same object, the newly corrected one yields.
  6. Update   each ID's state for the next chunk: tracking (carry its last mask) / lost / searching.

IDs come from the parsed task and never change; a re-found object is re-seeded under its own ID.

Frame indices: inside a chunk, `j` / `f` index the *extended* chunk (the previous chunk's last
frame prepended at position 0, when there is one); `_abs(j)` converts to the episode frame index.
"""

from __future__ import annotations

import json
import time
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import NamedTuple

import numpy as np

from . import console
from .assign import AssignConfig
from .boxes import clean_mask, covered, mask_iou, touches_edge
from .data import Dataset, Episode, iter_frames
from .device import device_name, get_device, synchronize
from .ground import FrameGrounder
from .io import MaskWriter, VideoWriter, write_json
from .parser import ParsedTask, Target, parse_task
from .track import VideoTracker
from .viz import SIDEBAR_WIDTH, render_track_frame

Masks = list[dict[int, np.ndarray]]         # per frame: {track id: mask}
Scores = list[dict[int, float]]             # per frame: {track id: SAM 2 presence score}


@dataclass
class PipelineConfig:
    chunk_size: int = 150
    refresh: dict[str, int] = field(default_factory=lambda: {"top": 15})   # camera → interval
    default_refresh: int = 5            # cameras not listed (wrist cameras: objects come and go)
    confirm_iou: float = 0.3            # two uncertain detections agree if their masks overlap this much
    drift_iou: float = 0.2              # tracked mask vs confident detection below this → drift
    conflict_iou: float = 0.5           # masks overlapping this much are the same object
    drift_min_score: float = 0.15       # a detection must score at least this (composite) to override a track
    seed_min_score: float = 0.08        # detections below this never start a track
    seed_strong_score: float = 0.15     # confident and at least this: start a track without confirmation
    near_window: int = 45               # frames: a lost ID's last mask counts as "where it just was"
    near_min_score: float = 0.02        # re-acquire threshold for a detection where the ID just was
    verify_failures: int = 2            # consecutive failed identity checks before a track is cut
    steady_iou: float = 0.3             # frame-to-frame IoU below this is a jump (track may have moved)
    # A detection candidate lying spill_inside within a track, making up at least spill_min of
    # it, means the track spread onto another object; the rest must be at least spill_keep.
    spill_inside: float = 0.5
    spill_min: float = 0.15
    spill_keep: float = 0.1
    # A close-up detection this similar to an earlier rejected object, and more similar to it
    # than to the ID's own memory, is ignored.
    veto_similarity: float = 0.9
    # A rejected candidate this similar to a target's own crops is that target misread, not a
    # distractor.
    known_similarity: float = 0.95
    start: int = 0
    max_frames: int | None = None
    write_video: bool = True
    assign: AssignConfig = field(default_factory=AssignConfig)

    def refresh_interval(self, camera: str) -> int:
        return self.refresh.get(camera, self.default_refresh)


@dataclass
class TrackState:
    target: Target
    status: str = "searching"                   # searching | tracking | lost
    last_seen: int | None = None                # episode frame index
    carry: np.ndarray | None = None             # mask on the last frame of the previous chunk
    pending: np.ndarray | None = None           # uncertain detection awaiting confirmation
    last_mask: np.ndarray | None = None         # mask on the last frame it was visible
    verify_fails: int = 0                       # consecutive failed identity checks so far


class Detection(NamedTuple):
    mask: np.ndarray
    status: str             # "confident" | "uncertain" (from assignment)
    score: float            # composite score
    at_edge: bool           # the candidate touches the frame border


class CameraTracker:
    """Runs the chunk loop for one camera, keeping per-ID state between chunks."""

    MAX_CORRECTION_ROUNDS = 4

    def __init__(self, camera: str, parsed: ParsedTask, grounder: FrameGrounder, tracker: VideoTracker,
                 cfg: PipelineConfig):
        self.camera, self.cfg = camera, cfg
        self.grounder, self.tracker = grounder, tracker
        self.interval = cfg.refresh_interval(camera)
        self.states = {t.id: TrackState(t) for t in parsed.targets}
        self.nouns = [t.noun for t in parsed.targets]
        self.prev_frame: np.ndarray | None = None
        self.log: list[dict] = []
        self.timing = defaultdict(float)
        self.refresh_count = 0
        self.skipped_refreshes = 0
        self._handed: set[tuple[int, int, int]] = set()     # hand-overs already logged
        # Appearance memory (SigLIP image embeddings), keyed by episode frame: each ID's crops on
        # identity checks it passed, and rejected detection candidates (bowls, plates...) as
        # distractors. Used only for close-ups, see _like_rejected.
        self.memory: dict[int, dict[int, np.ndarray]] = {t.id: {} for t in parsed.targets}
        self.distractors: dict[tuple[int, int], np.ndarray] = {}    # (frame, candidate) → embedding
        self.vetoed: set[tuple[int, int]] = set()       # (frame, track id): detection ignored as distractor
        self.excluded: set[tuple[int, int]] = set()     # (frame, track id): check passed because the
        #                                                 target(s) it read as are tracked elsewhere
        # Current chunk: episode frame indices, and 1 if the previous chunk's last frame is prepended.
        self.frame_idx = np.zeros(0, int)
        self.offset = 0

    # ── chunk loop ──────────────────────────────────────────────────────────────────────────

    def process(self, frames: np.ndarray, frame_idx: np.ndarray) -> tuple[Masks, Scores, list[dict[int, str]]]:
        """Returns per-frame (masks, scores, states) for this chunk."""
        # Prepend the previous chunk's last frame so carried masks seed on the exact frame they
        # came from; its outputs are dropped at the end.
        self.frame_idx, self.offset = frame_idx, int(self.prev_frame is not None)
        ext = np.concatenate([self.prev_frame[None], frames]) if self.offset else frames

        masks, scores = self._track_carried(ext)
        detections, candidates, destinations = self._detect(frames, masks)
        corrected = self._correct(ext, masks, scores, detections, candidates, destinations)
        self._resolve_duplicates(masks, scores, corrected)

        out_masks, out_scores = masks[self.offset:], scores[self.offset:]
        out_states = self._update_states(out_masks)
        self.prev_frame = frames[-1]
        return out_masks, out_scores, out_states

    def _track_carried(self, ext: np.ndarray) -> tuple[Masks, Scores]:
        """Step 1: track the IDs carried over from the previous chunk."""
        t0 = time.perf_counter()
        carried = {tid: (0, s.carry) for tid, s in self.states.items()
                   if self.offset and s.status == "tracking" and s.carry is not None}
        masks, scores = self.tracker.track(ext, carried, backward=False)
        self.timing["tracking"] += time.perf_counter() - t0
        return masks, scores

    def _detect(self, frames: np.ndarray, masks: Masks):
        """Step 2: ground all targets on refresh frames where some target is not tracked.

        Checking the tracker's state is free; detection costs ~1 s. Skipping it while every
        target is tracked means identity errors that arise then (two IDs swapping) are only
        caught once some target is lost again.

        Returns three dicts keyed by frame (in the extended chunk):
          detections    {track id: Detection}
          candidates    every candidate as (mask, SigLIP's best target noun, or None if rejected)
          destinations  destination masks (D, H, W)
        """
        t0 = time.perf_counter()
        detections: dict[int, dict[int, Detection]] = {}
        candidates: dict[int, list[tuple[np.ndarray, str | None]]] = {}
        destinations: dict[int, np.ndarray] = {}
        for i, f in enumerate(self.frame_idx):
            j = i + self.offset
            if f % self.interval != 0:
                continue
            if all(tid in masks[j] for tid in self.states):
                self.skipped_refreshes += 1
                continue
            g = self.grounder(frames[i])
            destinations[j] = g.destination_masks
            candidates[j] = [(g.masks[c], None if g.result.rejected[c] else self.nouns[int(g.siglip[c].argmax())])
                             for c in range(len(g.masks))]
            detections[j] = {}
            for m in g.result.matches:
                if m.candidate is None:
                    continue
                mask, emb = g.masks[m.candidate], g.embeddings[m.candidate]
                if mask.mean() >= self.grounder.context_min_frac and self._like_rejected(m.target_id, emb, int(f)):
                    continue
                detections[j][m.target_id] = Detection(clean_mask(mask, keep_largest=True), m.status,
                                                       float(np.exp(m.score)), touches_edge(mask))
            for c in np.nonzero(g.result.rejected)[0]:
                tracked_here = any(self._same_object(g.masks[c], m) for m in masks[j].values())
                if not tracked_here and not self._known(g.embeddings[c]):
                    self.distractors[(int(f), int(c))] = g.embeddings[c]
        self.refresh_count += len(detections)
        self.timing["grounding"] += time.perf_counter() - t0
        return detections, candidates, destinations

    def _correct(self, ext, masks: Masks, scores: Scores, detections, candidates, destinations) -> set[int]:
        """Steps 3–4: corrections, then re-track the corrected IDs. Repeated, because re-tracking
        can reveal later refresh frames where the same ID needs another correction (lost again).
        Returns the IDs corrected in this chunk."""
        t0 = time.perf_counter()
        self._cut_on_destination(masks, scores, destinations)
        after: dict[int, int] = {}          # track id → frame of its latest correction this chunk
        corrected: set[int] = set()
        start_pending = {tid: s.pending for tid, s in self.states.items()}
        end_pending = dict(start_pending)
        refresh_frames = [i + self.offset for i, f in enumerate(self.frame_idx) if f % self.interval == 0]
        verified: dict[tuple[int, int], tuple] = {}
        for _ in range(self.MAX_CORRECTION_ROUNDS):
            self._verify_tracks(ext, masks, scores, refresh_frames, verified)
            corrections, end_pending, cuts = self._corrections(detections, candidates, masks, after, corrected,
                                                               start_pending)
            if not corrections:
                break
            for tid, f in cuts.items():
                self._drop(masks, scores, tid, f)
                self._event(f, tid, "cut_mislabeled")
            prompts = {tid: (f, m, self._backfill_stop(masks, tid, f, kind))
                       for tid, (f, m, kind) in corrections.items()}
            retracked = self.tracker.track(ext, prompts)
            for tid, (f, _, kind) in corrections.items():
                self._apply_retrack(masks, scores, retracked, tid, f, kind)
                after[tid] = f
                corrected.add(tid)
            for tid, (f, _, kind) in corrections.items():
                self._event(f, tid, kind)
            self._cut_on_destination(masks, scores, destinations)
        self._verify_tracks(ext, masks, scores, refresh_frames, verified, final=True)
        for tid, pending in end_pending.items():
            self.states[tid].pending = pending
        synchronize(self.tracker.device)
        self.timing["tracking"] += time.perf_counter() - t0
        return corrected

    @staticmethod
    def _apply_retrack(masks: Masks, scores: Scores, retracked: tuple[Masks, Scores], tid: int, f: int,
                       kind: str) -> None:
        """Merge an ID's re-tracked masks (prompted at frame f) into the chunk's masks."""
        m2, s2 = retracked
        # The backward fill keeps only the unbroken run of frames just before f: once SAM 2 loses
        # the object going back (e.g. through motion blur), anything it finds further back is a
        # jump to another object.
        gap_start = next((j for j in range(f - 1, -1, -1) if tid not in m2[j]), -1)
        for j in range(len(masks)):
            if j <= gap_start and tid not in masks[j]:
                continue
            if j >= f:
                masks[j].pop(tid, None)
                scores[j].pop(tid, None)
            elif kind in ("drift", "trim") or tid in masks[j]:
                continue            # before the correction: keep earlier tracking where it existed
            if tid in m2[j]:
                masks[j][tid], scores[j][tid] = m2[j][tid], s2[j][tid]

    def _resolve_duplicates(self, masks: Masks, scores: Scores, corrected: set[int]) -> None:
        """Step 5: two IDs on one object, the one that was already on it stays.

        SAM 2 can move a lost ID onto a similar-looking object nearby (a white cup lifted out of
        view → the yellow cup next to it); the ID that arrived there from elsewhere is wrong.
        """
        for j in range(len(masks)):
            tids = list(masks[j])
            for a in tids:
                for b in tids:
                    if not (a < b and a in masks[j] and b in masks[j]
                            and self._same_object(masks[j][a], masks[j][b])):
                        continue
                    ca, cb = self._continuity(masks, j, a), self._continuity(masks, j, b)
                    if abs(ca - cb) > 0.2:
                        loser = a if ca < cb else b
                    elif (a in corrected) != (b in corrected):
                        loser = a if a in corrected else b
                    else:
                        loser = a if scores[j][a] < scores[j][b] else b
                    masks[j].pop(loser)
                    scores[j].pop(loser)

    def _update_states(self, out_masks: Masks) -> list[dict[int, str]]:
        """Step 6: per-frame states for the output, then each ID's state for the next chunk."""
        out_states = []
        for j, f in enumerate(self.frame_idx):
            frame_states = {}
            for tid, s in self.states.items():
                if tid in out_masks[j]:
                    s.last_seen = int(f)
                    s.last_mask = out_masks[j][tid]
                    frame_states[tid] = "tracking"
                else:
                    frame_states[tid] = "lost" if s.last_seen is not None else "searching"
            out_states.append(frame_states)
        for tid, s in self.states.items():
            last = out_masks[-1].get(tid)
            s.carry = clean_mask(last, keep_largest=True) if last is not None else None
            s.status = "tracking" if last is not None else ("lost" if s.last_seen is not None else "searching")
        return out_states

    # ── corrections ─────────────────────────────────────────────────────────────────────────

    def _corrections(self, detections, candidates, masks: Masks, after: dict[int, int], seen: set[int],
                     start_pending: dict[int, np.ndarray | None]):
        """Per ID, the first refresh frame (after its latest correction) needing a correction.

        Returns three dicts keyed by track id:
          corrections  (frame, mask, "seed" | "reacquire" | "drift" | "reclaim" | "trim")
          pending      uncertain detection still awaiting confirmation at the chunk end
          cuts         frame from which the track is mislabeled and must be cut
        """
        cfg = self.cfg
        detections = self._hand_over(detections, masks)
        out, pendings, cuts = {}, {}, {}
        for tid, s in self.states.items():
            pending = start_pending[tid] if tid not in after else None
            drift_pending = claim_pending = None
            for f in sorted(detections):
                if f <= after.get(tid, -1):
                    continue
                tracked = masks[f].get(tid)
                trimmed = self._trim_spill(tracked, candidates.get(f, []), s.target.noun) \
                    if tracked is not None else None
                if trimmed is not None:
                    out[tid] = (f, trimmed, "trim")
                    break
                det = detections[f].get(tid)
                if det is None:
                    pending = drift_pending = None
                    continue
                mask, status, score, at_edge = det
                if tracked is not None and mask_iou(tracked, mask) >= cfg.drift_iou:
                    pending = drift_pending = None      # tracker and detector agree
                    continue
                holders = [t for t, m in masks[f].items() if t != tid and self._same_object(mask, m)]
                if holders:
                    # The detection sits on an object tracked under another ID. Normally the
                    # tracker is trusted (no ID stealing). But if the detector strongly says this
                    # object is `tid`, does not itself put the holder's label there, and says so on
                    # two consecutive refreshes, the holder's track is the mislabeled one.
                    holder = holders[0]
                    holder_det = detections[f].get(holder)
                    holder_ok = (holder_det is not None
                                 and mask_iou(holder_det.mask, masks[f][holder]) >= cfg.drift_iou)
                    # A holder that has stayed on the same object without a break is trusted over
                    # the detector's color reading: in some views SigLIP consistently reads a pale
                    # gray cup as "white cup" and never finds "gray cup" at all.
                    # Likewise, an ID steadily tracked on another object is not moved by a
                    # detection: reclaiming is for IDs that are lost or have jumped.
                    steady = (self._steady(masks, holder, f)
                              or (tracked is not None and self._steady(masks, tid, f)))
                    if status == "confident" and score >= cfg.drift_min_score and not holder_ok and not steady:
                        if claim_pending is not None and mask_iou(claim_pending, mask) >= cfg.confirm_iou:
                            cuts[holder] = min(cuts.get(holder, f), f)
                            out[tid] = (f, mask, "reclaim")
                            break
                        claim_pending = mask
                    else:
                        claim_pending = None
                    pending = drift_pending = None
                    continue
                if tracked is not None:
                    # Overriding a working track needs more than finding a lost one: a strong,
                    # confident detection, confirmed in the same place on the next refresh.
                    if status == "confident" and score >= cfg.drift_min_score:
                        if drift_pending is not None and mask_iou(drift_pending, mask) >= cfg.confirm_iou:
                            out[tid] = (f, mask, "drift")
                            break
                        drift_pending = mask
                    else:
                        drift_pending = None
                    continue
                # Starting a track: "confident" is relative (nothing better competed), so also
                # require absolute strength. Weak-but-plausible detections need a second refresh
                # agreeing in the same place; a missed seed is cheap, a wrong one is tracked onward.
                # Where the ID just was (e.g. lost while the wrist camera zoomed in): a weak
                # detection overlapping or containing its last mask is very likely the same object.
                last = self._recent_seen(masks, tid, f)[0]
                if last is not None and score >= cfg.near_min_score and self._near(last, mask):
                    out[tid] = (f, mask, "reacquire")
                    pending = None
                    break
                if score < cfg.seed_min_score:
                    pending = None
                    continue
                # An object cut off by the frame edge shows only part of its shape: a yellow bowl
                # half out of view reads as "yellow cup". Such seeds always need confirmation.
                strong = status == "confident" and score >= cfg.seed_strong_score and not at_edge
                if strong or (pending is not None and mask_iou(pending, mask) >= cfg.confirm_iou):
                    out[tid] = (f, mask, "reacquire" if (s.last_seen is not None or tid in seen) else "seed")
                    pending = None
                    break
                pending = mask
            pendings[tid] = pending
        return out, pendings, cuts

    def _hand_over(self, detections, masks: Masks):
        """Give a detection to the same-noun ID that was just lost right there.

        A cup held in the gripper stays put in the wrist camera; if its track is lost (cut, or
        blurred) and the detector then names that same cup with a sibling attribute ("white cup"
        for a gray cup seen against the light), the ID lost on that spot moments ago is far more
        likely than one last seen elsewhere. Color readings flip with lighting; position does not.
        """
        out = {f: dict(d) for f, d in detections.items()}
        for f, dets in out.items():
            for tid, det in list(dets.items()):
                own = masks[f].get(tid)
                if own is None:
                    own = self._recent_seen(masks, tid, f)[0]
                if own is not None and self._near(own, det.mask):
                    continue
                noun = self.states[tid].target.noun
                best, best_seen = None, None
                for o, s in self.states.items():
                    if o == tid or s.target.noun != noun or o in masks[f]:
                        continue
                    last, seen = self._recent_seen(masks, o, f)
                    if last is None or not self._near(last, det.mask):
                        continue
                    if o in dets and not self._same_object(dets[o].mask, det.mask):
                        continue            # the detector placed o somewhere else
                    if best_seen is None or seen > best_seen:
                        best, best_seen = o, seen
                if best is None:
                    continue
                dets[best] = det
                del dets[tid]
                key = (self._abs(f), tid, best)
                if key not in self._handed:
                    self._handed.add(key)
                    self._event(f, best, "hand_over", from_track_id=tid)
        return out

    def _trim_spill(self, tracked: np.ndarray, candidates, noun: str) -> np.ndarray | None:
        """The tracked mask minus other objects it has spread onto, or None if it has not.

        SAM 2 can extend a mask over an object touching it, typically when the object comes
        back into view (a spoon reappearing under a cup's rim returns as spoon + cup, mostly
        cup). A detection candidate that lies mostly inside the track and that SigLIP reads as a
        different target noun is such an object; the remainder must still be a real piece.
        """
        area = tracked.sum()
        spill = np.zeros_like(tracked)
        for m, other_noun in candidates:
            if other_noun is None or other_noun == noun:
                continue        # rejected candidates can be parts of this very object (a cup's inside as "plate")
            if covered(m, tracked) > self.cfg.spill_inside:
                spill |= m
        if np.logical_and(spill, tracked).sum() < self.cfg.spill_min * area:
            return None
        trimmed = clean_mask(tracked & ~spill, keep_largest=True)
        return trimmed if trimmed.sum() >= self.cfg.spill_keep * area else None

    def _cut_on_destination(self, masks: Masks, scores: Scores, destinations) -> None:
        """Cut a track at the first refresh frame where its mask *is* the destination (high IoU).

        SAM 2 can spread a mask from a cup onto the clear container holding it. IoU, not
        coverage, is used: a cup inside the container is covered by the container's mask but
        is a legitimate, much smaller mask. The cut frees the ID for re-acquisition.
        """
        for f in sorted(destinations):
            for tid in list(masks[f]):
                if any(mask_iou(masks[f][tid], d) > self.cfg.conflict_iou for d in destinations[f]):
                    self._drop(masks, scores, tid, f)
                    self._event(f, tid, "cut_on_destination")

    # ── identity check ──────────────────────────────────────────────────────────────────────

    def _verify_tracks(self, ext, masks: Masks, scores: Scores, refresh_frames: list[int], cache,
                       final: bool = False) -> None:
        """Identity check on every refresh frame: cut a track whose mask stops looking like its
        target on `verify_failures` consecutive checks, from the first failed check up to the
        next check it passes.

        Catches a track that has slid onto another object (a spoon's mask onto a cup at the
        frame edge) when the real object is out of view, which the detector-based corrections
        cannot catch because there is no detection elsewhere to correct toward.
        """
        t0 = time.perf_counter()
        for f in refresh_frames:
            todo = {tid: m for tid, m in masks[f].items() if cache.get((f, tid), (None,))[0] is not m}
            if todo:
                for tid, (ok, emb, read_as) in self.grounder.verify(ext[f], todo).items():
                    cache[(f, tid)] = (todo[tid], ok, emb, read_as)
        # In frame order, so memory only holds crops from before the frame being judged.
        passed = {}
        for f in refresh_frames:
            a = self._abs(f)
            for tid in masks[f]:
                _, ok, emb, read_as = cache[(f, tid)]
                if ok:
                    self.memory[tid][a] = emb
                else:
                    self.memory[tid].pop(a, None)
                    if read_as is not None and self._accounted_for(read_as, tid, masks[f]):
                        ok = True
                        self.excluded.add((a, tid))
                passed[(f, tid)] = ok
        for tid, s in self.states.items():
            checks = [(f, passed[(f, tid)]) for f in refresh_frames if tid in masks[f]]
            fails, first = s.verify_fails, (0 if s.verify_fails else None)
            for k, (f, ok) in enumerate(checks):
                if ok:
                    fails, first = 0, None
                    continue
                fails += 1
                first = f if first is None else first
                if fails < self.cfg.verify_failures:
                    continue
                # Cut only the failing stretch: up to the next check this ID passes (e.g. a later
                # re-acquisition), not to the chunk end, which would erase that too...
                end = next((g for g, good in checks[k + 1:] if good), len(masks))
                # ...and back to just after the last check it passed: frames between that check
                # and the first failure were never verified and lead into the failure.
                start = next((g + 1 for g, good in reversed(checks[:k]) if good and g < first), 0)
                run = first
                while run > start and tid in masks[run - 1]:
                    run -= 1            # never past a gap: an earlier, separate stretch stands on its own
                self._drop(masks, scores, tid, run, end)
                self._event(first, tid, "cut_unverified")
                fails, first = 0, None
            if final:
                s.verify_fails = fails if tid in masks[-1] else 0
        self.timing["verify"] += time.perf_counter() - t0

    def _accounted_for(self, ids: tuple[int, ...], tid: int, frame_masks: dict[int, np.ndarray]) -> bool:
        """Every target in `ids` (the ones SigLIP read tid's mask as) is tracked on this frame
        on an object other than tid's.

        SigLIP misreads small utensils as each other (a fork 30 px wide from above as "spoon")
        and gray as white. If the spoon is tracked elsewhere, this mask cannot be the spoon, so
        the misreading is no evidence against it. A track that has slid onto another target
        overlaps that target's own mask, or that target is lost, so it is not excused."""
        others = [o for o in ids if o != tid and o in self.states]
        return bool(others) and all(o in frame_masks and not self._same_object(frame_masks[o], frame_masks[tid])
                                    for o in others)

    def _known(self, emb: np.ndarray) -> bool:
        """The crop closely matches some target's memory: a lost target that the detector
        rejected, not a distractor. Banking it would make that target's own appearance count
        against it from then on."""
        mem = [e for m in self.memory.values() for e in m.values()]
        return bool(mem) and float((np.stack(mem) @ emb).max()) >= self.cfg.known_similarity

    def _like_rejected(self, tid: int, emb: np.ndarray, frame: int) -> bool:
        """The detection looks like an object rejected on an earlier refresh (a bowl that SigLIP
        read as "bowl" there) more than like this ID's own earlier crops. A bowl close-up reads as
        "cup" on some frames and "bowl" on most; without this, each "cup" frame re-seeds it.
        Applied to close-ups only: from far away, different objects of one color look alike (a
        white cup and the white bowl from above), and an ID with no memory yet would never seed."""
        rejected = [e for (a, _), e in self.distractors.items() if a < frame]
        if not rejected:
            return False
        sim_rejected = float((np.stack(rejected) @ emb).max())
        if sim_rejected < self.cfg.veto_similarity:
            return False
        own = [e for a, e in self.memory[tid].items() if a < frame]
        sim_own = float((np.stack(own) @ emb).max()) if own else 0.0
        if sim_rejected > sim_own:
            self.vetoed.add((frame, tid))
            return True
        return False

    # ── helpers ─────────────────────────────────────────────────────────────────────────────

    def _event(self, j: int, tid: int, kind: str, **extra) -> None:
        """Log a tracker decision for events.jsonl."""
        self.log.append({"camera": self.camera, "frame_index": self._abs(j), "track_id": tid,
                         "label": self.states[tid].target.label, "event": kind, **extra})

    @staticmethod
    def _drop(masks: Masks, scores: Scores, tid: int, start: int, end: int | None = None) -> None:
        """Remove an ID's masks on frames [start, end) (default: to the chunk end)."""
        for k in range(start, len(masks) if end is None else end):
            masks[k].pop(tid, None)
            scores[k].pop(tid, None)

    def _near(self, last: np.ndarray, mask: np.ndarray) -> bool:
        """`mask` is where `last` was: overlapping it, or containing most of it."""
        return mask_iou(last, mask) >= self.cfg.confirm_iou or covered(last, mask) > 0.5

    def _recent_seen(self, masks: Masks, tid: int, f: int) -> tuple[np.ndarray | None, int | None]:
        """The ID's most recent mask within `near_window` frames before f (this chunk or the
        last), and the episode frame it was seen on; (None, None) if there is none."""
        for j in range(f - 1, max(f - 1 - self.cfg.near_window, -1), -1):
            if tid in masks[j]:
                return masks[j][tid], self._abs(j)
        s = self.states[tid]
        if s.last_mask is not None and s.last_seen is not None:
            if self._abs(f) - s.last_seen <= self.cfg.near_window:
                return s.last_mask, s.last_seen
        return None, None

    @staticmethod
    def _backfill_stop(masks: Masks, tid: int, f: int, kind: str) -> int | None:
        """How far back to re-track a corrected ID (None: not at all).

        The backward pass only fills the gap before the correction: frames that already have
        this ID keep their earlier tracking, so computing them again would be thrown away.
          drift/trim → None: frames before keep the original track
          re-acquire → the frame after the ID was last visible before f
          seed       → the chunk start: it may have been in view but not yet identified
        """
        if kind in ("drift", "trim"):
            return None
        for j in range(f - 1, -1, -1):
            if tid in masks[j]:
                return j + 1
        return 0

    def _steady(self, masks: Masks, tid: int, f: int) -> bool:
        """The ID has been on one object without a break from the chunk start (or its start in
        this chunk) up to frame f: present on every frame, overlapping itself frame to frame."""
        for j in range(f, 0, -1):
            prev = masks[j - 1].get(tid)
            if prev is None:
                # Absent before: a track that started here (a seed) is steady; one that went
                # missing and came back may have come back on a different object.
                return not any(tid in masks[k] for k in range(j - 1))
            if mask_iou(prev, masks[j][tid]) < self.cfg.steady_iou:
                return False
        return True

    @staticmethod
    def _continuity(masks: Masks, j: int, tid: int) -> float:
        """IoU of an ID's mask with its own mask on the previous frame (0 if it wasn't there)."""
        prev = masks[j - 1].get(tid) if j > 0 else None
        return mask_iou(prev, masks[j][tid]) if prev is not None else 0.0

    def _same_object(self, a: np.ndarray, b: np.ndarray) -> bool:
        """a and b cover the same object: high IoU, or one covers most of the other (a detection
        box merging two touching objects covers the smaller one entirely at low IoU)."""
        inter = np.logical_and(a, b).sum()
        smaller = min(a.sum(), b.sum())
        return mask_iou(a, b) > self.cfg.conflict_iou or (smaller > 0 and inter / smaller > self.cfg.conflict_iou)

    def _abs(self, j: int) -> int:
        """Episode frame index of frame j in the extended chunk."""
        return int(self.frame_idx[j - self.offset]) if j >= self.offset else int(self.frame_idx[0]) - 1


# ── episode driver ──────────────────────────────────────────────────────────────────────────

def run_episode(
    task: str,
    episode_index: int,
    output_dir: str | Path,
    cameras: list[str] | None = None,
    cfg: PipelineConfig | None = None,
    data_dir: str | Path | None = None,
    repo_id: str | None = None,
    device: str | None = None,
) -> dict:
    cfg = cfg or PipelineConfig()
    out = Path(output_dir) / f"episode_{episode_index:03d}"
    out.mkdir(parents=True, exist_ok=True)
    t_start = time.perf_counter()

    parsed = parse_task(task)
    write_json(out / "targets.json", parsed.to_dict())
    ds = Dataset(repo_id, data_dir) if repo_id else Dataset(data_dir=data_dir)
    episode = ds.episode(episode_index, cameras=cameras)
    dev = get_device(device)
    console.header(task, parsed, device_name(dev))

    t0 = time.perf_counter()
    grounder = FrameGrounder(parsed, cfg.assign, device=dev)
    tracker = VideoTracker(device=dev)
    t_load = time.perf_counter() - t0
    console.models_loaded(t_load)

    stats = {"task": task, "episode_index": episode_index, "device": device_name(dev),
             "model_load_s": round(t_load, 1), "cameras": {}}
    log: list[dict] = []
    masks_writer = MaskWriter(out / "masks.parquet")
    try:
        for camera in episode.cameras:
            cam = CameraTracker(camera, parsed, grounder, tracker, cfg)
            stats["cameras"][camera] = _run_camera(cam, episode, cfg, masks_writer, out)
            log += cam.log
    finally:
        masks_writer.close()

    stats["total_seconds"] = round(time.perf_counter() - t_start, 1)
    total_frames = sum(c["frames"] for c in stats["cameras"].values())
    stats["sec_per_frame_all_cameras"] = round(stats["total_seconds"] / max(total_frames, 1), 3)
    write_json(out / "stats.json", stats)
    (out / "events.jsonl").write_text("".join(json.dumps(e) + "\n" for e in log))
    console.summary(stats, {t.id: t.label for t in parsed.targets}, out)
    return stats


def _run_camera(cam: CameraTracker, episode: Episode, cfg: PipelineConfig, masks_writer: MaskWriter,
                out: Path) -> dict:
    """Track one camera through the episode, writing masks and video. Returns its stats."""
    camera, labels = cam.camera, {tid: s.target.label for tid, s in cam.states.items()}
    stop = episode.length if cfg.max_frames is None else min(cfg.start + cfg.max_frames, episode.length)
    console.camera_start(camera, stop - cfg.start, cam.interval)

    video = None
    visible = defaultdict(int)
    n_frames, t_decode, t_write = 0, 0.0, 0.0
    t_cam = t0 = time.perf_counter()
    for chunk in iter_frames(episode, camera, chunk_size=cfg.chunk_size, start=cfg.start, stop=stop):
        t_decode += time.perf_counter() - t0
        n_events = len(cam.log)
        masks, scores, states = cam.process(chunk.frames, chunk.frame_indices)

        t0 = time.perf_counter()
        h, w = chunk.frames.shape[1:3]
        cleaned = [_clean_frame(m) for m in masks]
        masks_writer.write_frames(episode.index, camera, chunk.frame_indices.tolist(), cleaned,
                                  scores, states, labels, (h, w))
        if cfg.write_video:
            video = video or VideoWriter(out / f"{camera}.mp4", episode.fps, w + SIDEBAR_WIDTH, h)
            for frame, f, m, st in zip(chunk.frames, chunk.frame_indices, cleaned, states):
                video.write(render_track_frame(frame, camera, int(f), m, st, labels))
        for m in cleaned:
            for tid in m:
                visible[tid] += 1
        n_frames += len(chunk.frames)
        t_write += time.perf_counter() - t0
        console.chunk(int(chunk.frame_indices[0]), int(chunk.frame_indices[-1]),
                      [labels[t] for t in sorted(cleaned[-1])], cam.log[n_events:])
        t0 = time.perf_counter()
    if video:
        video.close()

    elapsed = time.perf_counter() - t_cam
    console.camera_done(n_frames, elapsed)

    def per_target(count) -> dict[str, float]:
        return {labels[tid]: count(tid) for tid in labels}

    return {
        "frames": n_frames,
        "seconds": round(elapsed, 1),
        "sec_per_frame": round(elapsed / max(n_frames, 1), 3),
        "stage_seconds": {
            "decode": round(t_decode, 1),
            "grounding": round(cam.timing["grounding"], 1),
            "verify": round(cam.timing["verify"], 1),
            "tracking": round(cam.timing["tracking"], 1),
            "write": round(t_write, 1),
        },
        "refresh_interval": cam.interval,
        "refresh_frames": cam.refresh_count,
        "refreshes_skipped_all_tracked": cam.skipped_refreshes,
        "events": {k: sum(e["event"] == k for e in cam.log) for k in sorted({e["event"] for e in cam.log})},
        "visible_fraction": per_target(lambda tid: round(visible[tid] / max(n_frames, 1), 3)),
        "identity_checks_passed_by_exclusion": per_target(lambda tid: sum(t == tid for _, t in cam.excluded)),
        "detections_vetoed_as_distractor": per_target(lambda tid: sum(t == tid for _, t in cam.vetoed)),
    }


def _clean_frame(masks: dict[int, np.ndarray]) -> dict[int, np.ndarray]:
    """Drop specks and give overlapping pixels to the smaller mask (usually the object in front,
    e.g. a spoon resting in a cup), so each pixel belongs to at most one track."""
    cleaned = {tid: clean_mask(m) for tid, m in masks.items()}
    for tid in sorted(cleaned, key=lambda t: cleaned[t].sum()):
        for other in cleaned:
            if other != tid and cleaned[other].sum() > cleaned[tid].sum():
                cleaned[other] = cleaned[other] & ~cleaned[tid]
    return {tid: m for tid, m in cleaned.items() if m.any()}
