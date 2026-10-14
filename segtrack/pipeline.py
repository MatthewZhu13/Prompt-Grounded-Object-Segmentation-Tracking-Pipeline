"""End-to-end pipeline: task + episode → per-frame masks with persistent track IDs.

Per camera, the episode is processed in chunks (bounded memory). For each chunk:

  1. Detect   ground all targets on "refresh" frames (every N frames; denser for wrist cameras).
  2. Track    SAM 2 follows the IDs that were visible at the end of the previous chunk.
  3. Correct  per ID, the first refresh frame where a detection should replace the tracker:
                seed /      ID not tracked there (never found yet, or lost) and the detection is
                re-acquire  strong (confident, composite ≥ seed_strong_score), or plausible
                            (≥ seed_min_score) on two consecutive refreshes in the same place;
                drift       ID tracked there but a strong, confident detection is elsewhere on
                            two consecutive refreshes.
              Detections sitting on another ID's tracked object are ignored (no ID stealing).
  4. Re-track SAM 2 from each correction frame, forward to the chunk end and backward to fill
              the frames before it was re-detected (offline, so allowed).
  5. Resolve  if two IDs cover the same object, the newly corrected one yields.
  6. Update   each ID's state for the next chunk: tracking (carry its last mask) / lost / searching.

IDs come from the parsed task and never change; a re-found object is re-seeded under its own ID.
"""

from __future__ import annotations

import json
import time
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from .assign import AssignConfig
from .boxes import clean_mask
from .data import Dataset, iter_frames
from .device import device_name, get_device, synchronize
from .ground import FrameGrounder
from .io import MaskWriter, VideoWriter, write_json
from .parser import ParsedTask, Target, parse_task
from .track import VideoTracker
from .viz import annotate_frame


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


def iou(a: np.ndarray, b: np.ndarray) -> float:
    union = np.logical_or(a, b).sum()
    return float(np.logical_and(a, b).sum() / union) if union else 0.0


class CameraTracker:
    """Runs the chunk loop for one camera, keeping per-ID state between chunks."""

    MAX_CORRECTION_ROUNDS = 4

    def __init__(self, camera: str, parsed: ParsedTask, grounder: FrameGrounder, tracker: VideoTracker,
                 cfg: PipelineConfig):
        self.camera, self.cfg = camera, cfg
        self.grounder, self.tracker = grounder, tracker
        self.interval = cfg.refresh_interval(camera)
        self.states = {t.id: TrackState(t) for t in parsed.targets}
        self.prev_frame: np.ndarray | None = None
        self.log: list[dict] = []
        self.timing = defaultdict(float)
        self.refresh_count = 0

    def process(self, frames: np.ndarray, frame_idx: np.ndarray):
        """Returns per-frame (masks, scores, states) for this chunk."""
        cfg = self.cfg
        # Prepend the previous chunk's last frame so carried masks seed on the exact frame they
        # came from; its outputs are dropped at the end.
        offset = 1 if self.prev_frame is not None else 0
        ext = np.concatenate([self.prev_frame[None], frames]) if offset else frames

        # 1. Detect on refresh frames (indices into `ext`).
        t0 = time.perf_counter()
        detections: dict[int, dict[int, tuple[np.ndarray, str, float]]] = {}
        destinations: dict[int, np.ndarray] = {}
        for i, f in enumerate(frame_idx):
            if f % self.interval == 0:
                g = self.grounder(frames[i])
                destinations[i + offset] = g.destination_masks
                detections[i + offset] = {
                    m.target_id: (clean_mask(g.masks[m.candidate], keep_largest=True), m.status, float(np.exp(m.score)))
                    for m in g.result.matches if m.candidate is not None
                }
        self.refresh_count += len(detections)
        self.timing["grounding"] += time.perf_counter() - t0

        # 2. Track IDs carried over from the previous chunk.
        t0 = time.perf_counter()
        carried = {tid: (0, s.carry) for tid, s in self.states.items()
                   if offset and s.status == "tracking" and s.carry is not None}
        masks, scores = self.tracker.track(ext, carried, backward=False)
        self._cut_on_destination(masks, scores, destinations, frame_idx, offset)

        # 3–4. Corrections, then re-track the corrected IDs. Repeated, because re-tracking can
        # reveal later refresh frames where the same ID needs another correction (lost again).
        after: dict[int, int] = {}          # track id → frame of its latest correction this chunk
        corrected: set[int] = set()
        start_pending = {tid: s.pending for tid, s in self.states.items()}
        end_pending = dict(start_pending)
        for _ in range(self.MAX_CORRECTION_ROUNDS):
            corrections, end_pending, cuts = self._corrections(detections, masks, after, corrected, start_pending)
            if not corrections:
                break
            for tid, f in cuts.items():
                for k in range(f, len(masks)):
                    masks[k].pop(tid, None)
                    scores[k].pop(tid, None)
                self.log.append({"camera": self.camera, "frame_index": int(self._abs(f, frame_idx, offset)),
                                 "track_id": tid, "label": self.states[tid].target.label, "event": "cut_mislabeled"})
            m2, s2 = self.tracker.track(ext, {tid: (f, m) for tid, (f, m, _) in corrections.items()}, backward=True)
            for tid, (f, _, kind) in corrections.items():
                for j in range(len(ext)):
                    if j >= f:
                        masks[j].pop(tid, None)
                        scores[j].pop(tid, None)
                    elif kind == "drift" or tid in masks[j]:
                        continue            # before the correction: keep earlier tracking where it existed
                    if tid in m2[j]:
                        masks[j][tid], scores[j][tid] = m2[j][tid], s2[j][tid]
                after[tid] = f
                corrected.add(tid)
            for tid, (f, _, kind) in corrections.items():
                self.log.append({"camera": self.camera, "frame_index": int(self._abs(f, frame_idx, offset)),
                                 "track_id": tid, "label": self.states[tid].target.label, "event": kind})
            self._cut_on_destination(masks, scores, destinations, frame_idx, offset)
        for tid, pending in end_pending.items():
            self.states[tid].pending = pending
        synchronize(self.tracker.device)
        self.timing["tracking"] += time.perf_counter() - t0

        # 5. Two IDs on one object: the one that was already on it yields last. SAM 2 can move
        # a lost ID onto a similar-looking object nearby (a white cup lifted out of view → the
        # yellow cup next to it); the ID that arrived there from elsewhere is the wrong one.
        for j in range(len(ext)):
            tids = list(masks[j])
            for a in tids:
                for b in tids:
                    if a < b and a in masks[j] and b in masks[j] and self._same_object(masks[j][a], masks[j][b]):
                        ca, cb = self._continuity(masks, j, a), self._continuity(masks, j, b)
                        if abs(ca - cb) > 0.2:
                            loser = a if ca < cb else b
                        elif (a in corrected) != (b in corrected):
                            loser = a if a in corrected else b
                        else:
                            loser = a if scores[j][a] < scores[j][b] else b
                        masks[j].pop(loser)
                        scores[j].pop(loser)

        out_masks, out_scores = masks[offset:], scores[offset:]

        # 6. Per-frame states, then state for the next chunk.
        out_states = []
        for j, f in enumerate(frame_idx):
            frame_states = {}
            for tid, s in self.states.items():
                if tid in out_masks[j]:
                    s.last_seen = int(f)
                    frame_states[tid] = "tracking"
                else:
                    frame_states[tid] = "lost" if s.last_seen is not None else "searching"
            out_states.append(frame_states)
        for tid, s in self.states.items():
            last = out_masks[-1].get(tid)
            s.carry = clean_mask(last, keep_largest=True) if last is not None else None
            s.status = "tracking" if last is not None else ("lost" if s.last_seen is not None else "searching")
        self.prev_frame = frames[-1]
        return out_masks, out_scores, out_states

    def _corrections(self, detections, masks, after: dict[int, int], seen: set[int],
                     start_pending: dict[int, np.ndarray | None]):
        """Per ID, the first refresh frame (after its latest correction) needing a correction.
        Returns ({track id: (frame in ext, mask, "seed" | "reacquire" | "drift" | "reclaim")},
                 {track id: uncertain detection still awaiting confirmation at the chunk end},
                 {track id: frame from which its track is mislabeled and must be cut})."""
        cfg = self.cfg
        out, pendings, cuts = {}, {}, {}
        for tid, s in self.states.items():
            pending = start_pending[tid] if tid not in after else None
            drift_pending = claim_pending = None
            for f in sorted(detections):
                if f <= after.get(tid, -1):
                    continue
                det = detections[f].get(tid)
                if det is None:
                    pending = drift_pending = None
                    continue
                mask, status, score = det
                tracked = masks[f].get(tid)
                if tracked is not None and iou(tracked, mask) >= cfg.drift_iou:
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
                    holder_ok = holder_det is not None and iou(holder_det[0], masks[f][holder]) >= cfg.drift_iou
                    if status == "confident" and score >= cfg.drift_min_score and not holder_ok:
                        if claim_pending is not None and iou(claim_pending, mask) >= cfg.confirm_iou:
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
                        if drift_pending is not None and iou(drift_pending, mask) >= cfg.confirm_iou:
                            out[tid] = (f, mask, "drift")
                            break
                        drift_pending = mask
                    else:
                        drift_pending = None
                    continue
                # Starting a track: "confident" is relative (nothing better competed), so also
                # require absolute strength. Weak-but-plausible detections need a second refresh
                # agreeing in the same place; a missed seed is cheap, a wrong one is tracked onward.
                if score < cfg.seed_min_score:
                    pending = None
                    continue
                strong = status == "confident" and score >= cfg.seed_strong_score
                if strong or (pending is not None and iou(pending, mask) >= cfg.confirm_iou):
                    out[tid] = (f, mask, "reacquire" if (s.last_seen is not None or tid in seen) else "seed")
                    pending = None
                    break
                pending = mask
            pendings[tid] = pending
        return out, pendings, cuts

    def _cut_on_destination(self, masks, scores, destinations, frame_idx, offset) -> None:
        """Cut a track at the first refresh frame where its mask *is* the destination (high IoU).

        SAM 2 can spread a mask from a cup onto the clear container holding it. IoU, not
        coverage, is used: a cup inside the container is covered by the container's mask but
        is a legitimate, much smaller mask. The cut frees the ID for re-acquisition.
        """
        for f in sorted(destinations):
            if not len(destinations[f]):
                continue
            for tid in list(masks[f]):
                if any(iou(masks[f][tid], d) > self.cfg.conflict_iou for d in destinations[f]):
                    for k in range(f, len(masks)):
                        masks[k].pop(tid, None)
                        scores[k].pop(tid, None)
                    self.log.append({"camera": self.camera, "frame_index": int(self._abs(f, frame_idx, offset)),
                                     "track_id": tid, "label": self.states[tid].target.label,
                                     "event": "cut_on_destination"})

    @staticmethod
    def _continuity(masks, j: int, tid: int) -> float:
        """IoU of an ID's mask with its own mask on the previous frame (0 if it wasn't there)."""
        prev = masks[j - 1].get(tid) if j > 0 else None
        return iou(prev, masks[j][tid]) if prev is not None else 0.0

    def _same_object(self, a: np.ndarray, b: np.ndarray) -> bool:
        """a and b cover the same object: high IoU, or one covers most of the other (a detection
        box merging two touching objects covers the smaller one entirely at low IoU)."""
        inter = np.logical_and(a, b).sum()
        smaller = min(a.sum(), b.sum())
        return iou(a, b) > self.cfg.conflict_iou or (smaller > 0 and inter / smaller > self.cfg.conflict_iou)

    @staticmethod
    def _abs(j: int, frame_idx: np.ndarray, offset: int) -> int:
        return int(frame_idx[j - offset]) if j >= offset else int(frame_idx[0]) - 1


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
    labels = {t.id: t.label for t in parsed.targets}
    write_json(out / "targets.json", parsed.to_dict())
    print(f"task: {task}")
    print("targets: " + ", ".join(f"{t.id}={t.label}" for t in parsed.targets))

    ds = Dataset(repo_id, data_dir) if repo_id else Dataset(data_dir=data_dir)
    episode = ds.episode(episode_index, cameras=cameras)
    dev = get_device(device)
    print(f"device: {device_name(dev)}")

    t0 = time.perf_counter()
    grounder = FrameGrounder(parsed, cfg.assign, device=dev)
    tracker = VideoTracker(device=dev)
    t_load = time.perf_counter() - t0

    stop = None if cfg.max_frames is None else cfg.start + cfg.max_frames
    masks_writer = MaskWriter(out / "masks.parquet")
    stats = {"task": task, "episode_index": episode_index, "device": device_name(dev),
             "model_load_s": round(t_load, 1), "cameras": {}}
    log: list[dict] = []
    try:
        for camera in episode.cameras:
            cam = CameraTracker(camera, parsed, grounder, tracker, cfg)
            video = None
            visible = defaultdict(int)
            n_frames, t_cam, t_decode, t_write = 0, time.perf_counter(), 0.0, 0.0
            print(f"\n[{camera}] refresh every {cam.interval} frames")
            t0 = time.perf_counter()
            for chunk in iter_frames(episode, camera, chunk_size=cfg.chunk_size, start=cfg.start, stop=stop):
                t_decode += time.perf_counter() - t0
                masks, scores, states = cam.process(chunk.frames, chunk.frame_indices)

                t0 = time.perf_counter()
                h, w = chunk.frames.shape[1:3]
                cleaned = [_clean_frame(m) for m in masks]
                masks_writer.write_frames(episode_index, camera, chunk.frame_indices.tolist(), cleaned,
                                          scores, states, labels, (h, w))
                if cfg.write_video:
                    video = video or VideoWriter(out / f"{camera}.mp4", episode.fps, w, h)
                    for frame, f, m in zip(chunk.frames, chunk.frame_indices, cleaned):
                        video.write(annotate_frame(frame, m, labels, caption=f"{camera} f{f}"))
                for m in cleaned:
                    for tid in m:
                        visible[tid] += 1
                n_frames += len(chunk.frames)
                t_write += time.perf_counter() - t0
                tracked = ", ".join(f"{tid}" for tid in sorted(masks[-1])) or "-"
                print(f"  frames {chunk.frame_indices[0]}–{chunk.frame_indices[-1]}: "
                      f"tracking [{tracked}], {sum(1 for e in cam.log if e['frame_index'] >= chunk.frame_indices[0] - 1)} events")
                t0 = time.perf_counter()
            if video:
                video.close()
            elapsed = time.perf_counter() - t_cam
            log += cam.log
            stats["cameras"][camera] = {
                "frames": n_frames,
                "seconds": round(elapsed, 1),
                "sec_per_frame": round(elapsed / max(n_frames, 1), 3),
                "stage_seconds": {"decode": round(t_decode, 1), "grounding": round(cam.timing["grounding"], 1),
                                  "tracking": round(cam.timing["tracking"], 1), "write": round(t_write, 1)},
                "refresh_interval": cam.interval,
                "refresh_frames": cam.refresh_count,
                "events": {k: sum(1 for e in cam.log if e["event"] == k) for k in sorted({e["event"] for e in cam.log})},
                "visible_fraction": {labels[tid]: round(visible[tid] / max(n_frames, 1), 3) for tid in labels},
            }
            print(f"  {n_frames} frames in {elapsed:.0f}s ({elapsed / max(n_frames, 1):.2f}s/frame); "
                  f"visible: " + ", ".join(f"{labels[t]} {100 * visible[t] / max(n_frames, 1):.0f}%" for t in labels))
    finally:
        masks_writer.close()

    stats["total_seconds"] = round(time.perf_counter() - t_start, 1)
    total_frames = sum(c["frames"] for c in stats["cameras"].values())
    stats["sec_per_frame_all_cameras"] = round(stats["total_seconds"] / max(total_frames, 1), 3)
    write_json(out / "stats.json", stats)
    (out / "events.jsonl").write_text("".join(json.dumps(e) + "\n" for e in log))
    print(f"\noutputs in {out}")
    return stats


def _clean_frame(masks: dict[int, np.ndarray]) -> dict[int, np.ndarray]:
    """Drop specks and give overlapping pixels to the smaller mask (usually the object in front,
    e.g. a spoon resting in a cup), so each pixel belongs to at most one track."""
    cleaned = {tid: clean_mask(m) for tid, m in masks.items()}
    for tid in sorted(cleaned, key=lambda t: cleaned[t].sum()):
        for other in cleaned:
            if other != tid and cleaned[other].sum() > cleaned[tid].sum():
                cleaned[other] = cleaned[other] & ~cleaned[tid]
    return {tid: m for tid, m in cleaned.items() if m.any()}
