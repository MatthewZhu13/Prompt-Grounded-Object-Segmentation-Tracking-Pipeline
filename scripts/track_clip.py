"""Seed targets by grounding, then track them through a clip with SAM 2 and write an overlay video.

    python scripts/track_clip.py --episode 0 --camera top --frames 300

This is tracking only (no re-detection of lost objects yet): each target is seeded from its
most confident grounding in the first `--seed-window` frames, then followed chunk by chunk.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from segtrack.boxes import clean_mask                         # noqa: E402
from segtrack.data import Dataset, iter_frames                # noqa: E402
from segtrack.device import device_name, get_device           # noqa: E402
from segtrack.ground import FrameGrounder                     # noqa: E402
from segtrack.io import VideoWriter                           # noqa: E402
from segtrack.parser import parse_task                        # noqa: E402
from segtrack.seeding import choose_seeds                     # noqa: E402
from segtrack.track import VideoTracker                       # noqa: E402
from segtrack.viz import annotate_frame                       # noqa: E402


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--episode", type=int, default=0)
    p.add_argument("--camera", default="top")
    p.add_argument("--start", type=int, default=0)
    p.add_argument("--frames", type=int, default=300)
    p.add_argument("--chunk", type=int, default=150, help="frames per SAM 2 session")
    p.add_argument("--seed-window", type=int, default=60)
    p.add_argument("--seed-step", type=int, default=10)
    p.add_argument("--task")
    p.add_argument("--data-dir", default="data/cup-01")
    p.add_argument("--out", default="out/track")
    args = p.parse_args()

    ds = Dataset(data_dir=args.data_dir)
    episode = ds.episode(args.episode, cameras=[args.camera])
    parsed = parse_task(args.task or ds.annotated_task(args.episode))
    labels = {t.id: t.label for t in parsed.targets}
    device = get_device()
    print(f"device: {device_name(device)} | targets: {labels}")

    chunks = list(iter_frames(episode, args.camera, chunk_size=args.frames,
                              start=args.start, stop=args.start + args.frames))
    frames, frame_idx = chunks[0].frames, chunks[0].frame_indices
    n = len(frames)

    t0 = time.perf_counter()
    grounder = FrameGrounder(parsed, device=device)
    window = min(args.seed_window, n)
    seeds = choose_seeds(grounder, frames[:window], frame_idx[:window], step=args.seed_step)
    t_seed = time.perf_counter() - t0
    for tid, s in sorted(seeds.items()):
        print(f"  seed {tid}: {labels[tid]:12} frame {s.frame_index} [{s.status}]")
    missing = [labels[t] for t in labels if t not in seeds]
    if missing:
        print(f"  not found in seed window (untracked for now): {missing}")

    tracker = VideoTracker(device=device)
    tracks: list[dict[int, np.ndarray]] = [{} for _ in range(n)]
    prompts = {tid: (s.frame_index - int(frame_idx[0]), s.mask) for tid, s in seeds.items()}
    t0 = time.perf_counter()
    start = 0
    while start < n and prompts:
        end = min(start + args.chunk, n)
        local = {tid: (f - start, m) for tid, (f, m) in prompts.items()}
        masks, _ = tracker.track(frames[start:end], local)
        for i, m in enumerate(masks):
            if start + i == start and start > 0:
                continue                                # first frame duplicates the previous chunk's last
            tracks[start + i] = m
        print(f"  tracked frames {frame_idx[start]}–{frame_idx[end - 1]} with {len(local)} objects")
        if end == n:
            break
        # Next chunk starts on this chunk's last frame, seeded with the masks there.
        prompts = {tid: (end - 1, clean_mask(m, keep_largest=True)) for tid, m in masks[-1].items()}
        start = end - 1
    t_track = time.perf_counter() - t0

    out = Path(args.out) / f"ep{args.episode:03d}_{args.camera}_{args.start:05d}-{args.start + n - 1:05d}.mp4"
    with VideoWriter(out, fps=episode.fps, width=frames.shape[2], height=frames.shape[1]) as w:
        for i in range(n):
            visible = {tid: clean_mask(m) for tid, m in tracks[i].items()}
            w.write(annotate_frame(frames[i], visible, labels, caption=f"{args.camera} f{frame_idx[i]}"))

    print("\nvisible frames per track:")
    for tid in labels:
        k = sum(tid in tr for tr in tracks)
        print(f"  {tid}: {labels[tid]:12} {k}/{n} ({100 * k / n:.0f}%)")
    print(f"\nseeding {t_seed:.1f}s | tracking {t_track:.1f}s = {t_track / n:.3f}s/frame | saved {out}")


if __name__ == "__main__":
    main()
