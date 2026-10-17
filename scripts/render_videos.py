"""Re-draw a finished run's overlay videos from its masks.parquet (no models needed).

    python scripts/render_videos.py out/demo/episode_000 --data-dir data/cup-01
    python scripts/render_videos.py out/demo/episode_000 --cameras top --start 600 --frames 900 --out clips/

Useful after changing the overlay style, or to cut a short clip around a moment for a
presentation. Writes <camera>.mp4 into --out (default: the run folder, replacing its videos).
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import pyarrow.parquet as pq

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from segtrack import rle                                    # noqa: E402
from segtrack.data import DEFAULT_REPO, Dataset, iter_frames  # noqa: E402
from segtrack.io import VideoWriter                         # noqa: E402
from segtrack.viz import SIDEBAR_WIDTH, render_track_frame  # noqa: E402


def load_rows(run: Path, camera: str, first: int, stop: int | None):
    """{frame index: [rows]} for one camera, restricted to [first, stop)."""
    filters = [("camera", "=", camera), ("frame_index", ">=", first)]
    if stop is not None:
        filters.append(("frame_index", "<", stop))
    rows = pq.read_table(run / "masks.parquet", filters=filters).to_pylist()
    by_frame = defaultdict(list)
    for r in rows:
        by_frame[r["frame_index"]].append(r)
    return by_frame


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("run", type=Path, help="run folder, e.g. out/demo/episode_000")
    p.add_argument("--data-dir", help="local copy of the dataset (default: download)")
    p.add_argument("--repo", default=DEFAULT_REPO)
    p.add_argument("--cameras", nargs="+", help="default: every camera in the run")
    p.add_argument("--start", type=int, default=0, help="first frame")
    p.add_argument("--frames", type=int, help="number of frames (default: to the end)")
    p.add_argument("--out", type=Path, help="output folder (default: the run folder)")
    args = p.parse_args()

    targets = json.loads((args.run / "targets.json").read_text())["targets"]
    labels = {t["id"]: t["label"] for t in targets}
    meta = pq.read_table(args.run / "masks.parquet", columns=["episode_index", "camera"]).to_pydict()
    episode_index = meta["episode_index"][0]
    cameras = args.cameras or list(dict.fromkeys(meta["camera"]))
    episode = Dataset(args.repo, args.data_dir).episode(episode_index, cameras=cameras)
    stop = None if args.frames is None else args.start + args.frames
    out_dir = args.out or args.run
    out_dir.mkdir(parents=True, exist_ok=True)

    for camera in cameras:
        by_frame = load_rows(args.run, camera, args.start, stop)
        path = out_dir / f"{camera}.mp4"
        video = None
        for chunk in iter_frames(episode, camera, start=args.start, stop=stop):
            h, w = chunk.frames.shape[1:3]
            video = video or VideoWriter(path, episode.fps, w + SIDEBAR_WIDTH, h)
            for frame, f in zip(chunk.frames, chunk.frame_indices):
                rows = by_frame.get(int(f), [])
                masks = {r["track_id"]: rle.decode(r["rle"], r["height"], r["width"]) for r in rows if r["visible"]}
                states = {r["track_id"]: r["state"] for r in rows}
                video.write(render_track_frame(frame, camera, int(f), masks, states, labels))
        if video:
            video.close()
            print(f"wrote {path}")


if __name__ == "__main__":
    main()
