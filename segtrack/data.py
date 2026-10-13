"""Load episodes from a LeRobot v3 dataset.

In LeRobot v3, many episodes are concatenated into each MP4. `meta/episodes/*.parquet`
records, per episode and camera, which MP4 file holds it and its [from, to) timestamps.
We read that metadata, fetch only the files an episode needs (or use a local copy),
and decode its frames in fixed-size chunks so the whole episode is never held in memory.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

import av
import numpy as np
import pyarrow.parquet as pq
from huggingface_hub import hf_hub_download, snapshot_download

DEFAULT_REPO = "allenai/15122025-cup-01"
VIDEO_KEY_PREFIX = "observation.images."


@dataclass(frozen=True)
class VideoSegment:
    """Where one camera's footage for an episode lives."""

    path: Path          # local path to the MP4 (may contain other episodes too)
    from_ts: float      # episode start within the MP4, seconds
    to_ts: float        # episode end (exclusive), seconds


@dataclass(frozen=True)
class Episode:
    index: int
    length: int                         # number of frames
    fps: float
    cameras: dict[str, VideoSegment]    # e.g. {"top": ..., "left": ..., "right": ...}


@dataclass(frozen=True)
class FrameChunk:
    """A contiguous run of decoded frames from one camera."""

    frame_indices: np.ndarray   # (N,) int, episode-relative, matches the dataset's frame_index
    frames: np.ndarray          # (N, H, W, 3) uint8 RGB


class Dataset:
    """A LeRobot v3 dataset, read from a local directory with Hugging Face Hub as fallback."""

    def __init__(self, repo_id: str = DEFAULT_REPO, data_dir: str | Path | None = None):
        self.repo_id = repo_id
        self.data_dir = Path(data_dir) if data_dir else None
        self.info = json.loads(self._fetch("meta/info.json").read_text())
        self.fps = float(self.info["fps"])
        self.camera_names = [
            k.removeprefix(VIDEO_KEY_PREFIX)
            for k, v in self.info["features"].items()
            if v.get("dtype") == "video"
        ]
        self._episode_rows: dict[int, dict] | None = None

    # ---------- file access ----------

    def _fetch(self, rel_path: str) -> Path:
        """Return a local path for a dataset file, downloading it if not present locally."""
        if self.data_dir is not None:
            local = self.data_dir / rel_path
            if local.exists():
                return local
        return Path(hf_hub_download(
            self.repo_id, rel_path, repo_type="dataset",
            local_dir=self.data_dir,
        ))

    def _episodes_meta_dir(self) -> Path:
        if self.data_dir is not None and (self.data_dir / "meta/episodes").exists():
            return self.data_dir / "meta/episodes"
        root = snapshot_download(
            self.repo_id, repo_type="dataset",
            allow_patterns=["meta/episodes/*"], local_dir=self.data_dir,
        )
        return Path(root) / "meta/episodes"

    # ---------- metadata ----------

    def _load_episode_rows(self) -> dict[int, dict]:
        if self._episode_rows is None:
            # Only read the columns we need: the per-episode `stats/*` columns are deeply
            # nested lists that some pandas/pyarrow versions fail to convert.
            wanted = ["episode_index", "length"]
            for cam in self.camera_names:
                key = f"videos/{VIDEO_KEY_PREFIX}{cam}"
                wanted += [f"{key}/chunk_index", f"{key}/file_index",
                           f"{key}/from_timestamp", f"{key}/to_timestamp"]
            rows = {}
            for f in sorted(self._episodes_meta_dir().glob("*/*.parquet")):
                for row in pq.read_table(f, columns=wanted).to_pylist():
                    rows[row["episode_index"]] = row
            self._episode_rows = rows
        return self._episode_rows

    @property
    def num_episodes(self) -> int:
        return len(self._load_episode_rows())

    def episode(self, index: int, cameras: list[str] | None = None) -> Episode:
        """Resolve an episode's video files (downloading them if needed) and time ranges."""
        rows = self._load_episode_rows()
        if index not in rows:
            raise ValueError(f"episode {index} not in dataset (has {len(rows)} episodes)")
        row = rows[index]
        cameras = cameras or self.camera_names
        unknown = set(cameras) - set(self.camera_names)
        if unknown:
            raise ValueError(f"unknown cameras {sorted(unknown)}; available: {self.camera_names}")

        segments = {}
        for cam in cameras:
            key = f"{VIDEO_KEY_PREFIX}{cam}"
            rel = self.info["video_path"].format(
                video_key=key,
                chunk_index=row[f"videos/{key}/chunk_index"],
                file_index=row[f"videos/{key}/file_index"],
            )
            segments[cam] = VideoSegment(
                path=self._fetch(rel),
                from_ts=row[f"videos/{key}/from_timestamp"],
                to_ts=row[f"videos/{key}/to_timestamp"],
            )
        return Episode(index=index, length=row["length"], fps=self.fps, cameras=segments)

    def annotated_task(self, index: int) -> str | None:
        """Per-episode instruction from `meta/tasks_annotated.parquet`, if the dataset has one."""
        try:
            table = pq.read_table(self._fetch("meta/tasks_annotated.parquet"))
        except Exception:
            return None
        for row in table.to_pylist():
            if row.get("episode_index") == index:
                return row["task"]
        return None


def iter_frames(
    episode: Episode,
    camera: str,
    chunk_size: int = 300,
    start: int = 0,
    stop: int | None = None,
) -> Iterator[FrameChunk]:
    """Decode one camera's frames for `episode`, yielding chunks of up to `chunk_size` frames.

    `start`/`stop` are episode-relative frame indices (stop exclusive, default: episode end).
    """
    seg = episode.cameras[camera]
    stop = episode.length if stop is None else min(stop, episode.length)
    if not 0 <= start < stop:
        return
    half_frame = 0.5 / episode.fps

    with av.open(str(seg.path)) as container:
        stream = container.streams.video[0]
        stream.thread_type = "AUTO"
        # Seek to the nearest keyframe at or before the first frame we want, then decode forward.
        target_ts = seg.from_ts + start / episode.fps
        container.seek(int(target_ts / stream.time_base), stream=stream, backward=True)

        indices: list[int] = []
        frames: list[np.ndarray] = []
        for frame in container.decode(stream):
            t = frame.time
            if t < seg.from_ts - half_frame:
                continue                        # still before the episode (seek landed early)
            if t >= seg.to_ts - half_frame:
                break                           # past the episode
            idx = round((t - seg.from_ts) * episode.fps)
            if idx < start:
                continue
            if idx >= stop:
                break
            indices.append(idx)
            frames.append(frame.to_ndarray(format="rgb24"))
            if len(frames) == chunk_size:
                yield FrameChunk(np.array(indices), np.stack(frames))
                indices, frames = [], []
        if frames:
            yield FrameChunk(np.array(indices), np.stack(frames))


def _main() -> None:
    """Sanity check: decode an episode and save a few sample frames per camera."""
    from PIL import Image

    p = argparse.ArgumentParser(description=_main.__doc__)
    p.add_argument("--episode", type=int, default=0)
    p.add_argument("--data-dir", default="data/cup-01")
    p.add_argument("--out", default="out/samples")
    p.add_argument("--num-samples", type=int, default=5)
    args = p.parse_args()

    ds = Dataset(data_dir=args.data_dir)
    ep = ds.episode(args.episode)
    print(f"episode {ep.index}: {ep.length} frames @ {ep.fps} fps, cameras {list(ep.cameras)}")
    print(f"task: {ds.annotated_task(ep.index)}")

    out = Path(args.out) / f"episode_{ep.index:03d}"
    out.mkdir(parents=True, exist_ok=True)
    wanted = set(np.linspace(0, ep.length - 1, args.num_samples).astype(int).tolist())
    for cam in ep.cameras:
        decoded = 0
        expected_next = 0
        for chunk in iter_frames(ep, cam):
            assert chunk.frame_indices[0] == expected_next, "gap or overlap in decoded frames"
            assert np.all(np.diff(chunk.frame_indices) == 1), "non-contiguous frame indices"
            expected_next = chunk.frame_indices[-1] + 1
            decoded += len(chunk.frame_indices)
            for i, idx in enumerate(chunk.frame_indices):
                if idx in wanted:
                    Image.fromarray(chunk.frames[i]).save(out / f"{cam}_{idx:05d}.png")
        status = "OK" if decoded == ep.length else f"MISMATCH (expected {ep.length})"
        print(f"  {cam}: decoded {decoded} frames, shape {chunk.frames.shape[1:]} — {status}")
    print(f"sample frames written to {out}")


if __name__ == "__main__":
    _main()
