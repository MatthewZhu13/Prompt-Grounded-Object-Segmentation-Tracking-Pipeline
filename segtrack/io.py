"""Output writers: overlay videos (MP4) and per-frame masks (Parquet)."""

from __future__ import annotations

import json
from pathlib import Path

import av
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from . import rle
from .boxes import mask_to_box

# One row per (camera, frame_index, track_id), for every target on every frame — invisible
# targets get a row too (visible=False, empty mask), so the table is dense and easy to join
# with the dataset's own per-frame parquet on (episode_index, frame_index).
MASK_SCHEMA = pa.schema([
    ("episode_index", pa.int32()),
    ("camera", pa.string()),
    ("frame_index", pa.int32()),          # matches the dataset's frame_index
    ("track_id", pa.int32()),             # stable target ID (see targets.json)
    ("label", pa.string()),               # e.g. "gray cup"
    ("visible", pa.bool_()),              # False: object not found / out of view on this frame
    ("state", pa.string()),               # "tracking" | "lost" (seen before) | "searching" (never seen yet)
    ("score", pa.float32()),              # SAM 2 object-presence probability (0 when not visible)
    ("bbox_x", pa.int32()), ("bbox_y", pa.int32()),     # tight box around the mask, pixels
    ("bbox_w", pa.int32()), ("bbox_h", pa.int32()),
    ("area", pa.int32()),                 # mask pixels
    ("height", pa.int32()), ("width", pa.int32()),
    ("rle", pa.list_(pa.int32())),        # COCO uncompressed RLE, column-major (segtrack.rle)
])


class MaskWriter:
    """Appends mask rows to a Parquet file, one row group per call to `write_frames`."""

    def __init__(self, path: str | Path):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        self.writer = pq.ParquetWriter(str(path), MASK_SCHEMA, compression="zstd")

    def write_frames(
        self,
        episode_index: int,
        camera: str,
        frame_indices: list[int],
        masks: list[dict[int, np.ndarray]],
        scores: list[dict[int, float]],
        states: list[dict[int, str]],
        labels: dict[int, str],
        shape: tuple[int, int],
    ) -> None:
        h, w = shape
        empty = rle.encode(np.zeros((h, w), bool))
        cols: dict[str, list] = {name: [] for name in MASK_SCHEMA.names}
        for f, frame_masks, frame_scores, frame_states in zip(frame_indices, masks, scores, states):
            for tid, label in labels.items():
                mask = frame_masks.get(tid)
                box = mask_to_box(mask) if mask is not None else None
                visible = box is not None
                x0, y0, x1, y1 = (int(v) for v in box) if visible else (0, 0, 0, 0)
                row = {
                    "episode_index": episode_index, "camera": camera, "frame_index": int(f),
                    "track_id": tid, "label": label, "visible": visible,
                    "state": frame_states.get(tid, "searching"),
                    "score": float(frame_scores.get(tid, 0.0)) if visible else 0.0,
                    "bbox_x": x0, "bbox_y": y0, "bbox_w": x1 - x0, "bbox_h": y1 - y0,
                    "area": int(mask.sum()) if visible else 0, "height": h, "width": w,
                    "rle": rle.encode(mask) if visible else empty,
                }
                for k, v in row.items():
                    cols[k].append(v)
        self.writer.write_table(pa.table(cols, schema=MASK_SCHEMA))

    def close(self) -> None:
        self.writer.close()


def write_json(path: str | Path, obj) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(obj, indent=2, default=_json_default))


def _json_default(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    raise TypeError(type(o))


class VideoWriter:
    """Write RGB frames to an H.264 MP4 (plays in browsers, QuickTime, VLC)."""

    def __init__(self, path: str | Path, fps: float, width: int, height: int, crf: int = 20):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        self.container = av.open(str(path), mode="w")
        self.stream = self.container.add_stream("libx264", rate=round(fps))
        self.stream.width, self.stream.height = width, height
        self.stream.pix_fmt = "yuv420p"
        self.stream.options = {"crf": str(crf), "preset": "medium"}

    def write(self, frame: np.ndarray) -> None:
        for packet in self.stream.encode(av.VideoFrame.from_ndarray(frame, format="rgb24")):
            self.container.mux(packet)

    def close(self) -> None:
        for packet in self.stream.encode():
            self.container.mux(packet)
        self.container.close()

    def __enter__(self) -> "VideoWriter":
        return self

    def __exit__(self, *exc) -> None:
        self.close()
