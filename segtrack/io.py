"""Output writers."""

from __future__ import annotations

from pathlib import Path

import av
import numpy as np


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
