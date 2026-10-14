"""Run-length encoding of binary masks, in the COCO "uncompressed RLE" convention.

The mask is flattened in column-major (Fortran) order and stored as alternating run lengths,
starting with a run of 0s (which may be length 0). An empty H×W mask is [H*W].
Compatible with pycocotools: `{"size": [H, W], "counts": counts}` can be passed to
`pycocotools.mask.frPyObjects` / `decode`.
"""

from __future__ import annotations

import numpy as np


def encode(mask: np.ndarray) -> np.ndarray:
    flat = np.asarray(mask, dtype=bool).ravel(order="F")
    change = np.flatnonzero(flat[1:] != flat[:-1]) + 1
    runs = np.diff(np.concatenate(([0], change, [flat.size])))
    if flat.size and flat[0]:
        runs = np.concatenate(([0], runs))
    return runs.astype(np.int32)


def decode(counts, height: int, width: int) -> np.ndarray:
    counts = np.asarray(counts, dtype=np.int64)
    values = np.repeat(np.arange(len(counts)) % 2, counts).astype(bool)
    if values.size != height * width:
        raise ValueError(f"RLE covers {values.size} pixels, expected {height * width}")
    return values.reshape((height, width), order="F")
