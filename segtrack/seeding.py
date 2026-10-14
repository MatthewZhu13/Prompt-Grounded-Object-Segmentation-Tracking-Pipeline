"""Choose where each target's track starts: its most confident grounding in a window of frames.

Seeding from the first frame alone would fail for objects that are hidden or out of view at
the start (common in the wrist cameras); scanning several frames and keeping each target's
best match also avoids seeding from a frame where its identification was a near-tie.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .boxes import clean_mask
from .ground import FrameGrounder


@dataclass
class Seed:
    target_id: int
    frame_index: int        # episode frame index
    mask: np.ndarray
    status: str             # "confident" | "uncertain"
    score: float            # composite (log space)

    @property
    def rank(self) -> tuple[bool, float]:
        return (self.status == "confident", self.score)


def choose_seeds(
    grounder: FrameGrounder,
    frames: np.ndarray,
    frame_indices: np.ndarray,
    step: int = 10,
) -> dict[int, Seed]:
    """Ground every `step`-th frame and keep each target's best match (confident first, then score)."""
    best: dict[int, Seed] = {}
    for i in range(0, len(frames), step):
        g = grounder(frames[i])
        for m in g.result.matches:
            if m.candidate is None:
                continue
            seed = Seed(m.target_id, int(frame_indices[i]), clean_mask(g.masks[m.candidate], keep_largest=True),
                        m.status, m.score)
            if m.target_id not in best or seed.rank > best[m.target_id].rank:
                best[m.target_id] = seed
    return best
