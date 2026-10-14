"""Box and mask geometry helpers (NumPy only). Boxes are (x0, y0, x1, y1) in pixels."""

from __future__ import annotations

import numpy as np


def iou_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Pairwise IoU between boxes a (N, 4) and b (M, 4) → (N, M)."""
    if len(a) == 0 or len(b) == 0:
        return np.zeros((len(a), len(b)))
    x0 = np.maximum(a[:, None, 0], b[None, :, 0])
    y0 = np.maximum(a[:, None, 1], b[None, :, 1])
    x1 = np.minimum(a[:, None, 2], b[None, :, 2])
    y1 = np.minimum(a[:, None, 3], b[None, :, 3])
    inter = np.clip(x1 - x0, 0, None) * np.clip(y1 - y0, 0, None)
    area_a = (a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1])
    area_b = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
    return inter / np.maximum(area_a[:, None] + area_b[None, :] - inter, 1e-9)


def nms(boxes: np.ndarray, scores: np.ndarray, iou_threshold: float) -> np.ndarray:
    """Greedy non-maximum suppression. Returns kept indices, highest score first."""
    order = np.argsort(-scores)
    keep = []
    while len(order):
        i = order[0]
        keep.append(i)
        if len(order) == 1:
            break
        ious = iou_matrix(boxes[i:i + 1], boxes[order[1:]])[0]
        order = order[1:][ious <= iou_threshold]
    return np.array(keep, dtype=int)


def mask_to_box(mask: np.ndarray) -> np.ndarray | None:
    """Tight box around a boolean mask, or None if the mask is empty."""
    ys, xs = np.nonzero(mask)
    if len(xs) == 0:
        return None
    return np.array([xs.min(), ys.min(), xs.max() + 1, ys.max() + 1], dtype=float)


def clean_mask(mask: np.ndarray, keep_largest: bool = False, min_piece: float = 0.02) -> np.ndarray:
    """Remove stray specks from a mask.

    keep_largest: keep only the largest connected piece (used for tracker prompts, so specks on
    the gripper or glare are not tracked along with the object). Otherwise drop pieces smaller
    than `min_piece` of the mask's area (used for outputs, so a legitimately split mask, e.g. an
    object partly hidden behind the gripper, keeps both visible parts).
    """
    from scipy import ndimage

    labels, n = ndimage.label(mask)
    if n <= 1:
        return mask
    sizes = np.bincount(labels.ravel())[1:]
    if keep_largest:
        keep = [int(sizes.argmax()) + 1]
    else:
        keep = np.nonzero(sizes >= min_piece * sizes.sum())[0] + 1
    return np.isin(labels, keep)


def containment(masks: np.ndarray) -> np.ndarray:
    """(N, N) matrix: fraction of mask i's pixels that lie inside mask j."""
    flat = masks.reshape(len(masks), -1).astype(np.float32)
    inter = flat @ flat.T
    area = flat.sum(1)
    return inter / np.maximum(area[:, None], 1.0)
