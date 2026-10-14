"""Drawing helpers: colored masks, boxes and ID labels over frames."""

from __future__ import annotations

import numpy as np
from PIL import Image, ImageDraw, ImageFont

# Distinct, colorblind-friendly-ish colors indexed by track ID (ID 1 → PALETTE[0]).
PALETTE = [
    (230, 25, 75), (60, 180, 75), (0, 130, 200), (245, 130, 48), (145, 30, 180),
    (70, 240, 240), (240, 50, 230), (210, 245, 60), (0, 128, 128), (170, 110, 40),
]


def id_color(track_id: int) -> tuple[int, int, int]:
    return PALETTE[(track_id - 1) % len(PALETTE)]


def _font(size: int = 12):
    try:
        return ImageFont.load_default(size=size)
    except TypeError:                       # Pillow < 10.1
        return ImageFont.load_default()


def overlay_masks(frame: np.ndarray, masks: dict[int, np.ndarray], alpha: float = 0.45) -> np.ndarray:
    """Blend each track's mask over the frame in its ID color."""
    out = frame.astype(np.float32).copy()
    for tid, mask in masks.items():
        if mask is not None and mask.any():
            out[mask] = (1 - alpha) * out[mask] + alpha * np.array(id_color(tid), dtype=np.float32)
    return out.clip(0, 255).astype(np.uint8)


def annotate_frame(
    frame: np.ndarray,
    masks: dict[int, np.ndarray],
    labels: dict[int, str],
    caption: str | None = None,
    font_size: int = 12,
) -> np.ndarray:
    """Frame with each track's mask filled and outlined in its ID color and labeled "id: name"."""
    from scipy import ndimage

    out = overlay_masks(frame, masks)
    for tid, mask in masks.items():
        edge = mask & ~ndimage.binary_erosion(mask, iterations=2)
        out[edge] = id_color(tid)
    image = Image.fromarray(out)
    draw = ImageDraw.Draw(image)
    font = _font(font_size)
    for tid, mask in masks.items():
        ys, xs = np.nonzero(mask)
        if len(xs):
            x, y = float(xs.min()), float(ys.min()) - font_size - 3
            draw_label(draw, (x, max(y, 0)), f"{tid}: {labels.get(tid, '?')}", id_color(tid), font)
    if caption:
        draw_label(draw, (4, 4), caption, (0, 0, 0), font)
    return np.asarray(image)


def draw_label(draw: ImageDraw.ImageDraw, xy: tuple[float, float], text: str, color, font) -> None:
    x, y = xy
    l, t, r, b = draw.textbbox((x, y), text, font=font)
    draw.rectangle((l - 2, t - 1, r + 2, b + 1), fill=color)
    draw.text((x, y), text, fill=(255, 255, 255), font=font)


def draw_boxes(
    image: Image.Image,
    boxes: list[tuple[np.ndarray, str, tuple[int, int, int], int]],
    font_size: int = 12,
) -> Image.Image:
    """Draw (box, label, color, line width) tuples. Labels go above the box when there's room."""
    draw = ImageDraw.Draw(image)
    font = _font(font_size)
    for box, label, color, width in boxes:
        x0, y0, x1, y1 = [float(v) for v in box]
        draw.rectangle((x0, y0, x1, y1), outline=color, width=width)
        if label:
            ty = y0 - font_size - 3 if y0 > font_size + 3 else y1 + 2
            draw_label(draw, (x0, ty), label, color, font)
    return image
