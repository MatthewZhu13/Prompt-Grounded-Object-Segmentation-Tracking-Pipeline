"""Drawing helpers: coloured masks, ID tags, a status sidebar, and boxes."""

from __future__ import annotations

import numpy as np
from PIL import Image, ImageDraw, ImageFont

# Distinct colours indexed by track ID (ID 1 → PALETTE[0]). They identify tracks, not object
# colours: the "yellow bowl" may well be drawn green.
PALETTE = [
    (230, 25, 75), (60, 180, 75), (0, 130, 200), (245, 130, 48), (145, 30, 180),
    (70, 240, 240), (240, 50, 230), (210, 245, 60), (0, 128, 128), (170, 110, 40),
]

SIDEBAR_WIDTH = 180
SIDEBAR_BG = (28, 28, 30)
TEXT = (240, 240, 240)
MUTED = (140, 140, 145)
STATE_TEXT = {"tracking": "visible", "lost": "lost", "searching": "not seen yet"}


def id_color(track_id: int) -> tuple[int, int, int]:
    return PALETTE[(track_id - 1) % len(PALETTE)]


def _font(size: int):
    try:
        return ImageFont.load_default(size=size)
    except TypeError:                       # Pillow < 10.1
        return ImageFont.load_default()


def overlay_masks(frame: np.ndarray, masks: dict[int, np.ndarray], alpha: float = 0.45) -> np.ndarray:
    """Blend each track's mask over the frame in its ID colour."""
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
    states: dict[int, str] | None = None,
    font_size: int = 13,
) -> np.ndarray:
    """Frame with each track's mask filled, outlined and tagged "id label".

    With `states` ({track id: tracking | lost | searching}), a sidebar is added to the right
    listing every target and whether it is visible, with `caption` at its top; the output is
    then SIDEBAR_WIDTH pixels wider than the frame. Without it, `caption` goes on the frame.
    """
    from scipy import ndimage

    out = overlay_masks(frame, masks)
    for tid, mask in masks.items():
        out[mask & ~ndimage.binary_erosion(mask, iterations=2)] = id_color(tid)

    image = Image.fromarray(out)
    draw = ImageDraw.Draw(image)
    font = _font(font_size)
    for tid, mask in masks.items():
        ys, xs = np.nonzero(mask)
        if len(xs):
            draw_tag(draw, (float(xs.min()), float(ys.min()) - font_size - 4), f"{tid} {labels.get(tid, '?')}",
                     id_color(tid), font, image.width)
    if states is None:
        if caption:
            draw_tag(draw, (4, 4), caption, (0, 0, 0), font, image.width)
        return np.asarray(image)
    return np.hstack([np.asarray(image), _sidebar(frame.shape[0], labels, states, caption, font_size)])


def _sidebar(height: int, labels: dict[int, str], states: dict[int, str], caption: str | None,
             font_size: int) -> np.ndarray:
    """Panel listing every target: colour swatch, "id label", and its state on this frame."""
    panel = Image.new("RGB", (SIDEBAR_WIDTH, height), SIDEBAR_BG)
    draw = ImageDraw.Draw(panel)
    font, small = _font(font_size), _font(font_size - 1)
    y = 10
    if caption:
        draw.text((12, y), caption, fill=TEXT, font=font)
        y += font_size + 14
    for tid, label in labels.items():
        state = states.get(tid, "searching")
        visible = state == "tracking"
        box = (12, y + 2, 12 + font_size, y + 2 + font_size)
        if visible:
            draw.rectangle(box, fill=id_color(tid))
        else:
            draw.rectangle(box, outline=id_color(tid), width=2)
        draw.text((20 + font_size, y), f"{tid} {label}", fill=TEXT if visible else MUTED, font=font)
        draw.text((20 + font_size, y + font_size + 3), STATE_TEXT.get(state, state), fill=MUTED, font=small)
        y += 2 * font_size + 14
    return np.asarray(panel)


def draw_tag(draw: ImageDraw.ImageDraw, xy: tuple[float, float], text: str, color, font,
             max_width: int | None = None) -> None:
    """White text on a filled box in `color`, kept inside the image."""
    l, t, r, b = draw.textbbox((0, 0), text, font=font)
    x, y = xy
    if max_width is not None:
        x = min(x, max_width - (r - l) - 4)
    x, y = max(x, 2), max(y, 1)
    draw.rectangle((x - 2, y + t - 2, x + r + 2, y + b + 2), fill=color)
    draw.text((x, y), text, fill=(255, 255, 255), font=font)


def draw_boxes(
    image: Image.Image,
    boxes: list[tuple[np.ndarray, str, tuple[int, int, int], int]],
    font_size: int = 12,
) -> Image.Image:
    """Draw (box, label, colour, line width) tuples. Labels go above the box when there's room."""
    draw = ImageDraw.Draw(image)
    font = _font(font_size)
    for box, label, color, width in boxes:
        x0, y0, x1, y1 = [float(v) for v in box]
        draw.rectangle((x0, y0, x1, y1), outline=color, width=width)
        if label:
            ty = y0 - font_size - 3 if y0 > font_size + 3 else y1 + 2
            draw_tag(draw, (x0, ty), label, color, font, image.width)
    return image


def render_track_frame(frame: np.ndarray, camera: str, frame_index: int, masks: dict[int, np.ndarray],
                       states: dict[int, str], labels: dict[int, str]) -> np.ndarray:
    """One frame of a pipeline output video: masks, tags and the status sidebar.

    A target whose mask was emptied by output cleaning is shown as lost, so the sidebar always
    agrees with what is drawn."""
    shown = {tid: "tracking" if tid in masks else ("lost" if s == "tracking" else s) for tid, s in states.items()}
    return annotate_frame(frame, masks, labels, caption=f"{camera} · frame {frame_index}", states=shown)
