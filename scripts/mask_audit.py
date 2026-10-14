"""Audit mask quality for every assigned target on frames sampled across an episode.

    python scripts/mask_audit.py --episode 0 --cameras top left right --step 200

Per mask it reports simple warning signs (no ground truth needed):
  pieces     connected components; >1 with a small largest piece suggests a fragmented mask
  fill       mask area / its tight box area; very low suggests a thin or ragged mask
  sam_q      SAM 2's own predicted IoU for the mask
  overlap    largest IoU with another assigned mask in the same frame
  edge       mask touches the frame border (object partly out of view)
and saves a close-up mosaic with each mask's outline drawn, for visual review.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw
from scipy import ndimage

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from segtrack.boxes import mask_to_box                         # noqa: E402
from segtrack.data import Dataset, iter_frames                 # noqa: E402
from segtrack.device import get_device                         # noqa: E402
from segtrack.ground import FrameGrounder                      # noqa: E402
from segtrack.parser import parse_task                         # noqa: E402
from segtrack.viz import id_color                              # noqa: E402

TILE = 220


def mask_stats(mask: np.ndarray, others: list[np.ndarray]) -> dict:
    labels, n = ndimage.label(mask)
    sizes = np.bincount(labels.ravel())[1:] if n else np.array([0])
    x0, y0, x1, y1 = mask_to_box(mask)
    area = int(mask.sum())
    overlap = max((float((mask & o).sum()) / float((mask | o).sum()) for o in others), default=0.0)
    h, w = mask.shape
    return {
        "area": area,
        "pieces": int(n),
        "largest": float(sizes.max() / max(area, 1)),
        "fill": area / max((x1 - x0) * (y1 - y0), 1),
        "overlap": overlap,
        "edge": bool(mask[0].any() or mask[-1].any() or mask[:, 0].any() or mask[:, -1].any()),
        "box": (x0, y0, x1, y1),
        "w": w, "h": h,
    }


def flags(s: dict, q: float) -> list[str]:
    out = []
    if s["pieces"] > 1 and s["largest"] < 0.9:
        out.append("fragmented")
    if s["fill"] < 0.3:
        out.append("sparse")
    if q < 0.8:
        out.append("low-sam-q")
    if s["overlap"] > 0.1:
        out.append("overlap")
    return out


def tile(frame: np.ndarray, mask: np.ndarray, s: dict, color, caption: str) -> Image.Image:
    x0, y0, x1, y1 = s["box"]
    pad = 0.25 * max(x1 - x0, y1 - y0)
    l, t = int(max(x0 - pad, 0)), int(max(y0 - pad, 0))
    r, b = int(min(x1 + pad, s["w"])), int(min(y1 + pad, s["h"]))
    crop = frame[t:b, l:r].copy()
    outline = mask[t:b, l:r] & ~ndimage.binary_erosion(mask[t:b, l:r], iterations=1)
    outline = ndimage.binary_dilation(outline, iterations=1)
    crop[outline] = color
    img = Image.fromarray(crop)
    scale = TILE / max(img.size)
    img = img.resize((max(int(img.width * scale), 1), max(int(img.height * scale), 1)), Image.NEAREST)
    canvas = Image.new("RGB", (TILE, TILE + 28), (20, 20, 20))
    canvas.paste(img, ((TILE - img.width) // 2, (TILE - img.height) // 2))
    ImageDraw.Draw(canvas).multiline_text((3, TILE), caption, fill=(255, 255, 255), spacing=1)
    return canvas


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--episode", type=int, default=0)
    p.add_argument("--cameras", nargs="+", default=["top", "left", "right"])
    p.add_argument("--step", type=int, default=200)
    p.add_argument("--task")
    p.add_argument("--data-dir", default="data/cup-01")
    p.add_argument("--out", default="out/ground")
    p.add_argument("--cols", type=int, default=8)
    args = p.parse_args()

    ds = Dataset(data_dir=args.data_dir)
    episode = ds.episode(args.episode, cameras=args.cameras)
    parsed = parse_task(args.task or ds.annotated_task(args.episode))
    labels = {t.id: t.label for t in parsed.targets}
    grounder = FrameGrounder(parsed, device=get_device())

    rows, tiles = [], []
    print(f"{'cam':5} {'frame':>5} {'id':>2} {'label':12} {'area':>6} {'pieces':>6} {'largest':>7} "
          f"{'fill':>5} {'sam_q':>5} {'overlap':>7} edge  flags")
    for cam in args.cameras:
        for chunk in iter_frames(episode, cam, chunk_size=args.step):
            f, frame = int(chunk.frame_indices[0]), chunk.frames[0]
            g = grounder(frame)
            owner = g.result.candidate_to_target()
            for c, tid in owner.items():
                others = [g.masks[o] for o in owner if o != c]
                s = mask_stats(g.masks[c], others)
                q = float(g.mask_quality[c])
                fl = flags(s, q)
                rows.append((cam, f, tid, fl))
                print(f"{cam:5} {f:>5} {tid:>2} {labels[tid]:12} {s['area']:>6} {s['pieces']:>6} {s['largest']:>7.2f} "
                      f"{s['fill']:>5.2f} {q:>5.2f} {s['overlap']:>7.2f} {'yes' if s['edge'] else 'no':4}  {','.join(fl)}")
                caption = f"{cam} f{f} {tid}:{labels[tid]}\nq={q:.2f} pcs={s['pieces']} fill={s['fill']:.2f} {' '.join(fl)}"
                tiles.append(tile(frame, g.masks[c], s, id_color(tid), caption))

    n_flagged = sum(1 for r in rows if r[3])
    print(f"\n{len(rows)} masks audited, {n_flagged} flagged")
    for name in ["fragmented", "sparse", "low-sam-q", "overlap"]:
        print(f"  {name:10}: {sum(name in r[3] for r in rows)}")

    if tiles:
        th = tiles[0].height
        n_rows = (len(tiles) + args.cols - 1) // args.cols
        sheet = Image.new("RGB", (TILE * args.cols, th * n_rows), (20, 20, 20))
        for k, t in enumerate(tiles):
            sheet.paste(t, ((k % args.cols) * TILE, (k // args.cols) * th))
        out = Path(args.out) / f"mask_audit_ep{args.episode:03d}.jpg"
        out.parent.mkdir(parents=True, exist_ok=True)
        sheet.save(out, quality=92)
        print(f"saved {out}")


if __name__ == "__main__":
    main()
