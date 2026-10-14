"""Run single-frame grounding on frames sampled across an episode and tile them into one image.

    python scripts/ground_sweep.py --episode 0 --cameras left right --step 200

Models load once; each tile is one frame's grounding result (same drawing as ground_frame.py).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from PIL import Image, ImageDraw

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ground_frame import render                               # noqa: E402
from segtrack.data import Dataset, iter_frames                # noqa: E402
from segtrack.device import get_device                        # noqa: E402
from segtrack.ground import FrameGrounder                     # noqa: E402
from segtrack.parser import parse_task                        # noqa: E402


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--episode", type=int, default=0)
    p.add_argument("--cameras", nargs="+", default=["left", "right"])
    p.add_argument("--step", type=int, default=200)
    p.add_argument("--task")
    p.add_argument("--data-dir", default="data/cup-01")
    p.add_argument("--out", default="out/ground")
    p.add_argument("--cols", type=int, default=4)
    p.add_argument("--tile-width", type=int, default=480)
    args = p.parse_args()

    ds = Dataset(data_dir=args.data_dir)
    episode = ds.episode(args.episode, cameras=args.cameras)
    parsed = parse_task(args.task or ds.annotated_task(args.episode))
    grounder = FrameGrounder(parsed, device=get_device())
    labels = {t.id: t.label for t in parsed.targets}

    for cam in args.cameras:
        tiles = []
        for chunk in iter_frames(episode, cam, chunk_size=args.step):
            idx, frame = int(chunk.frame_indices[0]), chunk.frames[0]
            g = grounder(frame)
            found = {m.target_id: m.status for m in g.result.matches if m.candidate is not None}
            print(f"{cam} f{idx:05d}: " + ", ".join(
                f"{labels[t]}{'?' if s == 'uncertain' else ''}" for t, s in found.items()) or "-")
            tile = render(frame, g, parsed)
            ImageDraw.Draw(tile).text((4, tile.height - 14), f"{cam} f{idx}", fill=(255, 255, 0))
            tiles.append(tile.resize((args.tile_width, int(tile.height * args.tile_width / tile.width))))

        rows = (len(tiles) + args.cols - 1) // args.cols
        tw, th = tiles[0].size
        sheet = Image.new("RGB", (tw * args.cols, th * rows), (20, 20, 20))
        for k, tile in enumerate(tiles):
            sheet.paste(tile, ((k % args.cols) * tw, (k // args.cols) * th))
        out = Path(args.out) / f"sweep_ep{args.episode:03d}_{cam}.jpg"
        out.parent.mkdir(parents=True, exist_ok=True)
        sheet.save(out, quality=90)
        print(f"saved {out}")


if __name__ == "__main__":
    main()
