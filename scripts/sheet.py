"""Tile every Nth frame of a video into one contact-sheet image.

    python scripts/sheet.py out/demo/episode_000/top.mp4 --every 150 --start 300

Saves <video>_sheet_<start>.jpg next to the video.
"""

from __future__ import annotations

import argparse

import av
from PIL import Image


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("video")
    p.add_argument("--every", type=int, default=25, help="take every Nth frame")
    p.add_argument("--start", type=int, default=0, help="first frame")
    p.add_argument("--count", type=int, default=12, help="number of tiles")
    p.add_argument("--cols", type=int, default=4)
    p.add_argument("--width", type=int, default=480, help="tile width in pixels")
    args = p.parse_args()

    tiles = []
    with av.open(args.video) as container:
        for i, frame in enumerate(container.decode(video=0)):
            if i < args.start or (i - args.start) % args.every:
                continue
            img = frame.to_image()
            tiles.append(img.resize((args.width, round(img.height * args.width / img.width))))
            if len(tiles) == args.count:
                break
    if not tiles:
        raise SystemExit("no frames in range")

    w, h = tiles[0].size
    rows = -(-len(tiles) // args.cols)
    sheet = Image.new("RGB", (w * args.cols, h * rows))
    for k, tile in enumerate(tiles):
        sheet.paste(tile, ((k % args.cols) * w, (k // args.cols) * h))
    out = args.video.rsplit(".", 1)[0] + f"_sheet_{args.start}.jpg"
    sheet.save(out, quality=90)
    print(out)


if __name__ == "__main__":
    main()
