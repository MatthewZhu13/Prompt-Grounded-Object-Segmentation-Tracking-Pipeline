"""Tile every Nth frame of a video into a contact sheet.  python scripts/sheet.py VIDEO [--every 25] [--start 0]"""
import argparse
import av
from PIL import Image

p = argparse.ArgumentParser(); p.add_argument("video"); p.add_argument("--every", type=int, default=25)
p.add_argument("--start", type=int, default=0); p.add_argument("--count", type=int, default=12)
p.add_argument("--cols", type=int, default=4); p.add_argument("--width", type=int, default=480)
a = p.parse_args()
c = av.open(a.video); tiles = []
for i, f in enumerate(c.decode(c.streams.video[0])):
    if i >= a.start and (i - a.start) % a.every == 0:
        img = f.to_image(); tiles.append(img.resize((a.width, int(img.height * a.width / img.width))))
        if len(tiles) == a.count: break
w, h = tiles[0].size; rows = (len(tiles) + a.cols - 1) // a.cols
sheet = Image.new("RGB", (w * a.cols, h * rows))
for k, t in enumerate(tiles): sheet.paste(t, ((k % a.cols) * w, (k // a.cols) * h))
out = a.video.rsplit(".", 1)[0] + f"_sheet_{a.start}.jpg"; sheet.save(out, quality=90); print(out)
