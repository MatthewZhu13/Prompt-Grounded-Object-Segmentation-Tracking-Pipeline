"""Ground every target in a single frame and save an annotated image.

    python scripts/ground_frame.py --episode 0 --camera top --frame 0

Prints the score matrices behind each decision and per-model timings.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from segtrack.assign import AssignConfig                     # noqa: E402
from segtrack.data import Dataset, iter_frames                # noqa: E402
from segtrack.device import device_name, get_device           # noqa: E402
from segtrack.ground import FrameGrounder, FrameGrounding     # noqa: E402
from segtrack.parser import ParsedTask, parse_task            # noqa: E402
from segtrack.viz import draw_boxes, id_color, overlay_masks  # noqa: E402

GRAY = (160, 160, 160)


def render(frame: np.ndarray, g: FrameGrounding, parsed: ParsedTask) -> Image.Image:
    by_id = {t.id: t for t in parsed.targets}
    owner = g.result.candidate_to_target()
    image = Image.fromarray(overlay_masks(frame, {tid: g.masks[c] for c, tid in owner.items()}))
    boxes = []
    for c, box in enumerate(g.boxes):
        if c in owner:
            tid = owner[c]
            m = next((m for m in g.result.matches if m.candidate == c), None)
            flag = "?" if m is not None and m.status == "uncertain" else ""
            boxes.append((box, f"{tid}: {by_id[tid].label}{flag}", id_color(tid), 2))
        else:
            boxes.append((box, "x" if g.result.rejected[c] else "", GRAY, 1))
    return draw_boxes(image, boxes)


def print_report(g: FrameGrounding, parsed: ParsedTask) -> None:
    targets = parsed.targets
    col = lambda s: f"{s[:11]:>11}"
    print("\nCandidates (DINO score for each target's noun | SigLIP score for each target's phrase)")
    print(f"{'cand':>5} {'box':>22} | " + " ".join(col(t.label) for t in targets)
          + " | best negative")
    for c in range(len(g.boxes)):
        box = "[" + ",".join(f"{v:.0f}" for v in g.boxes[c]) + "]"
        neg = f"{g.negatives[g.siglip_neg[c].argmax()]} {g.siglip_neg[c].max():.3f}" if g.negatives else ""
        flag = "  REJECTED" if g.result.rejected[c] else ""
        print(f"{c:>5} {box:>22} | " + " ".join(f"{g.dino[c, t]:>5.2f}/{g.siglip[c, t]:<5.3f}"
                                                  for t in range(len(targets))) + f" | {neg}{flag}")
    print("\nAssignment")
    for m, t in zip(g.result.matches, targets):
        if m.candidate is None:
            print(f"  {t.id}: {t.label:16} → not found")
            continue
        fmt = lambda x: "inf" if not np.isfinite(x) else f"{np.exp(x):.2f}x"
        print(f"  {t.id}: {t.label:16} → cand {m.candidate:<3} composite {np.exp(m.score):.3f}  "
              f"vs next cand {fmt(m.margin):>6}  vs sibling phrase {fmt(m.sibling_margin):>6}  [{m.status}]")
    for tid, cands in g.result.extra.items():
        print(f"  {tid}: extra instances → candidates {cands}")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--episode", type=int, default=0)
    p.add_argument("--camera", default="top")
    p.add_argument("--frame", type=int, default=0)
    p.add_argument("--task", help="instruction (default: the episode's annotated task)")
    p.add_argument("--data-dir", default="data/cup-01")
    p.add_argument("--out", default="out/ground")
    p.add_argument("--device")
    p.add_argument("--dino-threshold", type=float, default=AssignConfig.dino_threshold)
    p.add_argument("--plain-siglip-threshold", type=float, default=AssignConfig.plain_siglip_threshold)
    p.add_argument("--repeat", type=int, default=2, help="runs (first includes warm-up) for timing")
    args = p.parse_args()

    ds = Dataset(data_dir=args.data_dir)
    episode = ds.episode(args.episode, cameras=[args.camera])
    task = args.task or ds.annotated_task(args.episode)
    if not task:
        sys.exit("no --task given and the dataset has no annotated task for this episode")
    parsed = parse_task(task)
    frame = next(iter_frames(episode, args.camera, start=args.frame, stop=args.frame + 1)).frames[0]

    device = get_device(args.device)
    print(f"device: {device_name(device)}")
    print(f"task:   {task}")
    print("targets: " + ", ".join(f"{t.id}={t.label}" for t in parsed.targets))
    cfg = AssignConfig(dino_threshold=args.dino_threshold, plain_siglip_threshold=args.plain_siglip_threshold)
    grounder = FrameGrounder(parsed, cfg, device=device)
    print(f"negatives: {grounder.negatives}")

    for run in range(args.repeat):
        g = grounder(frame)
        print(f"run {run + 1} timings: " + ", ".join(f"{k} {v:.3f}s" for k, v in g.timings.items())
              + f", total {sum(g.timings.values()):.3f}s")
    print_report(g, parsed)

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    path = out / f"ep{args.episode:03d}_{args.camera}_f{args.frame:05d}.png"
    render(frame, g, parsed).save(path)
    print(f"\nsaved {path}")


if __name__ == "__main__":
    main()
