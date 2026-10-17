"""Human-readable progress and summary lines printed during a run."""

from __future__ import annotations

from pathlib import Path

from .parser import ParsedTask

# Tracker decisions (events.jsonl), grouped for the progress lines.
EVENT_GROUPS = {
    "found": ("seed", "reacquire", "hand_over", "reclaim"),
    "corrected": ("drift", "trim"),
    "cut": ("cut_unverified", "cut_on_destination", "cut_mislabeled"),
}


def duration(seconds: float) -> str:
    """42 s · 17 min 18 s · 1 h 02 min"""
    s = round(seconds)
    if s < 60:
        return f"{s} s"
    if s < 3600:
        return f"{s // 60} min {s % 60:02d} s"
    return f"{s // 3600} h {s % 3600 // 60:02d} min"


def header(task: str, parsed: ParsedTask, device: str) -> None:
    print(f"Task     {task}")
    print("Targets  " + " · ".join(f"{t.id} {t.label}" for t in parsed.targets))
    if parsed.destinations:
        print("Into     " + ", ".join(parsed.destinations))
    print(f"Device   {device}")


def models_loaded(seconds: float) -> None:
    print(f"Models   loaded in {duration(seconds)}")


def camera_start(camera: str, n_frames: int, interval: int) -> None:
    title = f" {camera} camera · {n_frames:,} frames · detection every {interval} frames "
    print(f"\n──{title}{'─' * max(0, 76 - len(title))}")
    print(f"  {'frames':<11}  {'visible at chunk end':<44}  tracker decisions")


def chunk(first: int, last: int, visible: list[str], events: list[dict]) -> None:
    counts = [(name, sum(e["event"] in kinds for e in events)) for name, kinds in EVENT_GROUPS.items()]
    decisions = ", ".join(f"{n} {name}" for name, n in counts if n) or "–"
    print(f"  {f'{first}–{last}':<11}  {', '.join(visible) or '–':<44}  {decisions}")


def camera_done(n_frames: int, seconds: float) -> None:
    print(f"  done in {duration(seconds)} ({seconds / max(n_frames, 1):.2f} s per frame)")


def summary(stats: dict, labels: dict[int, str], out: Path) -> None:
    cams = stats["cameras"]
    width = max(len(label) for label in labels.values()) + 2
    line = "─" * 78
    print(f"\n{line}\nShare of frames where each target has a mask\n")
    print(" " * width + "".join(f"{c:>12}" for c in cams))
    for label in labels.values():
        print(f"{label:<{width}}" + "".join(f"{100 * c['visible_fraction'][label]:>11.0f}%" for c in cams.values()))
    print()
    print(f"{'time':<{width}}" + "".join(f"{duration(c['seconds']):>12}" for c in cams.values()))
    print(f"{'s / frame':<{width}}" + "".join(f"{c['sec_per_frame']:>12.2f}" for c in cams.values()))
    total_frames = sum(c["frames"] for c in cams.values())
    print(f"\nTotal {duration(stats['total_seconds'])} for {total_frames:,} frames "
          f"({stats['sec_per_frame_all_cameras']:.2f} s per frame)")
    print(f"Outputs in {out}\n{line}")
