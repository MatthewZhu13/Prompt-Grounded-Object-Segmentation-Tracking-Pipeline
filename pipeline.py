"""Prompt-grounded object segmentation and tracking for LeRobot episodes.

    python pipeline.py --task "Place the yellow cup, white cup, and gray cup into the container. Add the spoon and fork" \\
                       --episode 0 --output-dir out/

Writes to <output-dir>/episode_<NNN>/:
    <camera>.mp4       overlay video per camera (masks + "id: label")
    masks.parquet      one row per (camera, frame_index, track_id); RLE masks, see README for schema
    targets.json       parsed task: track id → phrase, noun, attributes
    stats.json         runtime (sec/frame per camera and stage), device, visibility per track
    events.jsonl       every tracker decision (seed, re-acquire, drift, cut...), with frame and track id
"""

from __future__ import annotations

import argparse

from segtrack.assign import AssignConfig
from segtrack.data import DEFAULT_REPO
from segtrack.pipeline import PipelineConfig, run_episode


def parse_refresh(spec: str) -> dict[str, int]:
    """"top=15,left=5" → {"top": 15, "left": 5}"""
    out = {}
    for part in filter(None, spec.split(",")):
        name, _, value = part.partition("=")
        out[name.strip()] = int(value)
    return out


def quiet_libraries() -> None:
    """Hide Hugging Face loading bars and config warnings, which bury the progress lines."""
    import warnings

    from huggingface_hub.utils import disable_progress_bars
    from transformers.utils import logging as hf_logging

    hf_logging.set_verbosity_error()
    hf_logging.disable_progress_bar()
    disable_progress_bars()
    warnings.filterwarnings("ignore", category=UserWarning)
    warnings.filterwarnings("ignore", category=FutureWarning)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--task", required=True, help="natural-language instruction naming the objects")
    p.add_argument("--episode", type=int, required=True, help="episode index")
    p.add_argument("--output-dir", required=True)
    p.add_argument("--cameras", nargs="+", help="subset of cameras (default: all)")
    p.add_argument("--repo", default=DEFAULT_REPO, help="Hugging Face dataset repo (LeRobot v3)")
    p.add_argument("--data-dir", help="local copy of the dataset; missing files are downloaded")
    p.add_argument("--device", help="cuda / mps / cpu (default: best available)")
    p.add_argument("--start", type=int, default=0, help="first frame")
    p.add_argument("--max-frames", type=int, help="process at most this many frames per camera")
    p.add_argument("--chunk-size", type=int, default=150, help="frames per tracking chunk (memory bound)")
    p.add_argument("--refresh", default="top=15",
                   help="re-detection interval per camera, e.g. 'top=15,left=5' (others: --default-refresh)")
    p.add_argument("--default-refresh", type=int, default=5)
    p.add_argument("--no-video", action="store_true", help="skip writing overlay videos")
    p.add_argument("--verbose", action="store_true", help="show model-loading progress bars and warnings")
    args = p.parse_args()
    if not args.verbose:
        quiet_libraries()

    cfg = PipelineConfig(
        chunk_size=args.chunk_size,
        refresh=parse_refresh(args.refresh),
        default_refresh=args.default_refresh,
        start=args.start,
        max_frames=args.max_frames,
        write_video=not args.no_video,
        assign=AssignConfig(),
    )
    run_episode(args.task, args.episode, args.output_dir, cameras=args.cameras, cfg=cfg,
                data_dir=args.data_dir, repo_id=args.repo, device=args.device)


if __name__ == "__main__":
    main()
