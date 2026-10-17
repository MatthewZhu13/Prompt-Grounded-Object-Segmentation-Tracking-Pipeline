# Prompt-Grounded Object Segmentation & Tracking

An offline pipeline that reads a natural-language robot instruction, finds every object it names
in a LeRobot episode, and writes a per-frame binary mask with a persistent track ID for each
target on every frame of every camera.

```bash
python pipeline.py \
  --task "Place the yellow cup, white cup, and gray cup into the container. Add the spoon and fork" \
  --episode 0 --output-dir out/demo
```

Dataset: [`allenai/15122025-cup-01`](https://huggingface.co/datasets/allenai/15122025-cup-01)
(LeRobot v3, 62 episodes, cameras `top`, `left` and `right`, 640×360 at 30 fps).
All models are open-source, run on a single consumer GPU, and are used zero-shot.

See [`report.md`](report.md) for runtime and failure modes with frame references.

---

## Install

Python 3.11, from a fresh clone:

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt        # torch, transformers, spaCy + en_core_web_sm, PyAV, pyarrow
```

Model weights download from the Hugging Face Hub on first use.
The dataset is fetched from the Hub.

## Usage

| Flag | Meaning |
|---|---|
| `--task` | instruction text (required) |
| `--episode` | episode index (required) |
| `--output-dir` | results go to `<output-dir>/episode_XXX/` |
| `--cameras top left` | subset of cameras (default: all) |
| `--data-dir` | local copy of the dataset |
| `--start`, `--max-frames` | process a frame window, for quick checks |
| `--refresh top=15` | re-detection interval per camera (default 5 for unlisted cameras) |
| `--chunk-size` | frames per tracking chunk, bounds memory (default 150) |
| `--no-video` | skip the overlay videos |

## Outputs

Each run writes `<output-dir>/episode_XXX/`:

| File | Content |
|---|---|
| `targets.json` | parsed task: `text`, `targets` (one per target with `id`, `label`, `noun`, `attributes`) and `destinations` |
| `masks.parquet` | one row per (camera, frame, target), dense: invisible targets get a row too |
| `top.mp4`, `left.mp4`, `right.mp4` | frames with coloured masks and `id: label` tags |
| `events.jsonl` | every tracker decision (seed, reacquire, drift, cut, hand-over) with frame and ID |
| `stats.json` | timing per stage and camera, visible fraction per target, decision counts |

`masks.parquet` columns:

| Column | Type | Notes |
|---|---|---|
| `episode_index`, `frame_index` | int32 | join with the dataset's own parquet on these |
| `camera` | string | `top`, `left` or `right` |
| `track_id`, `label` | int32, string | stable for the whole episode, the same across cameras |
| `visible` | bool | False when the object is not found or out of view |
| `state` | string | `tracking`, `lost` (seen before) or `searching` (not seen yet) |
| `score` | float32 | SAM 2 object-presence probability, 0 when not visible |
| `bbox_x`, `bbox_y`, `bbox_w`, `bbox_h`, `area` | int32 | pixels |
| `height`, `width` | int32 | mask size |
| `rle` | list\<int32\> | COCO uncompressed RLE, column-major |

Decoding a mask:

```python
from segtrack import rle
mask = rle.decode(row["rle"], row["height"], row["width"])   # bool H×W
# or with pycocotools: mask.decode(mask.frPyObjects({"size": [h, w], "counts": counts}, h, w))
```

---

## How it works

```
task text ──► parser ──► targets (id, noun, attributes) + destinations
                              │
per camera, per 150-frame chunk:
  1. track    SAM 2 video propagates every ID visible at the end of the last chunk
  2. detect   on refresh frames (top every 15, wrist every 5):
                Grounding DINO boxes for the task's nouns
              → SAM 2 masks for each box
              → SigLIP 2 scores each masked crop against the target phrases and negatives
              → gates + composite score + Hungarian assignment (one object per target)
  3. verify   SigLIP identity check on every tracked mask at each refresh frame
  4. correct  seed / re-acquire / drift / hand-over / cut, from steps 2–3
  5. re-track SAM 2 from each correction, forward to the chunk end and back across the gap
```

**Parsing** (`segtrack/parser.py`). The parser is rule-based on spaCy part-of-speech tags,
with no hardcoded object names. It splits the instruction into clauses and strips the verb.
It splits off the destination at the first destination preposition, then splits the object
list on commas and conjunctions. Each item is parsed on its own. Its head noun becomes the
target noun and the modifiers before it become attributes. "Blue, white and pink bowls" gives
three bowls. A plural without a count ("cups") becomes one open-ended target. Destinations
("container") are never targets, and they are used as negatives.

**Grounding** (`segtrack/ground.py`, `segtrack/assign.py`). Grounding DINO proposes boxes for
the nouns only. It is good at "cup" but cannot tell "yellow cup" from "gray cup". SAM 2 turns
each box into a mask. SigLIP 2 then scores a masked crop against every target phrase. The crop
is shown on a gray background, so neighbouring objects cannot vote. SigLIP also scores
negatives built from the task such as other nouns with the task's colours ("yellow bowl"), the
destination, and "robot gripper". For close-ups, SigLIP
sees the unblanked crop instead, because a held object's shape needs its context. Hungarian
matching on a weighted product of both scores gives each target at most one object with the highest possible score.

**Tracking and corrections** (`segtrack/pipeline.py`, `segtrack/track.py`). SAM 2 video
carries masks between refresh frames. On each refresh frame, the detections and the identity
check decide per ID:
- **seed / reacquire** when the ID is not tracked and the detection is confident;
- **drift** when a confident detection sits elsewhere on two consecutive refreshes;
- **hand-over** when a detection lands where a different same-noun ID was just lost;
- **cut** when the mask stops looking like its target on two consecutive checks, or lands
  on the destination.

The identity check fails a track when SigLIP reads the masked crop as a different noun, or
as another instance's hue (a white-cup track that reads "yellow cup"). The failure is excused
when the target it read as is tracked elsewhere on that frame. Masks only ever disappear
through cuts.

---

## Model selection

| Role | Model | Why this one |
|---|---|---|
| Open-vocabulary boxes | `IDEA-Research/grounding-dino-tiny` | pip-installable through `transformers` , strong on nouns, fast enough to run on every refresh frame |
| Masks and tracking | `facebook/sam2.1-hiera-small` (image + video) | best available promptable video segmenter that is not gated. The video model keeps identity between refreshes, so detection runs on 1 frame in 5–15 |
| Attributes and identity | `google/siglip2-so400m-patch14-384`, fp16 | the only component that binds colour to object, used on masked crops |
| Task parsing | spaCy `en_core_web_sm` + rules | deterministic, CPU-only, no extra GPU memory |

What was tested and measured:

- **SigLIP 2 so400m vs base.** On hand-labelled crops, so400m got 7 of 7 right and base got
  4 of 7. Masked crops on a gray background beat crops that kept the surroundings or the true
  scale. The cost is about 2.4 s per frame against 0.26 s for base. This is affordable because
  SigLIP runs only on refresh frames.
- **Grounding DINO as a second opinion for small objects.** This was calibrated on 533 tracked
  masks and rejected. For 41% of masks under 1,000 px, Grounding DINO returned no box at all,
  which covers almost every small fork. When it did answer, it was often wrong. It called the
  top-camera gray cup in the container "spoon" (0.46 against 0.18). SigLIP's noun verdict was
  right on 94–100% of masks at every size.
- **Grounding DINO threshold.** Lowered from 0.25 to 0.2 so that held and partly occluded
  objects get a box at all. SigLIP and the negatives reject the extra boxes.

## Design decisions backed by experiments

Each rule below was added for a specific failure. Each was then checked on chunks of several
episodes (0, 5, 18, 24 and 31), and kept only if it did not create wrong labels elsewhere.

- **Masked crops for small objects, unblanked crops for close-ups.** A held bowl's masked crop
  read as "white cup". The unblanked crop shows its shape (episode 0, left camera).
- **The hue rule in the identity check.** A white-cup track sat on the yellow cup nested
  inside it for 1,440 frames (episode 0, top camera). A first version of the rule also cut
  correct gray and yellow cups that glare made read "white" on the wrist cameras. Only
  readings that add a hue now count, and never on close-ups.
- **Removed after an ablation.** An appearance-memory rescue and a small-mask "is it a bowl?"
  test were removed. The bowl test is a coin flip from above: it cut correct cups on episodes 0
  and 31. The memory existed only to undo those cuts. A per-ID size prior was measured and
  rejected, because an object's apparent area changes about 2× between table and container.

## Repository layout

```
pipeline.py            CLI entry point
segtrack/
  parser.py            task text → targets, destinations
  detector.py          Grounding DINO wrapper
  segmenter.py         SAM 2 image (box → mask)
  attributes.py        SigLIP 2 scoring, crops, negatives
  assign.py            gates, composite score, Hungarian matching (NumPy/SciPy only)
  ground.py            one frame: detect → segment → score → assign; identity check
  track.py             SAM 2 video propagation in chunks
  pipeline.py          per-camera tracking, corrections, outputs
  data.py, io.py, rle.py, viz.py, boxes.py, device.py
  seeding.py           best-frame seeding, used only by scripts/track_clip.py
scripts/               single-frame grounding, sweeps, clip tracking, contact sheets
tests/                 parser, negatives, RLE
```
