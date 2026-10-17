# Report

Demo run: episode 0, all three cameras, 2,452 frames each.

```
Place the yellow cup, white cup, and gray cup into the container. Add the spoon and fork
```

Robustness run: episode 24, which has a different object set (three bowls, a fork and a
spoon, with cups present as distractors). Chunks of episodes 2, 5, 18 and 31 were used during
development.

There is no ground-truth segmentation for this dataset. The numbers below are measured from
the pipeline's own outputs. Every claim about correctness comes from inspecting the annotated
videos at the frames cited.

## Runtime

Measured on an Apple M5 laptop, episode 0, all cameras.

| Camera | Frames | Seconds | s / frame | Detection refresh |
|---|---|---|---|---|
| top | 2,452 | 2,424 | 0.99 | every 15 frames |
| left | 2,452 | 1,152 | 0.47 | every 5 frames |
| right | 2,452 | 2,548 | 1.04 | every 5 frames |
| **all** | 7,356 | 6,134 | **0.83** | |

Time by stage, in seconds:

| Camera | Grounding (DINO + SAM 2 + SigLIP) | Identity check | SAM 2 tracking | Decode + write |
|---|---|---|---|---|
| top | 250 | 61 | 2,103 | 67 |
| left | 724 | 23 | 409 | 18 |
| right | 720 | 106 | 1,783 | 44 |

SAM 2 video propagation dominates the computation time. Its cost grows with the number of objects tracked and with
re-tracking after each correction. Detection runs only on refresh frames. On the top camera
that is 164 of 2,452 frames.

## What works

Episode 0, share of frames where each target has a mask:

| Target | top | left | right |
|---|---|---|---|
| yellow cup | 90% | 6% | 29% |
| white cup | 16% | 0% | 13% |
| gray cup | 74% | 7% | 27% |
| spoon | 91% | 23% | 61% |
| fork | 70% | 0% | 46% |

The white cup's 16% on the top camera is correct behaviour. From frame 450 the yellow cup is
nested inside the white cup in the container, so only the yellow cup is visible from above.
The wrist cameras see each object for only part of the episode, so their low numbers are
mostly expected. Missed frames are covered under failure modes.

Checked visually on the top camera at frames 300, 450, 600, 750, 900, 1050, 1200, 1830 and 1900:
- every visible target is on the correct object;
- IDs stay the same through grasps, moves and occlusion by the arm;
- no distractor (bowls, the container, the gripper) carries a target label.

## Failure modes

Ordered by how much they hurt. The pipeline is tuned to prefer a miss over a wrong label, so
most failures are missing masks.

1. **Cup versus bowl of the same colour.** This is the main source of persistently wrong labeling
   At 360p from above, a cup and a bowl are both coloured circles. Grounding DINO scores them
   about equally and SigLIP reads many cup crops as bowls. When the task
   names a yellow bowl and a yellow cup is in the scene, the yellow cup can win. 
2. **Colour on wrist-camera close-ups.** Glare and blur wash colours out. In episode 0, right
   camera, frames 2110–2155, the "white cup" tag sits on the held yellow cup until the identity
   check cuts it. The hue rule deliberately skips close-ups, because on these views it cut more
   correct tracks than wrong ones.
3. **Low recall on the wrist cameras, especially for cups in the clear container.** The
   container's reflections and the steep angle make SigLIP's noun verdict unreliable, so the
   identity check cuts these tracks. In episode 0 the left camera tracks the cups on under 10%
   of frames. The right camera also loses the cups in the container near the end of the episode
   (from about frame 2170).
4. **Gray read as another colour.** In episode 24 the gray bowl reads as "green bowl" from
   above. "Green bowl" is a negative there, because the task uses green for the spoon and names
   bowls in three colours. The rule exists to stop a gray bowl being labelled "white bowl"
   (episode 5). Its cost here is about 24 points of top-camera recall on the gray bowl.
5. **Ambiguous first frames.** In episode 5, left camera, frames 0–100, the yellow cup is read
   as "yellow bowl". Both models get this view wrong. It corrects itself from frame 150, once
   the cup is seen at an angle. Similar issue to number 1.
6. **No cross-camera reasoning.** Each camera is tracked on its own. The same object gets the
   same ID on every camera because IDs come from the task, but nothing checks that the cameras
   agree.

## Robustness run: episode 24

```
Place the yellow bowl, white bowl, and gray bowl into the container. Then, put the pink fork
and green spoon inside the container as well.
```

The scene has three bowls, a fork and a spoon on the right, and three cups of the same colours
(yellow, white, gray) on the left as distractors. The left arm moves the cups into the
container first, then the right arm moves the bowls. The "pink fork" and "green spoon" are
pale, nearly white. Output: `out/full_ep24_v2/episode_024`, 1,672 frames per camera,
0.92 s per frame across all cameras.

| Target | top | left | right |
|---|---|---|---|
| yellow bowl | 89% | 70% | 18% |
| white bowl | 1% | 20% | 50% |
| gray bowl | 82% | 28% | 24% |
| pink fork | 73% | 29% | 62% |
| green spoon | 66% | 9% | 68% |

These are shares of frames with a mask, not accuracy. Checked visually:

- **Right camera (the arm that moves the bowls): mostly correct.** All three bowls are labelled
  correctly at the start. The white bowl and the pale fork and spoon are followed through the
  grasps. The pale colours do not stop the fork and spoon being found, because single-instance
  objects keep their colour descriptive rather than decisive.
- **Top camera: one long wrong label.** The yellow-bowl tag is right while the bowl is on the
  table and in the gripper. At frame 615 it drifts onto the yellow cup in the container, while
  the real bowl is still held. It stays there until the identity check cuts it at frame 1530.
  That is about 55% of the episode. The gray bowl, fork and spoon are right whenever labelled.
- **Left camera: mostly wrong.** All three cups carry the
  matching bowl labels over frames 0–150. The yellow-bowl tag follows the yellow cup in the
  container from about frame 720 to the end. Only the gray bowl at the end of the episode is
  labelled correctly.

The pipeline is unreliable when the distractors are similar to the targets in colour or shape (the bowls in episode 0), and when objects are not seen from the side. That needs a cue these models do not use, such as relative size or depth.

## With two more weeks

1. **A size-aware tiebreaker for cup versus bowl (failure modes 1, 2 and 5).** 
   This is the first priority, because episode 24 shows it is the largest error. Call a mid-size open VLM only when the assignment is marked
   uncertain, or when the identity check and the detector disagree. That is a few dozen frames
   per episode. A VLM sees the whole frame, so it can use relative size, which tells a cup from
   a bowl.
2. **Cross-camera consistency.** Use the camera geometry and the robot's joint state from the
   dataset to project the gripper into each view. An object held in the gripper is in the
   gripper on every camera. This would recover wrist-camera misses, and
   reject wrist labels the top camera contradicts.