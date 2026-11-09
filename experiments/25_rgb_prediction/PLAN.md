# Plan: SQ decomposition from a single RGB view

Status: draft (2026-11-08), nothing implemented. Builds on experiment 22 (single-view RGB-D predictor) and on
experiments 12 / 15 (monocular depth for the fitting baselines).

## Question
How much is lost when depth is removed, and where should the 3D come from: an off-the-shelf metric depth network
(Depth Pro), or a depth head trained end to end with the SQs?

Reference points:
- Exp 22's DINO-only model already uses depth only for (i) the patch centroids (positional encoding, anchors) and
  (ii) the per-pixel assignment of targets to patches (this needs GT depth only at train time, which is fine).
  Its DINO-only run is as good as DINO + Sonata, so "RGB" here means replacing the centroids.
- Exp 15: fitting from Depth Pro (single view) gives vIoU 0.19 against 0.43 with RGB-D, and 0.33 with an
  oracle per-view scale. The metric scale error (2-10 %) was the bottleneck. A learned model sees GT targets,
  so it can learn to correct systematic scale errors.

## Variants (all exp 22's top-k, DINO only, lr 3e-3, warm-up 5 %, mask loss; targets unchanged, GT camera frame)
- **A, oracle:** GT depth centroids, i.e. exp 22's DINO-only run. Upper bound, no new run.
- **B, Depth Pro:** patch centroids from Depth Pro's metric depth (known focal of 600 px). Training on predicted
  depth too, so that the model learns its errors. Needs Depth Pro for all 1600 cached frames (cached now: every 20th
  frame of room_0 / room_1 / office_0) at the /4 resolution, about 10-15 min on the 5090. `Thesis/data/replica_mono_depth`.
- **C, learned depth:** a per-patch depth head on the DINO tokens (the median depth of the patch's pixels as target,
  log-L1). Centroid = the patch's ray x the predicted depth, detached or not (option). No external depth network.
- **D, no 3D input:** 2D (u, v) Fourier positions only. The SQ translation becomes the ray of the anchor patch times
  a regressed depth (as in monocular 3D detection, e.g. Cube R-CNN). Probably close to C, so only run it if C is
  interesting.

## Evaluation
As exp 22 (pair IoU seen / partial, occupancy IoU, recalls, masks, class), plus the depth error of the centroids
(AbsRel) and the pair IoU after aligning the scale per frame (separates scale from shape / layout errors, like
exp 15's oracle). Settings: f16 (train = eval; tests only whether it can fit), then full (test room_0).

Expectation: on f16, B ~ A. On full, everything fails as in exp 22 (pair IoU ~0.10); the interesting number is
then the gap A -> B -> C on the training frames and on `--split frames`.

## Steps
1. Cache Depth Pro for the 1600 frames (`cache_depth.py`, reuses exp 12's `depth.py`), plus per-scene AbsRel /
   scale-error statistics.
2. A swap of the centroid source in exp 22's `Frames` (`--centroids gt|depthpro|learned`). The target assignment
   keeps GT depth.
3. f16 for B and C, then full. Every run only after you say go.

## Code
Imports exp 22's code (as exp 23 does). New: the depth cache, the centroid option, and the depth head.
