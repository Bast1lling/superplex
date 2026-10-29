# 12 — SQs from rendered views: RGBD + instance labels, multi-view (and the shared method of 12–15)

**Question (12–15).** How far do we get with only rendered views + per-pixel instance labels (no reconstructed point
cloud / mesh)? Four settings: 12 RGBD multi-view, 13 RGB multi-view, 14 RGBD single view, 15 RGB single view.
Representation fixed to **one SQ per instance** (the presentation's `instances` method), so the experiments measure
the information, not the decomposition.

## TL;DR (all four settings)

Volumetric IoU per instance against the full 3D GT (mean over instances; for single views only the instances seen in
that view); "upper bound" = the same fit on the voxel grid of the complete GT point cloud, on the same instances.

| setting | runs | vIoU | precision | recall | upper bound | rel |
|---|---|---|---|---|---|---|
| 12 RGBD multi (100 views) | 3 scenes | 0.608 | 0.721 | 0.792 | 0.701 | **0.87** |
| 13 RGB multi, Depth Pro + multi-view scale alignment | 3 scenes | 0.401 | 0.500 | 0.679 | 0.701 | **0.57** |
| 14 RGBD single view | 3 scenes × 5 frames | 0.433 | 0.683 | 0.615 | 0.687 | **0.63** |
| 15 RGB single view, Depth Pro | 3 scenes × 5 frames | 0.191 | 0.402 | 0.269 | 0.688 | **0.28** |
| 15 control: Depth Pro with oracle per-view scale | 3 scenes × 5 frames | 0.333 | 0.636 | 0.456 | 0.687 | 0.49 |

- **RGBD multi-view is essentially solved** by carving a label grid from the views with two rules (solid up to
  10 cm behind an observed surface, nothing beyond the observed bounding box): 87 % of the 3D-GT upper bound, already
  with 40 views (10 views: 0.59 vs 0.62 on room_0). Renders look like the 3D-GT fit.
- **Single view RGBD** loses mostly recall (backs, occluded and out-of-frame parts are not observed): 63 %.
- **RGB = depth estimation.** With a focal-aware monocular metric depth network (Depth Pro, given the known focal
  length) RGB becomes RGBD with errors. Multi-view: the metric poses let us estimate a scale per view (+ a global
  one) from multi-view consistency, which brings Depth Pro's per-view scale error from 2.8 % to 0.7 % (median) and
  rescues Depth Anything V2 (27 % scale error → IoU 0.001 without, 0.17 with alignment). Single view: no alignment
  possible; 2–10 % scale error shifts surfaces by 5–30 cm, more than walls / table tops are thick → 0.19 (oracle
  scale: 0.33). The metric scale is the bottleneck of single-view RGB.
- **Depth-free (masks only) does not reach depth networks**: differentiable rendering of the SQ occupancies against
  the multi-view instance masks (13) gives 0.16, with a volume prior 0.27 (room_0), vs. 0.49 with Depth Pro;
  refining the Depth Pro fit with mask rendering makes it worse (0.44).

Scene-wise vIoU (12 / 13 / 14 / 15 / 15 oracle scale): room_0 0.63 / 0.49 / 0.45 / 0.20 / 0.36,
room_1 0.64 / 0.41 / 0.44 / 0.14 / 0.34, office_0 0.55 / 0.30 / 0.40 / 0.23 / 0.30.

## Method (`code/`, shared by 12–15)
- **Inputs** (`views.py`): NICE-SLAM Replica renderings (experiment 11; same world frame as the meshes), 4×
  downsampled to 300 × 170 (pixel centers, so depth stays exact), camera poses + intrinsics. Per-pixel instance
  labels as in 11 (back-projected GT depth → nearest labelled 1 cm point within 2 cm); in the RGB settings the GT
  depth is only used to make these labels (they stand in for a 2D instance segmentation).
  Multi-view: every 20th frame (100 views); single view: frames 0, 400, 800, 1200, 1600.
- **Label grid from views** (`grid.carve_grid`, 4 cm): every voxel center is projected into every view. Free vote if
  in front of the observed depth (by > 4 cm), solid vote with the pixel's instance if between 4 cm in front of and
  `thick` = 10 cm behind it. Free if free votes win, else the instance with most solid votes, else unknown (never
  seen; deeper inside objects). Beyond the bounding box of all observed points (+ 8 cm): free.
  Both rules mirror the GT grid (`grid.gt_grid` = the presentation's voxelization: k-NN normal vote, solid only up
  to 10 cm from the surface, outside = free) and are essential: without the 10 cm solid band 0.34, with it 0.62
  (room_0); outside = free mainly fixes walls bulging outwards (structure 0.62 → 0.69).
- **Fit** (`fit.py`): one SQ per instance with the presentation's instances loss (positives = the instance's
  voxels, negatives = box samples: free 1, unknown `w_unk` = 0.1, other instances 0), radial SDF, Adam 2000 steps.
- **Evaluation** (`fit.evaluate`): per GT instance, 4 cm GT grid, TP = GT solid voxels of k inside its SQ, FP = free
  voxels inside (space around the grid counted free), IoU = TP / (|GT_k| + FP). `compare.py` adds the upper bound on
  the same instances. Note: one SQ per instance caps precision at ~0.72 even with full 3D GT.
- **Predicted depth** (`depth.py`, experiments 13 / 15): Depth Pro (Apple, HF `apple/DepthPro-hf`) rescaled from its
  own focal estimate to the true one; Depth Anything V2 Metric Indoor for comparison. `align_scales`: global scale
  by grid search (share of reprojected neighbor-view points agreeing within 5 %), then per-view scales by Jacobi
  iterations (median depth ratio of label-consistent reprojections from ±4 neighbor views).
- **Mask rendering** (`render_fit.py`, experiment 13): see its docstring and 13's README.

## Results of 12 (RGBD multi-view, room_0 sweeps)

| variant | vIoU | prec | rec | note |
|---|---|---|---|---|
| first version (surface band only, outside unknown) | 0.492 | 0.633 | 0.687 | walls / floor bulge outwards |
| + solid 10 cm behind surfaces, outside free | 0.623 | 0.682 | 0.892 | |
| w_unk 0 / 0.03 / 0.3 / 1 | 0.599 / 0.613 / 0.618 / 0.594 | | | flat around 0.1 |
| thick 0.05 / 0.2 | 0.544 / 0.624 | | | |
| 400 / 40 / 10 views | 0.630 / 0.626 / 0.586 | | | saturates at ~40 views |
| erode 1 (final) | 0.630 | 0.746 | 0.804 | label erosion, see 13 |

All scenes: room_0 0.630, room_1 0.641, office_0 0.551 (upper bound 0.707 / 0.694 / 0.700).

## Files
- `code/views.py` inputs, `grid.py` GT / carved label grids, `sq.py` SQ model, `fit.py` fit + evaluation + render,
  `depth.py` monocular depth + scale alignment, `render_fit.py` mask rendering, `compare.py` vs. upper bound.
- `run_all.sh` main runs of 12–15 on all scenes; `results/runs/<run>/{config.json, metrics.json, sqs.npz,
  render.jpg}` (12: `<scene>_gt3d` = upper bound, `<scene>_rgbd`); `results/logs/` sweeps and the all-runs table.

      cd code && uv run python fit.py --exp 12 --name <run> --set scene=room_1 [--set frames=stride50 --set w_unk=0.3 ...]
      uv run python compare.py 12:room0_rgbd 13:room0_depthpro ...
