# 22 — RGBD prediction: SuperFlex targets for the NICE-SLAM scenes, and a single-frame predictor

**Goal.** A target decomposition for every scene with RGB-D sequences (NICE-SLAM's 8 Replica renderings: room 0–2,
office 0–4), as training data for predicting SQs from RGB-D later.

**Targets** (`code/fit_targets.py`). Experiment 16's best run, `network / theirs`, with plain SQs, on all 8 scenes: per GT
instance (labelled class, >= 80 points of the 1 cm cloud) 4096 surface points and 100k query points labelled from the
4 cm GT grid, SuperFlex robust as initialization with tapering and bending set to 0, then 1000 Adam steps of SuperFlex's own
objective without them (on room_0 in experiment 16: vIoU 0.817 against 0.831 with them), best state per instance. Scored with experiment 12's per-instance vIoU. Primitives in the Replica world frame:
`results/targets/<scene>/sqs.npz` (per instance 16 slots, `exist` > 0.5 used; `tapering` and `bending` are 0), per-instance metrics in `metrics.json`.

    ./run_all.sh                                              (one process per scene, skips finished ones)
    uv run python viewer/export.py
    python -m http.server -d experiments/22_rgbd_prediction/viewer 8022      -> http://localhost:8022

**Results** (`results/targets/` = filled, `results/targets_solid/` = solid only)

| scene | instances | primitives | vIoU filled | vIoU solid only | precision filled / solid |
|---|---|---|---|---|---|
| room_0 | 79 | 250 | 0.819 | 0.808 | 0.94 / 0.95 |
| room_1 | 51 | 149 | 0.783 | 0.820 | 0.87 / 0.94 |
| room_2 | 58 | 201 | 0.781 | 0.821 | 0.90 / 0.95 |
| office_0 | 40 | 162 | 0.753 | 0.751 | 0.90 / 0.93 |
| office_1 | 37 | 101 | 0.720 | 0.799 | 0.82 / 0.94 |
| office_2 | 49 | 178 | 0.789 | 0.831 | 0.87 / 0.94 |
| office_3 | 66 | 237 | 0.791 | 0.829 | 0.88 / 0.94 |
| office_4 | 43 | 188 | 0.772 | 0.791 | 0.89 / 0.93 |
| mean | | | 0.776 | 0.806 | |

**Walls in the filled targets are slabs.** Experiment 16's occupancy gives every UNKNOWN voxel (behind a surface,
> 10 cm from it) to the nearest solid instance. Behind a wall that is all space up to the grid's edge: 72-95 % of a
wall's occupied query points in office_0, room_1, office_3 come from this fill (room_0: 29-64 %, its walls lie at the
scene's edge), and the fit covers it (wall half-thickness up to 1.6 m, precision down to 0.07). Experiment 16's room_1
and office_0 numbers include this. Solid only: UNKNOWN counts as outside, which removes the slabs (precision 0.93-0.95).

**Iterative fill** (`code/fill_grid.py`): an UNKNOWN voxel joins instance k when >= 10 of its 26 neighbours are solid
voxels of k, repeated until nothing changes (a flat surface gives at most 9). It fills sofas (+87 % voxels in room_0,
+213 % in office_0) but leaks: once a concave corner fills, the next voxel along it again has >= 10, so it runs along
walls layer by layer (walls +136-319 %, office_3's ceiling +406 %). Not used for fitting.

**Winding numbers** (`code/winding.py`, `uv run --with libigl`, all 8 scenes, `results/large_files/winding/`). Fast
winding number of oriented point clouds (Barill et al. 2018, libigl) at every UNKNOWN voxel, area per point from its
16th neighbour. Per instance (max over instances of the winding number of their own points), share of UNKNOWN voxels
each class would gain:

| | w >= 0.5 | w >= 0.75 |
|---|---|---|
| walls, floors, ceilings | 0-1 % (office_3 ceiling 6 %) | 0 % |
| sofas | +31 to +74 % | 0 to +33 % |
| cabinets, nightstands | +41 to +60 % | +3 to +22 % |
| chairs, benches, tables, desks | +2 to +39 % | 0 to +17 % |

So it closes objects and leaves flat walls alone, unlike both rules above. Over all points of the scene at once it
gives ~0 everywhere: the room's own surface faces inwards (w = -1 inside the room), which cancels the objects in it.
Not used for fitting yet.

**Unlabelled instances.** Replica leaves some objects without a class (class id -1: `non-plane`, `undefined`,
`anonymize_picture`), and experiment 12's GT drops them. Largest: office_4's conference table (`non-plane`, 155k
points), `undefined` instances of 72k (office_0) and 66k points (office_1).

**Viewer** (`viewer/`, experiment 16's page). Per scene both targets (keys 1, 2) colored by instance / primitive /
instance IoU, GT points and solid voxels, the occupancy grids (filled, grown, winding >= 0.5; voxels made solid from UNKNOWN are dark),
a height cut; click an object for its class and IoU. Orbs at every n-th camera of the NICE-SLAM renderings, click one
for its RGB frame (`uv run python viewer/export_views.py` links them). Points in their Replica colors (sRGB).

**Per-instance occupancy** (`code/occupancy.py`, `uv run --with libigl`): per GT instance, the voxels (4 cm, experiment
12's grid) holding >= 1 of its points (a voxel can belong to several instances) plus the voxels in its box where the
winding number of its own points is >= 0.5. No normal vote, no UNKNOWN. Walls and floors stay at their surface voxels
(+0-28 %, office_3's ceiling +48 %), sofas +195-316 %, cabinets about +270 %. `fit_targets.py --occ winding` fits on it
and evaluates on it (FP = voxels no instance occupies): vIoU 0.749-0.824, mean 0.791, precision 0.82-0.88 (the old
grids' 10 cm shell behind every surface is free space now), recall 0.89-0.92. These are the targets of the predictor.

## Single-frame predictor (design: `PLAN.md`)

**Per-frame targets** (`code/frame_stats.py`, `results/frame_stats.json`): every target SQ intersecting the camera
frustum. Median 30-98 per frame, max 187 (office_3); 20-35 % belong to instances not visible at all. -> 200 queries.

**Cache** (`/home/sebastian/Documents/Uni/Thesis/data/replica_rgbd_frames`, 15 GB, every 10th frame, 1600 frames):
`code/cache_frames.py` (DINOv3 ViT-L/16 tokens 42 x 75, computed in bfloat16 since float16 overflows to NaN; the /4
view; targets in the camera frame), then `code/cache_sonata.py` in experiment 04's environment (Sonata on the frame's
own depth, world frame, its 3 coarsest stages).

**Model** (`code/model.py`): frozen DINOv3 + frozen Sonata -> per-patch tokenizer (tiny MLP on Sonata features,
shared point MLP max/mean-pooled over the patch's 16 pixels, fused with the DINO token, Fourier encoding of the patch's
3D centroid) -> SuperFlex's decoder (3 layers, d 128, 1 head), 200 queries -> plain SQ, existence, class, mask
embedding. **Training** (`code/train.py`, loss in its docstring): 7 scenes, test room_0, trackio dashboard
`code/dashboard.py` (port 7863). **Viewer** `viewer/frames.html` (`uv run python viewer/export_frames.py <run>`): per
test frame RGB with predicted / GT instance masks, depth points, predicted and target SQs.

Not yet: crops / scale jitter (the cache holds whole frames), the unsupervised terms, the later ideas of `PLAN.md`.

**First run `base`** (150 epochs, 55 min, `results/base.json`). Test room_0 / 100 training frames:

| | matched Chamfer | targets found (exist > 0.5) | class acc. | occupancy IoU | mask coverage |
|---|---|---|---|---|---|
| test | 0.82 m | 0.27 | 0.51 | 0.045 | 0.17 |
| train | 0.23 m | 0.40 | 0.97 | 0.046 | 0.23 |

It fails on the training frames too. Targets line up with the depth points (viewer), so the data is right. Causes seen:
- Existence too low: matched queries median 0.37 (unmatched 0.18), walls / floors / ceilings found 8 %, objects 29 %.
  About 33 of 98 SQs per test frame are switched on, so the occupancy IoU stays near 0.
- Shape does not learn: predicted exponents stay at ~1.0 (the sigmoid's start), targets ~0.2 (boxy). The regular
  (eta, omega) grid puts boxy SQs' samples near their edges, so Chamfer hardly constrains the exponents.
- Translation: matched SQs are a median 0.3 m off. Queries regress absolute camera-frame positions from their
  sinusoidal encoding alone.

**Overfitting one frame** (`run_overfit.sh`, room_2_000500, 112 targets, 3000 steps; base = 200 learned queries,
patch = a query per patch with its centroid as anchor). NMS only with `iou_exist`.

| run | matched Chamfer | pair IoU | occupancy IoU (exist > 0.5 / 0.3) | found (0.5 / 0.3) | predicted (0.3) | mask coverage |
|---|---|---|---|---|---|---|
| base, Chamfer | 0.053 m | 0.48 | 0.65 / 0.64 | 1.00 / 1.00 | 115 | 0.83 |
| base, volumetric | 3.73 m | 0.00 | 0 / 0 | 0.92 / 0.94 | 113 | 0.46 |
| patch, Chamfer | 0.031 m | **0.74** | 0.48 / **0.73** | 0.96 / 0.99 | 113 | 0.75 |
| patch, volumetric | 0.23 m | 0.64 | 0.15 / 0.16 | 0.96 / 0.99 | 112 | 0.75 |
| patch, volumetric + IoU-aware existence | 0.23 m | 0.64 | 0.15 / 0.17 | 0.59 / 0.70 | 147 | 0.51 |
| patch, Chamfer + volumetric | 0.035 m | **0.81** | 0.10 / 0.43 | 0.64 / 0.88 | 113 | 0.41 |
| patch, Chamfer, mask = visible part of the SQ | 0.029 m | 0.75 | 0.56 / 0.68 | 0.96 / 0.99 | 113 | (0.52) |
| patch, Chamfer, mask = whole SQ's silhouette | 0.030 m | 0.73 | 0.51 / 0.72 | 0.95 / 0.99 | 112 | (0.30) |

- The 800-step overfit runs were simply too short: with 3000 steps both Chamfer runs fit the frame, and existence
  sorts itself out (patch: 113 SQs above 0.3 for 112 targets, without NMS).
- Patch anchors give the better geometry (pair IoU 0.74 against 0.48).
- The volumetric loss alone fails: truncated at 0.1 m it has no gradient for SQs further than that from their target,
  so learned queries never find them (3.7 m off) and with anchors the far ones stay far (Chamfer 0.23 m). It needs a
  far-field term (Chamfer) next to it.
- IoU-aware existence does not help here: fewer targets found, more duplicates. Dropped (code removed).
- Chamfer + volumetric: the best shapes (pair IoU 0.81 against 0.74, Chamfer about the same), but existence lags
  (found 0.88 at 0.3 against 0.99), so the union IoU and the masks are lower. Its occupancy IoU at 0.3 stayed at
  0.07-0.09 until step 2250 and then jumped to 0.35-0.43: existence was still catching up when the run ended.
- Per-SQ masks instead of instance masks (`--mask sq`, `--mask sq_amodal`) leave the geometry where it was (pair IoU
  0.73-0.75, occupancy 0.68-0.72 at 0.3; single runs on one frame, so differences of this size are noise). The masks
  themselves: visible part 0.48 IoU per SQ (many visible parts are smaller than a 16 px patch), silhouette 0.83 (it
  follows from the geometry). Mask coverage (in brackets) groups queries by mask overlap into instances, which per-SQ
  masks are not meant for, so it is not comparable there.

**Full run, patch queries + Chamfer** (`patch_cd`: 150 epochs, 26k steps, 77 min; train = 7 scenes, test = room_0;
train eval = 100 training frames; `results/patch_cd.{json,log}`, viewer `frames.html`).

| set | matched Chamfer | pair IoU | class acc. | found (0.5 / 0.3) | predicted (0.5 / 0.3) / targets | occupancy IoU (0.3) | mask coverage |
|---|---|---|---|---|---|---|---|
| train frames | 0.12 m | 0.30 | 1.00 | 0.08 / 0.30 | 6 / 45 / 60 | 0.02 | 0.03 |
| test (room_0) | 0.35 m | 0.10 | 0.66 | 0.04 / 0.16 | 4 / 40 / 98 | 0.03 | 0.02 |

- A failure on the training frames too, not only a generalization gap: the model underfits geometry and existence
  while it memorizes classes (1.00 train, 0.66 test).
- Existence: matched test queries median 0.18 (10 % / 90 %: 0.07 / 0.48); the 10th-best query of a frame 0.45, the
  100th 0.23. Focal-loss scores are low by design, but even at 0.3 only 30 % of training targets are found.
- Test metrics plateau from epoch ~60 on (Chamfer 0.37 -> 0.35); the loss keeps falling (29.7 -> ~4).
- Contrast with the one-frame overfit (pair IoU 0.74, 99 % found): one-to-one Hungarian matching over 3150
  neighbouring, near-identical patch queries is fine for one fixed frame but probably unstable across frames (the
  query matched to a target changes between steps, so existence gets contradictory labels).

**Two-stage selection, and 16 frames** (`run_twostage.sh`; `--two_stage`: the 300 best patch predictions by existence
are decoded again, loss on both, see `code/model.py`; `--frames 16`: train = test = 16 evenly spaced training frames,
6000 steps, so each frame is seen 3000 times against 24000 in the one-frame overfit).

| run | frames | matched Chamfer | pair IoU | found (0.5 / 0.3) | predicted (0.3) / targets | occupancy IoU (0.3) | mask coverage | class acc. |
|---|---|---|---|---|---|---|---|---|
| of_patch_cd | 1 | 0.031 m | 0.74 | 0.96 / 0.99 | 113 / 112 | 0.73 | 0.75 | 1.00 |
| of_twostage_cd | 1 | 0.041 m | 0.60 | 0.82 / 0.98 | 128 / 112 | 0.60 | 0.63 | 1.00 |
| f16_patch_cd | 16 | 0.079 m | 0.47 | 0.13 / 0.50 | 63 / 62 | 0.10 | 0.09 | 1.00 |
| f16_twostage_cd | 16 | 0.135 m | 0.31 | 0.11 / 0.61 | 87 / 62 | 0.12 | 0.07 | 1.00 |

- 16 frames already break it, and existence is what breaks: classes are memorized at once (1.00), geometry is half
  there and still improving at the end (pair IoU 0.44 -> 0.47 over the last 1200 steps), but only 13 % of the
  matched queries pass 0.5 and 50 % pass 0.3, so the union IoU stays at 0.1. Mask coverage is low for the same
  reason: it groups only queries with exist > 0.5. Consistent with the dense one-to-one matching hypothesis (neighbouring
  patches that look the same get opposite existence labels).
- Partly a budget effect too (8x fewer views per frame than the one-frame overfit), but existence lags far
  behind geometry at every evaluation.
- Two-stage does not fix it: worse geometry (0.31 against 0.47 pair IoU), more targets above 0.3 (61 %) but also
  40 % more predictions than targets. Its proposals are still matched one-to-one, so the selection inherits the
  problem.

**One-to-many (dense) assignment, one frame** (`of_dense`: `--dense`, `code/dense.py` + `model.DenseModel`: simplified
tokenizer = LayerNorm(DINO || patch-max-pooled Sonata) -> MLP -> 128 + 3D Fourier position, 1 pre-norm encoder
layer, each patch assigned the visible SQ owning >= 50 % of its pixels; room_2_000500, 3000 steps; 68 visible SQs).

- Geometry is learned: a patch's SQ against its own target has pair IoU 0.57 on average; the best patch per SQ 0.70
  (77 % of the SQs above 0.5).
- Existence is useless for selection: 99.5 % of the patches are positive, all score ~0.995, and the tiny
  differences are correlated within an SQ, so the 300 best patches all belong to **one** SQ: proposal recall@0.5
  0.015 (1 / 68), occupancy IoU of the top 300 0.10. "Is this patch on a target SQ" does not rank proposals.

**Heatmap existence + top-k decoder, one frame** (`of_dense_heat`: existence = CenterNet heatmap, one peak per SQ
at the patch deepest inside its region, penalty-reduced focal loss, proposals = top 300 of the 3 x 3 local maxima;
`of_topk`: + 1 decoder layer over those 300 (query = encoder token, reference = proposal SQ detached, refinement
heads), Hungarian loss against the visible SQs + the one-to-many proposal loss. room_2_000500, 68 visible SQs, 3000
steps, the two runs in parallel).

| run | proposals recall@0.5 / best IoU | pair IoU | found (0.5 / 0.3) | predicted (0.5 / 0.3) | occupancy IoU (0.5 / 0.3) | mask coverage |
|---|---|---|---|---|---|---|
| of_dense (flat existence) | 0.015 / 0.02 | – | – | 3149 / – | 0.11 / – | – |
| of_dense_heat | 0.62 / 0.51 | – | – | 112 / – | 0.57 / – | – |
| of_topk (decoder) | 0.44 / 0.41 (its encoder) | 0.62 | 0.68 / 0.97 | 47 / 77 | 0.11 / 0.68 | 0.55 |
| of_patch_cd (reference, all 112 targets) | – | 0.74 | 0.96 / 0.99 | – / 113 | 0.48 / 0.73 | 0.75 |

- The heatmap fixes selection: about one existence peak per SQ (112 above 0.5 for 68 visible SQs), 62 % of the SQs
  have a proposal with IoU > 0.5 (0.015 before).
- The decoder, matched one-to-one on top of these proposals, gets pair IoU 0.62 and finds 97 % of the visible SQs at
  0.3 (68 % at 0.5); its occupancy IoU at 0.3 (0.68) is close to the old patch-query overfit's (0.73), with 1 encoder +
  1 decoder layer. Not yet as tight as of_patch_cd (pair IoU 0.74), which had 3 layers and all its 3150 queries.

**Heatmap dense + top-k, 16 frames and full** (`run_dense.sh`: f16 = 16 training frames, 6000 steps; full = 7
scenes, 150 epochs, test room_0. full_topk stopped at epoch 63 / 150, its numbers are from the eval at epoch 60).

| run | frames evaluated | proposals recall@0.5 / best IoU | pair IoU | found (0.5 / 0.3) | predicted (0.5 / 0.3) / visible | occupancy IoU (0.5 / 0.3) | mask coverage |
|---|---|---|---|---|---|---|---|
| f16_dense | the 16 | 0.42 / 0.41 | – | – | 107 / – / 42 | 0.38 (top-k 0.15) | – |
| f16_topk | the 16 | 0.28 / 0.32 (encoder) | 0.54 | 0.55 / 0.85 | 25 / 50 / 42 | 0.16 / 0.28 | 0.48 |
| full_dense | 100 train frames | 0.34 / 0.35 | – | – | 144 / – / 41 | 0.37 (top-k 0.21) | – |
| full_dense | room_0 | 0.014 / 0.13 | – | – | 169 / – / 68 | 0.15 (top-k 0.13) | – |
| full_topk (epoch 60) | room_0 | 0.016 / 0.12 (encoder) | 0.08 | 0.27 / 0.67 | 25 / 86 / 68 | 0.04 / 0.14 | 0.21 |

- On 16 frames both fit reasonably (f16_topk pair IoU 0.54, found@0.3 0.85; the old f16_patch_cd had found 0.13 /
  0.50), so the heatmap + top-k setup fixes the multi-frame existence problem of the patch queries.
- On the full set they do not generalize at all: full_dense reaches recall@0.5 0.34 on training frames but 0.014 on
  room_0 (best IoU 0.13, i.e. the proposals hardly overlap any target), flat from epoch 30 on. full_topk on room_0:
  pair IoU 0.08, encoder recall 0.016. Stopped (2026-11-06): the full runs are a clear failure, a generalization (not
  a capacity) problem: 7 scenes are too few, and room_0 is the densest scene (98 targets / 68 visible per frame against
  ~60 / 41 in training).

**Training speed-ups** (`code/speedups.py`, `code/latency_breakdown.py`, `code/cache_assign.py`; details in
`results/latency.md` and `results/speedups.md`). Before: 155 ms (dense) / 230 ms (top-k) per step of 8 frames,
overhead-bound. Each change alone and together, batch 8, step ms (speed-up):

| config | dense | top-k | exact? |
|---|---|---|---|
| base | 155 | 230 | – |
| 1 plain surface sampler | 121 (1.28x) | 162 (1.42x) | yes (points equal to 2e-7) |
| 1b batched losses (one geometric pass per batch) | 100 (1.55x) | 124 (1.85x) | yes |
| 2 cached assignment + heatmap | 103 (1.50x) | 191 (1.20x) | yes |
| 3 unique-pair Sonata pooling | 148 (1.05x) | 227 (1.01x) | yes |
| 1 + 1b + 2 + 3 | 48 (3.2x) | 64 (3.6x) | yes (grad cosine 1.000000) |
| + bf16 autocast | 44 (3.5x) | 54 (4.2x) | no (grad cosine 0.99999 / 0.9995) |
| + torch.compile | 47 (3.3x) | 65 (3.5x) | yes |
| + bf16 + compile | 46 (3.4x) | 47 (4.9x) | no |

Kept: 1, 1b, 2, 3 (2026-11-06) as the only code path, the switches removed (`speedups.py` with them; the code from
before is not versioned). Checked against the original code on 2 fixed batches: losses equal to 1e-7, gradient cosine
1.0000000 and identical evaluation metrics for dense, top-k and the patch-query model. `--loss vol / both` and
`--mask sq / sq_amodal` keep the per-frame loop (`train.losses_loop`). `cache_assign.py` must be rerun when the targets
change (without the cache `dense.assign` computes the assignment per step, as before). In train.py on the full set:
8 s / epoch for dense (was 26 s), 11 s for top-k (was 41 s), so 150 epochs take ~20 / ~28 min plus evaluation. With
`--frames 16` the gain is smaller (27 s instead of 58 s for 320 steps): an epoch is only 8 steps there, and the
DataLoader restarts its prefetch at every epoch.

**Evaluation speed-ups** (2026-11-06, `results/speedups.md`). Kept, exact (every per-frame metric identical to the
code before): pair IoUs only for proposal / target pairs whose sample boxes overlap; shared inside-tests for the
occupancy IoUs; instance grouping on the CPU; vectorized mask coverage; no double final evaluation (the last epoch's
evaluation is the final one). Per 8 room_0 frames 143 -> 105 ms (dense), 313 -> 145 ms (top-k). Options:
`--eval_subsample n` (periodic evaluations on every n-th frame, the final one on all; with 2 the metrics move by
≤ 0.002 on room_0, ≤ 0.014 on the 100 training frames) and `train.FAST_SDF` (compiled inside test, 56 / 91 ms per
batch; over 80 room_0 frames the only difference was one point in one occupancy IoU, |Δ| 2e-5; ~4 s compilation). Both on since 2026-11-06 (`--eval_subsample` default 2).

**16-frame exploration (2026-11-06 afternoon)**: `run_occupancy.sh`, then `worker.sh` (2-3 runs in parallel from
`queue.txt`; `code/compare.py` prints the table). All 6000 steps, train = evaluation frames (42 visible SQs on
average), single seed. New options: `--loss both` for dense / top-k (Chamfer + pairwise volumetric loss),
`--union target|observed` (soft-IoU loss of the decoder's union, see train.py), `--exist bce` (decoder),
`--width`; top-k also reports `dec_recall@0.5` (proposal recall computed on the decoder's 300 SQs).

| run | proposal recall@0.5 | dec. recall@0.5 | pair IoU | found 0.5 / 0.3 | occ. IoU 0.5 / 0.3 | mask cov. |
|---|---|---|---|---|---|---|
| f16_topk (lr 1e-4, baseline) | 0.28 | (0.63, 8 frames) | 0.54 | 0.55 / 0.85 | 0.16 / 0.28 | 0.48 |
| + Chamfer + volumetric | 0.33 | 0.57 | 0.54 | 0.41 / 0.84 | 0.10 / 0.31 | 0.39 |
| + union vs target SQs | 0.24 | 0.58 | 0.54 | 0.66 / 0.90 | 0.21 / 0.38 | 0.55 |
| + union vs target SQs, weight 3 | 0.26 | 0.51 | 0.51 | 0.70 / 0.91 | 0.30 / 0.38 | 0.60 |
| + union vs depth map | 0.29 | 0.56 | 0.54 | 0.63 / 0.91 | 0.22 / 0.30 | 0.57 |
| decoder BCE instead of focal | 0.28 | 0.55 | 0.54 | 0.66 / 0.77 | 0.14 / 0.21 | 0.54 |
| 3 decoder layers (aux losses) | 0.21 | 0.35 | 0.40 | 0.68 / 0.91 | 0.16 / 0.31 | 0.57 |
| width 256, 2 encoder layers | 0.40 | 0.73 | 0.64 | 0.75 / 0.92 | 0.32 / 0.48 | 0.65 |
| **lr 3e-4** | 0.44 | 0.83 | 0.71 | 0.90 / 0.97 | 0.49 / 0.62 | 0.78 |
| **lr 1e-3** | **0.54** | **0.90** | **0.78** | **0.93 / 0.99** | **0.61 / 0.78** | **0.82** |
| f16_dense (lr 1e-4) | 0.42 | – | – | – | 0.38 (existence > 0.5) | – |
| f16_dense + Chamfer + volumetric | 0.48 | – | – | – | 0.39 | – |

- The learning rate dominates everything else: with OneCycle max lr 1e-4 the model was badly under-trained. 1e-4 ->
  3e-4 -> 1e-3 improves every metric (pair IoU 0.54 -> 0.71 -> 0.78, occupancy IoU at 0.3 0.28 -> 0.62 -> 0.78).
  All other comparisons in this table are at 1e-4 and need repeating at the new lr (in progress / queued).
- At lr 1e-4: the union losses improve existence (found, occupancy IoU, mask coverage) without changing the
  geometry, and the depth-only variant does as well as the one against the target SQs; the volumetric loss improves the
  proposals but hurts the decoder's existence; BCE only shifts the calibration; 3 decoder layers are worse (maybe
  also under-trained); width 256 + 2 layers helps (consistent with under-fitting).

**Learning-rate sweep (2026-11-06 evening, step 1 of `viewer/ablations.html`)**: default top-k, 16 frames, 6000
steps, OneCycle with max lr and warm-up share (`--warmup`, default 0.3) as given; single seed.

| max lr | warm-up | proposal recall@0.5 | dec. recall@0.5 | pair IoU | found 0.5 / 0.3 | occ. IoU 0.5 / 0.3 | mask cov. |
|---|---|---|---|---|---|---|---|
| 1e-4 | 30 % | 0.28 | – | 0.54 | 0.55 / 0.85 | 0.16 / 0.28 | 0.48 |
| 3e-4 | 30 % | 0.44 | 0.83 | 0.71 | 0.90 / 0.97 | 0.49 / 0.62 | 0.78 |
| 1e-3 | 30 % | 0.54 | 0.90 | 0.78 | 0.93 / 0.99 | 0.61 / 0.78 | 0.82 |
| 1e-3 | 5 % | 0.54 | 0.91 | 0.78 | 0.96 / 1.00 | 0.71 / 0.79 | 0.88 |
| 2e-3 | 30 % | 0.60 | 0.94 | 0.80 | 0.95 / 0.99 | 0.73 / 0.81 | 0.88 |
| 2e-3 | 5 % | 0.60 | 0.95 | 0.81 | 0.96 / 1.00 | 0.76 / 0.83 | 0.88 |
| 3e-3 | 30 % | 0.64 | 0.96 | 0.82 | 0.97 / 0.99 | 0.75 / 0.83 | 0.88 |
| **3e-3** | **5 %** | 0.66 | 0.95 | 0.82 | **0.97 / 1.00** | **0.79 / 0.87** | **0.89** |
| 5e-3 | 30 % | **0.70** | **0.97** | **0.83** | 0.96 / 0.99 | 0.72 / 0.84 | 0.87 |

- Up to 3e-3 every metric improves; at 5e-3 the geometry still improves (proposal recall 0.70, pair IoU 0.83) while
  existence / occupancy / masks drop slightly: the optimum is around 3e-3 - 5e-3.
- A 5 % warm-up instead of OneCycle's 30 % helps at every lr, mostly existence (occupancy IoU at 0.5 +0.03 to +0.10).
- Stopped before the end (deleted): 5e-3 with 5 % warm-up, 1e-2, and the first seed repeat of 3e-3 / 5 %. So the noise
  level is still unknown: differences of ~0.01-0.02 between neighbouring settings are not interpretable yet.
- Proposed new baseline: max lr 3e-3, warm-up 5 % (best existence, near-best geometry).

**Ablations at the new baseline** (lr 3e-3, warm-up 5%, f16, one seed each; `--w_mask` new, 0 = no mask / Dice loss)

| run | proposal recall@0.5 | pair IoU | found 0.5 / 0.3 | occ. IoU 0.5 / 0.3 | mask cov. |
|---|---|---|---|---|---|
| baseline | 0.66 | 0.82 | 0.97 / 1.00 | 0.79 / 0.87 | 0.89 |
| no mask loss | 0.65 | 0.83 | 0.97 / 0.98 | 0.78 / 0.82 | (0.13, untrained) |
| + union vs targets | 0.58 | 0.80 | 0.94 / 0.98 | 0.71 / 0.77 | 0.86 |
| + union vs depth | 0.63 | 0.82 | 0.96 / 0.99 | 0.71 / 0.71 | 0.87 |

- The mask loss costs nothing on the SQ metrics (differences within 0.01) and gives the instance masks: keep.
- At the proper lr the union loss no longer helps (at lr 1e-4 / 3e-4 it did): both variants lower occupancy IoU
  by 0.08, union vs targets also proposal recall. Its earlier gain made up for the too-low lr.

**Full runs at the new lr** (7 scenes, 150 epochs, lr 3e-3, warm-up 5%, mask loss; test room_0, train = 100 training frames)

| run | set | proposal recall@0.5 | pair IoU | found 0.5 / 0.3 | occ. IoU 0.5 / 0.3 | mask cov. | class acc. |
|---|---|---|---|---|---|---|---|
| full_topk (old, lr 1e-4, epoch 60) | room_0 | 0.016 | 0.08 | 0.27 / 0.67 | 0.04 / 0.14 | 0.21 | – |
| full_topk_lr3e-3_wu05 | train | 0.44 | 0.45 | 0.46 / 0.77 | 0.19 / 0.43 | 0.55 | 1.00 |
| full_topk_lr3e-3_wu05 | room_0 | 0.035 | 0.10 | 0.21 / 0.47 | 0.05 / 0.14 | 0.27 | 0.65 |
| + union vs depth | train | 0.41 | 0.45 | 0.51 / 0.77 | 0.30 / 0.38 | 0.60 | 1.00 |
| + union vs depth | room_0 | 0.039 | 0.11 | 0.24 / 0.47 | 0.11 / 0.15 | 0.32 | 0.64 |

- The lr fix barely helps on room_0 (pair IoU 0.08 -> 0.10). Test metrics plateau by epoch ~50 and stay flat.
- Large generalization gap (pair IoU 0.45 train vs 0.10 test, class accuracy 1.00 vs 0.65), and the training frames
  are not fit well either (f16 reached 0.82).
- The union loss helps occupancy IoU@0.5 and mask coverage a little on the full set (unlike on f16).
- Next (running): within-scene hold-out (`--split frames`: train on the first 75% of each training scene's
  trajectory, test on the last 20%) to separate cross-scene from cross-view generalization; width 256 + 2 encoder
  layers (capacity); DINO only / Sonata only (`--inputs`, which feature generalizes).

**Why the full runs fail: diagnostics** (2026-11-06, all lr 3e-3, warm-up 5%, mask loss, no union, 150 epochs;
new switches `--split frames`, `--inputs dino|sonata`, `--flip p`)

| run | train pair IoU | test set | proposal recall@0.5 | pair IoU | found 0.5 / 0.3 | occ. IoU 0.5 / 0.3 | mask cov. | class acc. |
|---|---|---|---|---|---|---|---|---|
| baseline | 0.45 | room_0 | 0.035 | 0.10 | 0.21 / 0.47 | 0.05 / 0.14 | 0.27 | 0.65 |
| hold-out views (same scenes) | 0.46 | last 20% of each training trajectory | 0.15 | 0.19 | 0.32 / 0.50 | 0.07 / 0.16 | 0.39 | 0.91 |
| width 256, 2 encoder layers | 0.40 | room_0 | 0.046 | 0.11 | 0.20 / 0.52 | 0.07 / 0.21 | 0.24 | 0.63 |
| DINO only | 0.42 | room_0 | 0.025 | 0.10 | 0.20 / 0.56 | 0.07 / 0.19 | 0.22 | 0.65 |
| Sonata only | 0.42 | room_0 | 0.022 | 0.09 | 0.17 / 0.45 | 0.04 / 0.14 | 0.16 | 0.60 |

- Already new views of the training scenes fail (0.46 -> 0.19): the model memorizes frames rather than learning
  geometry that transfers; a new room halves it again (0.10). Test metrics plateau (no late overfitting).
- Not a capacity problem: the bigger model fits the training frames worse (lr 3e-3 may be too high for it) and
  barely improves the test set.
- Inputs: the two features are largely redundant, neither alone generalizes better; DINO alone ~ both, Sonata alone
  slightly worse (masks, class).
- Likely cause: too little / too redundant data (7 trajectories, neighbouring frames near-duplicates), and possibly
  ambiguous targets (amodal SQ extents of partly visible objects, arbitrary decomposition of large surfaces).
- Flip augmentation (`--flip 0.5`: frames mirrored left-right incl. targets; the DINO map is mirrored, not
  recomputed): mirrored assignment agrees with the recomputed one on 99.7% of patches (CX is 0.25 px off center).
- Flip run stopped at epoch 83 / 150 (2026-11-07). Its eval at epoch 80 on room_0 is on par with the baseline's:
  pair IoU 0.11 (baseline 0.11 at epoch 80), recall@0.5 0.033 (0.033), occ. IoU 0.05 / 0.18 (0.04 / 0.15), mask cov.
  0.23 (0.22). The flip doesn't help: more of the same views is not the missing diversity.

**Visibility-aware targets and diverse frames** (2026-11-07; new: `train.target_visibility` / `visible_fraction`,
`--partial clip`, `--vis_x`, `--min_patches`, `--train_frames diverse`, `select_frames.py`; `cache_assign.py` now also
stores f and the per-pixel SQ owner `pixown`)

- Visible fraction f per target SQ = visible / camera-facing surface area (16 x 16 samples, area-weighted). A sample is
  visible if it projects into the image, faces the camera (analytic SQ normal) and is not hidden: z <= depth + 3 cm,
  or its pixel is owned by this SQ (`pixel_owner`). Costs 0.4 ms per batch on the GPU (training step ~125 ms); cached.
- room_1: 41% of the visible SQs have f < 0.5; small SQs are mostly well seen (1 patch: median f 0.71, > 16 patches:
  0.50). In-frustum SQs owning no pixel are mostly tiny SQs (< 3 cm) on visible surfaces, subsumed by a bigger SQ of
  the same instance in `pixel_owner` (f ~ 1, but not targets).
- `--partial clip`: for targets with f < 0.5 the target -> prediction Chamfer direction only over the visible
  samples, prediction -> target against the full SQ (in the loss, in the Hungarian cost and the eval matching);
  forced Chamfer weighted by min(f / 0.5, 1). Eval adds pair_iou_seen / pair_iou_partial (f >= / < 0.5), cd_clip.
- `--min_patches N` (training frames only): visible SQs owning < N patches are dropped, their patches ignored by the
  proposal existence loss (the decoder still pushes queries on them to 0). Not run yet.

f16 (16 frames, train = eval, 750 epochs, lr 3e-3, warm-up 5%); baseline re-evaluated with the new metrics:

| run | pair IoU | seen / partial | occ. IoU 0.5 / 0.3 | proposal recall@0.5 | decoder recall@0.5 | mask cov. | partial: vol. precision / recall |
|---|---|---|---|---|---|---|---|
| f16 baseline (DINO + Sonata, full Chamfer) | 0.82 | 0.82 / 0.83 | 0.79 / 0.86 | 0.66 | 0.95 | 0.89 | 0.93 / 0.87 |
| f16_topk_dino_clip (DINO, clipped) | 0.65 | 0.79 / 0.47 | 0.29 / 0.35 | 0.46 | 0.72 | 0.86 | 0.82 / 0.51 |

- As designed, partially visible predictions stop covering the unseen part (volume recall 0.51, median volume 0.64 x
  the target's) while staying mostly inside it (precision 0.82); amodal metrics (pair / occupancy IoU) count that as
  error. The baseline memorizes the 16 frames, partial SQs included, so f16 cannot show the intended benefit (less
  ambiguity across views), only the cost.
- But the clipped run also fits worse where it should not: its clipped Chamfer on partial pairs is 0.071 vs the
  baseline's 0.055 (trained with the full loss), decoder recall 0.72 vs 0.95, proposal recall 0.46 vs 0.66. Part of
  this may be DINO only (no f16 DINO-only baseline yet).

Frame selection (`select_frames.py`, pool = every 2nd trajectory frame, 200 per scene, unit = (SQ, octant of the
viewing direction in the SQ's frame), value f, greedy with sqrt diminishing returns):

| scene | units diverse / stride 10 / pool | SQs seen | octants per SQ (of 8) | views per unit diverse / stride 10 |
|---|---|---|---|---|
| room_1 | 224 / 225 / 229 | 129 / 128 / 129 | 1.74 / 1.76 | 50 / 35 |
| room_2 | 338 / 333 / 343 | 172 / 171 / 173 | 1.97 / 1.95 | 37 / 29 |
| office_0 | 229 / 232 / 234 | 128 / 128 / 128 | 1.79 / 1.81 | 42 / 29 |
| office_1 | 121 / 118 / 121 | 78 / 77 / 78 | 1.55 / 1.53 | 55 / 41 |
| office_2 | 296 / 301 / 308 | 161 / 161 / 162 | 1.84 / 1.87 | 48 / 29 |
| office_3 | 399 / 400 / 409 | 206 / 203 / 206 | 1.94 / 1.97 | 47 / 33 |
| office_4 | 263 / 258 / 265 | 149 / 149 / 149 | 1.77 / 1.73 | 36 / 25 |

- Every 10th frame already covers ~98% of the units the whole trajectory offers; each SQ is seen from < 2 of 8
  sides. The selection only shows the same units more fully (more views per unit, ~40 of 200 frames shared with
  stride 10). The NICE-SLAM trajectories themselves are the diversity bottleneck, not the subsampling. The selected
  frames are not cached yet (`cache_frames.py --select`, ~10 GB).

**Clipping ablations and failure diagnosis** (2026-11-07, all DINO only, lr 3e-3, warm-up 5%, mask loss, no union)

f16 (train = eval):

| run | pair IoU | seen / partial | clipped CD partial | occ. IoU 0.5 / 0.3 | proposal / decoder recall@0.5 |
|---|---|---|---|---|---|
| full Chamfer (f16_topk_dino) | 0.83 | 0.82 / 0.85 | 0.030 | 0.65 / 0.83 | 0.64 / 0.95 |
| clip, f < 0.5 (hidden or outside) | 0.65 | 0.79 / 0.47 | 0.071 | 0.29 / 0.35 | 0.46 / 0.72 |
| clip, only outside the image (`--clip_what trunc`) | 0.73 | 0.81 / 0.66 | 0.051 | 0.42 / 0.53 | 0.56 / 0.84 |
| clip, no forced weighting (`--forced_weight none`) | 0.70 | 0.84 / 0.57 | 0.046 | 0.41 / 0.49 | 0.49 / 0.77 |
| clip, f < 0.25 (`--vis_x 0.25`) | 0.73 | 0.82 / 0.42 | 0.084 | 0.49 / 0.58 | 0.56 / 0.82 |

Full (7 scenes, test room_0, 150 epochs):

| run | set | pair IoU | seen / partial | occ. IoU 0.5 / 0.3 | proposal / decoder recall@0.5 | mask cov. | class acc. |
|---|---|---|---|---|---|---|---|
| full Chamfer (full_topk_lr3e-3_wu05_dino) | train | 0.42 | – | 0.07 / 0.36 | 0.38 / 0.45 | 0.38 | 1.00 |
| full Chamfer | room_0 | 0.10 | 0.10 / 0.09 (diagnose) | 0.07 / 0.19 | 0.025 / 0.022 | 0.22 | 0.65 |
| clip (full_topk_dino_clip) | train | 0.40 | 0.46 / 0.34 | 0.09 / 0.33 | 0.38 / 0.43 | 0.42 | 1.00 |
| clip | room_0 | 0.08 | 0.09 / 0.07 | 0.07 / 0.17 | 0.035 / 0.019 | 0.25 | 0.60 |

- Clipping costs fit on f16 in proportion to how much it clips, and does not help generalization: on room_0 even the
  well-seen SQs get worse (0.09 vs 0.10). Dropped.

Failure diagnosis of full_topk_lr3e-3_wu05_dino (`diagnose.py`, results/diagnose_*.json), per visible target:

| | train frames | room_0 |
|---|---|---|
| pair IoU (decoder, matched) | 0.39 | 0.10 |
| ... with the target's center (predicted shape) | 0.58 | 0.25 |
| ... with the target's shape (predicted center) | 0.44 | 0.12 |
| median center error | 5 cm | 24 cm (1.15 x the largest half-axis) |
| best proposal among the target's own patches (oracle selection) | 0.45 | 0.20 |
| ... for targets with > 16 patches | 0.76 | 0.36 |
| targets with a patch among the 300 selected (1-2 patch targets) | 0.76 (0.51) | 0.63 (0.28) |
| pair IoU by f: < 0.25 / 0.25-0.5 / 0.5-0.75 / >= 0.75 | 0.37 / 0.42 / 0.42 / 0.37 | 0.09 / 0.11 / 0.10 / 0.10 |
| test targets of a class unseen in training | – | 5% |

Linear probes on the frozen DINO tokens + patch centroid (`probe.py`, results/probe.json; patches of visible targets,
train 300k / room_0 100k): class accuracy 0.98 train / **0.92 room_0**; log scale R^2 0.74 / 0.34; center offset R^2
0.63 / 0.41.

- Visibility does not explain the failure: well-seen SQs fail as much as partial ones on room_0.
- Position is the main error (fixing the center gives 2.5x the IoU, fixing the shape barely helps), and the
  regression itself does not transfer (oracle-selected proposals 0.76 -> 0.36 even for large SQs).
- The class is in the features (linear probe 0.92 on room_0; the proposals' own class accuracy on room_0 is ~0.90);
  the decoder's 0.65 comes from bad matches (queries in the wrong place matched to the wrong target).
- Small SQs (1-2 patches, 37% of the targets) are rarely even selected (28% on room_0) and their proposals are
  useless (best IoU 0.05).

**Regularization and learning curve** (2026-11-07; new switches `--dropout`, `--feat_drop`, `--wd`, `--train_limit`,
`--train_scenes`; DINO only, lr 3e-3, warm-up 5%, same number of steps as 150 epochs on all frames)

| run | train pair IoU | room_0 pair IoU | occ. IoU 0.5 / 0.3 | found 0.5 | mask cov. | class acc. |
|---|---|---|---|---|---|---|
| baseline (1400 frames, 7 scenes) | 0.42 | 0.103 | 0.07 / 0.19 | 0.20 | 0.22 | 0.65 |
| dropout 0.1, input dropout 0.2, weight decay 0.05 | 0.37 | 0.104 | 0.01 / 0.18 | 0.13 | 0.15 | 0.65 |
| half the frames (700, 7 scenes) | 0.52 | 0.091 | 0.07 / 0.15 | 0.24 | 0.26 | 0.63 |

- Regularization does not help (same test, worse fit). Halving the frames of the same trajectories costs little
  (0.103 -> 0.091): within a trajectory the frames are redundant, so more of them will not close the gap. Running: a
  quarter of the frames, and 3 / 5 training scenes (the number of scenes as the likely lever).
- Target decompositions are inconsistent across scenes (`targets_winding`, SQs per instance): floor 1 SQ (room_0) to
  5 (office_0), door 1 to 5, chair 2.0 (office_1) to 9.6 (office_2) on average, a wall plug ~5 SQs. A model cannot
  learn an arbitrary split count; but the decomposition-free union occupancy IoU on room_0 is low too (0.07), so this
  is not the whole story.

| run (continued) | train pair IoU | room_0 pair IoU | occ. IoU 0.5 / 0.3 | found 0.5 | mask cov. | class acc. |
|---|---|---|---|---|---|---|
| a quarter of the frames (350) | 0.44 | 0.083 | 0.13 / 0.17 | 0.34 | 0.33 | 0.63 |
| `--min_patches 2` | 0.40 | 0.100 | 0.08 / 0.16 | 0.16 | 0.20 | 0.60 |
| input dropout 0.5 | 0.22 | 0.087 | 0.02 / 0.15 | 0.10 | 0.11 | 0.64 |
| lr 1e-3 | 0.43 | 0.091 | 0.07 / 0.16 | 0.21 | 0.24 | 0.63 |

- Frames learning curve 350 / 700 / 1400: room_0 pair IoU 0.083 / 0.091 / 0.103, ~+0.01 per doubling. The baseline's
  room_0 pair IoU peaks at epoch ~50 (0.108) and stays flat (no late overfitting).
- Dropping the 1-patch SQs (`--min_patches 2`, evaluated on all visible SQs) changes nothing; neither does stronger
  input dropout or a lower lr.

Learning curve over scenes (`--train_scenes N`: the first N of room_1, room_2, office_0, office_1, office_2, ...;
same number of steps):

| training scenes | frames | train pair IoU | room_0 pair IoU | occ. IoU 0.5 / 0.3 | class acc. |
|---|---|---|---|---|---|
| 3 | 600 | 0.53 | 0.064 | 0.07 / 0.12 | 0.50 |
| 5 | 1000 | 0.52 | 0.080 | 0.10 / 0.15 | 0.59 |
| 7 | 1400 | 0.42 | 0.103 | 0.07 / 0.19 | 0.65 |
| 7 (half the frames) | 700 | 0.52 | 0.091 | 0.07 / 0.15 | 0.63 |

- **The number of scenes is the lever, not the number of frames**: at ~the same frame count, 7 scenes beat 3 by
  0.091 vs 0.064, and the curve over scenes is not flattening (+0.016 for 3 -> 5, +0.023 for 5 -> 7). Everything
  else tried (loss variants, regularization, lr, capacity, inputs, flip, frame selection) moves room_0 by < 0.01.
- Available: full Replica v1 has 10 more scenes in Thesis/data/replica_v1 (apartment_0-2, frl_apartment_0-5,
  hotel_0), with meshes and instance point clouds (replica_v1_points, voxel 10 mm) but no rendered trajectories (the
  NICE-SLAM renders exist only for the 8 scenes) and no renderer installed (only trimesh).

### Capacity (full runs, DINO only, room_0 test)

| run | room_0 pair IoU | train pair IoU |
|---|---|---|
| baseline (w128, 1 enc / 1 dec layer, lr 3e-3) | 0.103 | 0.42 |
| 3 decoder layers | 0.085 | 0.39 |
| width 256, 2 encoder layers, lr 1e-3 | 0.110 | 0.51 |

Wider model: better fit on train, test gain within (unmeasured) seed noise. `diagnose.py` on it: same picture as the
baseline: median centre error 24.5 cm on room_0 (3.8 cm train), IoU with GT centre 0.23, oracle best proposal for
large SQs 0.34 (train 0.81), small SQs (1–2 patches) selected 26 %. Capacity does not address the generalisation gap.
Follow-ups (seed repeats, w256 + flip, w384/3 layers, w256 + regularisation) were queued but stopped at 17:11 before any finished.

## Projection loss: rendering the predicted SQs (`code/render.py`)

**Renderer.** A differentiable ray caster in plain PyTorch, with no CUDA extension. 3DSQS (Daniel-MacSwayne/3DSQS_code)
is a 3DGS-style splatter: it sorts per splat by centre depth and uses an approximate rim point. That suits ~10^5 small
splats but not our ~100–300 large, overlapping SQs (walls, floors), where a single depth per primitive gives the wrong
occlusion order. Per (ray, SQ) pair:
1. Cull against the SQ's box.
2. Take 24 SDF samples (SuperFlex radial SDF, whose sign is exact) and refine the argmin.
3. Bisect to the entry point.

Gradients:
- **Silhouette.** sigmoid(−SDF at the argmin / τ), with τ = half a rendered pixel at that depth. By the envelope
  theorem, the argmin point itself needs no gradient.
- **Depth.** The entry depth with one Newton correction (implicit surface gradient, as in IDR).
- **Compositing.** Front-to-back per ray, sorted per ray, with alpha = existence × silhouette.

Rays every 2 px (84 × 150).

Checks (scratch scripts):
- **Target SQs.** Rendered at existence 1, the front-most SQ's instance equals the GT instance label on 94 % of
  covered pixels; coverage is 99.8 % and the median depth error 1.4 %.
- **Gradients.** Autograd matches central finite differences for translation x/z, scale and shape (within ~3 %).
- **Speed.** 17 ms per frame (forward + backward) for 300 SQs. In training, a step takes 2.5× longer
  (0.45 s vs 0.18 s).

**Losses** (final decoder only; `--render_inst/--render_cls/--render_depth` weights, `--w_dec_cd` factor on the decoder's Chamfer):
- **inst.** Each query takes the instance of its Hungarian-matched target. The loss is −log Σ w_q over the queries of
  the pixel's GT instance, so it does not depend on how the target decomposes an instance into SQs.
- **cls.** −log Σ_q w_q p_q(the pixel's class). It needs no matching.
- **depth.** Relative L1 of the rendered expected depth on covered pixels.

**Eval** (every run now): r_cover, r_inst_acc (front-most query's matched instance = GT), r_cls_acc,
r_depth_absrel.

**Ablation (16 frames of room_1, train = eval, 750 epochs)** (`--frames 16 --train_scenes 1`): base / + inst /
+ inst + cls + depth / the same with the decoder Chamfer off (`--w_dec_cd 0`; proposals keep their Chamfer, and the
matching still uses the Chamfer cost).

## Occlusion viewer (`viewer/occlusion.html`)

Run `uv run python experiments/22_rgbd_prediction/viewer/server.py`, then open http://localhost:8022/occlusion.html.
Visibility is computed per frame on request, for every cached frame (every 10th).

The viewer colours every target SQ in the frustum by an area-weighted ratio:
- visible / whole surface (at most ~0.5, so colour-mapped on [0, 0.5]);
- visible / camera-facing surface (the training f);
- occluded / camera-facing;
- outside the image / camera-facing.

Other modes:
- Colour by surface sample state: back-facing, outside the image, occluded or visible.
- Ratio range filter and a per-frame histogram.
- Image overlay: each pixel coloured by the ratio of the SQ that owns it.
- Click an SQ for its numbers.

## Render viewer (`viewer/render.html`)

Open http://localhost:8022/render.html on the same server. It renders on request, from the predictions a run saves at
its final evaluation.

The page is a 3 × 3 grid:
- **Rows:** the GT; the rendered target SQs; the rendered prediction.
- **Columns:** the segmentation of the front SQ (by instance, class or SQ index); the rendered depth; and a check
  column (agreement with the GT instance or class, signed relative depth error, or coverage).

Controls:
- Ray stride: 1, 2 or 4 px.
- Which predicted queries are rendered: as in training, existence > 0.5, or matched only.
- A cursor shared by all panels, with the per-pixel values.
- Metrics per frame.

First look: `full_topk_dino_dec3` on room_0, frame 0. With the training query set, 98 % of the labelled pixels are
covered, but the instance is right on only 15 %. Most of the wall goes to unmatched (duplicate) queries. The matched
queries alone cover 21 % of the pixels.

## Visibility fix: hidden SQs inside bigger ones of the same instance (2026-11-07)

**Bug (found in the occlusion viewer).** In room_1 frame 130, target SQ 1 (a 2.4 cm thin wall piece) lies inside the
big wall SQ 0: its front face is ~1.6 cm behind SQ 0's. It still had f = 0.94 and owned 324 pixels. Two causes:
- **Pixel owner.** Each pixel went to the SQ of its instance with the smallest SDF at the back-projected point, which
  is the SQ the point lies *deepest in*, so the hidden piece won.
- **Visibility test.** The test against the observed depth (3 cm tolerance) cannot see coplanar overlaps.

**New rule** (`train.OWNER_RULE = "zbuffer"`, the default; `--owner_rule sdf` is the old rule):
- **Pixel owner.** Among the SQs of the pixel's GT instance, the first one the pixel ray enters (hard ray casting,
  `render.entry_depths`). If the ray misses them all, the one with the smallest |SDF|.
- **Visible sample.** As before, plus two tests:
  - not behind the surface of its pixel's owner, if that is another SQ (`ownz` + 5 mm);
  - not inside another SQ of its instance.

A first version tested against the z-buffer of *all* target SQs and removed 45 % of the samples that pass the
observed-depth test. Those were mostly fits of other instances sticking out in front of the observed surface: window
behind blinds, book inside table, lamp inside ceiling. The GT label decides which instance a pixel shows, so only SQs
of the same instance can hide a sample.

**Impact** (320 frames, every 5th cached frame, without rewriting the caches):

| | old rule | new rule |
|---|---|---|
| visible targets / frame | 44.8 | 43.8 (1.8 lost, 0.9 gained) |
| labelled pixels changing owner | | 2.5 % |
| mean f of visible targets | 0.61 | 0.46 |
| share with f < 0.5 | 39 % | 56 % |

About 13 targets per frame lose more than 0.3 of f, almost all through overlap with siblings of the same instance
(wall plugs, candles, windows, books, cushions, plants, chairs).

**Not yet applied to the caches.** The visible-target sets, patch owners and heatmaps (`cache_assign.py`) all depend
on the pixel owner, and all earlier runs used the old rule. The f16s_* render ablation runs with `--owner_rule sdf`.
The occlusion viewer has an old / new switch.

## Surface sampling leaves the pole caps of boxy SQs empty (2026-11-07)

`train.surface` samples a regular (η, ω) grid at cell centres, as SuperFlex's `parametric_to_points_extended` does.
The last η row sits at radius cos(π/2 − π/2n)^ε1 of the pole face, so for boxy SQs (small ε1) the inside of both pole
faces gets no samples. Example: office_1 frame 720, floor SQ 15 (ε1 = 0.15, thin axis z). Its top face is sampled
only at normalized radius ≥ 0.87, i.e. on the rim. For half the SQs (office_1, n = 16), the unsampled cap reaches past
64 % of the half-axes. The same happens in ω for small ε2: samples crowd onto the edges.

Every SQ showed holes in the occlusion viewer for two reasons: the mesh had no cap triangles (viewer bug, now fixed),
and the caps carry no samples (now drawn white, "not sampled").

**Effect on f.** Compared with a cube grid projected radially onto the SQ, x = s ⊙ u · F(u)^(−ε1/2), where u is on
the unit cube and F is the inside–outside function. For boxy SQs this places samples almost uniformly on the box
faces, its area matches 8(ab + bc + ca), and it is differentiable in scale and shape. Measured on 3603 pixel-owning
SQs:
- median |Δf| 0.04;
- 7 % of SQs move by more than 0.2 (11 % of the boxy SQs with ε1 < 0.3, none of the round ones with ε1 > 0.8);
- mean f 0.47 → 0.48.

For the example floor, f stays ≈ 0: 99 % of its camera-facing area really is outside the image.

**Larger concern (not measured).** The Chamfer losses, the matching cost and `cd_clip` use the same grid, so the
targets of boxy SQs are represented mostly by their rims and edges. A radial cube sampling for the losses would be a
training change, to be decided.

### Done: new visibility rule cached, cube sampling implemented (2026-11-07)

**Caches.** `cache_assign.py` was rerun with `OWNER_RULE = "zbuffer"` for 1584 frames (1.5 min). The 16 room_1 frames
of the running f16s_* ablation are rebuilt automatically once it ends. The caches now contain:
- `pixown`, `ownz`, `owner`, `heat`, `visible` under the new rule;
- `f` (grid samples) and `f_cube` (cube samples).

`Frames` loads the f of `--sampling`. The old caches are in
`Thesis/data/backup/replica_rgbd_frames_assign_sdfrule_2026-11-07.tgz`. Visible targets: mean f 0.467 (grid) / 0.484
(cube); f < 0.5 for 56 % / 52 %.

**`--sampling grid|cube`** (`train.surface`, default grid). The cube option is a radial cube grid, 6 m² samples with
m = round(n / √6), so n = 16 gives 294 samples against the grid's 256. It applies to the Chamfer and forced terms,
the matching cost, `vol_loss` and the visibility / f. Metrics (pair IoU boxes, cd, cd_clip, occupancy boxes) stay on
the grid so they remain comparable with earlier runs.

Checks:
- Normals agree with autograd of the inside–outside function (cosine ≥ 0.9999997).
- Areas: unit sphere 12.81 (n = 16) vs 4π = 12.57. A 1.5 × 1 × 0.02 box gives 12.33 vs 12.40; the grid's cells sum
  to only 5.56 there.
- Gradients are finite. A NaN at the cube's axis directions (lateral sum 0) is fixed by a clamp.
- Same speed as the grid.
- 3-epoch smoke run OK.

**Viewer `viewer/sampling.html`** (on the same server):
- Grid and cube side by side with a shared camera; translucent reference surface (fine cube mesh) and depth points.
- Samples coloured by visibility state, area per sample, or SQ.
- Frame mode: SQs sorted by |f grid − f cube|, with the max gap = largest distance from the reference surface to the
  nearest sample. Click a row to isolate an SQ.
- Single-SQ mode with sliders for a, b, c, ε1, ε2 and the sample budget.

Example: room_1 frame 130, wall SQ 0. The max gap is 146 cm with the grid (its large faces carry samples only near the
edges and corners) vs 28 cm with the cube, and f is 0.23 vs 0.60. Median max gap in that frame: 7 cm vs 2 cm.

### Projection-loss overfit ablation (16 room_1 frames, train = eval, 750 epochs, old caches / `--owner_rule sdf`)

| run | pair IoU | occupancy IoU | dec recall@0.5 | r_inst_acc | r_cls_acc | r_cover | r_depth absrel |
|---|---|---|---|---|---|---|---|
| f16s_base | **0.802** | **0.855** | **0.950** | 0.905 | 0.914 | 0.980 | 0.029 |
| + inst (w 1) | 0.763 | 0.714 | 0.886 | 0.905 | 0.910 | 0.984 | 0.034 |
| + inst + cls + depth (w 1 each) | 0.750 | 0.714 | 0.890 | **0.934** | **0.940** | **0.999** | **0.028** |
| render only (decoder Chamfer off) | 0.000 | 0.000 | 0.002 | 0.041 | 0.335 | 0.858 | 0.392 |

The baseline already renders well (90.5 % of the pixels show the right instance), so there is little left for the loss
to fix on training frames. With all three terms the rendered image gets better (93.4 %, coverage 99.9 %), but the 3D fit
gets worse (pair IoU −0.05, occupancy IoU −0.14). The weights (1 each, untuned) probably pull the SQs towards covering
pixels at the cost of their volume. Whether it helps generalization (room_0) is not tested.

Render only collapses: without the 3D Chamfer the decoder never learns to place SQs (r_inst_acc 4 %, depth error
39 %). The loss is too weak or ill-conditioned to train from scratch. It can only act as an auxiliary term next to the
3D supervision.

### Heatmap ablation (full benchmark, new caches; stopped after 2 of 4 runs)

Is the existence heatmap's Gaussian clipped to the SQ's own patches? Checked on all 1600 cached frames: no patch
outside an SQ's region gets heat from it, and every SQ has exactly one positive. 8517 SQ regions are split into several
8-connected pieces, where the Euclidean distance treats the pieces as near.

`cache_heat.py` builds three alternatives with the same peak and σ (`--heat`):
- **linear:** 1 − d / 3σ, own patches only;
- **geodesic:** the Gaussian of the shortest path within the region;
- **nearest:** every patch takes the Gaussian of the nearest peak of any SQ, not clipped.

New eval metrics: `sel_recall` (targets with ≥ 1 own patch among the 300 selected) and `sel_recall_small` (targets
owning 1–2 patches). Runs: full_heat_{gauss, nearest, geodesic, linear}. full_heat_gauss is also the baseline on the
new visibility caches.

| run (test room_0) | pair IoU | occupancy IoU | dec recall@0.5 | sel_recall | sel_recall_small | cls acc |
|---|---|---|---|---|---|---|
| full_heat_gauss | 0.102 | 0.055 | 0.023 | 0.653 | 0.273 | 0.669 |
| full_heat_nearest | 0.104 | 0.049 | 0.024 | 0.644 | 0.267 | 0.653 |
| geodesic, linear | stopped mid-run (2026-11-07) | | | | | |

Gauss and nearest are equal within noise. The falloff around the peak does not limit selection; small targets (27 %
recalled) do.
