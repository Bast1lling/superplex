# 25 — SQ decomposition from a single RGB view (no 3D input)

**Question.** How much is lost when exp 22's predictor gets no depth, and can it get its 3D from the image alone?
Decision (2026-11-08): variant D of `PLAN.md`. The model takes no 3D input at all, except optionally the camera pose
(ablated). GT depth is only used as supervision (targets, patch assignment, depth loss, render depth loss).

## Code (`code/`, a copy of exp 24's code; `model.py` is new)
- Tokens: DINOv3 + Fourier encoding of the patch's **2D** grid position. `--pose` adds, to every token, the encoding
  of the gravity direction in the camera frame (row 2 of R_c2w; Replica's world z is up) and the camera height above
  the scene's floor (1st percentile of the world z of the depth points). This is what an IMU plus a known mounting
  height would give.
- `--depth pred` (default): a per-patch depth head after the encoder (log depth, L1 against the patch's mean GT depth,
  weight 1). Anchor = the patch ray × predicted depth (detached). From there on everything is exp 22's top-k model:
  the tokens get the Fourier encoding of their predicted 3D anchor, proposal heads give translation = anchor +
  offset, then the decoder.
- `--depth none`: no depth head, anchor = 3 m along the ray, no 3D encoding (all 3D from the regressed offsets).
- `--depth gt`: anchor = GT patch centroid. This is the RGB-D oracle (exp 22's / exp 24's inputs, plus the 2D
  position).
- Extra metrics: `depth_absrel` of the patch depths, `depth_scale` = median predicted / GT depth.

Same defaults, settings, losses and metrics as exp 24 (1 view). f16 runs use 3000 steps.

    ./worker.sh &
    uv run python experiments/25_rgb_prediction/viewer/server.py       -> http://localhost:8025

## Results (2026-11-08, one seed each; run times 25-40 min because 5-6 jobs shared the GPU)

**f16** (16 training frames, train = eval, 3000 steps):

| run | 3D source | pose | pair IoU | seen / partial | occ. IoU 0.5 / 0.3 | dec. recall@0.5 | mask cov. | patch depth AbsRel |
|---|---|---|---|---|---|---|---|---|
| `f16_gt` (RGB-D oracle) | GT depth | – | 0.725 | 0.739 / 0.718 | **0.69 / 0.77** | 0.89 | **0.89** | – |
| `f16_pred` | depth head | – | 0.743 | 0.764 / 0.738 | 0.42 / 0.55 | 0.87 | 0.69 | 0.026 |
| `f16_pred_pose` | depth head | yes | 0.727 | 0.728 / 0.733 | 0.37 / 0.58 | 0.86 | 0.68 | 0.025 |
| `f16_none` | none (offsets only) | – | 0.762 | 0.783 / 0.756 | 0.56 / 0.71 | 0.90 | 0.83 | – |
| `f16_none_pose` | none | yes | **0.774** | 0.800 / 0.763 | 0.58 / 0.72 | **0.92** | 0.80 | – |
| `f16_pred_pose_render` | depth head + render loss | yes | 0.625 | 0.653 / 0.616 | 0.57 / 0.56 | 0.73 | 0.88 | 0.030 |

- On f16 every variant fits the 16 frames about equally well, even the one without any 3D. A network can memorize 16
  frames' SQ positions from the image alone, so f16 does not separate the 3D sources. Only hold-out views do (below).
- The render loss shows the same trade-off as in exp 24 (pair IoU -0.10, rendered instance accuracy 0.45 -> 0.89).

**Hold-out views** (`--setting frames`, 4000 steps; see exp 24):

| run | 3D source | pose | pair IoU | seen / partial | occ. IoU 0.5 / 0.3 | dec. recall@0.5 | mask cov. | patch depth AbsRel (train views) |
|---|---|---|---|---|---|---|---|---|
| `frames_gt` (RGB-D oracle) | GT depth | – | **0.161** | 0.225 / 0.125 | 0.03 / 0.14 | 0.078 | 0.18 | – |
| ... on its training views | | | 0.314 | | 0.05 / 0.29 | 0.295 | 0.33 | |
| `frames_pred_pose` | depth head | yes | 0.085 | 0.114 / 0.071 | 0.01 / 0.08 | 0.031 | 0.09 | 0.118 (0.057) |
| ... on its training views | | | 0.234 | | 0.02 / 0.21 | 0.189 | 0.23 | |
| `frames_none_pose` | none (offsets only) | yes | 0.088 | 0.116 / 0.073 | 0.01 / 0.11 | 0.031 | 0.13 | – |
| ... on its training views | | | 0.249 | | 0.04 / 0.24 | 0.213 | 0.28 | |
| `frames_pred` | depth head | – | 0.083 | 0.110 / 0.067 | 0.01 / 0.09 | 0.028 | 0.12 | 0.129 (0.056) |
| ... on its training views | | | 0.228 | | 0.03 / 0.22 | 0.186 | 0.25 | |

- **Pose (gravity + camera height) barely matters:** 0.085 vs 0.083 pair IoU. It slightly helps the depth head on
  held-out views (AbsRel 11.8 % vs 12.9 %, median scale 1.02 vs 1.04). That is not because the pose is constant: the
  camera tilt against gravity spans 2-50° (10th-90th percentile per scene), the height 1.4-2.4 m. The model just does
  not exploit it. The likely reason is that DINO tokens already encode the image-plane layout of floors, walls and
  the horizon, and the error is dominated by what does not generalize anyway (the SQ positions of new views).

- **Without depth, half of the (already low) held-out quality is lost:** pair IoU 0.085 vs 0.161 for the RGB-D
  oracle. On the training views the gap is smaller (0.23 vs 0.31).
- The depth head itself generalizes poorly. AbsRel is 11.8 % on held-out views vs 5.7 % on training views, with an
  unbiased scale (median ratio 1.02). For comparison, exp 12 measured Depth Pro (off the shelf, known focal length)
  at 2.4-4.6 % AbsRel on room_0, a scene it never saw. A depth head trained on 7 scenes cannot compete with a
  monocular network trained on millions of images.
- **The depth head adds nothing over plain offset regression:** without it (`frames_none_pose`, anchor = 3 m along
  the ray) the result is the same, 0.088 vs 0.085, and slightly better on the training views. With 7 scenes, the
  explicit depth anchor does not make the 3D regression transfer better.
- So **the next RGB step should be variant B** of `PLAN.md` (Depth Pro depth as input, the same pipeline as the
  RGB-D oracle), not more of D. Its expected gap to the oracle is set by Depth Pro's ~3 % depth error (exp 12/15:
  metric scale errors of 2-10 % shift surfaces by 5-30 cm).
