# 27 — Slot Attention on DINO patches: one RGB-D frame → instances → SQs (results)

**Question.** Can a slot model on frozen DINOv3 patch tokens (+ 3D positions from depth) decompose one RGB-D view into
instances (one slot each) and fit superquadrics to them? First step: **overfit a single Replica frame**
(`room_2/000500`, exp 22's cache; 42 × 75 patches, 168 × 300 depth). Design discussion: `../PLAN.md`; exact loss and
metric definitions: `../code/train.py`'s docstring.

    uv run python experiments/27_slot_attention/code/train.py --name X --mask_source inst1 ...   (runs: ../queue.done)
    uv run python experiments/27_slot_attention/code/dashboard.py      -> http://127.0.0.1:7827  (trackio, by group)
    uv run python experiments/27_slot_attention/viewer/server.py       -> http://localhost:8027  (runs by group, steps)

Run summaries: `runs/<run>.json` (args, eval history every 250 steps, best step). Predictions per eval step, models
(`model.pt`, `best.pt`): `large_files/runs/<run>/`. All numbers: one frame, mostly one seed (train = eval frame).

## Main findings

1. **Slot Attention works for the grouping, at a 100× lower learning rate than first used.** Masks-only, Slot
   Attention overfits the frame's 28 instances at lr 1e-5 (mask mIoU 0.93, ARI 0.98, every matched slot active); at
   5e-3 / 1e-3 it never got past 0.59 and collapsed slots. Competition between slots is not the problem: Mask2Former-
   style queries behave the same (0.90 at lr 3e-5, total collapse at 5e-3 / 1e-3).
2. **Given masks, one SQ per instance is fit well with the pair Chamfer alone.** With GT masks and single-SQ targets:
   frustum IoU 0.78, points 1.9 cm from their SQ. `pair_cd` is the only essential term; the SDF / free-space / overlap
   / render terms add nothing on top of it (or hurt), and the old IoU loss hurt.
3. **Jointly (predicted masks + SQs) it nearly matches GT masks:** Slot Attention at lr 3e-5 + `pair_cd`: mask mIoU
   0.92, frustum IoU 0.68 (best eval 0.78), points 2.1 cm from their SQ.
4. **Without target SQs** (visibility-aware Chamfer to the observed points), the SQs fit the visible surface very
   closely (6–7 mm) but become **sheets**: the hidden extent is unconstrained (frustum IoU vs. the targets ~0.15).
5. **Several SQs per instance (set decoder, SuperFlex targets), with GT masks: frustum IoU 0.86–0.88** over three
   seeds, part IoU 0.81, the right number of parts per instance — after fixing a late training collapse (below).

## Setup that changed along the way (read before comparing numbers)

- **Distance.** SuperFlex's radial SDF is not metric (1 cm beside a 3 m × 3 cm floor slab reads 0.5 m); exp 20's
  scaled distance is wrong *inside* an SQ. `common.sdf_scaled` = the scaled distance outside, inside the smallest of it
  and the exact distances along the 3 local axes to the surface (median error vs. brute force 0.1 mm inside, 0.5 mm
  outside). All runs before this fix are invalid and archived in `large_files/invalid_radial_sdf/`. Two NaN-gradient
  bugs of the new distance (powers at 0, exp overflow far outside) were fixed later; training skips non-finite steps.
- **Targets / labels** (`--mask_source`): per-SQ labels (`sdf`, `sdf_inst`, `proj`; a pixel = an exp 22 SuperFlex SQ),
  `instance` (GT instance = the frame's point labels; target = the union of the instance's SuperFlex SQs), `inst1`
  (the same labels, target = **one SQ fit per instance** to the GT points, `../code/fit_single_sq.py`,
  `data/replica_single_sq/`), `instance_proj` (labels = the rendered union of the instance's SuperFlex SQs). The
  single-SQ fits were refit with half-axes ≥ 1.5 cm and the body behind the surface (GT normals): the first fits made
  walls 2–8 mm sheets, for which volumetric IoU is meaningless (`room_2_thin.npz` keeps them).
- **Pair IoU metric.** Uniform points in an axis-aligned box almost never hit a 3 cm slab (noisy, NaN on the main wall).
  Now importance sampling in the SQs' own oriented boxes (`group_pair_iou`; exact on test cases); `pair_iou_fr` (main
  metric) counts only the camera frustum. Numbers before `D_gt_masks_10k` use the old estimator.

## 1 — Masks: learning rate, Slot Attention vs. Mask2Former (`M_mask_diagnostic`, masks only, `inst1`, K = 128, 10k)

| | lr 5e-3 | 1e-3 | 3e-4 | 1e-4 | 3e-5 | 1e-5 |
|---|---|---|---|---|---|---|
| Slot Attention, mask mIoU (final) | 0.59 | 0.43 | 0.57 | 0.65 | 0.88 | **0.93** |
| Mask2Former-style queries | 0.00 | 0.00 | | **0.90** | **0.90** | 0.86 |

At high lr both fluctuate or collapse: with Adam every weight moves ~lr per step, a large step changes which slot
wins patches, the Hungarian matching then swaps the slots' targets (moving targets), and attention saturates to one
winner. Slot Attention at lr 1e-5: ARI 0.98, 27 of 28 instances own patches, all matched slots active, none extra.

## 2 — Geometry given GT masks (`D_fixes`, `G_loss_ablation`, `U_unsupervised`; `inst1`, `--gt_masks`, 10k)

Loss-term ablation (frustum pair IoU vs. the single-SQ targets; points → own SQ):

| losses | frustum IoU | sdf_own |
|---|---|---|
| all (sdf, pair_cd, pair_iou [old sampling], free, overlap) | 0.713 | 1.9 cm |
| without pair_cd | 0.000 (diverges) | 5 m |
| without pair_iou | 0.760 | 1.6 cm |
| all, pair_iou with box sampling | 0.753 | 1.8 cm |
| **pair_cd only** | **0.778** | 1.9 cm |
| pair_cd + box-sampled pair_iou | 0.745 (1.85× slower) | 1.8 cm |
| observation terms only (sdf, free, overlap) | 0.000 (escape: truncated SDF saturates) | 3.8 m |

Without target SQs (no target term at all):

| losses | frustum IoU | sdf_own | |
|---|---|---|---|
| obs_cd (visibility-aware Chamfer to the slot's points) | 0.144 | 0.6 cm | sheets: smallest half-axis 6–9 mm |
| obs_cd + render | 0.212 | 1.7 cm | 4.5× slower |
| obs_cd + free | 0.179 | 0.7 cm | |
| obs_cd + render + free | 0.209 | 2.2 cm | |
| render only | 0.010 | 56 cm | no long-range gradient |

The observed surface alone does not constrain thickness / back side; next candidates: a minimum thickness, priors,
or other frames of the sequence.

## 3 — Joint: predicted masks + geometry (`J_joint`, `inst1`, 10k)

| run | mask mIoU | frustum IoU | sdf_own |
|---|---|---|---|
| reference: GT masks + pair_cd | 1.00 | 0.778 | 1.9 cm |
| Slot Attention + pair_cd, lr 1e-5 / heads 1e-3 | 0.94 | 0.651 | 1.9 cm |
| **Slot Attention + pair_cd, lr 3e-5** | 0.92 | **0.677** (0.78 at 7.5k) | 2.1 cm |
| Mask2Former + pair_cd, lr 3e-5 / heads 1e-3 | 0.86 | 0.561 | 2.1 cm |
| Slot Attention + obs_cd, lr 1e-5 / heads 1e-3 | 0.95 | 0.125 | 0.7 cm |

A separate (higher) learning rate for the heads is not needed. Part of the remaining gap to GT masks may come from
the single-SQ targets disagreeing with the instance masks (one wall slab covers part of the door; rounded corners).

## 4 — Several SQs per instance: set decoder (`P_parts`, `C_collapse`; `instance` labels, SuperFlex targets, GT masks)

`--parts 16`: per slot 16 part queries (slot feature + learned embeddings), 2 layers of self-attention among them and
cross-attention to the slot's patches; per part an SQ (translation = slot centroid + offset) and an existence. Per
matched slot a Hungarian matching of its parts to the instance's SuperFlex SQs (Chamfer cost); `part_cd`
(visibility-weighted Chamfer per matched pair) + `part_exist` (BCE). `--part_targets visible`: only the instance's
SQs with a visible surface sample (77 of 94 in this frame).

| run | final frustum IoU | best (step) | part IoU | active parts / target SQs per instance |
|---|---|---|---|---|
| visible targets, lr 1e-3 | 0.388 (collapsed) | 0.878 (5750) | 0.20 | 1.54 / 2.75 |
| visible targets, lr 1e-4 | 0.868 | 0.869 | 0.82 | 2.75 / 2.75 |
| all targets, lr 1e-3 | 0.283 (collapsed) | 0.840 (6750) | 0.21 | 2.57 / 3.36 |
| all targets, lr 1e-4 | 0.837 | 0.857 | 0.78 | 3.36 / 3.36 |
| rendered-union masks (`instance_proj`), visible, lr 1e-3 | 0.875 | 0.875 | 0.83 | 2.72 / 2.72 (25 instances) |

**The collapse** (lr 1e-3, `P_vis_lr1e-3` at step 8574, `P_all_lr1e-3` at 7886): the part-existence BCE had been
~0 for thousands of steps (saturated logits), Adam's running RMS of those gradients ~0; a swap in the part matching
labels a confidently "off" part "on" → a huge Adam step (gradient norm 3 → 31), and the inflated RMS (β₂ = 0.999) then
leaves the model crawling instead of recovering. Fixes, lr 1e-3, visible targets:

| fix | final frustum IoU | best | spikes |
|---|---|---|---|
| none | 0.388 | 0.878 | 50 |
| `--exist_smooth 0.01` (existence targets 0.01 / 0.99) | 0.454 | 0.882 | 6 |
| `--adam_eps 1e-6` | 0.514 | 0.851 | 50 |
| `--part_match_exist` (matching cost − log σ(part exist)) | 0.846 | 0.871 | 15 |
| all three, seeds 0 / 1 / 2 | **0.875 / 0.860 / 0.860** | 0.877 / 0.867 / 0.866 | 42 / 4 / 1 |

The existence-aware matching is what makes training recover from spikes; all three are the defaults since then, and
every run keeps its best eval state (`best.pt`). lr 1e-4 has no spikes at all, but fits the points less tightly
(5.5–5.8 cm vs. 2.3–4.8 cm).

## 5 — Several SQs per instance with predicted masks (`R_parts_pred`)

Slot Attention (K = 128) at lr 3e-5 predicts the masks, the part decoder of section 4 the SQs; collapse fixes on.
`instance` labels unless noted, 10k steps, one seed.

| run | mask mIoU | final frustum IoU | best (step) | part IoU | sdf_own |
|---|---|---|---|---|---|
| reference: GT masks (`C_all`) | 1.00 | 0.875 | 0.877 | 0.82 | 2.3 cm |
| **part decoder lr 1e-3, visible targets** (`R_vis_heads1e-3`) | 0.94 | **0.878** | 0.878 (10000) | 0.84 | 4.0 cm |
| part decoder lr 1e-4 | 0.95 | 0.678 | 0.844 (9750) | 0.60 | 5.9 cm |
| one lr 3e-5 for everything | 0.94 | 0.736 | 0.737 | 0.65 | 1.9 cm |
| part decoder lr 1e-3, all targets | 0.94 | 0.850 | 0.864 | 0.79 | 2.5 cm |
| part decoder lr 1e-3, `instance_proj` labels (25 instances) | 0.96 | 0.885 | 0.900 (9250) | 0.85 | 7.7 cm |

**With predicted masks the part decoder matches GT masks** (0.878 vs. 0.875), every matched slot active, none
extra, the right number of parts per instance (2.75). The part decoder needs its own, higher learning rate (1e-3).
`instance_proj` is not comparable one-to-one (3 instances fewer, labels = the rendered targets themselves).
Between evaluations the frustum IoU moves by ±0.05 (single seed), so differences below that are noise.

**Anchoring and K** (`A_anchor`, `K_slots`; visible targets, part decoder lr 1e-3):

| run | mask mIoU | final frustum IoU | best | part IoU |
|---|---|---|---|---|
| GT masks, anchor (`C_all`) / no anchor (`A_noanchor_gt`) | 1.00 / 1.00 | 0.875 / 0.801 | 0.877 / 0.815 | 0.82 / 0.73 |
| predicted masks, anchor (`R_vis_heads1e-3`) / no anchor (`A_noanchor_pred`) | 0.94 / 0.94 | 0.878 / 0.828 | 0.878 / 0.840 | 0.84 / 0.77 |
| predicted masks, K = 64 (`K64_vis_heads1e-3`) | 0.94 | 0.847 | 0.867 | 0.82 |

`--anchor none`: parts predict their translation directly (camera frame, init 2 m ahead). The anchor is worth
0.05–0.07 frustum IoU even when overfitting one frame; with GT masks it also learns faster (0.68 vs. 0.59 at step
1000), with predicted masks the early curves are similar (the masks are the bottleneck there). K = 64 is within the noise of K = 128 (28 instances; masks identical). Exp 28 continues with the anchor
and K = 64.

## Open

- Hidden extent without target SQs (minimum thickness, priors, multi-view consistency).
- More than one frame: one scene, then 7 scenes, then held-out frames.
- Mask resolution (patch level, 4 × 4 pixels at the cached /4 view) limits small instances.
