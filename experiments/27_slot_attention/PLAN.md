# 27 — Supervised Slot Attention on DINO patches: visible SQs of one RGB-D view

**Question.** Does Slot Attention (patches compete for slots) give a clean one-slot-per-SQ decomposition of a view, so
that each slot can regress its SQ with SuperFlex-style geometric losses? First step: can it **overfit a single
Replica frame**?

## Data (reused, nothing new rendered)
- Frame: `room_2/000500` of `data/replica_rgbd_frames` (exp 22's cache): DINOv3 ViT-L/16 last layer, 42 x 75 tokens,
  depth / GT instance label / RGB at 168 x 300 (4 x 4 pixels per patch), target SQs in the camera frame.
- Targets: exp 22's SuperFlex + optimization fits per GT instance (winding occupancy). 112 SQs in the frustum, 67 own
  pixels. Targets = SQs owning >= 0.5 patch: **75** with the default label-free `sdf` labels (57 proj, 61 sdf_inst;
  exp 22's z-buffer owner rule gave 51). Smaller ones: their pixels are *ignore*.
- Precomputed per frame (`code/precompute.py`, in `data/replica_slot_targets/` or the experiment's large_files):
  pixel labels, patch shares t[n, m], visible targets, target cube samples with visibility class and distance
  outside the frustum, IoU sample points.

## Model
```
DINO token (1024) -> LN -> MLP -> d=128   (+)   Linear(Fourier(patch centroid xyz, camera frame, m))
  -> Slot Attention: K = 128 slots, learned init, T = 3 (BO-QSA: detach before the last iteration,
     straight-through to the init), softmax over slots
  -> [0 self-attention layers among slots]
  -> per slot: SuperFlex head (plain SQ) + existence logit
```
- Translation: `--anchor centroid` (soft 3D centroid of the slot's mask, detached, + offset) | `none`.
- Scale: `0.005 + softplus(x)` (no upper bound), shape `0.1 + 1.8 sigmoid`, rotation from a quaternion.
- Patch centroid xyz from depth; Fourier frequencies covering ~2 cm .. 10 m.

## Losses (all weights 1; Adam, constant lr 5e-3, 3000 steps, grad clip 1)
1. **Matching** (no grad): Hungarian on `C = CE[k, m] + Dice[k, m]` between the slot masks A (N x K) and the patch
   shares t (N x M).
2. **L_mask**: CE over slots per patch (target T[n, σ(m)] = t[n, m], unmatched slots 0) + Dice per matched pair.
3. **L_exist**: BCE, matched = 1, unmatched = 0.
4. **L_sdf** (SuperFlex): the observed points of instance m against slot σ(m)'s SQ only (GT assignment),
   truncated |SDF|.
5. **L_pair** vs the matched target SQ: Chamfer on **cube samples** + soft IoU, each target sample weighted by
   visibility:
   - visible: 1
   - in the frustum, occluded: `w_occ` (default 1: inferable, e.g. depth of a chair)
   - outside the frustum: `exp(-d / λ)`, d = metric distance to the frustum, λ = 10 cm (default)
6. **L_free**: points sampled on the pixel rays in front of the observed depth, `sigmoid(-sdf / τ)` of every
   existing SQ there. Ablation: **L_render** (depth + instance of the rendered SQs) instead.
7. **L_overlap** (SuperFlex): points inside more than one existing SQ.

## Metrics
Matched mask mIoU and ARI against the GT patch labels; SQ fit to the visible points (Chamfer, mean |SDF|); pair IoU and
Chamfer against the matched targets; active slots (exist > 0.5) vs the 51 targets.

## Runs (single frame, overfitting; nothing starts without approval)
1. Base: all losses as above, anchor centroid.
2. Ablations, one at a time: anchor none; L_free -> L_render; without L_pair / L_sdf / L_free / L_overlap;
   predicted assignment in L_sdf; mask source; w_occ, λ.

## Mask source (ablation)
- **sdf (default):** the SQ with the smallest |SDF| at the pixel's back-projected point, over all target SQs (no
  instance labels needed).
- sdf_inst: the same, restricted to the SQs of the pixel's GT instance.
- proj: z-buffered projection of the target SQs.

## Later
One scene (all frames) -> 7 scenes -> held-out frames; self-attention among slots; amodal SQs; finer masks.

## Code
`code/train.py` (losses, metrics, ablation flags; docstring = exact definitions), `code/model.py`, `code/data.py`
(labels, visibility weights), `code/common.py` (exp 26's SQ geometry, Replica camera), `code/dashboard.py`
(trackio, port 7827, runs grouped by `--group`).
