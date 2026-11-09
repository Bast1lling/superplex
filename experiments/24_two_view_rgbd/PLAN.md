# Plan: SQ decomposition from two RGB-D views

Status: draft (2026-11-08), nothing implemented. Builds on experiment 22 (single-view predictor).

## Question
Does a second view with a known relative pose (a) improve the SQs of the first view's objects, mostly the partly
visible ones (f < 0.5, 52-56 % of the visible targets), and (b) do better than merging two single-view
predictions? A side question: does it narrow exp 22's generalization gap? Probably not, because it uses the same 7 training scenes.

## Data (no new caching needed)
- Exp 22's frame cache (`Thesis/data/replica_rgbd_frames`, every 10th frame): DINOv3 tokens, Sonata, `c2w`, targets
  in the camera frame, assignment/heatmaps. The cache stores `c2w` per frame, so every pair of frames gets its
  relative pose for free.
- Pairs: (i, i + Δ) within a trajectory, Δ sampled from {10, 20, 40, 80} frames, kept if the view overlap is in
  [0.2, 0.8]. Overlap = the share of view-1 depth points that project into view 2 without occlusion.
  Statistics of Δ vs overlap come first (`pair_stats.py`).
- Reference frame: view 1's camera. View 2's patch centroids and SQs are moved into it with T_12.
- Targets: the SQs visible in view 1 **or** view 2, i.e. the union of exp 22's per-view visible sets. f per view and
  f_pair = the visible area of the union over the camera-facing area of the union.

## Model (exp 22's top-k, DINO only, lr 3e-3, warm-up 5 %, mask loss)
- Same per-view tokenizer, shared weights, so 2 x 3150 tokens. Positional encoding = the Fourier features of the
  patch's 3D centroid **in the common frame**, plus a learned view embedding (0/1).
- Encoder: a joint self-attention layer over both views' tokens, so the views exchange information through
  3D-consistent positions. Cost: 6300 tokens, about 4x the attention FLOPs, so check memory at batch 8.
  Alternative: per-view self-attention + cross-attention between the views.
- Proposals: heatmap peaks of both views (top 300 of all of them). The same SQ peaks in both views, so there are
  duplicates. Two options: one peak per SQ, only in the view with the larger f, or let both stay and rely on the
  decoder's one-to-one matching.
- Decoder and heads as in exp 22. Mask head per view: a query predicts its instance mask in both images.

## Baselines and evaluation
1. Exp 22's single-view model on view 1 alone, scored on the view-1 targets. This is the reference for (a).
2. **Late fusion**: exp 22's model on each view, view 2's SQs moved into view 1, merged by NMS on pair IoU. This is
   the reference for (b), and it costs no training.
3. Two-view model, scored on view-1 targets (split into seen / partial by the view-1 f) and on the union targets.
- Metrics as in exp 22: pair IoU (seen / partial), occupancy IoU @0.5/0.3, proposal / decoder recall@0.5, mask coverage.
- Settings: f16 (16 fixed pairs, train = eval), then full (7 scenes, test room_0) and `--split frames`.

## Steps
1. `pair_stats.py`: overlap vs Δ, how much the second view raises f of the view-1 targets (the upper bound on
   what (a) can gain).
2. Late fusion baseline (eval only, uses an exp 22 checkpoint).
3. Two-view model: overfit 1 pair, then f16, then full. Every run only after you say go.

## Code
Imports exp 22's `train.py` / `model.py` / `dense.py` (as exp 23 does), so exp 22's fixes carry over. The new
code covers only the pair dataset, the joint encoder, and pair targets and assignment.
