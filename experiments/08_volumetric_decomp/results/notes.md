# 08 — notes (chronological)

Hardware: this session had no GPU (4 CPU cores, torch CPU). One room_0 run (78 instances with >= 20 points,
4 cm voxels, ~280k target voxels) takes ~9-12 min with 2 threads.

## room0_v1 — first working version
Strict negatives (free = other instance = 1, unknown = 0.3), δ = 0.03.
- Before the region constraint (run v0, deleted) wall / floor SQs grew freely into the unknown space behind them and
  beyond the 30 cm margin where negatives are sampled -> added the extent penalty (`w_region`).
- 137 SQs (1.76 / instance), vIoU 0.73. Top view already "human-like": one SQ per blind / picture / door / stool,
  1-2 per chair, sofa 1-2, cushions 1-2. Failures:
  - floor split into 5 SQs: the rug lies on it, and covering the rug's voxels is as expensive as covering free space,
    so no single floor slab is allowed (no amodal completion);
  - windows with vIoU 0: a window instance is a hollow *frame* (glass is not scanned), the box over the frame
    covers the free voxels of the opening, the SQ shrinks to nothing;
  - coffee table (0.34) and an L-shaped wall (0.48) stay at K = 1 although they need more.

## Amodal overlap (w_other 0.3; unknown 0 or 0.1)
| run | sqs | K/inst | vIoU | precision | recall | coverage | point mIoU |
|---|---|---|---|---|---|---|---|
| v1 (w_other 1, w_unk 0.3) | 137 | 1.76 | 0.732 | 0.822 | 0.801 | 0.855 | 0.633 |
| amodal (0.3, 0) | 147 | 1.89 | 0.806 | 0.870 | 0.871 | 0.881 | 0.674 |
| amodal_unk01 (0.3, 0.1) | 140 | 1.80 | 0.814 | 0.881 | 0.878 | 0.899 | 0.646 |
Note that vIoU itself changes with the weights (other-instance voxels count as false positives with weight w_other),
so compare renders too: the floor went from 5 to 3 SQs, walls stay thin slabs with w_unknown = 0.1.

## δ (price of a primitive in IoU), w_other 0.3, w_unknown 0.1
| δ | sqs | K/inst | vIoU | K by size (<250/<1250/<5000/>5000 pts) | vIoU by size |
|---|---|---|---|---|---|
| 0.05 | 124 | 1.59 | 0.801 | 1.6 / 1.9 / 1.4 / 1.3 | 0.87 / 0.72 / 0.83 / 0.83 |
| 0.03 | 140 | 1.80 | 0.814 | 1.7 / 2.2 / 1.8 / 1.4 | 0.87 / 0.74 / 0.85 / 0.84 |
| 0.015 | 170 | 2.18 | 0.833 | 2.3 / 2.5 / 2.1 / 1.7 | 0.88 / 0.77 / 0.86 / 0.85 |
δ is a clean parsimony knob; the gain per extra SQ is small, i.e. the search, not the price, limits quality.

## Debugging the search (`--set only=17,29,11,57,6 --set verbose=true`)
- Proposal outcomes are very noisy: the same "add" for the coffee table was rejected in round 1 (IoU −0.02) and
  accepted in round 2 (+0.44); the L-shaped wall went 0.49 -> 0.94 over 6 accepted proposals after an early
  near-rejection. With max_fail = 2 many instances stop too early.
- Windows 17 / 29 get no proposal at all: their single SQ has collapsed (frame problem above).
Changes (v2): free voxels outside the room box (oriented box of wall / floor / ceiling points) or within 6 cm of its
faces -> unknown (space seen through windows); max_fail 3 and alternate add / split after a failure;
prune-with-refit rounds (drop each instance's cheapest SQ, re-optimize 250 steps, keep the removal if IoU fell by
< δ): lets neighbours grow into the gap, i.e. a merge move.

## v2 (room-box unknown, alternation, prune-with-refit) and the other scenes
| run | sqs | K/inst | vIoU | precision | recall | coverage | point mIoU | K by size | vIoU by size |
|---|---|---|---|---|---|---|---|---|---|
| room0_v2 (δ 0.03) | 146 | 1.87 | 0.809 | 0.882 | 0.867 | 0.898 | 0.644 | 1.8 / 2.1 / 1.7 / 1.8 | 0.86 / 0.70 / 0.85 / 0.88 |
| room0_v2_d02 (δ 0.02) | 164 | 2.10 | 0.830 | 0.894 | 0.883 | 0.903 | 0.688 | 1.9 / 2.4 / 1.8 / 2.0 | 0.87 / 0.74 / 0.86 / 0.89 |
| room1_v2 | 86 | 1.69 | 0.808 | 0.868 | 0.869 | 0.910 | 0.637 | 1.6 / 1.2 / 1.8 / 2.0 | 0.88 / 0.63 / 0.84 / 0.87 |
| office0_v2 | 74 | 1.85 | 0.872 | 0.927 | 0.928 | 0.946 | 0.583 | 1.6 / 1.5 / 2.2 / 2.0 | 0.93 / 0.93 / 0.83 / 0.82 |
- room_0: the floor is now **one** SQ (prune-with-refit merged the fragments), rug one, blinds / pictures / doors /
  stools one each, sofas and chairs 1-2. Visually this is what I would draw by hand. Left: the front wall with door
  and window openings is 3-4 slabs, one of them tilted out of the wall plane near the doorway.
- room_1 / office_0: same picture (bed, nightstands, desk, chairs, sofa, rug sensible). Two systematic artifacts:
  SQs with ε > 1 (pinched / diamond profiles; the room_1 rug renders as concentric rings) and a floor slab of office_0
  leaking past the wall into the unknown region. -> v3 restricts ε to [0.1, 1.2].
- Point mIoU (most-inside assignment) is lower than in 06 because amodal overlap is now allowed (rug points are
  inside the floor slab, cushions inside the sofa). I tried a nearest-|d| assignment as well; it is meaningless with
  the radial SDF (|d| not comparable across SQ sizes) and was removed.

## Baseline: experiment 06 scored with these metrics (`eval_external.py`, run `ext06_k6_exist_pars5mm`)
06's best surface fit (GT instances, K=6 + existence, parsimony 5 mm): **307 SQs (3.9 / instance), vIoU 0.371**,
precision 0.80, recall 0.39, coverage 0.93, point mIoU 0.78. Surface fits explain the points (coverage) but not the
volume (recall 0.39: hollow / patch-like), and the render shows the floor and rug as patchworks of 6 SQs.
=> volumetric targets: 2x fewer SQs, 2.2x vIoU.

## Fixed slots + existence + parsimony (`fit_exist.py`) — what a feed-forward model would optimize
K = 8 k-means SQs per instance, Bernoulli existence (expected loss of the first existing SQ along increasing SDF),
λ Σ e per instance, 3000 steps, no structure search.
| run | sqs | K/inst | vIoU | precision | recall | coverage | point mIoU | K by size | vIoU by size |
|---|---|---|---|---|---|---|---|---|---|
| exist λ 0.3 | 227 | 2.91 | 0.849 | 0.914 | 0.913 | 0.949 | 0.759 | 1.3 / 3.0 / 2.4 / 4.4 | 0.87 / 0.79 / 0.84 / 0.92 |
| exist λ 1 | 151 | 1.94 | 0.827 | 0.896 | 0.897 | 0.946 | 0.721 | 1.1 / 2.2 / 1.5 / 2.6 | 0.85 / 0.78 / 0.80 / 0.89 |
| greedy v2_d02 | 164 | 2.10 | 0.830 | 0.894 | 0.883 | 0.903 | 0.688 | 1.9 / 2.4 / 1.8 / 2.0 | 0.87 / 0.74 / 0.86 / 0.89 |
- Unlike with the surface loss of 06, existence + parsimony with the volumetric loss reaches the same budget /
  quality trade-off as the greedy search; small objects are even cleaner (1.1 SQs). Pruning decisions are made by
  step ~500 and are stable.
- But the render shows 06's failure mode on large instances: the floor stays a k-means patchwork of ~5 slabs.
  Existence prunes redundant SQs, it cannot merge. Large instances: 2.6 SQs vs 2.0 for greedy.

| run | sqs | K/inst | vIoU | precision | recall | coverage | point mIoU | K by size | vIoU by size |
|---|---|---|---|---|---|---|---|---|---|
| exist λ 1, multiscale init | 136 | 1.74 | 0.806 | 0.886 | 0.878 | 0.922 | 0.685 | 1.0 / 2.3 / 1.2 / 2.0 | 0.80 / 0.78 / 0.80 / 0.86 |
| exist λ 3 | 123 | 1.58 | 0.801 | 0.878 | 0.879 | 0.931 | 0.666 | 1.0 / 2.0 / 1.2 / 1.8 | 0.84 / 0.76 / 0.80 / 0.83 |
| greedy v2 (δ 0.03) | 146 | 1.87 | 0.809 | 0.882 | 0.867 | 0.898 | 0.644 | 1.8 / 2.1 / 1.7 / 1.8 | 0.86 / 0.70 / 0.85 / 0.88 |
- Multi-scale init (slot 0 = whole-instance box, slots 1-7 = k-means parts) only partly helps: floor 5 -> 4 pieces.
  The whole-instance slot co-adapts with the parts (it shrinks to its own share instead of making them redundant).
- λ 3: fewest SQs at about the greedy quality, but the floor is still 3 slabs.
- Take-away for the feed-forward direction: the volumetric loss + existence + parsimony is a viable *training*
  objective (budget / quality on par with the search, small objects even cleaner), but it inherits the inability to
  merge, so large planar instances end up as a few slabs. Options: supervise the network with the greedy results
  (distillation), or add a merge-friendly term; the search result is the better target for now.
