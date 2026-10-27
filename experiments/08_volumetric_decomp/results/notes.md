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
