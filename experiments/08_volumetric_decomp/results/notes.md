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

## "Human-likeness" proxy: plausible K per class (`code/plausible.py`)
A hand-written table of the number of SQs I would draw per class (floor / rug / door / picture / cushion / book:
1; window, ceiling, stool, cabinet: 1-2; chair, lamp, bed: 1-3; sofa / wall: 1-4; table: 1-5; ...). Reported as the
share of instances whose K lies in the range (and the share over-split). It is crude (my judgment, class-level), but
it makes over-splitting visible, which vIoU rewards. Shown in `compare.py` output:
- 06 baseline: 0.24 plausible (76% over-split). Greedy v2: 0.74 (26% over-split, mostly *small* objects: books,
  cushions, wall plugs with 2-3 SQs; at 4 cm voxels one voxel is a big relative IoU change, so δ in IoU is too cheap
  for tiny objects). Existence λ 3: 0.85.
- -> v4: a primitive must also explain >= `min_gain_vox` voxels (IoU gain x |T_k|), for adding and for pruning.

## v3: ε in [0.1, 1.2], 100 warm-up steps in which only the new / split SQs move
| run | sqs | K/inst | vIoU | precision | recall | coverage | point mIoU | plausible | K by size | vIoU by size |
|---|---|---|---|---|---|---|---|---|---|---|
| room0_v2 | 146 | 1.87 | 0.809 | 0.882 | 0.867 | 0.898 | 0.644 | 0.74 | 1.8 / 2.1 / 1.7 / 1.8 | 0.86 / 0.70 / 0.85 / 0.88 |
| room0_v3 | 147 | 1.89 | 0.833 | 0.895 | 0.887 | 0.918 | 0.687 | 0.74 | 1.5 / 2.2 / 1.8 / 1.8 | 0.88 / 0.73 / 0.86 / 0.91 |
| room1_v2 | 86 | 1.69 | 0.808 | 0.868 | 0.869 | 0.910 | 0.637 | 0.73 | 1.6 / 1.2 / 1.8 / 2.0 | 0.88 / 0.63 / 0.84 / 0.87 |
| room1_v3 | 95 | 1.86 | 0.814 | 0.875 | 0.869 | 0.908 | 0.643 | 0.63 | 1.8 / 1.5 / 2.0 / 2.1 | 0.88 / 0.64 / 0.86 / 0.87 |
- room_0: same budget, +0.024 vIoU; renders lose the ring artifacts of ε > 1 shapes and the front wall becomes one
  slab with no tilted pieces. Floor: 2 slabs in this run (1 in v2): the merge via prune-with-refit is not reliable.
- room_1: more SQs (warm-up makes adds succeed more often), plausibility down (over-splitting), vIoU barely up.

## Decision noise and the merge move (debug subset floor / coffee table / L-wall / sofa)
- Added an explicit **merge** move (before pruning): per instance, the pair of SQs whose owned voxels are best
  explained by one box (min vol(box(A ∪ B)) / (vol(box(A)) + vol(box(B)))) is replaced by that box, refitted (warm-up +
  joint), kept if the IoU fell by < δ (or < min_gain_vox voxels).
- While testing it, the mean IoU of the *unchanged* configuration fluctuated by ±0.03 from round to round
  (0.79 / 0.76 / 0.79 / 0.76): Adam at a constant lr keeps jittering the boxes. That is as large as δ, i.e. most
  accept / reject decisions so far were partly coin flips (explains the noisy proposal outcomes seen earlier).
- Fix: cosine lr decay to 5% within every round, so each decision compares settled states. On the subset the IoU
  now rises monotonically over the rounds and ends at 0.83 with 11 SQs (before: ~0.80 with fluctuations).
- v4 (= v3 + min_gain_vox 15) runs were stopped in favour of v5 = v3 + min_gain_vox 15 + merge + lr decay.

## v5 = v3 + min_gain_vox 15 + merge move + per-round lr decay
| run | sqs | K/inst | vIoU | precision | recall | coverage | point mIoU | plausible | K by size | vIoU by size |
|---|---|---|---|---|---|---|---|---|---|---|
| room0_v3 | 147 | 1.89 | 0.833 | 0.895 | 0.887 | 0.918 | 0.687 | 0.74 | 1.5 / 2.2 / 1.8 / 1.8 | 0.88 / 0.73 / 0.86 / 0.91 |
| room0_v5 | 136 | 1.74 | 0.830 | 0.893 | 0.882 | 0.934 | 0.710 | 0.81 | 1.0 / 1.9 / 1.8 / 2.0 | 0.80 / 0.77 / 0.85 / 0.92 |
| room1_v3 | 95 | 1.86 | 0.814 | 0.875 | 0.869 | 0.908 | 0.643 | 0.63 | 1.8 / 1.5 / 2.0 / 2.1 | 0.88 / 0.64 / 0.86 / 0.87 |
| room1_v5 | 84 | 1.65 | 0.805 | 0.866 | 0.866 | 0.902 | 0.627 | 0.78 | 1.0 / 1.4 / 1.9 / 1.9 | 0.83 / 0.64 / 0.85 / 0.88 |
| office0_v2 | 74 | 1.85 | 0.872 | 0.927 | 0.928 | 0.946 | 0.583 | 0.65 | 1.6 / 1.5 / 2.2 / 2.0 | 0.93 / 0.93 / 0.83 / 0.82 |
| office0_v5 | 65 | 1.63 | 0.868 | 0.928 | 0.923 | 0.957 | 0.574 | 0.82 | 1.0 / 1.5 / 2.3 / 1.8 | 0.90 / 0.94 / 0.84 / 0.83 |
- Consistent on all scenes: fewer SQs, same vIoU, plausibility 0.63-0.74 -> 0.78-0.82. Small objects get exactly 1
  SQ. room_0 render: floor still 2 slabs, one cushion an elongated sliver; otherwise clean.

## Semantic labels only (`labels=semantic`: pseudo-instances = connected components of each class, 5 cm)
room_0: 70 components (78 GT instances). `room0_v5_semantic`: 118 SQs, vIoU 0.819, plausible 0.83 — close to the
instance run, *except* one failure: all walls are one ring-shaped component (77k points); its PCA box is the whole
room, which collapses into a sausage-shaped blob inside the room (vIoU 0.04), and every "add" proposal box-fits the
whole ring again, so the search never recovers.
Fix (v6): initial boxes and add proposals use a *compact part*: recursive 2-means until the PCA box is filled to
>= `fill_min` = 0.1 (voxel volume / box volume), largest part. On the wall ring alone: 4 slabs (one per wall), IoU 0.80.

## v6 = v5 + compact-part init / proposals (fill_min 0.1)
| run | sqs | K/inst | vIoU | precision | recall | coverage | point mIoU | plausible | K by size | vIoU by size |
|---|---|---|---|---|---|---|---|---|---|---|
| room0_v5 | 136 | 1.74 | 0.830 | 0.893 | 0.882 | 0.934 | 0.710 | 0.81 | 1.0 / 1.9 / 1.8 / 2.0 | 0.80 / 0.77 / 0.85 / 0.92 |
| room0_v6 | 135 | 1.73 | 0.833 | 0.898 | 0.887 | 0.933 | 0.724 | 0.83 | 1.0 / 1.9 / 1.8 / 2.0 | 0.80 / 0.77 / 0.87 / 0.92 |
| room0_v5_semantic | 118 | 1.69 | 0.819 | 0.882 | 0.877 | 0.910 | 0.529 | 0.83 | 1.0 / 1.1 / 2.4 / 1.8 | 0.71 / 0.82 / 0.86 / 0.87 |
| room0_v6_semantic | 118 | 1.69 | 0.825 | 0.886 | 0.882 | 0.912 | 0.666 | 0.84 | 1.0 / 1.1 / 2.3 / 2.0 | 0.71 / 0.82 / 0.84 / 0.92 |
- Semantic-only: the wall-ring blob is gone, one slab per wall face; point mIoU 0.53 -> 0.67. Floor 3 slabs here.
- Instance labels: unchanged (no ring-shaped instances there), slightly better.

## Degraded input (fit on the perturbed cloud, metrics on the clean volume), room_0, v6
mild: 5 mm position noise, 5° normal noise, 30% random dropout. strong: 1 cm, 15°, 50% dropout, 15% of points removed
in 30 cm holes (occlusion-like).
| run | sqs | K/inst | vIoU | precision | recall | coverage | point mIoU | plausible | K by size | vIoU by size |
|---|---|---|---|---|---|---|---|---|---|---|
| clean | 135 | 1.73 | 0.833 | 0.898 | 0.887 | 0.933 | 0.724 | 0.83 | 1.0 / 1.9 / 1.8 / 2.0 | 0.80 / 0.77 / 0.87 / 0.92 |
| mild | 128 | 1.64 | 0.740 | 0.838 | 0.835 | 0.940 | 0.726 | 0.85 | 1.0 / 1.6 / 2.1 / 1.8 | 0.62 / 0.68 / 0.81 / 0.85 |
| strong | 130 | 1.67 | 0.689 | 0.823 | 0.797 | 0.894 | 0.631 | 0.83 | 1.0 / 1.9 / 1.8 / 1.8 | 0.60 / 0.61 / 0.74 / 0.82 |
- The *structure* is robust: same SQ count and plausibility, render of the strong case still looks like the clean
  one (floor 3-4 slabs, otherwise the same objects). The vIoU loss is concentrated in small objects (0.80 -> 0.60),
  whose clean target (a few voxels, thin objects = voxels that contain points) changes under dropout; large
  instances stay at 0.82-0.85.

## v6 on all scenes (current default config)
| run | sqs | K/inst | vIoU | precision | recall | coverage | point mIoU | plausible | K by size | vIoU by size |
|---|---|---|---|---|---|---|---|---|---|---|
| room0_v6 | 135 | 1.73 | 0.833 | 0.898 | 0.887 | 0.933 | 0.724 | 0.83 | 1.0 / 1.9 / 1.8 / 2.0 | 0.80 / 0.77 / 0.87 / 0.92 |
| room1_v6 | 91 | 1.78 | 0.832 | 0.897 | 0.898 | 0.934 | 0.634 | 0.75 | 1.0 / 1.9 / 1.9 / 2.0 | 0.83 / 0.76 / 0.85 / 0.87 |
| office0_v6 | 67 | 1.68 | 0.870 | 0.929 | 0.925 | 0.958 | 0.576 | 0.82 | 1.0 / 1.5 / 2.4 / 1.9 | 0.90 / 0.94 / 0.85 / 0.83 |

What is still over-split (K above my plausible range):
- **floors** (3-5 slabs in all scenes) and **windows** (4-5 bars each in room_0): both are amodal-completion cases.
  One floor slab must pass under the rug / furniture (their voxels cost w_other = 0.3 each), one window box must
  include the unscanned glass (unknown, 0.1). Under its own costs the search is right to prefer pieces.
- lamps (5-6), baskets, plants, blankets / comforters: genuinely complex or soft shapes; arguably fine.
- chairs with 4 SQs (seat, back, two arms): arguably plausible, my table says <= 3.
-> v7: cheaper amodal overlap (w_other 0.1, w_unknown 0.03). Because vIoU depends on these weights, runs with other
weights are re-scored with the default weights (`code/reeval.py`, `metrics_std.json`; compare.py prefers it).

## v7: cheap amodal overlap (w_other 0.1, w_unknown 0.03); scored with the default weights
| run | sqs | K/inst | vIoU | precision | recall | coverage | point mIoU | plausible | K by size | vIoU by size |
|---|---|---|---|---|---|---|---|---|---|---|
| room0_v6 | 135 | 1.73 | 0.833 | 0.898 | 0.887 | 0.933 | 0.724 | 0.83 | 1.0 / 1.9 / 1.8 / 2.0 | 0.80 / 0.77 / 0.87 / 0.92 |
| room0_v7 | 133 | 1.71 | 0.855 | 0.892 | 0.946 | 0.968 | 0.688 | 0.83 | 1.0 / 2.0 / 1.8 / 1.8 | 0.87 / 0.80 / 0.86 / 0.91 |
| room1_v6 | 91 | 1.78 | 0.832 | 0.897 | 0.898 | 0.934 | 0.634 | 0.75 | 1.0 / 1.9 / 1.9 / 2.0 | 0.83 / 0.76 / 0.85 / 0.87 |
| room1_v7 | 83 | 1.63 | 0.837 | 0.885 | 0.934 | 0.965 | 0.653 | 0.80 | 1.0 / 1.8 / 1.8 / 1.7 | 0.81 / 0.80 / 0.84 / 0.88 |
- Better on both scenes even under the stricter default weights (recall up, precision slightly down).
- room_0 floor: 1 slab. room_1 floor still 4, windows still 4 bars each (frame instances: the glass is not a target,
  so one box over the frame is never better than four bars).

## v8: amodal planar completion of the targets (`fill_planar`)
Thin planar instances (PCA thickness < 20 cm, size > 50 cm) get the cells inside the 2D convex hull of their
projection as extra targets (with their 5-95% thickness): floor under rug / furniture, window glass, wall behind
windows and pictures. (Plain 2D hole filling was tried first: it misses holes that touch the border, e.g. the floor
under a sofa that stands at the wall, and open window frames.) On room_0 it adds e.g. 4.7k voxels per window
(frame had ~370), 28k to the floor (21k), 10.8k to a wall with windows. Run with the v7 weights.

| run (scored with the strict default weights) | sqs | K/inst | vIoU | precision | recall | coverage | point mIoU | plausible | K by size | vIoU by size |
|---|---|---|---|---|---|---|---|---|---|---|
| room0_v7 | 133 | 1.71 | 0.855 | 0.892 | 0.946 | 0.968 | 0.688 | 0.83 | 1.0 / 2.0 / 1.8 / 1.8 | 0.87 / 0.80 / 0.86 / 0.91 |
| room0_v8_planar | 112 | 1.44 | 0.784 | 0.819 | 0.952 | 0.981 | 0.539 | **0.91** | 1.0 / 1.3 / 1.8 / 1.7 | 0.87 / 0.62 / 0.84 / 0.89 |
| room1_v7 | 83 | 1.63 | 0.837 | 0.885 | 0.934 | 0.965 | 0.653 | 0.80 | 1.0 / 1.8 / 1.8 / 1.7 | 0.81 / 0.80 / 0.84 / 0.88 |
| room1_v8_planar | 73 | 1.43 | 0.781 | 0.824 | 0.935 | 0.973 | 0.507 | **0.88** | 1.0 / 1.1 / 1.9 / 1.5 | 0.81 / 0.63 / 0.84 / 0.83 |
- Renders: the most human-like decompositions so far. room_0: floor 1 slab, rug 1, each window **one box set into
  its wall**, each wall face one slab; room_1: each window / door one box, floor 1 large slab.
- The strict metric drops (0.855 -> 0.784) *because* of the amodal completion it was designed to reward: the window
  box covers unscanned glass (free / unknown voxels), the floor slab passes under the rug and furniture (other
  instances), which the strict vIoU counts as false positives; mid-size bin (windows) 0.80 -> 0.62. Point mIoU drops
  for the same reason (rug points are inside the floor slab). I.e. vIoU against the *visible* volume is the wrong
  yardstick for amodal decompositions; the plausibility proxy and the renders say v8 is better.
- Renderer fix: flat SQs with ε1 > 1 showed concentric rings (side faces sampled with too few rows; the radially
  projected top face shrinks inwards). Now every face is sampled with as many rows as the largest extent; key runs
  were re-rendered (`code/rerender.py`).
- **v8 is now the default** (`fit.py` CFG: w_other 0.1, w_unknown 0.03, fill_planar on). Strict metric weights are
  fixed in `reeval.py` (METRIC) and used by `eval_external.py`; `fit_exist.py` keeps the weights of its runs.

| run (strict weights) | sqs | K/inst | vIoU | precision | recall | coverage | point mIoU | plausible | K by size | vIoU by size |
|---|---|---|---|---|---|---|---|---|---|---|
| office0_v6 | 67 | 1.68 | 0.870 | 0.929 | 0.925 | 0.958 | 0.576 | 0.82 | 1.0 / 1.5 / 2.4 / 1.9 | 0.90 / 0.94 / 0.85 / 0.83 |
| office0_v8_planar | 61 | 1.53 | 0.812 | 0.850 | 0.944 | 0.950 | 0.470 | 0.88 | 1.0 / 1.0 / 2.1 / 1.8 | 0.87 / 0.82 / 0.79 / 0.76 |
- office_0 v8: inside the room clean (sofa 2, desk, chairs, bins 1 each), but the rug / floor slab and the front
  wall bulge *outside* the room by up to the 30 cm region margin: with w_unknown = 0.03 the space outside the room
  became nearly free to cover.

## v9: exterior label
Unknown voxels strictly outside the room box (oriented box of the wall / floor / ceiling points, + 1 voxel) are now
their own label EXTERIOR with full weight (w_exterior = 1); unknown *inside* the room (occluded, deep inside
objects, the band at the room faces where window glass sits) stays cheap. The strict metric keeps weight 0.1 for
both (identical to before, so old runs re-score the same).

| run (strict weights) | sqs | K/inst | vIoU | precision | recall | coverage | point mIoU | plausible | K by size | vIoU by size |
|---|---|---|---|---|---|---|---|---|---|---|
| room0_v8_planar | 112 | 1.44 | 0.784 | 0.819 | 0.952 | 0.981 | 0.539 | 0.91 | 1.0 / 1.3 / 1.8 / 1.7 | 0.87 / 0.62 / 0.84 / 0.89 |
| room0_v9_exterior | 116 | 1.49 | 0.784 | 0.818 | 0.951 | 0.979 | 0.539 | 0.88 | 1.0 / 1.3 / 2.0 / 1.8 | 0.87 / 0.62 / 0.84 / 0.90 |
| room1_v8_planar | 73 | 1.43 | 0.781 | 0.824 | 0.935 | 0.973 | 0.507 | 0.88 | 1.0 / 1.1 / 1.9 / 1.5 | 0.81 / 0.63 / 0.84 / 0.83 |
| room1_v9_exterior | 75 | 1.47 | 0.783 | 0.828 | 0.927 | 0.972 | 0.510 | 0.86 | 1.0 / 1.2 / 1.8 / 1.7 | 0.81 / 0.64 / 0.84 / 0.83 |
| office0_v8_planar | 61 | 1.53 | 0.812 | 0.850 | 0.944 | 0.950 | 0.470 | 0.88 | 1.0 / 1.0 / 2.1 / 1.8 | 0.87 / 0.82 / 0.79 / 0.76 |
| office0_v9_exterior | 57 | 1.43 | 0.810 | 0.851 | 0.939 | 0.940 | 0.474 | **0.93** | 1.0 / 1.0 / 2.0 / 1.6 | 0.88 / 0.83 / 0.79 / 0.74 |
- room_0 / room_1: unchanged within run-to-run variation (±4 SQs; e.g. the room_0 floor is 1 slab in v8 and 2 in
  v9 — the floor merge decision is close to δ). **v9 is the final default.**
- office_0 v9: 57 SQs, plausibility 0.88 -> 0.93; the front wall no longer bulges. The rug slab still pokes slightly
  past the right wall at one corner: it lies inside the room box, but it is rotated a few degrees against the walls.

## Resolution: 3 cm instead of 4 cm (`room0_v8_res3`, min_gain_vox scaled to keep ~1 l)
| run | sqs | K/inst | vIoU (own grid) | plausible | K by size | vIoU by size |
|---|---|---|---|---|---|---|
| room0_v8_planar (4 cm) | 112 | 1.44 | 0.784 | 0.91 | 1.0 / 1.3 / 1.8 / 1.7 | 0.87 / 0.62 / 0.84 / 0.89 |
| room0_v8_res3 (3 cm) | 112 | 1.44 | 0.764 | 0.92 | 1.0 / 1.2 / 1.8 / 1.8 | 0.85 / 0.60 / 0.81 / 0.88 |
- Same structure (same SQ count, plausibility), 2.4x the voxels and ~1.7x the runtime -> no reason to go finer for
  this coarse goal. Scoring the 3 cm SQs on the 4 cm grid gives 0.698 (recall 0.83): SQs fitted to 3 cm-thin
  objects miss the shifted 4 cm voxel centers, i.e. voxel IoU across grids is unreliable for thin objects.
