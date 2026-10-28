# 08 — Volumetric, label-driven SQ decomposition with greedy model selection

## TL;DR (current state, see `results/notes.md` for the full log)
- **Fit SQs to volumes, not to surfaces.** Per-instance *solid* voxels are derived from the oriented points
  (behind-the-surface test), and every instance's SQ union is fitted with a logistic occupancy loss against its own
  voxels vs. free space / other instances. Experiment 06's best surface fit scored with the same metrics:
  307 SQs, vIoU 0.37, 24% plausible primitive counts; this experiment (v6/v7, room_0): **133-135 SQs, vIoU
  0.83-0.86, 83% plausible**.
- **Structure by explicit model selection** (add / split / merge / prune, each kept only if it changes the instance's
  volumetric IoU by more than δ and explains ≥ 15 voxels).
- **Amodal completion** makes it human-like (v8; v9 = v8 + exterior label, default): overlap with other instances / unknown space is cheap and
  thin planar instances (floor, walls, windows, rugs) get the convex hull of their 2D projection as target. room_0:
  **112 SQs for 78 instances, 91% plausible**: floor 1 slab, rug 1, each wall face 1, each window 1 box in its wall,
  blinds / pictures / doors / cushions / books 1 each, chairs and sofas 1-3. room_1: 73 SQs / 51 instances, 88%.
  office_0 (v9): 57 SQs / 40 instances, 93%. Renders: `results/runs/*/render.jpg` (start with `room0_v8_planar`,
  `room1_v8_planar`, `office0_v9_exterior`).
- Works with **semantic labels only** (pseudo-instances = connected components per class; v6: vIoU 0.825) and is
  robust in structure to scan-like degradation (50% dropout, holes, 1 cm noise, 15° normal noise: same SQ count
  and plausibility, vIoU 0.83 -> 0.69, mostly on small objects).
- **Fixed slots + existence + parsimony** (what a feed-forward model would be trained with) reaches the same
  budget / quality with this loss, but cannot *merge*: large instances stay as 2-5 slabs. The search results are
  therefore the better pseudo-targets for a network for now.
- Key lessons: (1) amodal overlap must be cheap and planar holes filled; (2) free space seen through windows is not
  observed; (3) decisions must compare *settled* states (lr decay per round) — constant-lr jitter was as large as δ;
  (4) IoU-relative prices over-split tiny objects, add an absolute minimum volume; (5) ring / L-shaped instances need
  compact-part proposals; (6) strict vIoU against the visible volume penalizes amodal completion — judge with the
  plausibility proxy + renders.

Best config = defaults in `code/fit.py` (v9). Runtime on this CPU-only box: ~20 min per room with 2 threads.

**Question.** Can a classic per-scene optimization produce a *sensible, human-like* decomposition of a Replica
scene into few superquadrics (SQs), each belonging to one semantic instance?

**Why a new formulation.** Experiments 05–07 fit SQ *surfaces* to *points* (forward point→SQ, backward SQ→point
distances). With surface terms nothing penalizes an SQ for filling empty space or swallowing a neighbor
(06: flush objects swallowed by wall slabs, leakage), and large planar surfaces are cheapest to explain by
patchwork (07: 14–72 SQs per wall). Humans decompose *volumes*. So here:

1. **Volume from oriented points** (`code/volume.py`). Voxel grid (4 cm). A voxel is solid if it lies behind the
   surfaces of its k nearest points (normals point to free space, verified on Replica), else free. Solid voxels get the
   instance of the nearest point. Space behind walls / floor / ceiling (structural classes) is kept only 10 cm deep,
   every instance's solid is clipped to its gravity-aligned oriented box (+1 voxel, fixes space behind windows /
   blinds), anything deeper than 1 m is *unknown*. Voxels containing points always belong to their instance (thin
   objects). Labels: free / unknown / instance.
2. **Per-instance occupancy loss** (`code/fit.py`). For instance k: positives = its solid voxels, negatives = free
   voxels (w_free), other instances' voxels (w_other) and unknown voxels (w_unknown) inside its box + 30 cm margin.
   Union SDF d = min_j d_j (radial SDF), loss = logistic: softplus(d/τ) on positives, w·softplus(−d/τ) on negatives,
   normalized by |T_k| (every instance counts equally). The logistic is linear far from the boundary, so distant
   targets / deep intrusions always produce gradients. A penalty keeps each SQ's extent inside the region where
   negatives are sampled.
3. **Greedy structure search** (all instances in parallel, batched as instances × 8 slots). Start with one PCA box
   per instance (of its largest *compact part*: recursive 2-means until the box is ≥ 10% filled, so ring / L-shaped
   instances are not covered by one room-sized box). Each round proposes one change per instance: *add* an SQ on the
   (compact part of the) largest connected component of uncovered target voxels, or *split* the SQ that covers most
   negatives (2-means). New SQs first move alone (100 steps), then all SQs jointly (250 steps, lr cosine-decayed to
   5% so the decision compares settled states). The change is kept only if the instance's hard volumetric IoU rose
   by more than **δ = 0.03** (the price of a primitive) and the gain is ≥ 15 voxels (~1 l); otherwise the instance
   is restored. Three failed proposals end the search for an instance (after a failure the other kind is tried).
   Then *merge* rounds (replace the most box-like pair of SQs by one box, refit) and *prune* rounds (drop the
   cheapest SQ, refit), each kept if the IoU fell by < δ (or < 15 voxels). Finally a polish at lower temperature.

## Evaluation (`code/evaluate.py`)
Standard CD / SDF error rewards patchwork, so the main numbers are *per-instance* and judged at a given SQ budget:
- **K**: SQs per instance (parsimony), reported by instance size.
- **vIoU**: volumetric IoU of the instance's SQ union with its solid voxels (free + other instances count as
  false positives, unknown ignored). **precision** = 1 − leakage (share of the union volume that is the instance),
  **recall**.
- **coverage**: share of the instance's points inside or within 3 cm of its union.
- **point mIoU / acc**: points assigned to the instance whose union is most inside (as in 05–07).
- **Renders** (`render.jpg` per run): GT instances, SQs colored by instance, SQs colored per primitive, from an
  oblique and a top view (ceiling hidden). I judged "human-likeness" mostly from these.

Size bins are in 2 cm points: < 250 / 250–1250 / 1250–5000 / > 5000 (≈ 1k / 5k / 20k at 1 cm as in 06).

## Files
- `code/volume.py` scene loading (instance or semantic labels, scan-like degradation), labeled voxel volume.
- `code/fit.py` greedy volumetric fitter (main method; defaults = best config). `code/fit_exist.py` fixed slots +
  existence + parsimony variant. `code/sq.py` SQ parameters / SDF.
- `code/evaluate.py` metrics + renders, `code/plausible.py` plausible-K table, `code/compare.py` run table,
  `code/reeval.py` re-score runs with the strict metric weights, `code/eval_external.py` score experiment 06's
  snapshots, `code/rerender.py`, `code/export_viewer.py` + `viewer/server.py` (06's viewer on these runs; untested
  here because it needs the user's data paths / trackio).
- `results/runs/<run>/`: config.json, metrics.json (+ metrics_std.json = strict re-score), sqs.npz (final SQs:
  trans, rot (local-to-world columns), scale, eps, instance index; instance_ids), render.jpg.
- `results/notes.md`: chronological log of all runs and observations.

    cd code && python fit.py --name <run> [--set key=value ...]     # e.g. --set scene=room_1 --set labels=semantic
    python compare.py <run> ...                                     # summary table
The volume cache goes to /tmp/claude-0/volcache (~30 s per scene to rebuild).
