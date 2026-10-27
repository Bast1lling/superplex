# 08 — Volumetric, label-driven SQ decomposition with greedy model selection

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
3. **Greedy structure search** (all instances in parallel, batched as instances × 8 slots). Start with one PCA box per
   instance. Each round proposes one change per instance: *add* an SQ on the largest connected component of uncovered
   target voxels, or *split* the SQ that covers most negatives (2-means). After 250 joint Adam steps the change is
   kept only if the instance's hard volumetric IoU rose by more than **δ** (the price of a primitive, in IoU);
   otherwise the instance is restored. Two failed proposals in a row end the search for that instance. Finally SQs
   whose removal costs < δ/2 IoU are pruned, and everything is polished at a lower temperature.

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

## Runs
See `results/notes.md` for the log of runs and observations.

    cd code && python fit.py --name <run> [--set key=value ...]     # defaults in fit.py: CFG
