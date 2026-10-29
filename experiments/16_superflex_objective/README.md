# 16 — SuperFlex's test-time objective vs. ours, from three initializations

**Question.** On the presentation branch, SuperFlex (robust checkpoint, run per GT instance) beat all our
optimizations. Is that the network, the deformable primitives, or the objective?

**Setup** (`code/sfopt.py`, see its docstring). Per GT instance as in SuperFlex's ShapeNet setup: 4096 surface points,
100k query points in the instance box (+10 %) labelled from the 4 cm GT grid (experiment 12's `gt_grid`; other
instances = outside, deep voxels = nearest solid instance). SuperFlex's parametrization (tapering, bending, smooth
union), parameter groups and loop (1000 Adam steps, best state per instance). 3 initializations × 2 objectives:
- network: SuperFlex robust; kmeans_net: k-means boxes with the network's primitive count; kmeans16: 16 boxes +
  SuperFlex's pruning.
- theirs: -log soft IoU + 16 · truncated surface SDF + bounding-box term + 0.1 · overlap; ours: the presentation's
  instances loss (softplus occupancy, free 1 / deep 0.1 / other instances 0, tau 2 cm) on the same query points.
Evaluation: experiment 12's per-instance vIoU, FP counted within the instance box + 0.5 m (re-evaluating 12's fits
this way changes them by < 0.005, see experiment 17).

**TL;DR.** The objective is the lever, not the network or the deformations.

| init | theirs (room_0 / room_1 / office_0 → mean) | ours | primitives |
|---|---|---|---|
| network, no optimization | 0.330 / 0.386 / 0.337 → 0.351 | – | 250 / 149 / 162 |
| network | 0.831 / 0.844 / 0.789 → **0.821** | 0.761 / 0.751 / 0.574 → 0.695 | 250 / 149 / 162 |
| k-means (network's count) | 0.799 / 0.780 / 0.773 → 0.784 | 0.746 / 0.716 / 0.558 → 0.673 | 250 / 149 / 162 |
| k-means 16 + pruning | 0.791 / 0.818 / 0.777 → 0.795 | 0.617 / 0.703 / 0.567 → 0.629 | ~550–1130 |

- Ours overfills (recall 0.93–0.97, precision 0.58–0.79); theirs stays tight (precision 0.86–0.98).
- k-means with the network's count is within ~0.04 of the network init: the network mostly contributes the count.
- Pruning keeps almost all of 16 slots (only near-empty primitives are removed): no simplification.
- Plain SQs (`--plain`, no tapering / bending; room_0, their objective): network 0.817 (vs 0.831), kmeans_net 0.805
  (vs 0.799), kmeans16 0.787 (vs 0.791). The deformations are not needed.
- Reference: one SQ per instance with our loss (experiment 12 `gt3d`): 0.709.

**Viewer** (`viewer/`): `uv run python export.py` then `python -m http.server -d viewer 8016`; all runs per scene
(keys 1–9, 0), colored by instance / primitive / instance IoU, click an object for its IoU in every run.

Runs: `results/runs/<scene>_<init>_<objective>[_plain]/{metrics.json, sqs.npz}`, logs in `results/logs/`.

    cd code && uv run python sfopt.py --scene room_0 --init network|kmeans_net|kmeans16 --objective theirs|ours [--plain] [--steps 0]
