# 19 — Plain SQ fits for every count K = 1 … 16

**Goal.** For every chair of experiment 18's dataset, one fit with exactly K plain superquadrics per K = 1 … 16, so a
later metric can choose the best count per chair.

    bash experiments/19_sq_count_sweep/run_all.sh    # rerun to resume

Output: `/home/sebastian/data/ShapeNet-Chairs-RGBD/sq_fits/<split>/kKK.npz` (test, then train, then val), fields in the
docstring of `code/fit_sqs.py`.

**Setup.** One k-means start per K, 1000 Adam steps, SuperFlex's loss and learning rates, best state per chair.
- No tapering or bending. They added 0.02–0.07 IoU in a 32-chair test, but only plain SQs are wanted here.
- Exactly K primitives always: no pruning, and a k-means cluster with fewer than 4 points starts as a tiny primitive
  (`tiny_inits` counts them per chair).
- `w_bbox = K / 16`: SuperFlex averages this term over all slots, so this is its loss with K of 16 slots in use.
- Best state stored before the Adam step. Experiment 16 stores it after, one step off.

**Status.** Stopped on request during test K = 10: K = 1 … 9 are complete, batches of K = 10 are kept and
`run_all.sh` resumes there. Train and val have not started.

**Results, test split** (1355 chairs, mean IoU on the 100k query points, no NaNs)

| K | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 |
|---|---|---|---|---|---|---|---|---|---|
| k-means start | 0.356 | 0.526 | 0.586 | 0.618 | 0.652 | 0.682 | 0.710 | 0.734 | 0.754 |

No cluster was too small for a box at any of these K.

**SuperFlex on the same chairs** (`code/fit_superflex.py`): `superflex` checkpoint, existence > 0.5, on average 7.8
primitives (2–15), 1000 steps of the same loop, with `w_bbox = 1`. `--plain` drops tapering and bending from the
prediction and the optimization. Its `bending_k` values are then unused leftovers of the parametrization.

| | IoU | median | 5th percentile |
|---|---|---|---|
| prediction | 0.653 | 0.702 | 0.250 |
| prediction, plain | 0.630 | 0.670 | 0.235 |
| optimized | 0.850 | 0.893 | 0.585 |
| optimized, plain | 0.787 | 0.827 | 0.456 |

Same count, same chairs (the 1008 with at most 9 network primitives): optimized plain 0.807 against k-means 0.752, so
the network's start is worth about 0.055. It wins on 58–84 % of chairs per K. The other 347 chairs are not compared
yet because their K was not reached.

**Viewer** (`viewer/`): `uv run python export.py` then `python -m http.server -d experiments/19_sq_count_sweep/viewer 8019`.
All 13 decompositions of every test chair, ranked per chair by the IoU above (hover for the mean over all chairs).
Keys: ← → sample, ↑ ↓ rank, r random, p surface points. Exported at 16 × 16 per primitive, so sharp edges can look
jagged. `viewer/data/` is 369 MB, not versioned.

**Checks** (before the full run)
- 10 batches at K = 4: 11 s per 64 chairs throughout, so `torch.compile` does not fall back between batches.
- Benchmark chairs at K = 4: IoU 0.687, against 0.697 with the benchmark's `w_bbox = 1`.
- A chair forced to 4 tiny clusters at K = 16 keeps 16 primitives, without NaNs.
- Hard IoU recomputed from the saved parameters with separate code: 0.621, stored 0.622 (20 chairs).
