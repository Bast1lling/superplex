# 07 — Global top-down SQ fit (no instance labels)

## `room0_global_tol5mm`: stuck at 1 SQ
Config as in config.yaml (1 SQ for the whole scene, verified 2-means splits, local tolerance 5 mm, 4000 steps).

- The single SQ becomes a room-sized shell: its surface lies on walls / floor / ceiling (backward 3.5 cm, 18% of the
  surface in air), the furniture inside is unexplained (forward 18 cm).
- Every split (one attempt every other check) is undone with gain -17 to -25 mm: forward improves (18 -> 14 cm), but
  backward triples (3.5 -> 10 cm). Likely cause: the 2-means halves of a shell are two open half-shells whose box fits
  do not overlap, so their cut faces are exposed and lie in the middle of the room, far from any point.
- Final: 1 SQ, fwd 18.2 cm, bwd 3.5 cm, oracle mIoU 0.002.

## `room0_global_init8_tol5mm`: start with 8 SQs (k-means of the scene, box fit per cluster)
Grows 8 -> 638 SQs by step 3000; the final verification undoes 134 -> 504 SQs. 91 s.

| step | SQs | fwd mean (cm) | bwd (cm) | surface in air | oracle mIoU | point purity |
|---|---|---|---|---|---|---|
| 0 | 8 | 31.1 | 30.2 | 0.89 | 0.020 | 0.30 |
| 1000 | 42 | 5.2 | 8.0 | 0.40 | 0.075 | 0.38 |
| 2000 | 215 | 3.5 | 2.2 | 0.11 | 0.125 | 0.44 |
| 4000 | 504 | 3.0 (median 0.9, p90 6.5) | 1.2 | 0.03 | 0.168 | 0.46 |

Split gains (median) drop from 15-24 mm early to ~3 mm late; the share of undone splits grows (late: ~70%).

GT diagnostics at the end:
- Mean SQ purity 0.91, but point purity 0.46: most SQs are pure, but a few large impure SQs own most points
  (largest semi-axes 2.3 x 3.4 x 0.7 m, 1.4 x 1.3 x 2.2 m, 2.0 x 1.2 x 0.8 m).
- 61% of instances are the majority of no SQ: small instances (< 1k points) 0 SQs, 1k-5k 1.3, 5k-20k 1.9, but
  > 20k (walls, floor, ceiling, rug) 21.8 SQs each: patchwork on large surfaces, nothing for small objects.
- Assigning points to the nearest SQ surface instead of the most-inside SQ does not change this (0.169 / 0.473).
- 14% of points are > 2 cm inside the union, 4% > 10 cm.

Interpretation: the split candidate test uses the *mean* forward error of an SQ's owned points. A small object inside
or on a large SQ adds only a few high-error points, so the large SQ's mean stays below 5 mm and it is never split;
meanwhile large flat regions keep getting halved.

## `room0_global_residual_tol5mm`: residual split, start with 1 SQ
Proposal: parent untouched, new SQ box-fitted to the largest 26-connected cluster (5 cm voxels) of the parent's points
with forward error > 2 cm. Judged on the cluster C: gain = mean_C f(shadow) - mean_C f(main) - b_new - (b_parent change),
keep if > 5 mm.

Result: stalls at 3 SQs (fwd 17.3 cm, bwd 3.1 cm, oracle mIoU 0.004). Gains of the judged proposals: -0.4, 3.2, 2.5, 5.6
(kept), 5.9 (kept), then negative.

Diagnosis on the final state:
- All 3 SQs are room-sized (semi-axes 3.9 x 2.2 x 1.4, 3.2 x 1.8 x 1.1, 3.2 x 2.0 x 1.2 m): near-duplicate shells.
- 57% of all points are > 2 cm from the union. Their largest connected cluster holds 77% of them (785k points; 73% with
  2 cm voxels): the room shell does not hug walls / floor / ceiling within 2 cm, so the "residual" is the shell region
  itself plus everything touching it, and the new SQ is another room box.
- The other clusters are object-sized (47k, 40k, 39k, 26k, 24k points; 20 clusters >= 20 points), but only one
  proposal per parent and check is made, and the room shell is the only parent.
=> The residual split only works once the parent explains most of its points; a single room shell never does.

## `room0_global_dbscan_residual`: DBSCAN-6D init + residual splits
Init: DBSCAN on [x, 0.143 * n], eps 2 cm, minPts 5 -> 92 clusters >= 500 points (noise 5%, small clusters 2%),
one box-fitted SQ each. Splits: residual (threshold 2 cm, every 26-connected 5 cm-voxel cluster >= 20 points, up to
8 per parent and check), fair verification with tolerance 5 mm (rejected residual proposals only remove the new SQ).
642 s; one CUDA OOM warning (allocator retry, run continued).

| step | SQs | fwd mean / median / p90 (cm) | bwd (cm) | in air | oracle mIoU | point purity |
|---|---|---|---|---|---|---|
| 0 | 92 | 42.8 / - / - | 3.1 | 0.18 | 0.361 | 0.77 |
| 1000 | 780 | 2.1 | 2.1 | 0.08 | 0.415 | 0.72 |
| 2000 | 1630 | 1.3 | 1.4 | 0.04 | 0.450 | 0.75 |
| 4000 | 1969 | 0.9 / 0.5 / 2.1 | 1.1 | 0.02 | 0.475 | 0.77 |

The pool cap (2048) is reached at step 2400. Per size (< 1k / 1k-5k / 5k-20k / > 20k): oracle IoU 0.15 / 0.60 / 0.50 /
0.59, SQs per instance 0.7 / 7.2 / 13.2 / 72.2; instances without an SQ 22% (k-means start: 61%).

- Much better than the k-means start (oracle mIoU 0.168 -> 0.475, purity 0.46 -> 0.77, fwd p90 6.5 -> 2.1 cm),
  but most of it comes from the initialization (step 0: 0.361 / 0.77); fitting adds +0.11 oracle mIoU.
- Patchwork is back at scale: 72 SQs per large instance. The residual threshold of 2 cm turns every surface detail
  into a cluster, and ~40% of the proposals pass the 5 mm gain test (gain median 2-3 mm, p90 ~20 mm).

## `room0_global_dbscan_residual_strict`: stricter residual splits
As above, but residual threshold 5 cm, clusters >= 200 points, keep if gain > 15 mm. 121 s.

| run | SQs | fwd mean / median / p90 (cm) | bwd (cm) | in air | oracle mIoU | point purity | inst. without SQ |
|---|---|---|---|---|---|---|---|
| DBSCAN init only (step 0) | 92 | 42.8 / - / - | 3.1 | 0.18 | 0.361 | 0.77 | - |
| residual 2 cm / 20 pts / 5 mm | 1969 | 0.9 / 0.5 / 2.1 | 1.1 | 0.02 | 0.475 | 0.77 | 22% |
| residual 5 cm / 200 pts / 15 mm | 377 | 1.2 / 0.6 / 2.8 | 1.7 | 0.06 | 0.466 | 0.75 | 30% |

Per size (< 1k / 1k-5k / 5k-20k / > 20k): oracle IoU 0.19 / 0.55 / 0.50 / 0.58; SQs per instance 0.3 / 1.2 / 2.5 / 14.3.

- 5x fewer SQs at almost the same oracle mIoU (0.466 vs 0.475) and fit (fwd 1.2 vs 0.9 cm); SQ count saturates at
  ~380 after step 1200 (kept splits per check drop to 2-14).
- Mid-size instances now get 1-2.5 SQs; large ones still 14 (vs 72), small ones mostly none (0.3).
