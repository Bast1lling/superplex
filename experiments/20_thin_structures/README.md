# 20 — Why thin chairs fail, and the voxel occupancy loss

**Question.** Director's chair `fbc560d04067a6b58ce66f12ba927a2b`: k-means fails, SuperFlex is mediocre. Is it the loss,
the distance or the start? Plain SQs on 32 test chairs: this one, 15 of the thinnest quarter, 16 others (`code/variants.py`,
`results/grid.json`, `grid2.json`). 1000 steps, K = 10 for k-means.

**Variants.** Loss: SuperFlex's `iou`, or `vox`, the voxel occupancy loss of experiments 12-16, softplus(d / tau) on occupied
points and surface samples + `w_free` × softplus(-d / tau) on free ones. Distance: SuperFlex's `radial`, far too large
beside a long thin primitive (0.15 instead of 0.01), or `scaled`, g / |grad g|, exact outside a box. Start: k-means or network.

**Results.** IoU / IoU ignoring query points within 0.01 of the surface / surface recall at 0.01:

| | target chair | thin chairs | other chairs |
|---|---|---|---|
| k-means, radial, IoU loss | 0.368 / 0.445 / 0.351 | 0.621 / 0.725 / 0.655 | 0.815 / 0.883 / 0.761 |
| k-means, scaled, IoU loss | 0.479 / 0.682 / 0.515 | 0.616 / 0.791 / 0.696 | 0.811 / 0.910 / 0.805 |
| k-means, radial, voxel 0.005 × 50 | 0.425 / 0.658 / 0.661 | 0.556 / 0.702 / 0.702 | 0.791 / 0.894 / 0.774 |
| network, radial, IoU loss | 0.663 / 0.800 / 0.505 | 0.714 / 0.841 / 0.642 | 0.825 / 0.894 / 0.751 |
| network, scaled, IoU loss | 0.649 / 0.811 / 0.642 | 0.683 / 0.853 / 0.676 | 0.824 / 0.911 / 0.784 |
| network, radial, voxel 0.005 × 50 | 0.601 / 0.787 / 0.484 | 0.646 / 0.810 / 0.648 | 0.802 / 0.899 / 0.780 |

- The voxel loss needs a large `w_free`: at 1 thin-chair IoU is 0.24–0.37, at 50 about 0.55–0.65. It does not beat the IoU loss.
- The start matters most on the target. The scaled distance helps k-means there and raises recall in every group.
- All of this scores against the flawed occupancy below, so the absolute numbers on sheet chairs are too low.

**Metric** (`code/metric_gap.py`, all 1355 test chairs, plain). Prediction → optimized: IoU 0.630 → 0.785, ignoring the
0.01 shell 0.764 → 0.866, surface recall 0.553 → 0.722. Only about a third of the gap is the shell at this tolerance.

**Occupancy flaw.** The viewer's voxels show the back and seat of the target mostly missing, see experiment 18's limits.
Occupied fraction of surface samples moved along their normals (mean of both directions, 0.5 = closed solid):

| offset | 0 | 0.001 | 0.002 | 0.004 | 0.008 |
|---|---|---|---|---|---|
| target | 0.856 | 0.855 | 0.705 | 0.361 | 0.357 |
| thin chairs | 0.996 | 0.995 | 0.767 | 0.490 | 0.420 |
| other chairs | 1.000 | 0.999 | 0.804 | 0.519 | 0.505 |

**Fix** (`code/fix_occupancy.py`, applied to all 6778 chairs): every query point within 0.005 of the 100k surface samples
is occupied. The old labels stay in `points.npz` as `occupancies_fusion`. Checked in `code/occupancy.py` on the 1355 test
chairs. Solid reference: inside ONet's watertight surface (behind the normal of the nearest sample); the ShapeNet mesh's own
winding number fails, doubled faces cancel to 0. Shell reference: query points within 0.005 of 1M mesh samples.

| labels | IoU, solid reference (median, 5th pct) | shell recall |
|---|---|---|
| fusion | 0.827 (0.895, 0.457) | 0.726 |
| fixed | 0.864 (0.923, 0.557) | 0.919 |

On the 48-chair pilot, tolerance 0.0075 and 0.01 overfill: IoU 0.832 and 0.783. The target goes from 0.521 to 0.693.

**Old fits on the new labels** (`code/rescore.py`, no refitting, reproduces the stored IoUs on the old labels):
SuperFlex prediction 0.653 → 0.670, optimized 0.850 → 0.807, plain 0.630 → 0.647 and 0.787 → 0.757, k-means K = 9
0.754 → 0.733. The fits were optimized on the flawed labels, so these numbers understate them. The pred → opt gap goes
from 0.197 to 0.136. Refit on the new labels, target only: network + IoU loss 0.660 → 0.751, k-means + scaled 0.499 → 0.537.

**Why the target's seat and back fail** (`code/diagnose.py`, `code/trim_test.py`, fixed labels, k-means 10, one chair).
"Covered" = not further than 0.01 outside any SQ. The scaled distance overestimates depth inside, so on / inside is not
a reliable split.

| fit | seat covered | back covered | IoU | precision | recall | SQs with ≤ 2 query points |
|---|---|---|---|---|---|---|
| voxel 0.005 × 50, radial | 0.96 | 0.92 | 0.509 | 0.72 | 0.63 | 0 |
| IoU loss, scaled | 0.19 | 1.00 | 0.525 | 0.92 | 0.55 | 2 |
| IoU loss, scaled, no cap (truncation 0.5) | 0.81 | 1.00 | 0.552 | 0.84 | 0.62 | 1 |
| IoU loss, scaled, surface term × 10 | 0.97 | 1.00 | 0.291 | 0.32 | 0.77 | 1 |
| voxel, radial, with cap | 0.83 | 1.00 | 0.569 | 0.82 | 0.65 | 0 |

- Seat: k-means boxes span seat and frame, and free space drives them off the seat in both fits (uncovered 0.15 → 0.5+ by
  step 100). The voxel fit recovers at steps 400-700, when one SQ flattens to a half-thickness of 0.006 and spreads over
  the seat. In the IoU fit its two seat SQs end as flakes (half-thickness 0.002-0.003, half-length 0.08-0.11) holding no
  query points, with the IoU term's shrinking and the surface term's growth in balance. Estimated cost of one free
  query point against one surface sample 0.01 outside: about 17:1 in SuperFlex's loss, about 1:1 in the voxel loss.
- Back: the voxel fit's sheet spans y -0.25 to 0.47, the panel 0.15 to 0.49. Trimming it to the panel raises the radial
  loss (4.67 → 6.06) but lowers the same loss with the scaled distance (3.94 → 3.70). The panel's own samples hardly
  move, so the radial overestimate at the few samples past the rim (1.5 % of the surface) holds the sheet long.
- The seat is 23 % of the surface and 28 % of the occupied query points, so IoU does count it. The IoU fit scores higher
  because it pays for the missing seat what the voxel fit pays for air: recall 0.55 against precision 0.72.

**Viewer** (`viewer/`): `uv run python export.py`, `uv run python export_voxels.py`, then
`python -m http.server -d experiments/20_thin_structures/viewer 8020`. The target with 41 decompositions, the 28 variants
refit on it alone (network rows match the grid closely, k-means rows differ by up to 0.05), voxels from a 128³ fusion
grid (`v`) and the 864 occupied query points (`o`).

**Findings viewer** (`findings/`, usage in its `export.py`, port 8021). Seat: 8 fits, 8 snapshots each, surface samples
green or red by coverage, per SQ its size, query points and the growth each loss term asks for. Back: the voxel fit's sheet
full or trimmed, with the samples lost and the free query points inside it.

**PartField** (`papers/PartField`, not versioned, separate environment on torch 2.7 for the RTX 5090, NVIDIA license:
non-commercial research only). Objaverse checkpoint, mesh mode with k-NN adjacency, levels of 1 to 20 parts. Results in
`/home/sebastian/data/ShapeNet-Chairs-RGBD/partfield`. `../19_sq_count_sweep/viewer/export_partfield.py` moves the labels
onto 16384 mesh samples (mean 0.002 from the dataset's surface on the target), shown as point colors in both chair viewers.

`code/grow.py` (adding primitives at uncovered surface) was never run.
