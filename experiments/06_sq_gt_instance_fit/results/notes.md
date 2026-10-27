# 06 — SQ fit to GT instances (limit test)

Setup: room_0, 81 labeled instances, K = 6 SQs per instance (486 SQs), k-means + box-fit init, radial SDF,
forward (points -> own instance's SQs) + backward (visible union surface -> own instance's points, per-instance
EDT at 1 cm), weight 1 : 1, 2000 steps, ~41 s per run. No cross-instance terms.

Metrics: own union SDF U_y(p) = min over the instance's SQs. enclosed: U_y <= 1 cm; leaked: > 1 cm inside an SQ of
another instance; mIoU: point -> instance with the most negative union SDF. `_inst` = mean over instances.

| forward mode | mIoU | fwd (cm) | bwd (cm) | enclosed_inst | leaked_inst |
|---|---|---|---|---|---|
| init (k-means + box fit) | 0.661 | 6.18 | 2.79 | 0.862 | 0.222 |
| union_abs  \|min_j d\| | 0.773 | 1.19 | 1.42 | 0.829 | 0.084 |
| enclosure  relu(min_j d) | 0.741 | 9.03 | 1.01 | 0.980 | 0.225 |
| min_abs  min_j \|d\| | 0.776 | 1.93 | 1.57 | 0.883 | 0.099 |

By instance size (points at 1 cm; n = 18 / 26 / 17 / 20), enclosed / leaked / IoU:

| mode | < 1k | 1k–5k | 5k–20k | > 20k |
|---|---|---|---|---|
| union_abs | 0.96 / 0.11 / 0.65 | 0.84 / 0.06 / 0.75 | 0.82 / 0.10 / 0.82 | 0.71 / 0.07 / 0.86 |
| enclosure | 1.00 / 0.46 / 0.51 | 0.98 / 0.18 / 0.78 | 0.98 / 0.25 / 0.74 | 0.97 / 0.06 / 0.91 |
| min_abs   | 0.98 / 0.16 / 0.66 | 0.88 / 0.09 / 0.73 | 0.91 / 0.09 / 0.86 | 0.78 / 0.06 / 0.87 |

For comparison, experiment 05 (`room0_rc1_ref`), enclosed per instance: 0.05 / 0.20 / 0.50 / 0.48.

## Observations
- With the assignment given, SQs enclose small instances almost perfectly (0.96–1.00 vs 0.05 in experiment 05).
  The enclosure problem of experiment 05 is therefore one of assignment / recruitment, not of the primitive or the SDF.
- The "upper bound" mIoU of ~0.77 with 6 SQs per instance is close to experiment 05's (0.67–0.76): the
  separation of instances via "most inside SQ" is limited by leakage, not by coverage.
- enclosure: forward relu(d) lets SQs bloat (points 9 cm inside the union on average). The backward term only sees the
  visible union surface, so it keeps the surface near points but cannot stop volume growing into neighbors:
  small instances leak 46% (they sit inside the SQs of e.g. the table below them).
- union_abs vs min_abs: nearly identical. min_abs encloses slightly more (internal faces allowed), leaks slightly more.
- Small instances leak 10–16% even in the surface modes: neighbors' SQs, fitted without knowledge of them, overlap them.
  This is the floor for any method that fits instances independently; reducing it needs a cross-instance term.

## Existence + parsimony (forward mode union_abs)

Existence e_j per SQ (init 0.9). Forward = expected value under Bernoulli existence (SQs sorted by SDF, Paschalidou
et al. 2019), backward weighted by e_j, parsimony lambda * mean_instances sum_j e_j. Evaluation uses SQs with e > 0.5.

| run | SQs / instance | mIoU | fwd (cm) | bwd (cm) | enclosed_inst | leaked_inst |
|---|---|---|---|---|---|---|
| no existence | 6.00 | 0.773 | 1.19 | 1.42 | 0.829 | 0.084 |
| lambda 2 mm  | 4.73 | 0.760 | 1.20 | 1.52 | 0.825 | 0.098 |
| lambda 5 mm  | 3.83 | 0.766 | 1.29 | 1.72 | 0.812 | 0.086 |
| lambda 10 mm | 3.07 | 0.738 | 1.37 | 1.94 | 0.802 | 0.111 |

Active SQs per instance by size (< 1k / 1k-5k / 5k-20k / > 20k points):
2 mm 3.3 / 4.4 / 5.3 / 6.0; 5 mm 2.3 / 3.4 / 4.1 / 5.6; 10 mm 1.3 / 3.1 / 2.6 / 5.1.

- Existence becomes binary (no SQ ends between 0.1 and 0.9); the pruning decisions are made by step ~500.
- Pruning halves the SQ count at almost no cost in fit quality, but it removes mostly redundant (overlapping) SQs.
  Large instances keep 5-6 SQs even at 10 mm: an SQ covering 1/6 of a wall cannot be removed, because its points
  would be ~0.5 m from the next SQ. Existence can only prune what neighbors already cover; it cannot merge patches
  that would first have to grow. Removing the patchwork needs a non-local move (merge proposals, or top-down
  splitting starting from 1 SQ per instance).

## Top-down splitting (`room0_split_gain5mm`)

Start with 1 box-fitted SQ per instance (8 slots). Every 200 steps (200-1800): per instance, split the SQ with the
largest share of the instance's mean forward error if the share > 5 mm (2-means of its owned points, box fit per half,
fresh Adam state). 2500 steps, 56 s.

| run | SQs / inst | mIoU | fwd (cm) | bwd (cm) | enclosed_inst | leaked_inst |
|---|---|---|---|---|---|---|
| k-means K=6 | 6.00 | 0.773 | 1.19 | 1.42 | 0.829 | 0.084 |
| K=6 + existence, lambda 5 mm | 3.83 | 0.766 | 1.29 | 1.72 | 0.812 | 0.086 |
| top-down, gain 5 mm | 4.63 | 0.684 | 1.01 | 1.94 | 0.845 | 0.164 |

SQs per instance by size (< 1k / 1k-5k / 5k-20k / > 20k): 1.3 / 4.4 / 5.2 / 7.4; histogram over 1..8 SQs:
14 11 9 9 6 3 4 25.

- Small instances: mostly 1 SQ, as intended (1.3 on average, enclosed 0.97).
- Large planar instances are over-split: the ceiling has 8 SQs, 5 of them near-identical 4 x 2.7 m slabs on top
  of each other. The criterion "largest error share" does not check whether a split reduces the error; relief on the
  ceiling (fixtures, recesses) keeps the share high and the children drift back to the same plane.
- Leakage doubles (0.084 -> 0.164): the most leaked instances are blinds, switches, vents and a lamp, i.e. objects flush
  with a wall or the ceiling. The whole-instance box fit of a wall / ceiling is a slab whose thickness (14-16 cm) spans
  the wall's own relief, and it swallows the flush objects. Forward and backward losses of the wall are both fine
  with that, since no term sees foreign points. The local k-means pieces were thinner and leaked less.

### Bug fix: backward normalization with free slots
The backward term was averaged over all K slots, including free ones, so in split mode an instance with n alive SQs
got only n/K of its backward weight (bloated single SQs were nearly free). Now: mean over the instance's alive SQs.
Runs without free slots (k-means, existence) are unaffected. `room0_split_gain5mm` and
`room0_split_verified_gain5mm` were run with the bug; `_fixed` reruns below.

## Verified splits (fix 1)
A split is judged `every` = 200 steps later: kept if the instance loss L_k (forward + backward, all points) dropped by
more than min_gain = 5 mm, else both slots are restored and the split SQ is excluded from further proposals.

| run | SQs / inst | mIoU | fwd (cm) | bwd (cm) | enclosed_inst | leaked_inst |
|---|---|---|---|---|---|---|
| K=6 + existence, lambda 5 mm | 3.83 | 0.766 | 1.29 | 1.72 | 0.812 | 0.086 |
| top-down, unverified (fixed) | 4.91 | 0.722 | 1.11 | 1.81 | 0.827 | 0.128 |
| top-down, verified (fixed) | 2.25 | 0.558 | 2.16 | 2.85 | 0.788 | 0.230 |

SQs per instance by size (< 1k / 1k-5k / 5k-20k / > 20k): unverified 1.7 / 4.9 / 5.7 / 7.3, verified 1.1 / 2.4 / 2.2 / 3.2.

- Verification rejects too much: large instances keep 3.2 SQs with 6 cm forward error and 36% leakage, so splits that
  are needed get undone. Likely causes: 200 steps (with decaying LR) are not enough for two fresh children to beat a
  parent that has been optimized all along; a rejected SQ is excluded for good; and resetting the Adam moments while
  the step counter stays high makes the first updates of the new rows ~3x the learning rate.
- Unverified top-down after the fix: better than before (mIoU 0.684 -> 0.722, leaked 0.164 -> 0.128) but still
  over-splits large instances and leaks more than the k-means + existence variant.

## Fair verification (option b, `room0_split_fair_gain5mm`)
- A shadow copy trains on the same batches without the trial splits; keep if L_k(split) < L_k(shadow) - 5 mm,
  else copy instance k back from the shadow (no lost progress).
- A rejected SQ cools down for 2 checks instead of being excluded for good.
- Halves inherit the parent's Adam second moments. 87 s (the shadow doubles the step cost during trials).

Result: mIoU 0.594, fwd 1.91 cm, bwd 3.01 cm, enclosed_inst 0.786, leaked_inst 0.184, 2.01 SQs / instance.
Only slightly better than the unfair verification (0.558, 2.25 SQs / inst).

Gain of judged splits, L_k(shadow) - L_k(split), p10 / median / p90 (mm):
step 400: -53 / 5.3 / 59; 600: -5 / 5.2 / 15; 800: 2 / 6.7 / 13; 1000: -56 / 0.3 / 5; 1200: -2 / 2.1 / 13;
1400: -3 / 2.4 / 7; 1600: -62 / -8.9 / 4; 1800: -6 / -0.5 / 3; 2000: -3 / 2.0 / 5.
- Most splits help, but by less than 5 mm after 200 steps; the median gain is ~5 mm early and ~2 mm later.
- A tail of splits hurts a lot (-55 mm): the 2-means halves with fresh box fits start far worse than the converged
  parent and do not recover in 200 steps, especially late in the cosine schedule (LR at step 1600 is 19% of the start).
