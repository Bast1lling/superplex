
# Speed-up ablation 2026-11-06 14:29, NVIDIA GeForce RTX 5090, batch 8, 80 steps

## dense

| config | step ms | speed-up | peak GB | Δloss (rel., 2 batches) | grad cosine | Δ|grad| (rel.) |
|---|---|---|---|---|---|---|
| base | 155 | 1.00x | 2.0 | 0.0e+00 | 1.000000 | 0.0e+00 |
| surface | 121 | 1.28x | 1.9 | 0.0e+00 | 1.000000 | 1.7e-07 |
| batched | 100 | 1.55x | 3.3 | 2.1e-07 | 1.000000 | 0.0e+00 |
| surface+batched | 100 | 1.55x | 3.2 | 2.1e-07 | 1.000000 | 1.7e-07 |
| assign | 103 | 1.50x | 2.0 | 4.5e-08 | 1.000000 | 0.0e+00 |
| pool | 148 | 1.05x | 2.0 | 0.0e+00 | 1.000000 | 0.0e+00 |
| 1-3 | 48 | 3.21x | 3.2 | 1.3e-07 | 1.000000 | 1.7e-07 |
| 1-3+bf16 | 44 | 3.51x | 3.1 | 3.1e-04 | 0.999994 | 2.6e-03 |
| 1-3+compile | 47 | 3.30x | 3.2 | 1.3e-07 | 1.000000 | 0.0e+00 |
| 1-3+bf16+compile | 46 | 3.35x | 3.4 | 3.2e-04 | 0.999994 | 2.5e-03 |

## topk

| config | step ms | speed-up | peak GB | Δloss (rel., 2 batches) | grad cosine | Δ|grad| (rel.) |
|---|---|---|---|---|---|---|
| base | 230 | 1.00x | 2.6 | 0.0e+00 | 1.000000 | 0.0e+00 |
| surface | 162 | 1.42x | 2.4 | 0.0e+00 | 1.000000 | 3.1e-07 |
| batched | 124 | 1.85x | 3.7 | 8.3e-08 | 1.000000 | 0.0e+00 |
| surface+batched | 116 | 1.98x | 3.6 | 1.3e-07 | 1.000000 | 3.1e-07 |
| assign | 191 | 1.20x | 2.5 | 3.2e-08 | 1.000000 | 0.0e+00 |
| pool | 227 | 1.01x | 2.3 | 0.0e+00 | 1.000000 | 0.0e+00 |
| 1-3 | 64 | 3.58x | 3.3 | 3.2e-08 | 1.000000 | 3.1e-07 |
| 1-3+bf16 | 54 | 4.23x | 3.2 | 1.3e-03 | 0.999488 | 3.1e-03 |
| 1-3+compile | 65 | 3.53x | 3.3 | 1.2e-07 | 1.000000 | 2.2e-06 |
| 1-3+bf16+compile | 47 | 4.90x | 3.5 | 1.1e-03 | 0.999488 | 3.1e-03 |

## Analysis (2026-11-06)

Switches: 1 surface = plain SQ sampler; 1b batched = one geometric pass over the batch (matching still per frame);
2 assign = owner / visible / heatmap from `cache_assign.py` (FRAMES/*_assign.npz, 50 MB, 17 s for all 1600 frames);
3 pool = Sonata patch max-pool over unique (patch, token) pairs; 4 bf16 autocast of the network, torch.compile of the
model. Noise between repeated runs ~5 % (base 152 / 155 ms, 1-3 45 / 48 ms in two runs).

- 1-3 are exact: loss within 2e-7 relative, gradient cosine 1.000000, i.e. the same computation up to float rounding.
  Together 155 -> 48 ms (dense, 3.2x) and 230 -> 64 ms (top-k, 3.6x). Peak memory 2.0 -> 3.2 GB (batched surfaces).
- Alone: batched 1.55x / 1.85x (it also removes most surface calls, so surface on top of batched adds nothing in
  dense, 8 % in top-k), assign 1.50x / 1.20x, surface 1.28x / 1.42x, pool 1.05x / 1.01x (it removes 11 ms of forward
  in the per-part profile, but the step is dominated elsewhere until 1-2 are in).
- 4: bf16 +9 % (dense) / +18 % (top-k) on top of 1-3, not exact (loss 3e-4 / 1e-3 relative, gradient cosine 0.99999 /
  0.9995). torch.compile alone adds nothing (overhead is in the Python loss and the per-frame matching, which are not
  compiled); bf16 + compile 47 ms for top-k (4.9x) but nothing for dense, and costs ~1 min of compilation per run.
- What is left with 1-3 (per-part profile, `latency_breakdown.py --fast`): backward 22 / 29 ms, forward 10 / 12 ms,
  loss 9 / 16 ms, host-to-device copy 4 ms.
- Evaluation is now the larger relative cost: 0.14 s (dense) / 0.59 s (top-k) per 8 frames, i.e. ~20 s per evaluation of
  300 frames, ~15 % of a full run at eval_every 10 (pair-IoU SDF sampling; top-k has ~0.3 s per batch unaccounted
  for by the wrapped functions).


# Evaluation speed-up ablation 2026-11-06 15:56, NVIDIA GeForce RTX 5090, room_0 frames 0-87, trained checkpoints

## full_dense

| config | ms / batch of 8 | speed-up | max |Δ| of any per-frame metric | warm-up s (2 batches) |
|---|---|---|---|---|
| base | 143 | 1.00x | 0.00e+00 | 0.5 |
| 1 pair filter | 108 | 1.33x | 0.00e+00 | 0.2 |
| 2 shared occupancy | 142 | 1.01x | 0.00e+00 | 0.3 |
| 3 grouping on CPU | 143 | 1.00x | 0.00e+00 | 0.3 |
| 4 vectorized coverage | 143 | 1.00x | 0.00e+00 | 0.3 |
| 1-4 | 105 | 1.36x | 0.00e+00 | 0.2 |
| 7 compiled inside test | 71 | 2.02x | 0.00e+00 | 3.6 |
| 1-4 + 7 | 56 | 2.55x | 0.00e+00 | 1.2 |

## full_topk

| config | ms / batch of 8 | speed-up | max |Δ| of any per-frame metric | warm-up s (2 batches) |
|---|---|---|---|---|
| base | 313 | 1.00x | 0.00e+00 | 0.5 |
| 1 pair filter | 276 | 1.14x | 0.00e+00 | 0.4 |
| 2 shared occupancy | 306 | 1.02x | 0.00e+00 | 0.5 |
| 3 grouping on CPU | 270 | 1.16x | 0.00e+00 | 0.4 |
| 4 vectorized coverage | 232 | 1.35x | 0.00e+00 | 0.4 |
| 1-4 | 145 | 2.16x | 0.00e+00 | 0.3 |
| 7 compiled inside test | 230 | 1.36x | 2.07e-05 | 0.4 |
| 1-4 + 7 | 91 | 3.45x | 2.07e-05 | 0.2 |

## 6: every 2nd frame (all switches on)

| run | set | metric | all frames | every 2nd | Δ |
|---|---|---|---|---|---|
| full_dense | room_0 | best_iou | 0.127 | 0.127 | -0.000 |
| full_dense | room_0 | recall@0.5 | 0.014 | 0.014 | -0.001 |
| full_dense | room_0 | occ_iou_topk | 0.134 | 0.134 | +0.000 |
| full_dense | train 100 | best_iou | 0.350 | 0.352 | +0.002 |
| full_dense | train 100 | recall@0.5 | 0.334 | 0.342 | +0.008 |
| full_dense | train 100 | occ_iou_topk | 0.209 | 0.211 | +0.003 |
| full_topk | room_0 | pair_iou | 0.079 | 0.078 | -0.001 |
| full_topk | room_0 | found@0.3 | 0.669 | 0.668 | -0.001 |
| full_topk | room_0 | mask_cov | 0.205 | 0.206 | +0.002 |
| full_topk | room_0 | enc_recall@0.5 | 0.016 | 0.015 | -0.001 |
| full_topk | train 100 | pair_iou | 0.195 | 0.195 | -0.001 |
| full_topk | train 100 | found@0.3 | 0.678 | 0.691 | +0.014 |
| full_topk | train 100 | mask_cov | 0.266 | 0.262 | -0.004 |
| full_topk | train 100 | enc_recall@0.5 | 0.132 | 0.130 | -0.002 |

## Evaluation: analysis (2026-11-06)

Before (trained checkpoints, room_0, after a warm-up): 143 ms (full_dense) / 313 ms (full_topk) per 8 frames, i.e.
~5 s / ~12 s per evaluation of 300 frames, 16 evaluations per full run (eval_every 10, the last one twice).

- 1 pair filter: pairs whose (10 % + 1 cm widened) sample boxes do not overlap are 0 without evaluating them; the
  random points are still drawn for all pairs, so the kept pairs are bit-identical. Keeps 26-42 % of the pairs (not
  5-10 % as hoped: walls, floor and large wrong proposals have large axis-aligned boxes). 1.33x / 1.14x.
- 2 shared occupancy inside-tests: 1.01x / 1.02x (the SDF cost is per point and SQ, and the sets barely overlap).
- 3 grouping on the CPU: 1.16x for top-k (dense does not group). 4 vectorized coverage: 1.35x for top-k.
- 1-4 together, exact (max |Δ| of every per-frame metric 0): 105 ms (1.36x) / 145 ms (2.16x).
- 7 (not in the original list) `torch.compile` of the inside test (SuperFlex's sdf_batch < 0): 4-5x faster per call,
  0 mismatches in 19 M test points; 1-4 + 7: 56 ms (2.55x) / 91 ms (3.45x), max |Δ| 2e-5 (one occupancy point).
  Costs ~4 s of compilation once per process.
- 5 no double final evaluation: 16 -> 15 evaluations per full run.
- 6 every 2nd frame for the periodic evaluations: half the time; metric changes on room_0 ≤ 0.002, on the 100
  training frames ≤ 0.014 (found@0.3 of top-k; recall@0.5 of dense 0.008).

Per full run (150 epochs, eval_every 10; training ~20 min dense / ~28 min top-k):

| | dense | top-k |
|---|---|---|
| before | 16 x 5.4 s = 87 s | 16 x 11.9 s = 190 s |
| 1-5 (kept) | 15 x 4.0 s = 60 s | 15 x 5.5 s = 83 s |
| 1-5 + 7 | 15 x 2.1 s = 32 s | 15 x 3.5 s = 52 s |
| 1-5 + 7 + 6 | 14 x 1.05 + 2.1 s = 17 s | 14 x 1.75 + 3.5 s = 28 s |
