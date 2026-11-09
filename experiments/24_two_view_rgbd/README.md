# 24 — SQ decomposition from two RGB-D views (known poses)

**Question.** Does a second view with a known relative pose (a) improve the SQs of the first view, mainly the partly
visible ones, and (b) beat merging two single-view predictions (late fusion)? Plan: `PLAN.md`. Decisions
(2026-11-08): Replica only, code written from scratch (no imports from exp 22), render loss and cube sampling
included and ablated, poses known, proposals from both views.

## Code (`code/`)
- `common.py`: plain-SQ geometry (grid / cube surface samples, SuperFlex's radial SDF in log space, pair IoU), Hungarian
  cost, the differentiable SQ ray caster (exp 22's algorithm, reimplemented).
- `data.py`: frames and pairs from exp 22's cache. Targets are matched across views by their global index in the
  scene's target list (world translation + instance, max error < 1 mm). An item holds V = 1 or 2 views, all in view
  0's camera frame. Targets = the union of the views' visible SQs (`--targets frustum`: all SQs in either frustum).
  The per-view patch owners are remapped to the union.
- `model.py`: exp 22's top-k model with DINO only. Tokens carry the Fourier encoding of their 3D patch centroid
  **in view 0's frame**, plus a view embedding. One joint self-attention encoder runs over both views' 6300 tokens.
  Proposals come from the 3 × 3 maxima of both views' heatmaps (top 300 over both), followed by 1 decoder layer with
  refinement heads.
- `train.py`: exp 22's losses (proposals one-to-many with the heatmap, decoder Hungarian against the union, masks
  over both views, `--render` projection losses into every view), metrics, the late-fusion baseline (`--fusion`:
  a 1-view model on each view, view 1 moved into view 0's frame, NMS at pair IoU 0.5), and `compare.py`.
- `pairs.py`: overlap table (`results/pairs.json`) and pair statistics (`results/pair_stats.json`).

    ./worker.sh &       (runs the lines of queue.txt; logs in results/large_files/logs)
    uv run python experiments/24_two_view_rgbd/viewer/server.py      -> http://localhost:8024

Defaults: lr 3e-3 OneCycle with 5 % warm-up, batch 8, d = 128, 1 encoder and 1 decoder layer, grid sampling. The
render loss (weights 1 / 1 / 1, stride 4) applies to the first 2 frames of each batch only, for speed.
**f16 runs use 3000 steps** (exp 22: 6000) to stay near 15 min while 3-4 jobs share the GPU, so their absolute
numbers sit below exp 22's f16. Compare within this experiment only.

## Pairs (`pairs.py`)
Candidates: i ± {10, 20, 40, 80, 160} trajectory frames. Overlap = the share of view 0's depth pixels that view 1 sees
(projection + depth test, 3 cm + 3 %). Training draws a random partner with overlap in [0.2, 0.8]. Evaluation uses
the candidate with overlap closest to 0.5.

Evaluation pairs (mean over each scene's 200 frames):

| scene | Δ frames | overlap | union targets | visible in v0 | only in v1 | v0 partial (f < 0.5) | ... also seen by v1 | ... f1 > f0 | mean f0 -> max(f0, f1) |
|---|---|---|---|---|---|---|---|---|---|
| room_1 | 90 | 0.51 | 46.5 | 38.3 | 8.2 | 64 % | 70 % | 40 % | 0.27 -> 0.34 |
| room_2 | 95 | 0.51 | 59.8 | 48.6 | 11.2 | 58 % | 68 % | 38 % | 0.25 -> 0.31 |
| office_0 | 91 | 0.50 | 42.2 | 33.2 | 9.0 | 59 % | 63 % | 38 % | 0.24 -> 0.31 |
| office_1 | 109 | 0.52 | 29.9 | 24.1 | 5.8 | 62 % | 72 % | 41 % | 0.24 -> 0.33 |
| office_2 | 76 | 0.51 | 51.9 | 43.0 | 8.9 | 60 % | 71 % | 40 % | 0.26 -> 0.32 |
| office_3 | 105 | 0.52 | 82.5 | 66.2 | 16.2 | 56 % | 70 % | 39 % | 0.25 -> 0.32 |
| office_4 | 79 | 0.51 | 39.7 | 31.5 | 8.2 | 63 % | 64 % | 35 % | 0.26 -> 0.32 |
| room_0 | 116 | 0.52 | 84.2 | 67.0 | 17.1 | 51 % | 62 % | 37 % | 0.27 -> 0.35 |

- The second view adds ~20 % more targets (only seen by v1).
- For view 0's partly visible SQs (> half of them), view 1 sees about two thirds again, but sees them better in only
  ~40 % of cases. The best visible fraction rises only from ~0.25 to ~0.32: along a trajectory the two views see an
  object from similar sides. So the second view mainly brings extra objects, not the backsides of known ones. This
  bounds what (a) can gain.

## Results (2026-11-08, one seed each)

Run times: 23-52 min instead of the planned ≤ 15 min. Five to six jobs shared the GPU (exps 24-26 plus the chair
fits of exp 19). Alone, a run takes roughly a third of that.

**f16** (16 fixed pairs of the training scenes, train = eval, 3000 steps). Targets: the union of both views' visible
SQs (55 per pair, 45 of them visible in view 0). Columns `v0` / `v0 partial` = view 0's targets (all / f0 < 0.5),
`v1 only` = targets only view 1 sees.

| run | pair IoU | v0 | v0 partial | v1 only | hidden | occ. IoU 0.5 / 0.3 | dec. recall@0.5 | mask cov. | rendered inst. acc. | rendered depth AbsRel |
|---|---|---|---|---|---|---|---|---|---|---|
| 1 view (`f16_v1`) | 0.718 | 0.718 | 0.714 | – | – | 0.50 / 0.65 | 0.87 | 0.80 | 0.57 | 0.055 |
| 2 views (`f16_v2`) | 0.740 | 0.741 | 0.732 | 0.738 | – | 0.44 / 0.67 | 0.88 | 0.77 | 0.52 | 0.054 |
| late fusion of f16_v1 | 0.592 | 0.713 | 0.709 | 0.066 | – | 0.34 / 0.37 | 0.70 | – | 0.43 | 0.100 |
| 2 views, cube sampling | **0.755** | **0.758** | **0.755** | 0.761 | – | 0.50 / **0.75** | **0.92** | 0.80 | 0.58 | 0.052 |
| 2 views + render loss | 0.641 | 0.637 | 0.635 | 0.665 | – | **0.62** / 0.62 | 0.73 | **0.91** | **0.92** | **0.028** |
| 1 view + render loss | 0.651 | 0.651 | 0.654 | – | – | 0.63 / 0.64 | 0.76 | 0.88 | 0.90 | 0.031 |
| 2 views, frustum targets (+ hidden SQs, 76 / pair) | 0.662 | 0.734 | 0.729 | 0.740 | 0.469 | 0.43 / 0.61 | 0.76 | 0.77 | 0.58 | 0.052 |

- **Two views vs one:** +0.02 on view 0's SQs (0.741 vs 0.718, partial 0.732 vs 0.714), and the extra targets of
  view 1 are fitted as well as view 0's. On f16 everything is memorized, so this mainly shows that the joint encoder
  over 6300 tokens trains fine.
- Late fusion on f16 is no fair test. The 1-view model has memorized only the 16 view-0 frames, while the partners
  are unseen frames (v1-only targets: 0.07). The hold-out split below is the fair comparison.
- **Cube sampling** (Chamfer, matching cost and proposals on the radial cube grid instead of the (η, ω) grid) is the
  best f16 variant: +0.015 pair IoU, +0.08 occupancy IoU at 0.3, +0.04 decoder recall. Its matched Chamfer, measured
  on the (η, ω) grid, gets *worse* (0.080 vs 0.037, not in the table). That metric samples boxy SQs mostly at their
  edges, so it is a questionable measure here. Not yet looked at per frame.
- **Render loss** (instance + class + depth, weight 1 each, 2 frames per batch, stride 4): the same trade-off as in
  exp 22, but larger. The rendered views get much better (instance accuracy 0.52 -> 0.92, depth AbsRel 5.4 % ->
  2.8 %, mask coverage 0.77 -> 0.91), and existence becomes calibrated (occupancy IoU at 0.5: 0.44 -> 0.62). But
  the SQs move away from the target decomposition (pair IoU -0.10, decoder recall -0.15). Consistent with 1 view
  (0.718 -> 0.651) and with exp 25. It doubles the step time.
- **Frustum targets (visibility ablation):** adding the hidden SQs (all SQs in either frustum) as decoder targets
  leaves the visible ones nearly unchanged (view 0: 0.734 vs 0.741). The hidden ones reach 0.47 (memorized
  amodal completion). The total drops because the hidden ones count.

**Hold-out views** (`--setting frames`: train on the first 75 % of each training trajectory with random partners,
evaluate on the last 20 % (160 pairs) with fixed partners; 4000 steps).

| run | pair IoU | v0 | v0 seen | v0 partial | v1 only | occ. IoU 0.5 / 0.3 | dec. recall@0.5 (v0) | predicted / targets |
|---|---|---|---|---|---|---|---|---|
| 1 view (`frames_v1`) | 0.153 | 0.153 | 0.221 | 0.110 | – | 0.04 / 0.14 | 0.084 | 7 / 26 |
| ... on its training views | 0.314 | 0.314 | 0.345 | 0.293 | – | 0.07 / 0.30 | 0.293 | 17 / 43 |
| late fusion of frames_v1 | 0.170 | **0.182** | 0.251 | 0.139 | 0.108 | 0.05 / 0.16 | 0.119 | 13 / 33 |
| 2 views (`frames_v2`) | 0.171 | **0.182** | 0.249 | **0.145** | 0.112 | 0.01 / 0.13 | 0.118 | 5 / 33 |
| ... on its training pairs | 0.316 | 0.336 | 0.368 | 0.318 | 0.231 | 0.02 / 0.27 | 0.356 | 15 / 52 |
| 2 views, cube sampling (`frames_v2_cube`) | **0.186** | **0.199** | 0.249 | **0.159** | 0.115 | 0.01 / 0.12 | 0.127 | 5 / 33 |
| ... on its training pairs | 0.330 | 0.349 | 0.377 | 0.329 | 0.249 | 0.02 / 0.26 | 0.363 | 14 / 52 |

| 1 view, cube (`frames_v1_cube`) | 0.161 | 0.161 | 0.211 | 0.123 | – | 0.04 / 0.14 | 0.073 | 7 / 26 |
| late fusion of frames_v1_cube | 0.179 | 0.191 | 0.242 | 0.152 | 0.121 | 0.06 / 0.16 | 0.107 | 13 / 33 |

- With cube sampling everywhere, the joint model is slightly ahead of late fusion (view 0 0.199 vs 0.191, partial
  0.159 vs 0.152). That is within what one seed can resolve, and fusion keeps the better-calibrated existence
  (occupancy IoU 0.06 / 0.16 vs 0.01 / 0.12).
- **Cube sampling helps on held-out views too** (+0.015 pair IoU, view 0 0.182 -> 0.199, partial 0.145 -> 0.159).
  It is consistent with f16, the second setting in which it wins. I would make it the default.

- The generalization gap of exp 22 is unchanged (training views 0.31, held-out views 0.15-0.18). Exp 22 got 0.19
  here after 150 epochs; these are 4000 steps.
- **A second view helps view 0's SQs: 0.153 -> 0.182**, partly visible ones 0.110 -> 0.14.
- **But the joint two-view network does not beat late fusion:** the same 0.182 on view 0 (partial 0.145 vs 0.139,
  within noise for one seed), and its existence is less calibrated (occupancy IoU at 0.5: 0.006 vs 0.05; 5
  predictions above 0.5 for 33 targets). At this budget the gain comes from having two views' predictions, not
  from fusing them inside the network.
- Consistent with the pair statistics: the second view mostly sees the same sides of the same objects, so there is
  little complementary information for an early fusion to exploit.

Not done: seeds (needed before reading the joint-vs-fusion difference); render loss on
hold-out views; pairs with a wider baseline (the trajectories hardly offer any, see the pair statistics).
