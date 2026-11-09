# 26 — Exp 22's single-view RGB-D predictor on chairs

**Question.** Exp 22's predictor fails to generalize across Replica scenes, and the number of scenes was the only
lever found (7 scenes). Does the same architecture generalize when it is trained on thousands of objects? The chair
dataset of exp 18 has 6778 chairs × 8 views × 4 textures, plus GT occupancy.

**Setup** (`code/`; a self-contained copy of exp 24's rewrite of exp 22's top-k model, DINO only).
- Input: one RGB-D view. DINOv3 ViT-L/16 runs frozen on the GPU, on the fly at 512² (32 × 32 tokens). That lets every
  training step pick a random view and one of the 4 textures (t00 = original materials). Depth comes at /4
  (128², pixels 4i + 1), with 4 × 4 pixels per patch.
- Targets: exp 19's SuperFlex fit of each chair (`sq_fits/<split>/superflex_opt_plain.npz`: SuperFlex's network on
  the full point cloud, exist > 0.5, then 1000 steps of its objective, plain SQs, ~7 SQs per chair). They are moved
  into the camera frame. The decoder is trained against **all** SQs of the chair (amodal), the proposals only on
  the visible ones.
- Pixel owner = the SQ with the smallest |SDF| at the back-projected pixel. Patch owner and CenterNet heatmap as in
  exp 22. The mask head predicts each SQ's visible part (instance = SQ).
- Metrics: exp 24's (pair IoU of Hungarian-matched SQs, occupancy IoU of the union against the target union,
  recalls), plus **vIoU** = IoU of the predicted union (exist > 0.5 / 0.3) against the chair's GT occupancy on 8192
  query points. **viou_target** is the same for the target fit (upper bound).

    uv run python experiments/26_chair_rgbd_prediction/code/train.py --name val677 --train_split val --steps 4000
    uv run python experiments/26_chair_rgbd_prediction/viewer/server.py         -> http://localhost:8026

Train fits for the `train` split were started on 2026-11-08 (`19_sq_count_sweep/code/fit_superflex.py --plain`, log
`sq_fits/superflex_plain_trainval.log`). The first run therefore trains on the 677 val chairs and evaluates on 200
test chairs.

## Results (2026-11-08, one seed each, eval = 200 test chairs × 1 view, t00)

| run | train chairs | steps | vIoU 0.5 / 0.3 | pair IoU | visible / hidden SQs | occ. IoU vs target union 0.5 / 0.3 | dec. recall@0.5 | proposal recall@0.5 | mask cov. | predicted (0.5 / 0.3) / targets |
|---|---|---|---|---|---|---|---|---|---|---|
| target fit (upper bound) | | | 0.749 | | | | | | | |
| `val677` | 677 (val split) | 4000 | 0.05 / 0.31 | 0.246 | 0.251 / 0.082 | 0.05 / 0.32 | 0.200 | 0.196 | 0.06 | 0.5 / 7.9 / 7.8 |
| ... on its training chairs | | | 0.06 / 0.34 | 0.287 | 0.294 / 0.099 | 0.06 / 0.36 | 0.233 | 0.241 | 0.07 | |
| `train4745` | 4745 | 7000 | **0.12 / 0.36** | **0.278** | 0.283 / **0.133** | 0.12 / 0.37 | 0.253 | 0.237 | 0.14 | 1.1 / 7.4 / 7.8 |
| ... on its training chairs | | | 0.09 / 0.32 | 0.284 | 0.292 / 0.093 | 0.10 / 0.35 | 0.219 | 0.228 | 0.13 | |

Evaluations of `train4745` during training (test chairs): pair IoU 0.157 / 0.199 / 0.270 / 0.278 at steps 2000 /
4000 / 6000 / 7000; vIoU at 0.3: 0.22 / 0.26 / 0.35 / 0.36.

- **No generalization gap:** pair IoU 0.284 on training chairs vs 0.278 on unseen test chairs (val677: 0.29 vs
  0.25). On Replica the same architecture gets 0.31 on training views vs 0.15 on held-out views and 0.10 on a new
  scene (exps 22, 24). With thousands of objects it learns something that transfers. This supports exp 22's
  conclusion that the number of distinct scenes, not the architecture, is the bottleneck there.
- **But it is far from converged:** still +0.08 pair IoU between steps 4000 and 6000. Existence is the weakest part
  (about 1 SQ above 0.5 per chair, 7.4 above 0.3 for 7.8 targets), so vIoU at 0.5 is meaningless for now. Use 0.3.
- Hidden SQs (the side of the chair facing away) are the hardest: 0.13 vs 0.28 for visible ones.
- Reference points (not the same input): SuperFlex's network on the **full** point cloud reaches 0.65 mean IoU, 0.79 after
  its optimization (exp 21's table: its 271-chair mini-test, plain SQs, all 100k query points). Here the input is one view.

**Longer training** (`train4745_30k`: 30k steps, 32 min with the GPU to itself, 0.06 s / step):

| run | vIoU 0.5 / 0.3 | pair IoU | visible / hidden SQs | occ. IoU vs targets 0.3 | dec. recall@0.5 | mask cov. | predicted (0.5 / 0.3) / targets |
|---|---|---|---|---|---|---|---|
| `train4745` (7k steps) | 0.12 / 0.36 | 0.278 | 0.283 / 0.133 | 0.37 | 0.253 | 0.14 | 1.1 / 7.4 / 7.8 |
| `train4745_30k` | **0.24 / 0.42** | **0.330** | 0.334 / **0.187** | 0.44 | 0.332 | 0.27 | 2.2 / 6.5 / 7.8 |
| ... on its training chairs | 0.20 / 0.39 | 0.328 | 0.336 / 0.166 | 0.43 | 0.312 | 0.24 | |

Test-chair evaluations at 5k / 10k / 15k / 20k / 25k / 30k steps: pair IoU 0.19 / 0.22 / 0.25 / 0.25 / 0.32 / 0.33,
vIoU at 0.3: 0.27 / 0.28 / 0.33 / 0.36 / 0.41 / 0.42.

- Still no gap between training and test chairs (0.328 vs 0.330). The model is capacity- or optimization-limited,
  not data-limited.
- Most of the gain comes late, while OneCycle anneals the lr (0.25 -> 0.33 between 20k and 30k). The curve has not
  saturated. Longer runs or a larger model (exp 22's width-256 / 2-layer variant) are the obvious next steps;
  existence (2 SQs above 0.5 per chair) is still the weakest part.

Not done: cube sampling, render loss, the two-view and RGB-only variants (exps 24, 25) on chairs.
