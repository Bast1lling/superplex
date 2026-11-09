# 23 — Debug viewers for exp 22's RGB-D → SQ predictor

**Question.** Exp 22's dense and top-k models fit 1 and 16 frames but fail on the full set (room_0 proposal recall@0.5
0.014 against 0.34 on training frames). Before changing the architecture: are the inputs sane, and what do the
failures look like?

    uv run python experiments/23_debug_viewers/code/predict.py        # export, ~30 s (results/large_files)
    uv run python experiments/23_debug_viewers/viewer/server.py       # -> http://localhost:8023

**Feature inspector** (`viewer/features.html`), any cached frame of the 8 scenes: RGB, depth, GT instances, and a PCA
(3 components as RGB, per frame or one basis per feature set over all scenes) of DINOv3 (42 × 75 × 1024), the Sonata
stages 2 / 3 / 4 and their concatenation (per pixel), the patch-max-pooled Sonata (DenseModel's input) and the encoder
output tokens of an exp 22 checkpoint. Clicking any image switches to cosine-similarity maps of the clicked feature,
in the same or another frame.

**Prediction inspector** (`viewer/results.html`): the six checkpoints of exp 22 (of_dense_heat, of_topk, f16_dense,
f16_topk, full_dense, full_topk at epoch 60) on 37 frames: the overfit frame, the 16 frames and every 10th frame of
room_0, each marked seen / not seen in training. 3D: the RGB point cloud, GT SQs (visible, hidden), proposals (top
300 local maxima of the existence map) and decoder SQs above an existence threshold, coloured by IoU with their best
visible target, class correctness, GT instance or existence; clicking an SQ shows its parameters, its best target and
(decoder) its Hungarian match. 2D: RGB with the predicted existence, the GT heatmap target or the patch owner, plus
the shown SQs' patches. Per-frame metrics and a seen / unseen summary of all runs (`results/predict_summary.json`).

## Results

Mean over the frames each run was / was not trained on (`results/predict_summary.json`; full: 17 seen = the 16 frames
+ the overfit frame, 20 unseen = room_0):

| run | seen: proposal recall@0.5 / best IoU | unseen: proposal recall@0.5 / best IoU | seen: decoder pair IoU / found@0.3 | unseen: decoder pair IoU / found@0.3 |
|---|---|---|---|---|
| of_dense_heat | 0.60 / 0.50 | 0.001 / 0.05 | – | – |
| of_topk | 0.44 / 0.42 | 0.000 / 0.05 | 0.63 / 0.97 | 0.01 / 0.50 |
| f16_dense | 0.42 / 0.41 | 0.000 / 0.05 | – | – |
| f16_topk | 0.28 / 0.32 | 0.001 / 0.07 | 0.55 / 0.85 | 0.03 / 0.61 |
| full_dense | 0.31 / 0.33 | 0.015 / 0.13 | – | – |
| full_topk (epoch 60) | 0.11 / 0.22 | 0.016 / 0.12 | 0.18 / 0.72 | 0.08 / 0.63 |

- full_topk at epoch 60 fits its own training frames poorly too (pair IoU 0.18), unlike full_dense (0.31 / 0.33).
- Visual inspection: to do.
