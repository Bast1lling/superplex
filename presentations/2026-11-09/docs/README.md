# Presentation 2026-11-09: superquadrics from a single RGB-D frame with Slot Attention

One pipeline (experiments 27 and 28, re-implemented compactly): frozen DINOv3 patch features + 3D patch positions →
Slot Attention groups the patches into instances → a relative part decoder predicts up to 16 superquadrics per
instance. The overview page (`viewer/index.html`) explains each stage.

## Run

    cd presentations/2026-11-09/code
    uv run python train.py --name overfit_parts --variant parts            # the 4 overfit runs: --variant render | render_obs | single | parts
    cd ../viewer && uv run python export.py [--copy_exp28 all]                 # -> viewer/data/ (not versioned)
    python -m http.server -d . 8030                                        # -> http://localhost:8030

`--copy_exp28` copies exp 28's checkpoints and eval histories of the generalization runs (they load unchanged into
`code/model.py`; outputs identical to exp 28's code, checked on room_2/001700).

## Viewers

1. `index.html`: the pipeline as a figure, and the losses.
2. `app.html#overfit`: one frame (room_2/000500), 10k steps, four variants that differ only in input, decoder and
   geometry loss:

   | variant | input | decoder | geometry loss / target |
   |---|---|---|---|
   | a) `render` | RGB (2D patch positions), no depth | one SQ per slot, no anchor | rendered SQs vs. the instance masks |
   | b) `render_obs` | RGB-D | one SQ per slot | rendered instance masks + depth, observed-point Chamfer |
   | c) `single` | RGB-D | one SQ per slot | Chamfer to one SQ fit per instance |
   | d) `parts` | RGB-D | relative part decoder | part Chamfer + part existence vs. the visible SuperFlex SQs |

3. `app.html#general`: exp 28's runs on room_2 (150 training frames, 30k steps), evaluated on 12 training frames and
   all 40 held-out views (the last 20 % of the trajectory): (a) camera frame, (b) gravity frame + yaw augmentation;
   (c) `S7_rel_grav_30k`: 7 scenes (1050 frames), room_0 held out, on 4 frames: one held-out view each of office_0,
   office_3 and room_1 and one room_0 frame, each the upper-median frustum IoU of 8 candidates of its scene, plus the
   middle training frame of office_0, office_3, room_1 and room_2.

## Layout

- `code/data.py` frames and targets; `sq.py` superquadric surface samples, signed distance, rendering; `model.py`
  Slot Attention + decoders; `losses.py` matching and losses; `metrics.py` mask IoU, frustum IoU; `train.py`.
- `results/runs/<run>.json` arguments and eval history; `results/large/<run>/model.pt` (not versioned);
  `results/logs/`.
