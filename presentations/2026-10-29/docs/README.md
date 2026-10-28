# Presentation 2026-10-29: decomposing a scene into superquadrics

Four deliberately simple methods, plus a variant of `volume`. Each one changes **one thing** relative to the
previous one, so the viewer shows the effect of exactly that change.

| method | target | labels | number of SQs | what it shows |
|---|---|---|---|---|
| `surface` | points (surface) | – | fixed: 128 room_0, 64 room_1, 64 office_0 | the baseline from experiments 05–07 |
| `volume` | solid / free voxels | – | same fixed K | volume instead of surface (experiment 08) |
| `volume_overlap` | solid / free voxels | – | same fixed K | + 0.1 per extra SQ covering a solid voxel |
| `instances` | solid / free voxels per instance | GT instances | 1 per instance | labels separate what geometry cannot |
| `adaptive` | same as instances | GT instances | as many per instance as its shape needs | adaptive complexity (experiment 10) |
| `superflex_<ckpt>` | points per instance | GT instances | network decides (≤ 16 per instance) | a learned decomposer (SuperFlex, ECCV 2026) run per instance |

- **surface → volume:** same SQs, init (k-means + box), optimiser and K; only the loss differs.
- **volume → instances:** same loss shape, but evaluated per instance and with one SQ per instance.
- **instances → adaptive:** same loss; a part is split while its single SQ explains it with IoU < 0.7.

Details of every step and where the choices come from: [methods.md](methods.md).

## Run

    cd presentations/2026-10-29/code
    uv run python surface.py   [--res 0.02] [scene ...]     # default: 4 cm voxels, room_0 room_1 office_0
    uv run python volume.py    [--res 0.02] [scene ...]
    uv run python instances.py [--res 0.02] [scene ...]
    uv run python adaptive.py  [--res 0.02] [scene ...]
    uv run python superflex.py [--res 0.02] [scene ...]     # checkpoints in Thesis/data/models/superflex/
    cd ../viewer && uv run python export.py [--res 0.02]  # -> viewer/data/<scene>_<res>cm/ (not versioned)
    python -m http.server -d . 8029           # -> http://localhost:8029


## Layout
- `code/data.py` points (1 cm) and the voxel grid; `code/sq.py` SQ parameters, signed distance, box init, optimiser;
  one file per method.
- `results/<res>/<scene>/<method>.npz` fitted SQs (`trans`, `scale`, `eps`, `rot`, `instance`), `results/logs/`.
- `viewer/` static three.js page: scene, method (keys 1–8), layers (SQs, points, solid voxels), cut height.
