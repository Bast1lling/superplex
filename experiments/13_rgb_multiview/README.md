# 13 — SQs from rendered views: RGB + instance labels, multi-view

Shared method, code and the overview of 12–15: [../12_rgbd_multiview/README.md](../12_rgbd_multiview/README.md).
Same as 12 (100 views, carved label grid, one SQ per instance), but without the rendered depth.

**TL;DR.** Two directions:
1. **Depth from a monocular network + multi-view scale alignment** (best): Depth Pro with the known focal length,
   per-view scales fitted from multi-view consistency, label erosion at mask boundaries. vIoU 0.49 / 0.41 / 0.30
   (room_0 / room_1 / office_0; 57 % of the 3D-GT upper bound on average, 12 with GT depth: 87 %).
2. **No depth at all: differentiable rendering of the instance masks** (`render_fit.py`): 0.16, with a volume prior
   0.27 (room_0). Masks constrain silhouettes only; the never-seen sides (back of walls, underside of the floor,
   thickness of flat objects) inflate. Refining the depth-network fit with mask rendering lowers it (0.49 → 0.44):
   one SQ per instance cannot match the masks exactly, so the rendering loss shrinks the SQs (recall 0.78 → 0.61).

## Depth networks (room_0, 100 views)

AbsRel against the rendered depth (frames 0 / 800 / 1600): Depth Pro with true focal 0.024–0.046, Depth Pro with
its own focal 0.031–0.068, Depth Anything V2 Metric Indoor 0.31–0.43 (0.05–0.06 after a per-image scale: it is
trained on Hypersim's ~60° FoV, ours is 90°, so its depth is ~25 % too far).

| variant | vIoU | prec | rec | rel |
|---|---|---|---|---|
| Depth Anything V2 (no alignment) | 0.001 | 0.001 | 0.003 | 0.00 |
| Depth Anything V2 + scale alignment (global scale recovered: 0.768 vs. true 0.733) | 0.171 | 0.235 | 0.365 | 0.24 |
| Depth Pro, own focal | 0.338 | 0.393 | 0.817 | 0.48 |
| Depth Pro, true focal | 0.400 | 0.456 | 0.863 | 0.57 |
| + scale alignment (per-view scale error 2.8 % → 0.7 % median) | 0.476 | 0.536 | 0.883 | 0.67 |
| + erode 1 (labels next to a mask boundary ignored; 13 small instances lost) | 0.532 | 0.664 | 0.766 | 0.72 |
| + erode 1, un-eroded fallback for instances that lose all voxels (**final**) | 0.490 | 0.609 | 0.777 | 0.69 |
| erode 2 + fallback / min. 3 solid votes / depth-proportional band 3 % | 0.454 / 0.470 / 0.405 | | | |

Erosion helps because monocular depth is smooth across object boundaries: boundary pixels get depths between fore-
and background and put their label into mid-air (carved instance voxels within 4 cm of their GT surface: 62 % with
GT depth, 37 % with Depth Pro).

## Mask rendering (room_0, 100 views)
One SQ per instance, initialised at the least-squares intersection of the rays through the instance's mask
centroids (size from mask area); every pixel ray is composited (occupancy sigma(-d/tau), tau annealed 0.2 → 0.01 m,
128 samples), cross-entropy with the pixel's instance.

| variant | vIoU | prec | rec |
|---|---|---|---|
| 500 steps / 3000 steps | 0.242 / 0.159 | 0.307 / 0.191 | 0.552 / 0.637 |
| + volume prior 0.3 / 1 / 3 (1500 steps) | 0.258 / 0.269 / 0.262 | 0.365 / 0.485 / 0.551 | 0.474 / 0.352 / 0.302 |
| init from the Depth Pro fit, w_vol 0 / 1 (1000 steps) | 0.437 / 0.407 | 0.609 / 0.697 | 0.613 / 0.479 |

Runs: `results/runs/<scene>_depthpro` (final), `room0_*` sweeps; tables in `results/logs/`.

    cd ../12_rgbd_multiview/code
    uv run python fit.py --exp 13 --name <run> --set scene=room_0 --set depth=depthpro --set align=1 --set erode=1
    uv run python render_fit.py --exp 13 --name <run> [--set w_vol=1 --set init=13:room0_depthpro ...]
