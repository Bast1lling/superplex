# 14 — SQs from rendered views: RGBD + instance labels, single view

Shared method, code and the overview of 12–15: [../12_rgbd_multiview/README.md](../12_rgbd_multiview/README.md).
Same as 12 with one view (frames 0, 400, 800, 1200, 1600 per scene); only instances seen in the view get an SQ and
are evaluated (against their full, amodal 3D GT).

**TL;DR.** vIoU 0.433 on average (63 % of the 3D-GT upper bound on the same instances; room_0 0.45, room_1 0.44,
office_0 0.40). Precision stays high (0.68), recall drops (0.62): the backs of objects, occluded parts and parts
outside the image are simply not observed, the carved grid only has the 10 cm solid band behind visible surfaces.
The unknown-space weight barely matters (room_0, 5 frames: w_unk 0 / 0.1 / 0.3 / 1 → 0.457 / 0.454 / 0.441 / 0.402);
default 0.1. Getting further would need shape priors (symmetry, class-typical extents) — i.e. learning.

| scene | frames 0 / 400 / 800 / 1200 / 1600 (vIoU) | instances seen | upper bound (same instances) |
|---|---|---|---|
| room_0 | 0.47 / 0.38 / 0.55 / 0.42 / 0.46 | 22 / 21 / 45 / 39 / 46 | 0.68–0.74 |
| room_1 | 0.45 / 0.43 / 0.48 / 0.40 / 0.44 | 11 / 23 / 25 / 18 / 24 | 0.65–0.81 |
| office_0 | 0.41 / 0.41 / 0.42 / 0.43 / 0.34 | 19 / 12 / 23 / 22 / 5 | 0.65–0.69 |

Runs: `results/runs/<scene>_f<frame>` (final), `room0_f<frame>_wunk<w>` (sweep, table in `results/logs/`).

    cd ../12_rgbd_multiview/code && uv run python fit.py --exp 14 --name <run> --set scene=room_0 --set frames=800
