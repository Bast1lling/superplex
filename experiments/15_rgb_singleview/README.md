# 15 — SQs from rendered views: RGB + instance labels, single view

Shared method, code and the overview of 12–15: [../12_rgbd_multiview/README.md](../12_rgbd_multiview/README.md).
As 14, but with Depth Pro's monocular depth (true focal length, label erosion 1) instead of the rendered depth.

**TL;DR.** vIoU 0.191 on average (28 % of the upper bound; 14 with GT depth: 0.433), and very uneven across frames
(0.04–0.43). The cause is the **metric scale of the single depth map**: it cannot be aligned without a second view,
and Depth Pro's per-image scale error of 0.6–10 % shifts surfaces 2 m away by 1–20 cm — more than walls, table tops
or pictures are thick. Frames with < 1 % scale error match RGBD (office_0 frame 1200: 0.43 vs. 0.43), frames with
7–10 % collapse (room_1 frame 0: 0.06, office_0 frame 400: 0.04). With the oracle per-view scale (control, uses GT
depth) the average rises to 0.333 (49 %); the rest is shape error of the predicted depth (AbsRel after scale 1–4 %).

| scene / frame | Depth Pro scale error | AbsRel after scale | vIoU RGBD (14) | vIoU RGB | vIoU RGB, oracle scale |
|---|---|---|---|---|---|
| room_0 0 / 400 / 800 / 1200 / 1600 | 2.6 / 4.5 / 2.2 / 2.8 / 4.4 % | 1.0–2.3 % | 0.47 / 0.38 / 0.55 / 0.42 / 0.46 | 0.32 / 0.20 / 0.22 / 0.14 / 0.14 | 0.37 / 0.40 / 0.34 / 0.30 / 0.39 |
| room_1 0 / 400 / 800 / 1200 / 1600 | 7.7 / 6.2 / 5.1 / 8.5 / 1.5 % | 1.3–4.0 % | 0.45 / 0.43 / 0.48 / 0.40 / 0.44 | 0.06 / 0.16 / 0.13 / 0.07 / 0.28 | 0.38 / 0.32 / 0.33 / 0.33 / 0.36 |
| office_0 0 / 400 / 800 / 1200 / 1600 | 6.6 / 10.5 / 0.8 / 0.6 / 1.1 % | 1.6–3.5 % | 0.41 / 0.41 / 0.42 / 0.43 / 0.34 | 0.16 / 0.04 / 0.27 / 0.43 / 0.23 | 0.23 / 0.37 / 0.28 / 0.39 / 0.23 |

Next steps would be a scale cue that a single view can provide: class-typical object sizes (door height, table
height) or the camera height above the floor, i.e. priors — or simply a second view (13 aligns scales from the
multi-view geometry).

Runs: `results/runs/<scene>_f<frame>` (final), `<scene>_f<frame>_gtscale` (oracle-scale control,
`--set depth=depthpro_gtscale`); tables in `results/logs/`.

    cd ../12_rgbd_multiview/code && uv run python fit.py --exp 15 --name <run> --set scene=room_0 --set frames=800 --set depth=depthpro --set erode=1
