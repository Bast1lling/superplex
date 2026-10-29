# 11 — Fitting superquadrics from renderings

**Question.** How well can SQs be fitted from RGB-D renderings alone (no reconstructed point cloud / mesh)?

## Step 1: view browser (`viewer/`)
Experiment 01's scene viewer plus the camera poses of the NICE-SLAM Replica renderings
(`Thesis/data/replica_nice_slam/Replica/<room0|room1|room2|office0..4>`: 2000 frames each, 1200 × 680 RGB + depth,
fx = fy = 600, depth in 1/6553.5 m, `traj.txt` = camera-to-world 4×4 per frame, OpenCV axes).

- The poses live in the same world frame as the Replica meshes / our point clouds: projecting the 1 cm points into a
  frame matches the rendered depth with a median error of 2–3 mm (room_0, frames 0 / 500 / 1500).
- A blob marks every n-th camera center (slider), with a short line in viewing direction.
- Click a blob: its rendering, and only the points visible in it (projected into the image and within 2 cm of the
  rendered depth). Esc: back to the full scene. Double-click elsewhere: orbit around the clicked point (01).

- Labelled renderings (checkbox / key L): GT instance per pixel, overlaid on the rendering. Every pixel is
  back-projected with the rendered depth and takes the instance of the nearest 1 cm point within 2 cm (else
  unlabelled: glass, unscanned). Colored like the points (instance, or class in "Semantic class" mode).

      uv run python experiments/11_render_views/viewer/server.py     # -> http://localhost:8011
