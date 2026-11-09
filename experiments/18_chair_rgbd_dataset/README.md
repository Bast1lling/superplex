# 18 — Chair dataset: textured RGB-D renderings + SuperFlex geometry

**Goal.** A dataset of ShapeNet chairs where every chair has RGB-D renderings under several textures and everything
SuperFlex's optimization needs (surface and occupancy samples), all in one frame.

    uv run python experiments/18_chair_rgbd_dataset/code/build_dataset.py --out /home/sebastian/data/ShapeNet-Chairs-RGBD

Output (a copy of this description is `README.md` in the dataset folder): `<out>/03001627/<model_id>/`

| File | Content |
|---|---|
| `pointcloud.npz` | 100k visible surface points, outward normals (float16; keys as in ONet) |
| `pointcloud_4096.npz` | farthest-point subset of them, float32 |
| `points.npz` | 100k uniform points in [-0.55, 0.55]³, packed `occupancies` |
| `cameras.npz` | `K`, `c2w` (8, 4, 4; OpenCV axes), `light` (4, 3), `depth_unit`, `mesh_loc`, `mesh_scale` |
| `depth/vVV.png` | 16-bit z-depth, multiply by `depth_unit` (1e-4); 0 = background |
| `rgb/tTT_vVV.png` | 512 × 512 RGB on white; `t00` original materials, `t01`–`t03` random textures |

Plus `03001627/{train,val,test}.lst`, so it is a drop-in SuperFlex `shapenet.path` (`normalize: False`, as in their
config). The split is ONet's, so objects held out from the SuperFlex checkpoints stay held out.

**Frame.** The ONet one, so the checkpoints see what they were trained on: `(v - mesh_loc) / mesh_scale`, then 270°
around y (`TO_ONET`), i.e. bbox centered, longest side 1, y up. Found by matching against ONet's point clouds: NN
distance ~0.015 for this rotation, ~0.07 for the other three. `mesh_scale` equals ONet's `scale`.

**Renderings.** 8 views per chair, the same 8 for every texture: stratified azimuth, elevation -10..50°, fov 36.9°,
distance so that the bounding sphere just fits (×1–1.2). Own z-buffer rasterizer in torch (`rasterize`): no renderer
installed, nvdiffrast not on PyPI, and per-pixel depth / face ids are all this needs. Lambert + ambient, one random
light per texture, no shadows, no anti-aliasing (keeps RGB and depth exactly aligned).

Random textures are procedural functions of the 3D position (solid, checker, stripes, noise, wood), drawn per
material, so they are consistent across views and need no UVs. Material groups often follow chair parts, but some
meshes have just one material and get one texture throughout.

**Occupancy: depth fusion, not binvox.** Tested first: ShapeNet's `solid.binvox`, cheap and of similar resolution to
SuperFlex's own ABO preparation. Checked against ONet's watertight surfaces (independent of Core), offsetting their
points along their normals by 0.01 (60 chairs):

| | occupied just inside (want 1) | occupied just outside (want 0) |
|---|---|---|
| solid binvox | 0.905 | 0.088 (one chair: 0.32) |
| depth fusion | 0.906 | 0.006 |

So binvox marks free space as solid, probably from its axis-aligned carving. Instead, as in ONet's mesh fusion: 100
depth maps all around (512², nearest depth in 3 × 3 so sub-pixel legs survive); a point is free if some view sees it
more than 0.005 in front of the surface. Sealed interiors stay solid, concavities get carved, and zero-thickness
sheets become a thin shell. The same maps give the surface samples: points no view sees (mesh interiors) are dropped,
normals are flipped towards a view that sees them. Interior faces would otherwise be surface samples inside the solid.

**Result.** All 6778 chairs of ONet's split (train 4746, val 677, test 1355), 271120 PNGs, 25 GB, 111 min with 12
workers. `config.json` holds the arguments, `quality.json` the per-chair results of `check_dataset.py`, and
`errors_first_pass.log` the only failures: the 8 ids starting with `u` broke the seed parsing and were built after the
fix.

**Checks**
- `check_dataset.py <out> 12` on every chair: sample counts, finite values, surface inside the cube, cameras, RGB
  against depth mask. One flag: `df51cb83f0e55b81d85934de5898f0b` touches the image border in one view of 8.
- 200 random chairs against ONet's surfaces: mean NN distance 0.0108 / 0.0106 in both directions. Depth back-projected
  to 3D lies 0.0048 from the surface samples, about the shell thickness.
- Occupied fraction of the cube: 0.3% / 2.7% / 38% (1st / 50th / 99th percentile), range 0.06% to 51%. The extremes
  look right: wire chairs and solid cube armchairs.
- Through SuperFlex's own `ShapeNet` dataset (`use_fps`), `superflex` checkpoint, `SuperQOptimization` with tapering
  and bending, 500 steps, 4 chairs of the pilot: IoU 0.65 / 0.65 / 0.78 / 0.27 → 0.80 / 0.85 / 0.94 / 0.60. Needs
  `pandas` and `torch_geometric`, which that dataloader imports and the project venv lacks (stubbed for this test).

**Limits**
- **Fixed:** query points within 0.005 of the surface samples are now occupied, the old labels are `occupancies_fusion` (experiment 20). Before that, sheets were largely missing. Found there: on the director's chair
  `fbc560d04067a6b58ce66f12ba927a2b` the back is striped with holes and most of the seat is gone, though both render
  solid. 14 % of its surface samples are themselves labelled free, and at ±0.004 from the surface the occupied fraction
  is 0.36 (0.5 is what a closed solid gives). The shell is thinner than the 0.005 margin suggests, on all chairs: the
  margin is measured along the ray, likely shrinking it in grazing views. Zero-thickness parts lose most.
- Occupancy is not ONet's: same idea, different implementation. ONet's `points.npz` is not on this machine, so there
  is no direct IoU against it.
- Fusion can't reach cavities no view sees into, those count as solid. The inside score above averages 0.906, not 1,
  because the 0.01 offset goes through very thin parts: worst chairs (0.18–0.30) are wire and sheet chairs.
- Thin chairs have few occupied query points: 22 chairs below 0.2% (under 200 of 100k).
- Where ONet keeps interior surfaces, the two disagree by design. On the worst chair of a sample of 200
  (`a598b87a9e37a438617c72672c86a49`, overlapping cushions) 53% of ONet's "just outside" points are in our solid; 95%
  of those lie over 0.02 from our visible surface, so inside the chair.
- Rendering is deliberately simple: not photorealistic, high-frequency textures alias, visibly on some flat faces.
- Only 4299 of the 6778 chairs have image textures, the others keep plain material colors in `t00`.
