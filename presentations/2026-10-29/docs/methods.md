# Methods in detail

## Shared by all methods

**Data** (`data.py`). Replica 1 cm point clouds of experiment 01 (mesh sampled, with normals and GT instances).
Points without a semantic class and instances with < 80 points (wall plugs …) are dropped. Scenes: room_0
(78 instances), room_1, office_0.

**Voxel grid** (`data.voxelize`; 4 cm, or 2 cm with `--res 0.02`), used by all methods except surface:
1. grid over the points + 8 cm margin. Space behind walls / floor / ceiling counts as solid only up to the grid
   edge, so they become thin slabs without knowing which points are walls;
2. a voxel is **solid** if most of its 8 nearest points see it from behind (n · (v − p) < 0), else **free**; solid
   voxels take the instance of the nearest point (only used by instances / adaptive). Solid voxels more than 10 cm
   from their nearest point are **deep** and ignored by every loss: far from surfaces the vote is random (room_0:
   the plant's label spread up to 1 m from its points, 3500 voxels vs. 364 in experiment 08, which hid this by
   clipping each instance to its box);
3. every voxel containing a point is solid (thin objects: table tops, shelves, lamp shades).

Samples beyond the grid count as **empty** in every loss: otherwise SQs grow out of the scan for free (first
version: walls 20 m long, blobs around the room).

**Superquadric** (`sq.py`). Center t, half extents a = exp(·), shape exponents ε₁, ε₂ ∈ [0.1, 1.9] (sigmoid, init 1),
rotation = PCA axes of the initial box × learned quaternion. Initialised as the PCA box of a point / voxel set.

**Signed distance**: radial, d = |q| (1 − G(q)^(−1)) for q = R^T (x − t), where G is the inside–outside function
(G = 1 on the surface). Fastest of the five estimators of experiment 02 and exact in sign; it overestimates |d| away
from the surface, which does not matter for fitting.

**Optimiser**: Adam (lr 0.01), cosine decay to 5 %, 2000 steps.

## surface (`surface.py`)
- K SQs (room_0 128, room_1 64, office_0 64: the counts of experiments 08 / 10 rounded to multiples of 32),
  initialised by k-means of the points + one box per cluster.
- Loss per step = forward + backward:
  - forward: 16k random points, mean |min_j d_j(p)| (every point lies on the union; experiment 06's best mode)
  - backward: 128 random points on each SQ surface (parametric angles), mean distance to the nearest scene point
    (every SQ surface lies on points). Plain: internal faces of overlapping SQs count too.
- Expected failure (experiments 05–07): nothing penalises filling empty space or covering several objects, and
  large flat surfaces are cheapest to explain by patchwork.

## volume (`volume.py`)
- Same K, init and optimiser as surface.
- Loss: logistic occupancy of the union d = min_j d_j, with τ = 2 cm: mean softplus(d/τ) on 16k solid voxel centers
  + mean softplus(−d/τ) on the empty ones of 16k uniform points in the grid box + 50 cm (= cross-entropy with
  p_inside = σ(−d/τ)).
- Variant `volume_overlap` (same file): + 0.1 · (Σ_j p_j − max_j p_j) on the solid samples, p_j = σ(−d_j/τ):
  every SQ beyond the first that covers a solid voxel costs 0.1.
- Origin: experiment 08, minus its unknown / exterior labels, slab cut by class, box clipping, planar hole filling.

## instances (`instances.py`)
- One SQ per GT instance, initialised as the PCA box of its voxels.
- Samples per instance: 256 of its own voxels (positives) and 256 uniform points in its axis-aligned box + 20 cm,
  of which the empty ones (free voxels, beyond the grid) are negatives. Voxels of other instances have **weight 0**:
  an SQ may overlap them for free (floor under a sofa, wall behind a picture = amodal completion without hole
  filling).
- Loss per instance as in volume (on the union of the instance's SQs), averaged over instances, so a small object
  counts as much as a wall.

## adaptive (`adaptive.py`)
- Search: every part (initially one per instance) gets one SQ, fitted for 300 steps with the instances loss on the
  part (other parts of the same instance: weight 0). Test: **tolerant IoU ≥ 0.7**, where a part voxel within half a
  voxel outside the SQ still counts as covered and a free voxel only counts as covered if it is more than half a
  voxel inside. Parts that fail are split in two by 2-means of their voxel coordinates and fitted again; a part is
  only split if both halves keep ≥ 512 cm³ (8 voxels at 4 cm, 64 at 2 cm; in voxels, 2 cm exploded to 1–4k SQs).
- Refinement: all SQs from the search are optimised together with the instances loss (2000 steps), each instance
  being the union of its SQs.
- Origin: experiment 10. Its lessons: the per-part quality test makes the count follow the *complexity* of a shape
  (an absolute price per SQ follows its *size*: large rugs get many SQs, plants few); without the half-voxel
  tolerance thin parts never pass and everything splits to the minimum size. Left out: merging of neighbouring
  parts after the split (in experiment 10 it removed about a third of the SQs), weights for other parts.

## superflex (`superflex.py`)
- SuperFlex (Tavernini, Fedele et al., ECCV 2026; code in `papers/superflex`): a feed-forward network that predicts
  up to 16 superquadrics with **tapering and bending** (+ existence) for one object point cloud. Run on every GT
  instance separately, results merged (as in the authors' scene pipeline, without its DBSCAN outlier filter).
- Per instance: 4096 of its 1 cm points (with repetition if fewer), centered and scaled to unit size, rotated from
  z-up to the network's y-up; all instances in one batch (seconds). Primitives with existence > 0.5 are kept.
- Three checkpoints (one training chain, same architecture):
  - `superflex`: ShapeNet, complete objects, unsupervised (primitives explain the points);
  - `unsup_robust`: + occlusion augmentation and fine-tuning on ASE (partial Aria scans), same unsupervised loss
    (fits what is visible); the authors' choice for scenes;
  - `robust`: same data, but supervised by `superflex`'s decomposition of the *complete* object (amodal).
- The viewer meshes the deformed primitives with SuperFlex's own mesher (`viewer/export.py`).
- Setup notes: its PVCNN CUDA ops are compiled on first use (needs nvcc); they ask for C++17, torch 2.14 needs
  C++20, so `superflex.py` swaps the flag without touching the submodule. Sanity check (room_0, unsup_robust): all
  primitive centers lie in their instance's box, mesh vertices are 2.3 cm (median) from the instance's points.
