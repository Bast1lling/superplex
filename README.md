Lessons learned:

1. Volume loss is much more powerful than surface loss, because it can deal with occlusions, wholes in the surface etc.
2. Superflex objective function is probably as good as it can get:
    a) -log IoU occupancy: 100k points with inside/outside labels
    b) sdf loss: 4k surface points with truncated leaky ReLU predicted signed distance (negative is inside)
    c) bbox loss: The extreme points along each axis must be inside the GT bbox
    d) overlap loss: no more than 5% overlap between individual SQs
3. Optimization does not seem to make a huge difference in terms of decomposition.
4. If the network had simply guessed the number of primitives needed, a k-means setup does almost as good out of the box: 0.80 vs. 0.83
5. even with fixed 16 count + pruning, not much changes.
6. Bending and tampering can be left out (+- 1% IoU max).


Initialization:

1. k-means init:
    a) Count: number of SQs the superflex network predicts for the instance (existence > 0.5)
    b) Cluster the instance's 4k surface points into that many clusters with k-means (positions only)
    c) One box per cluster: center + axes from PCA of its points, half extents from the point range along the axes, exponents 1
2. DBSCAN init:
    a) Features per point: position (normalized by instance size) + normal, both weighted equally
    b) DBSCAN (eps 0.15, 10 neighbors); clusters with < 40 of the 4k points become noise, keep the 16 largest
    c) The number of clusters is the count, one PCA box per cluster as in k-means
    d) Whole scene (no labels): same, but positions in meters and clusters with >= 20 of 50k points; walls / floors merge, so far fewer SQs than the network (81 vs. 239 in room_0)


Voxelization (GT grid, 4cm):

1. Grid over the scene's 1cm points + 8cm margin
2. For every voxel, look at its 8 nearest points: if most of their normals point away from the voxel, it is behind a surface (solid), else free
3. Solid voxels take the instance of the nearest point; solid voxels more than 10cm from any point are marked "deep" (the vote is unreliable that far away)
4. Voxels that contain a point are always solid (thin objects)
5. For the occupancy loss, "deep" voxels are filled with the instance of the nearest solid voxel; outside the grid is free


RGBD Fitting:

1. Generate a partial point cloud by backprojecting the pixel depth and labeling with closest GT point
2. Voxelize the 3D Scene bbox and then classify voxels into "front", surface", "back" for each views pose.
3. If you have multiple views, aggregate classifications from 2) over all views to obtain a single voxel grid
4. Adjust the superflex loss slightly:
    a) occupancy loss: sample query points in a bbox around an instance. "front" are negatives, "surface" positives, "back" are ignored in single-view, greedily claimed in multi-view (same as above)
    b) sdf loss: simply repeat all the backprojected depth values of an instance up to 4k
    c&d) stay the same, c) is dropped for single view


RGB Fitting:

1. Estimate D with Depth Pro
2. Guess the scale, optimize it in multi-view setting
3. Add erosion post-processing on top of the voxel grid to drop noisy depth predictions


Viewers (both pages from one server, run from the repo root):

    python -m http.server 8029 -d presentations/2026-10-29/viewer
    xdg-open http://localhost:8029/index.html && xdg-open http://localhost:8029/views.html

index.html = SQ decomposition (3D GT), views.html = SQs from rendered views

Reproduce (presentations/2026-10-29/code, then viewer/):

    uv run python network.py                 # SuperFlex predictions (counts for k-means)
    uv run python optimize.py                # per-instance fits on the 3D GT (methods 3-6)
    uv run python optimize.py --whole        # whole scene without labels (methods 7-8)
    uv run python optimize.py --views        # fits from rendered RGBD / RGB views
    uv run python optimize.py --views-whole  # the same without labels
    cd ../viewer && uv run python export.py && uv run python export_views.py
