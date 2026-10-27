Current research on superquadrics splits (roughly) into two directions:

1. Object-level decomposition, segmentation, primitive reconstruction etc. 
I.e. use a set of superquadrics to approximate an object’s parts
SuperDec still applies this to scenes, but it relies on some external method for decomposing the scene into objects.
2. Multi-view reconstruction, segmentation of scenes
I.e. given a set of images fit superquadrics for NVS (3DSQS) or occupancy prediction (QuadricFormer)

Neither takes scene level point clouds as input, probably because it is difficult to efficiently work on that scale. But in practice, point clouds are usually the backbone of Indoor BIM automation. Of course you could also extract multiple views from a Lidar’s panorama images, but they will not always be available and would require reconstructing the depth. So it seems most promising to work on the point clouds directly - even if they contain millions of points.

Moreover, this scale is also the biggest selling point of using superquadrics: We can “downsample” a huge chunk of unordered points into (potentially a hierarchy of) a few superquadrics, similar to what patches do for images. Importantly, here the focus is no longer on exact reconstruction/fitting of surfaces or points - it simply needs to be a coarse decomposition. So superquadrics at scene-level might even be a better fit than at object-level!
The advantages of such a compact representation are clear, we can use it instead of the point cloud on any downstream tasks.
The challenge is also clear: try to make this efficiently trainable, using a differentiable, feed-forward pipeline instead of optimization such that we can incorporate it into any downstream tasks end-to-end.

In the following I collected some thoughts on how to make this efficient, however this would of course be the main research of the thesis. If we find anything feasible, we could then also extend research toward adopting suitable downstream tasks → unlimited possibilities 😄
E.g. I would like to try:

1. Directly predict Scene Graphs from the active superquadrics.
2. Encode superquadrics into an LLM’s latent space, similar to what SceneScript does.
3. Simple 3D Semantic Segmentation might be a low-hanging fruit
4. Directly predict primitive shapes such as walls, doors, windows (but here superquadrics might not be ideal as representation).

- Training
    
    Our task is essentially to train an explicit variant of an **auto-encoder** for self-supervised point cloud compression: 
    
    1. The encoder takes a huge point cloud (each have a coordinate + a feature vector) and compresses it into a bottleneck.
    2. The bottleneck is not a latent space, but rather an explicit set of several thousand superquadrics. Each superquadric should also have a corresponding feature field. E.g. using a feature center and three feature directions aligned with the superquadric’s local axes let you sample features for any point inside the primitive.
    3. The decoder is explicit instead of learned. E.g. for reconstructing the point cloud, we could interpret each superquadric as an occupancy probability distribution and sample points from it (e.g. based on surface distance). Similarly, we could also sample corresponding features from our explicit fields
    
    Some thoughts on **pre-processing / initialization:**
    
    - Assuming we work on Lidar Scans. Thus, all point clouds are surface points. Thus, a region in space with a high point occupancy directly correlates with complex surfaces. Thus, we could initially distribute our superquadrics according to occupancy:
    Octrees are an efficient data structure for millions of points. We create them at the start, then place a superquad inside each cell center.
    - As long as we create it only once at the start, a dense voxelgrid does not limit us in terms of speed, only in terms of memory but should stay below e.g. 2GB.
    - Octrees also allow us efficient point lookups for superquadrics.
    - A graph structure on superquads could also come in handy, depending on the amount of primitives.
    
    **Objectives**:
    
    - We can’t directly copy the SuperFlex objectives, since they need a segmentation matrix of shape N, M over all N points and M superquadrics. But there might be a simpler solution for our purposes:
    - Precompute a (truncated) distance field over each point cloud once: a dense voxel grid storing the distance to the nearest point per cell (optionally k-nearest-averaged for robustness to outliers). Evaluate the loss in both directions analytically. **SQ → data**: sample points on each superquadric surface via its parametric form and look up the precomputed field by trilinear interpolation
    **Data → SQ**: for each occupied voxel, evaluate the superquadric's radial distance in closed form and penalize its deviation from the surface, weighted by the voxel's point count. 
    Simple L1/L2 on both terms.
    - Analogous formulation for point features: Store e.g. mean feature for each voxel cell.
    - If there are any semantic labels available, we can also add another objective that enforces that each superquadric only occupies points of the same class.
    - If there are BIM models, we could also explore a more direct SDF-based loss, similar to e.g. Marching Primitives paper
    - But of course we probably need to try many different approaches
    
    **Architecture:**
    
    - DETR style transformer over superquadric queries seems to be the obvious fit. However, a difficult challenge will be to come up with an efficient encoding scheme over the point cloud. E.g. using random downsampling, PointTransformerV3, or SparseConvs
    
    **Managing primitive count**:
    
    - Since it is feed-forward single-pass, probably best to stick to a learned existence gating as SuperDec does. However, it could definitely be an interesting direction to allow for more dynamic query sizes (e.g. adding/removing queries in each layer)
- Related Works
    1. SuperFlex: It is both the modernized variant of SuperDec and looks deeper into applying it to scenes
    2. The whole Superquadric-Occlusion-Prediction line for autonomous driving:
    QuadricFormer 1 and 2, SuperOcc, SuperQuadOcc etc…
    They could especially interesting in terms of making this feed-forward & efficient
    3. All papers that deal with the same objective - scene-level primitive decomposition - but use different primitives (e.g. Gaussians)
    I found
    Convex Decomposition of Indoor Scenes
     Self-Supervised_Learning_on_3D_Point_Clouds_by_Learning_Discrete_Generative
    and Quadric Representations for LiDAR
    They are a bit older (2021 and 2023) but basically tackle same problem. Probably there are more.
    4. Marching Primitives Interesting for their objective function