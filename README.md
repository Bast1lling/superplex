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

