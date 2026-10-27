# Test data (Replica, 2 cm)

Small subset of `Thesis/data/` for testing and remote runs (room_0, room_1, office_0). Same layout as `Thesis/data/`:

- `replica_v1_points/<scene>_voxel20mm.npz`: the 1 cm clouds subsampled to one point per 2 cm voxel
  (pos, col, nrm, instance, segment)
- `replica_v1_sonata/stage4/<scene>.npz`: only the frames (R, t, ...) of the Sonata tokens, without `feat`
- `replica_v1/<scene>/habitat/info_semantic.json`: class names and instance -> class lookup
