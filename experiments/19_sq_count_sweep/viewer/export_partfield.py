"""PartField's part labels as point clouds for index.html, into data/partfield/ (not versioned).

    papers/PartField/.venv/bin/python export_partfield.py [viewer data folder]

data/partfield/<id>.bin: POINTS surface points (f32), then their part label for 1 ... 20 parts (u8, one block per level).
Points are sampled on the mesh, so each one takes the label of its face, and moved into the dataset's frame. Runs in
PartField's environment: labels follow the face order of the trimesh that loaded the mesh there.
"""

import sys
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import trimesh

ROOT = Path("/home/sebastian/data/ShapeNet-Chairs-RGBD")
CORE = Path("/home/sebastian/data/ShapeNet-Core/03001627/03001627")
LABELS = ROOT / "partfield" / "clustering" / "chairs_test" / "cluster_out"
OUT = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(__file__).resolve().parent / "data"
POINTS, LEVELS = 16384, 20
TO_ONET = np.array([[0.0, 0, -1], [0, 1, 0], [1, 0, 0]])  # experiment 18


def export(model_id):
    if not (LABELS / f"{model_id}_0_{LEVELS:02d}.npy").exists():
        return None
    mesh = trimesh.load(CORE / model_id / "models" / "model_normalized.obj", force="mesh", process=False)
    np.random.seed(0)
    points, face = trimesh.sample.sample_surface(mesh, POINTS)
    cameras = np.load(ROOT / "03001627" / model_id / "cameras.npz")
    points = ((points - cameras["mesh_loc"]) / cameras["mesh_scale"]) @ TO_ONET.T
    parts = []
    for level in range(1, LEVELS + 1):
        labels = np.load(LABELS / f"{model_id}_0_{level:02d}.npy")
        assert len(labels) == len(mesh.faces), model_id
        parts.append(np.unique(labels[face], return_inverse=True)[1].astype("u1"))
    (OUT / "partfield" / f"{model_id}.bin").write_bytes(points.astype("<f4").tobytes() + np.concatenate(parts).tobytes())
    return model_id


def main():
    (OUT / "partfield").mkdir(parents=True, exist_ok=True)
    ids = sorted(p.name.split("_0_")[0] for p in LABELS.glob(f"*_0_{LEVELS:02d}.npy"))
    keep = [p.name[:-4] for p in sorted((OUT / "chairs").glob("*.bin"))]
    ids = [i for i in ids if i in keep]
    with Pool(12) as pool:
        done = [i for i in pool.imap_unordered(export, ids, chunksize=4) if i]
    print(len(done), "chairs")


if __name__ == "__main__":
    main()
