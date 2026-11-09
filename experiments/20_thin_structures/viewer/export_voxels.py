"""Occupancy of the target chair for the viewer: data/voxels.json with the centers of the occupied voxels of a dense grid
(the depth fusion of experiment 18 plus the fix for missing sheets) and the occupied query points the losses see.

    uv run python export_voxels.py
"""

import json
import sys
from pathlib import Path

import numpy as np
import torch
from scipy.spatial import cKDTree

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1] / "18_chair_rgbd_dataset" / "code"))
sys.path.insert(0, str(HERE.parent / "code"))
import build_dataset as dataset  # noqa: E402
from fix_occupancy import TOLERANCE  # noqa: E402
from variants import DATA, DEV, TARGET, load_chair  # noqa: E402

RES = 128
EXTENT = 0.55


def main():
    verts, faces, *_ = dataset.load_mesh(dataset.CORE / TARGET / "models" / "model_normalized.obj", DEV)
    fusion = dataset.Fusion(verts, faces)
    size = 2 * EXTENT / RES
    axis = torch.linspace(-EXTENT + size / 2, EXTENT - size / 2, RES, device=DEV)
    grid = torch.stack(torch.meshgrid(axis, axis, axis, indexing="ij"), -1).reshape(-1, 3)
    occupied = torch.cat([~fusion.query(chunk)[0] for chunk in grid.split(100_000)])
    surface = np.load(DATA / TARGET / "pointcloud.npz")["points"].astype(np.float32)
    occupied |= torch.tensor(cKDTree(surface).query(grid.cpu().numpy(), workers=-1)[0] <= TOLERANCE, device=DEV)
    chair = load_chair(TARGET)
    out = {"size": size, "voxels": np.round(grid[occupied].cpu().numpy(), 4).ravel().tolist(),
           "query": np.round(chair["q"][chair["occ"]], 4).ravel().tolist()}
    (HERE / "data" / "voxels.json").write_text(json.dumps(out))
    print(int(occupied.sum()), "voxels,", int(chair["occ"].sum()), "occupied query points")


if __name__ == "__main__":
    main()
