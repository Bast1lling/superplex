"""Replica points (1 cm, with normals and GT instances), the voxel grid built from them, and the command line."""

import json
import sys
from pathlib import Path

import numpy as np
import torch
from scipy.spatial import cKDTree

DATA = Path(__file__).resolve().parents[5] / "data"  # Thesis/data
RESULTS = Path(__file__).resolve().parents[1] / "results"
DEV = "cuda"
SCENES = ("room_0", "room_1", "office_0")
RES = 0.04  # voxel size, set by --res
# voxel labels; solid voxels carry their instance (>= 0). DEEP: behind a surface but far from it, ignored
FREE, OUTSIDE, DEEP = -1, -2, -3


def cli():
    """Arguments of every script: [--res 0.02] [scene ...]. Returns the scenes (default: all)."""
    global RES
    args = sys.argv[1:]
    if args[:1] == ["--res"]:
        RES, args = float(args[1]), args[2:]
    return args or list(SCENES)


def result_path(scene, method):
    return RESULTS / f"{RES * 100:g}cm" / scene / f"{method}.npz"


def load_points(scene, min_points=80):
    """1 cm Replica points (experiment 01). Points without a semantic class and instances with < min_points points
    (wall plugs etc.) are dropped; instances are renumbered 0..n-1."""
    d = np.load(DATA / "replica_v1_points" / f"{scene}_voxel10mm.npz")
    pos, nrm, col, inst = (d[k] for k in ("pos", "nrm", "col", "instance"))
    id_to_label = np.array(json.loads((DATA / "replica_v1" / scene / "habitat" / "info_semantic.json").read_text())["id_to_label"])
    labelled = id_to_label[np.minimum(inst, len(id_to_label) - 1)] > 0
    ids, count = np.unique(inst[labelled], return_counts=True)
    keep = labelled & np.isin(inst, ids[count >= min_points])
    _, inst = np.unique(inst[keep], return_inverse=True)
    nrm = nrm[keep].astype(np.float32)
    return {"pos": pos[keep], "nrm": nrm / np.linalg.norm(nrm, axis=1, keepdims=True), "col": col[keep], "inst": inst}


class Grid:
    """Voxel labels (instance, FREE or DEEP), with voxel centers at lo + index * res."""

    def __init__(self, label, lo, res):
        self.label_np, self.lo_np, self.res = label, lo, res
        self.label = torch.from_numpy(label).to(DEV)
        self.lo = torch.tensor(lo, device=DEV)
        self.dims = torch.tensor(label.shape, device=DEV)

    def centers(self, idx):
        return self.lo + torch.as_tensor(idx, device=DEV) * self.res

    def lookup(self, x):
        """Labels at points x (..., 3); OUTSIDE beyond the grid."""
        i = torch.round((x - self.lo) / self.res).long()
        inside = ((i >= 0) & (i < self.dims)).all(-1)
        i = torch.minimum(i.clamp_min(0), self.dims - 1)
        return torch.where(inside, self.label[i[..., 0], i[..., 1], i[..., 2]], OUTSIDE)

    def empty(self, x):
        """Free, or beyond the grid (nothing was scanned there; otherwise SQs grow out of the scan for free)."""
        lab = self.lookup(x)
        return (lab == FREE) | (lab == OUTSIDE)


def voxelize(pts, margin=0.08, k=8, near=0.1):
    """1. grid over the points + margin (behind walls / floor / ceiling the solid ends at the grid edge: slabs);
    2. a voxel is solid if most of its k nearest points see it from behind (n . (v - p) < 0), it takes the instance
    of the nearest point; far from surfaces this vote is unreliable (room_0: the plant's label spread up to 1 m from
    its points), so solid voxels more than `near` from their nearest point are DEEP (ignored);
    3. voxels containing points are solid (thin objects)."""
    pos, nrm, inst = pts["pos"], pts["nrm"], pts["inst"]
    lo = pos.min(0) - margin
    dims = np.ceil((pos.max(0) + margin - lo) / RES).astype(int) + 1
    centers = lo + np.indices(dims).reshape(3, -1).T * RES
    dist, nn = cKDTree(pos).query(centers, k=k, workers=-1)
    behind = np.einsum("vkj,vkj->vk", centers[:, None] - pos[nn], nrm[nn]) < 0
    label = np.where(behind.mean(1) > 0.5, np.where(dist[:, 0] <= near, inst[nn[:, 0]], DEEP), FREE).reshape(dims)
    label[tuple(np.round((pos - lo) / RES).astype(int).T)] = inst
    return Grid(label, lo.astype(np.float32), RES)
