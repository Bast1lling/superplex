"""Scene data and a labeled voxel volume built from the oriented point cloud.

Every voxel center v gets the nearest scene points (k-NN); it is *solid* if it lies behind their surfaces
(mean sign of n . (v - p) < 0) and *free* otherwise. Solid voxels take the instance of their nearest point.
Solid voxels whose nearest point belongs to a structural class (wall / floor / ceiling: everything behind them is
outside the room) are only kept up to `exterior_depth` behind the surface, so these become slabs; any solid voxel
deeper than `max_depth` is unknown as well. Finally, an instance's solid is clipped to its gravity-aligned oriented
bounding box (yaw from PCA of its points) padded by `box_pad` (removes e.g. space behind windows / blinds).
Free voxels outside the room box (oriented box of the structural points) or within `room_band` of its faces are
unknown: free space seen through windows / glass doors is not reliably observed (window frames have no glass).
Voxels that contain points of an instance always count as that instance's solid (thin objects).

Labels:  -1 free (observed empty space), -2 unknown (deep exterior), >= 0 instance index (0..n_inst-1).
"""

from pathlib import Path
import json

import numpy as np
from scipy.spatial import cKDTree

ROOT = Path(__file__).resolve().parents[3]
DATA = ROOT / "data"
FREE, UNKNOWN = -1, -2


def semantic_components(pos: np.ndarray, lab: np.ndarray, radius: float = 0.05) -> np.ndarray:
    """Pseudo-instances from semantic labels only: connected components (points closer than radius) per class."""
    from scipy.sparse import coo_matrix
    from scipy.sparse.csgraph import connected_components
    out = np.zeros(len(pos), np.int64)
    nxt = 0
    for c in np.unique(lab):
        idx = np.nonzero(lab == c)[0]
        pairs = cKDTree(pos[idx]).query_pairs(radius, output_type="ndarray")
        A = coo_matrix((np.ones(len(pairs)), (pairs[:, 0], pairs[:, 1])), shape=(len(idx), len(idx)))
        n, cc = connected_components(A, directed=False)
        out[idx] = nxt + cc
        nxt += n
    return out


def load_scene(scene: str, drop_unlabeled: bool = True, min_points: int = 1, labels: str = "instance") -> dict:
    """labels: "instance" (GT instances) or "semantic" (connected components of each GT class)."""
    d = np.load(DATA / "replica_v1_points" / f"{scene}_voxel20mm.npz")
    sem = json.loads((DATA / "replica_v1" / scene / "habitat" / "info_semantic.json").read_text())
    names = {c["id"]: c["name"] for c in sem["classes"]}
    i2l = np.array(sem["id_to_label"])
    pos, nrm, inst = d["pos"].astype(np.float32), d["nrm"].astype(np.float32), d["instance"].astype(np.int64)
    col = d["col"]
    lab = i2l[np.minimum(inst, len(i2l) - 1)]
    keep = lab > 0 if drop_unlabeled else np.ones(len(pos), bool)
    pos, nrm, inst, col, lab = pos[keep], nrm[keep], inst[keep], col[keep], lab[keep]
    if labels == "semantic":
        inst = semantic_components(pos, lab)
    ids, y, cnt = np.unique(inst, return_inverse=True, return_counts=True)
    if min_points > 1:  # drop tiny instances (a few points: wall plugs etc.)
        ok = cnt[y] >= min_points
        pos, nrm, inst, col, lab = pos[ok], nrm[ok], inst[ok], col[ok], lab[ok]
        ids, y = np.unique(inst, return_inverse=True)
    cls = np.array([lab[y == k][0] for k in range(len(ids))])
    return {"scene": scene, "pos": pos, "nrm": nrm / np.linalg.norm(nrm, axis=1, keepdims=True), "col": col,
            "y": y, "instance_ids": ids, "class_id": cls,
            "class_name": [names.get(int(c), "?") for c in cls]}


STRUCTURAL = ("wall", "floor", "ceiling")


def build_volume(sc: dict, res: float = 0.04, pad: float = 0.2, knn: int = 8,
                 exterior_depth: float = 0.1, max_depth: float = 1.0, box_pad: float = None,
                 room_band: float = 0.06) -> dict:
    pos, nrm, y = sc["pos"], sc["nrm"], sc["y"]
    lo = pos.min(0) - pad
    dims = np.ceil((pos.max(0) + pad - lo) / res).astype(int) + 1
    grid = np.stack(np.meshgrid(*[np.arange(n) for n in dims], indexing="ij"), -1).reshape(-1, 3)
    centers = lo + grid * res
    tree = cKDTree(pos)
    dist, idx = tree.query(centers, k=knn, workers=-1)
    s = np.einsum("vkj,vkj->vk", centers[:, None] - pos[idx], nrm[idx])
    w = 1.0 / (dist + 0.25 * res)
    sign = (np.sign(s) * w).sum(1) / w.sum(1)
    solid = (sign < 0).reshape(dims)
    d0 = dist[:, 0].reshape(dims)
    label = np.where(solid, y[idx[:, 0]].reshape(dims), FREE)
    structural = np.array([n in STRUCTURAL for n in sc["class_name"]])
    ext = solid & structural[y[idx[:, 0]]].reshape(dims)
    label[(ext & (d0 > exterior_depth)) | (solid & (d0 > max_depth))] = UNKNOWN
    box_pad = res if box_pad is None else box_pad
    flat = label.reshape(-1)
    sol = np.flatnonzero(flat >= 0)
    boxes = [yaw_box(pos[y == k]) for k in range(len(sc["instance_ids"]))]
    cand = y[idx[sol]]                                    # (S, knn) instances of the nearest points
    ok = np.zeros(cand.shape, bool)
    for k in np.unique(cand):
        R, c, half = boxes[k]
        m = cand == k
        r = np.nonzero(m.any(1))[0]
        inside = (np.abs((centers[sol[r]] - c) @ R) <= half + box_pad).all(1)
        ok[r] |= m[r] & inside[:, None]
    first = ok.argmax(1)
    has = ok.any(1)
    flat[sol] = np.where(has, cand[np.arange(len(sol)), first], UNKNOWN)
    # free space outside the room (seen through windows / doors / glass) is not reliably observed
    sp = np.isin(y, np.nonzero(structural)[0])
    if room_band is not None and sp.any():
        R, c, half = yaw_box(pos[sp])
        q = np.abs((centers - c) @ R)
        outside = (q > half - room_band).any(1).reshape(dims)
        label[outside & (label == FREE)] = UNKNOWN
    # voxels containing points
    vi = np.clip(np.round((pos - lo) / res).astype(int), 0, dims - 1)
    label[vi[:, 0], vi[:, 1], vi[:, 2]] = y  # (last writer wins for shared voxels)
    return {"label": label, "lo": lo.astype(np.float32), "res": res, "dims": dims, "dist": d0.astype(np.float32)}


def yaw_box(p: np.ndarray):
    """Gravity-aligned oriented box of points: rotation (columns = axes), center, half extents."""
    xy = p[:, :2] - p[:, :2].mean(0)
    _, vec = np.linalg.eigh(xy.T @ xy)
    R = np.eye(3)
    R[:2, :2] = vec
    q = p @ R
    lo, hi = q.min(0), q.max(0)
    return R, R @ ((lo + hi) / 2), (hi - lo) / 2


def voxel_centers(vol: dict, mask: np.ndarray) -> np.ndarray:
    return (vol["lo"] + np.argwhere(mask) * vol["res"]).astype(np.float32)
