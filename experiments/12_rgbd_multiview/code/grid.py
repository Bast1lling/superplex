"""Voxel grids with labels FREE, UNKNOWN or an instance index (its surface / solid).

gt_grid:    from the full GT point cloud (the presentation's voxelization: k-NN normal vote, solid voxels > 10 cm from
            the surface are UNKNOWN, voxels containing points are solid; beyond the grid: FREE). Upper bound and the
            reference of the evaluation.
carve_grid: from depth maps + per-pixel labels only, with the same two rules as gt_grid. Every voxel center is
            projected into every view: it gets a free vote if it lies in front of the observed depth (by more than
            `band`), a solid vote (with the pixel's instance) if it lies between `band` in front of and `thick`
            behind it (solid ends `thick` behind the surface, like gt_grid's 10 cm). Free if free votes outnumber
            solid votes, else the instance with most solid votes, else UNKNOWN (never seen, deeper inside objects).
            Beyond the grid (bounding box of all observed points + margin): FREE if outside_free (the scene ends
            where the observations end), else UNKNOWN.
"""

import numpy as np
import torch
from scipy.spatial import cKDTree

from views import DEV, gt, intrinsics

FREE, UNKNOWN = -1, -2


class Grid:
    def __init__(self, label, lo, res, outside):
        self.label_np, self.lo_np, self.res, self.outside = label, lo, res, outside
        self.label = torch.as_tensor(label, device=DEV)
        self.lo = torch.as_tensor(lo, dtype=torch.float32, device=DEV)
        self.dims = torch.tensor(label.shape, device=DEV)

    def centers(self, idx):
        return self.lo + torch.as_tensor(idx, device=DEV, dtype=torch.float32) * self.res

    def lookup(self, x):
        i = torch.round((x - self.lo) / self.res).long()
        inside = ((i >= 0) & (i < self.dims)).all(-1)
        i = torch.minimum(i.clamp_min(0), self.dims - 1)
        return torch.where(inside, self.label[i[..., 0], i[..., 1], i[..., 2]], self.outside)


def make_grid(lo, hi, res):
    dims = np.ceil((hi - lo) / res).astype(int) + 1
    return dims, lo + np.indices(dims).reshape(3, -1).T * res


def gt_grid(scene, res=0.04, margin=0.08, k=8, near=0.1):
    g = gt(scene)
    keep = g["inst"] >= 0
    pos, nrm, inst = g["pos"][keep], g["nrm"][keep], g["inst"][keep]
    lo = pos.min(0) - margin
    dims, centers = make_grid(lo, pos.max(0) + margin, res)
    dist, nn = cKDTree(pos).query(centers, k=k, workers=-1)
    behind = np.einsum("vkj,vkj->vk", centers[:, None] - pos[nn], nrm[nn]) < 0
    label = np.where(behind.mean(1) > 0.5, np.where(dist[:, 0] <= near, inst[nn[:, 0]], UNKNOWN), FREE).reshape(dims)
    label[tuple(np.round((pos - lo) / res).astype(int).T)] = inst
    return Grid(label, lo.astype(np.float32), res, FREE)


def backproject(view, depth):
    fx, fy, cx, cy = intrinsics()
    v, u = np.indices(depth.shape)
    cam = np.stack([(u - cx) / fx * depth, (v - cy) / fy * depth, depth], -1).reshape(-1, 3)
    return cam @ view["c2w"][:3, :3].T + view["c2w"][:3, 3]


@torch.no_grad()
def carve_grid(views, depths, n_inst, res=0.04, margin=0.08, band=None, thick=0.1, outside_free=True, erode=0,
               min_votes=1, rel_band=0.0):
    """views: dicts with label, c2w; depths: one depth map per view (GT or predicted, same pixel grid).
    Options against depth noise (predicted depth): erode = ignore labels within that many pixels of a label boundary
    (smoothed depth at object edges puts them in mid-air); instances that lose all voxels by this (small masks) keep
    their un-eroded votes; min_votes = solid votes needed for an instance voxel;
    rel_band = surface band grows with the depth (band = max(band, rel_band * z))."""
    from scipy.ndimage import maximum_filter, minimum_filter
    band = res if band is None else band
    pts = np.concatenate([backproject(v, d)[d.reshape(-1) > 0] for v, d in zip(views, depths)])
    lo, hi = np.quantile(pts, 0.001, 0) - margin, np.quantile(pts, 0.999, 0) + margin
    dims, centers = make_grid(lo, hi, res)
    x = torch.tensor(centers, dtype=torch.float32, device=DEV)
    fx, fy, cx, cy = intrinsics()
    n_free = torch.zeros(len(x), dtype=torch.int32, device=DEV)
    votes = torch.zeros(len(x), n_inst + 1, dtype=torch.int32, device=DEV)  # last column: unlabelled pixels
    plain = torch.zeros_like(votes) if erode else votes  # votes without erosion (fallback for small instances)
    for view, depth in zip(views, depths):
        w2c = torch.tensor(np.linalg.inv(view["c2w"]), dtype=torch.float32, device=DEV)
        c = x @ w2c[:3, :3].T + w2c[:3, 3]
        z = c[:, 2].clamp_min(1e-6)
        u = torch.round(fx * c[:, 0] / z + cx).long()
        v = torch.round(fy * c[:, 1] / z + cy).long()
        h, w = depth.shape
        ok = (c[:, 2] > 0.05) & (u >= 0) & (u < w) & (v >= 0) & (v < h)
        idx = ok.nonzero()[:, 0]
        d = torch.as_tensor(depth, device=DEV)[v[idx], u[idx]]
        lab0 = view["label"]
        lab = lab0
        if erode:
            size = 2 * erode + 1
            lab = np.where(maximum_filter(lab0, size) == minimum_filter(lab0, size), lab0, -1)
        lab0 = torch.as_tensor(lab0, device=DEV)[v[idx], u[idx]]
        lab = torch.as_tensor(lab, device=DEV)[v[idx], u[idx]]
        valid = d > 0
        b = torch.clamp_min(rel_band * d, band)
        n_free[idx[valid & (z[idx] < d - b)]] += 1
        surf = valid & (z[idx] >= d - b) & (z[idx] <= d + thick)
        for vt, lb in ((votes, lab), (plain, lab0)) if erode else ((votes, lab),):
            col = torch.where(lb[surf] >= 0, lb[surf], n_inst)
            vt.index_put_((idx[surf], col), torch.ones_like(col, dtype=torch.int32), accumulate=True)

    def decide(votes):
        n_surf = votes.sum(1)
        best = votes[:, :n_inst].argmax(1)
        ok = (n_surf > 0) & (votes[:, :n_inst].max(1).values >= votes[:, n_inst])
        ok &= votes[:, :n_inst].max(1).values >= min_votes
        label = torch.full((len(x),), UNKNOWN, device=DEV)
        label[ok] = best[ok]
        label[n_free > n_surf] = FREE
        return label

    label = decide(votes)
    if erode:
        fallback = decide(plain)
        lost = torch.bincount(label[label >= 0], minlength=n_inst) < 4
        take = (fallback >= 0) & lost[fallback.clamp_min(0)] & (label != FREE)
        label[take] = fallback[take]
    return Grid(label.reshape(*dims).cpu().numpy(), lo.astype(np.float32), res, FREE if outside_free else UNKNOWN)
