"""views: rendered RGBD views of a scene with per-pixel GT instance labels, and the voxel grid carved from them
(condensed from experiments 11 / 12).

Renderings (NICE-SLAM Replica): Thesis/data/replica_nice_slam/Replica/<room0|...>/results/{frame,depth}NNNNNN.{jpg,png},
1200 x 680, fx = fy = 600, camera-to-world poses in traj.txt (OpenCV axes), same world frame as the Replica meshes;
used at 300 x 170 (every 4th pixel). Instance label of a pixel: the GT instance of the nearest 1 cm point within 2 cm
of the back-projected pixel (stands in for a 2D instance segmentation). RGB settings: the same labels, but the depth
used afterwards (points, carving) is Depth Pro's estimate from the RGB image (depth.py) instead of the rendered one.
"""

import json

import numpy as np
import torch
from PIL import Image
from scipy.spatial import cKDTree

import data
import depth as mono
from data import DATA, DEEP, DEV, FREE, Grid

RENDERS = DATA / "replica_nice_slam" / "Replica"
CAM = json.loads((RENDERS / "cam_params.json").read_text())["camera"]
DOWN = 4
FX, FY, CX, CY = CAM["fx"] / DOWN, CAM["fy"] / DOWN, (CAM["cx"] - DOWN // 2) / DOWN, (CAM["cy"] - DOWN // 2) / DOWN
MULTI = list(range(0, 2000, 20))  # 100 views
SETTINGS = {"multi": (MULTI, "gt"), "f800": ([800], "gt"), "rgb_multi": (MULTI, "depthpro"), "rgb_f800": ([800], "depthpro")}
SCENE_SETTINGS = {"scene_multi": (MULTI, "gt"), "scene_rgb_multi": (MULTI, "depthpro")}  # without instance labels


def load_views(scene, frames, pts, depth_source="gt"):
    """Views with rgb, depth (rendered, or Depth Pro), labels (always from the rendered depth) and c2w."""
    folder = RENDERS / scene.replace("_", "")
    poses = np.loadtxt(folder / "traj.txt").reshape(-1, 4, 4)
    tree, views = cKDTree(pts["pos"]), []
    for f in frames:
        o = DOWN // 2
        depth = np.asarray(Image.open(folder / "results" / f"depth{f:06d}.png"))[o::DOWN, o::DOWN] / CAM["scale"]
        img = Image.open(folder / "results" / f"frame{f:06d}.jpg")
        view = {"frame": f, "c2w": poses[f], "depth": depth.astype(np.float32),
                "rgb": np.asarray(img.resize((img.width // DOWN, img.height // DOWN), Image.BOX))}
        dist, nn = tree.query(backproject(view), distance_upper_bound=0.02, workers=-1)
        view["label"] = np.where(np.isfinite(dist), pts["inst"][np.minimum(nn, len(pts["inst"]) - 1)], -1).reshape(depth.shape)
        if depth_source == "depthpro":
            view["depth"] = mono.predict(scene, f, folder / "results" / f"frame{f:06d}.jpg", CAM["fx"], DOWN)
        views.append(view)
    return views


def backproject(view):
    """World points of all pixels (rows of the image, then columns)."""
    d = view["depth"]
    v, u = np.indices(d.shape)
    cam = np.stack([(u - CX) / FX * d, (v - CY) / FY * d, d], -1).reshape(-1, 3)
    return cam @ view["c2w"][:3, :3].T + view["c2w"][:3, 3]


def view_points(views, labelled=True):
    """Back-projected pixels with depth (and, if labelled, an instance label): pos, inst, nrm (as data.load_points).
    Normals from the depth image: cross product of the neighboring pixels' points, turned towards the camera."""
    pos, inst, nrm, ok = [], [], [], []
    for v in views:
        P = backproject(v).reshape(*v["depth"].shape, 3)
        n = np.cross(np.gradient(P, axis=1), np.gradient(P, axis=0)).reshape(-1, 3)
        n /= np.linalg.norm(n, axis=1, keepdims=True) + 1e-9
        n *= np.sign(((v["c2w"][:3, 3] - P.reshape(-1, 3)) * n).sum(1, keepdims=True))
        pos.append(P.reshape(-1, 3)), nrm.append(n), inst.append(v["label"].reshape(-1))
        ok.append((v["depth"].reshape(-1) > 0) & ((v["label"].reshape(-1) >= 0) | (not labelled)))
    ok = np.concatenate(ok)
    return {"pos": np.concatenate(pos)[ok].astype(np.float32), "inst": np.concatenate(inst)[ok],
            "nrm": np.concatenate(nrm)[ok].astype(np.float32)}


@torch.no_grad()
def carve(views, n_inst, margin=0.08, thick=0.1, labelled=True):
    """Every voxel center is projected into every view. In front of the observed depth (by more than one voxel): a free
    vote; from one voxel in front to `thick` behind it: a solid vote for the pixel's instance. Free if free votes win,
    else the instance with most votes, else (never seen, or deeper inside objects) DEEP. Beyond the observed points'
    box: free (outside the grid). Without labels every pixel votes for instance 0 (solid)."""
    pts = np.concatenate([backproject(v)[v["depth"].reshape(-1) > 0] for v in views])
    res = data.RES  # set by --res
    lo = np.quantile(pts, 0.001, 0) - margin
    dims = np.ceil((np.quantile(pts, 0.999, 0) + margin - lo) / res).astype(int) + 1
    x = torch.tensor(lo + np.indices(dims).reshape(3, -1).T * res, dtype=torch.float32, device=DEV)
    free = torch.zeros(len(x), dtype=torch.int32, device=DEV)
    votes = torch.zeros(len(x), n_inst + 1, dtype=torch.int32, device=DEV)  # last column: unlabelled pixels
    for view in views:
        w2c = torch.tensor(np.linalg.inv(view["c2w"]), dtype=torch.float32, device=DEV)
        c = x @ w2c[:3, :3].T + w2c[:3, 3]
        z = c[:, 2].clamp_min(1e-6)
        u, v = torch.round(FX * c[:, 0] / z + CX).long(), torch.round(FY * c[:, 1] / z + CY).long()
        h, w = view["depth"].shape
        idx = ((c[:, 2] > 0.05) & (u >= 0) & (u < w) & (v >= 0) & (v < h)).nonzero()[:, 0]
        d = torch.as_tensor(view["depth"], device=DEV)[v[idx], u[idx]]
        lab = torch.as_tensor(view["label"] if labelled else 0 * view["label"], device=DEV)[v[idx], u[idx]]
        zi = z[idx]
        free[idx[(d > 0) & (zi < d - res)]] += 1
        solid = (d > 0) & (zi >= d - res) & (zi <= d + thick)
        col = torch.where(lab[solid] >= 0, lab[solid], n_inst)
        votes.index_put_((idx[solid], col), torch.ones_like(col, dtype=torch.int32), accumulate=True)
    best = votes[:, :n_inst].max(1)
    label = torch.where((best.values > 0) & (best.values >= votes[:, n_inst]), best.indices, DEEP)
    label[free > votes.sum(1)] = FREE
    return Grid(label.reshape(*dims).cpu().numpy(), lo.astype(np.float32), res)
