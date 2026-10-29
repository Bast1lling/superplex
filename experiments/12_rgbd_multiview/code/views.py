"""Inputs of experiments 12-15: rendered views (NICE-SLAM Replica renderings) with per-pixel GT instance labels,
and the full 3D GT that is only used for labels and evaluation.

Renderings: Thesis/data/replica_nice_slam/Replica/<room0|...>/results/{frame,depth}NNNNNN.{jpg,png}, 1200 x 680,
fx = fy = 600, depth = z-depth / 6553.5 m, traj.txt = camera-to-world (OpenCV axes), same world frame as the Replica
meshes (experiment 11). Views are downsampled by DOWN (pixel centers of the original grid, so depth / labels stay
exact; RGB is box-averaged).

Instance labels per pixel (as in experiment 11): back-project the pixel with its rendered depth, take the GT instance
of the nearest 1 cm point within 2 cm. Instances are indexed like the GT (labelled class, >= 80 points), -1 otherwise.
In the RGB settings the depth is only used for this labelling (the labels stand in for a 2D segmentation).
"""

import json
import sys
from functools import lru_cache
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from scipy.spatial import cKDTree

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "experiments" / "01_scene_viewer" / "code"))
from scenes import DATA, load_points, load_semantics  # noqa: E402

RENDERS = DATA / "replica_nice_slam" / "Replica"
CAM = json.loads((RENDERS / "cam_params.json").read_text())["camera"]
DOWN = 4  # 1200 x 680 -> 300 x 170
DEV = "cuda"
SCENES = ("room_0", "room_1", "office_0")


@lru_cache(maxsize=1)
def gt(scene, min_points=80):
    """GT points (1 cm) with instance index (-1: unlabelled class or < min_points points) and class names."""
    p = load_points(scene)
    sem = load_semantics(scene)
    i2l = np.array(sem["id_to_label"])
    raw = p["instance"].astype(np.int64)
    cls = i2l[np.minimum(raw, len(i2l) - 1)]
    ids, count = np.unique(raw[cls > 0], return_counts=True)
    ids = ids[count >= min_points]
    inst = np.full(len(raw), -1)
    ok = np.isin(raw, ids)
    inst[ok] = np.searchsorted(ids, raw[ok])
    nrm = p["nrm"].astype(np.float32)
    return {"pos": p["pos"].astype(np.float32), "nrm": nrm / np.linalg.norm(nrm, axis=1, keepdims=True), "inst": inst,
            "ids": ids, "names": [sem["classes"].get(int(i2l[i]), "?") for i in ids], "tree": cKDTree(p["pos"])}


def frames(scene):
    return np.loadtxt(RENDERS / scene.replace("_", "") / "traj.txt").reshape(-1, 4, 4)


def intrinsics(down=DOWN):
    """fx, fy, cx, cy of the downsampled image whose pixel (j, i) is pixel (down j + down // 2, ...) of the original."""
    o = down // 2
    return CAM["fx"] / down, CAM["fy"] / down, (CAM["cx"] - o) / down, (CAM["cy"] - o) / down


def load_view(scene, frame, down=DOWN):
    d = RENDERS / scene.replace("_", "") / "results"
    o = down // 2
    depth = np.asarray(Image.open(d / f"depth{frame:06d}.png"))[o::down, o::down] / CAM["scale"]
    img = Image.open(d / f"frame{frame:06d}.jpg")
    rgb = np.asarray(img.resize((img.width // down, img.height // down), Image.BOX)).astype(np.float32) / 255
    c2w = frames(scene)[frame]
    fx, fy, cx, cy = intrinsics(down)
    v, u = np.indices(depth.shape)
    cam = np.stack([(u - cx) / fx * depth, (v - cy) / fy * depth, depth], -1)
    g = gt(scene)
    dist, nn = g["tree"].query(cam.reshape(-1, 3) @ c2w[:3, :3].T + c2w[:3, 3], distance_upper_bound=0.02, workers=-1)
    label = np.where(np.isfinite(dist), g["inst"][np.minimum(nn, len(g["inst"]) - 1)], -1).reshape(depth.shape)
    return {"frame": frame, "rgb": rgb, "depth": depth.astype(np.float32), "label": label, "c2w": c2w}


def load_views(scene, frame_ids, down=DOWN):
    return [load_view(scene, f, down) for f in frame_ids]
