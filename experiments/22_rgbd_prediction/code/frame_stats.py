"""Per-frame target statistics on every 10th NICE-SLAM frame: SQs intersecting the camera frustum, their instances,
and how many are visible.

    uv run python frame_stats.py [--targets targets_winding]      -> results/frame_stats.json

An SQ intersects the frustum if any vertex of its mesh (SuperFlex's mesher) projects into the image in front of the
camera (z > 0.1 m). An instance is visible if >= 20 of its GT points (every 10th) project into the image within
3 cm of the rendered depth; an SQ is visible if its instance is.
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image

EXP = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(EXP.parent / "16_superflex_objective" / "code"))
from sfopt import PredictionHandler, gt  # noqa: E402
from views import CAM, RENDERS  # noqa: E402
from fit_targets import SCENES  # noqa: E402

STRIDE, W, H = 10, 1200, 680


def project(x, w2c):
    c = x @ w2c[:3, :3].T + w2c[:3, 3]
    u = CAM["fx"] * c[:, 0] / c[:, 2] + CAM["cx"]
    v = CAM["fy"] * c[:, 1] / c[:, 2] + CAM["cy"]
    return u, v, c[:, 2], (c[:, 2] > 0.1) & (u >= 0) & (u < W) & (v >= 0) & (v < H)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--targets", default="targets_winding")
    a = ap.parse_args()
    mesher = PredictionHandler.__new__(PredictionHandler)
    out = {}
    for scene in SCENES:
        z = np.load(EXP / "results" / a.targets / scene / "sqs.npz")
        verts, owner = [], []  # per existing SQ its mesh vertices and instance
        for b, k in enumerate(z["instance"]):
            for j in np.flatnonzero(z["exist"][b] > 0.5):
                v, _ = mesher._superquadric_mesh(z["scale"][b, j], z["exponents"][b, j], z["rotation"][b, j],
                                                 z["translation"][b, j], np.zeros(2), np.zeros(6), 8)
                verts.append(v), owner.append(k)
        nv = np.array([len(v) for v in verts])
        sq_of_vertex = np.repeat(np.arange(len(verts)), nv)
        verts, owner = np.concatenate(verts), np.array(owner)
        g = gt(scene)
        sel = np.flatnonzero(g["inst"] >= 0)[::10]
        P, I = g["pos"][sel], g["inst"][sel]
        root = RENDERS / scene.replace("_", "")
        c2w = np.loadtxt(root / "traj.txt").reshape(-1, 4, 4)
        rows = []
        for f in range(0, len(c2w), STRIDE):
            w2c = np.linalg.inv(c2w[f])
            ok = project(verts, w2c)[3]
            in_frustum = np.zeros(len(owner), bool)
            in_frustum[np.unique(sq_of_vertex[ok])] = True
            depth = np.asarray(Image.open(root / "results" / f"depth{f:06d}.png"), np.float32) / 6553.5
            u, v, zc, okp = project(P, w2c)
            d = depth[v[okp].astype(int), u[okp].astype(int)]
            seen = np.abs(zc[okp] - d) < 0.03
            ids, cnt = np.unique(I[okp][seen], return_counts=True)
            visible = set(ids[cnt >= 20].tolist())
            sq_vis = np.array([k in visible for k in owner])
            rows.append([int(in_frustum.sum()), int((in_frustum & sq_vis).sum()), len(np.unique(owner[in_frustum])),
                         len(visible)])
        r = np.array(rows)
        out[scene] = {"frames": len(r), "total_sqs": len(owner),
                      **{f"{n}_{q}": float(np.percentile(r[:, i], p)) for i, n in enumerate(
                          ["sqs_frustum", "sqs_frustum_visible_instance", "instances_frustum", "instances_visible"])
                         for q, p in (("median", 50), ("p95", 95), ("max", 100))}}
        o = out[scene]
        print(f"{scene}: {o['total_sqs']} SQs; per frame in frustum median {o['sqs_frustum_median']:.0f} / p95 "
              f"{o['sqs_frustum_p95']:.0f} / max {o['sqs_frustum_max']:.0f}, of visible instances "
              f"{o['sqs_frustum_visible_instance_median']:.0f}; instances in frustum {o['instances_frustum_median']:.0f}, "
              f"visible {o['instances_visible_median']:.0f}", flush=True)
    (EXP / "results" / "frame_stats.json").write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
