"""Exports GT points, GT voxels and the primitives of every run of experiment 16 for index.html into viewer/data/<scene>/
(not versioned).

    uv run python export.py [scene ...]
    python -m http.server -d experiments/16_superflex_objective/viewer 8016      -> http://localhost:8016
"""

import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "code"))
from sfopt import PredictionHandler, gt, gt_grid, instance_data  # noqa: E402
from scenes import load_points  # noqa: E402
from views import SCENES  # noqa: E402

RUNS = ("network_nooptim", "network_theirs", "network_ours", "kmeans_net_theirs", "kmeans_net_ours",
        "kmeans16_theirs", "kmeans16_ours", "network_theirs_plain", "kmeans_net_theirs_plain", "kmeans16_theirs_plain")


def surface_voxels(label):
    """Solid voxels with a non-solid 6-neighbor (the others are hidden anyway)."""
    solid = label >= 0
    P = np.pad(solid, 1)
    inner = solid.copy()
    for ax in range(3):
        for s in (-1, 1):
            inner &= np.roll(P, s, ax)[1:-1, 1:-1, 1:-1]
    return solid & ~inner


def mesh(z, norm, path, res=20):
    """Existing primitives meshed with SuperFlex's mesher -> vertices f32, faces u32, instance i16 and primitive
    index u16 per vertex. update_handler denormalizes scale / translation but not the bending curvature, hence / norm."""
    mesher = PredictionHandler.__new__(PredictionHandler)  # only its mesh methods are used
    verts, faces, inst, prim, n = [], [], [], [], 0
    for b, k in enumerate(z["instance"]):
        for j in np.flatnonzero(z["exist"][b] > 0.5):
            ben = z["bending"][b, j].copy()
            ben[0::2] /= norm[b]
            v, f = mesher._superquadric_mesh(z["scale"][b, j], z["exponents"][b, j], z["rotation"][b, j],
                                             z["translation"][b, j], z["tapering"][b, j], ben, res)
            verts.append(v), faces.append(f + n), inst.append(np.full(len(v), k)), prim.append(np.full(len(v), len(prim)))
            n += len(v)
    verts, faces = np.concatenate(verts), np.concatenate(faces)
    path.write_bytes(verts.astype("<f4").tobytes() + faces.astype("<u4").tobytes()
                     + np.concatenate(inst).astype("<i2").tobytes() + np.concatenate(prim).astype("<u2").tobytes())
    return [len(verts), len(faces), len(prim)]


def export(scene):
    g, grid = gt(scene), gt_grid(scene)
    data = instance_data(scene, grid)
    norm = {d["k"]: 2 * np.abs(d["surf"]).max() for d in data}  # SuperFlex's normalization scale per instance
    out = HERE / "data" / scene
    out.mkdir(parents=True, exist_ok=True)
    col = load_points(scene)["col"]
    (out / "points.bin").write_bytes(g["pos"].astype("<f4").tobytes() + g["inst"].astype("<i2").tobytes()
                                     + col.astype("u1").tobytes())
    idx = np.argwhere(surface_voxels(grid.label_np))
    (out / "voxels.bin").write_bytes(idx.astype("<i2").tobytes() + grid.label_np[tuple(idx.T)].astype("<i2").tobytes())
    runs = {}
    for r in RUNS:
        d = HERE.parent / "results" / "runs" / f"{scene.replace('_', '')}_{r}"
        if not (d / "metrics.json").exists():
            continue
        z = dict(np.load(d / "sqs.npz"))
        m = json.loads((d / "metrics.json").read_text())
        nv, nf, count = mesh(z, np.array([norm[k] for k in z["instance"]]), out / f"{r}.bin")
        iou = {int(z["instance"][i]): row["iou"] for i, row in enumerate(m["per_instance"])}
        runs[r] = {"mesh": [nv, nf], "count": count, "summary": m["summary"], "iou": iou}
    meta = {"points": len(g["pos"]), "voxels": len(idx), "lo": grid.lo_np.tolist(), "res": grid.res,
            "names": g["names"], "runs": runs}
    (out / "scene.json").write_text(json.dumps(meta))
    print(f"{scene}: {meta['points']} points, {meta['voxels']} voxels, runs {list(runs)}")


if __name__ == "__main__":
    for scene in sys.argv[1:] or SCENES:
        export(scene)
    names = sorted(p.parent.name for p in (HERE / "data").glob("*/scene.json"))
    (HERE / "data" / "index.json").write_text(json.dumps(names))
