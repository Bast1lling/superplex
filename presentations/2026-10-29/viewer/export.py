"""Exports points, voxels and the fitted SQs of every scene for index.html into viewer/data/<scene>_<res>cm/
(not versioned).

    python export.py [--res 0.02] [scene ...]
    python -m http.server -d presentations/2026-10-29/viewer 8029      -> http://localhost:8029
"""

import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "code"))
import data  # noqa: E402

METHODS = ("superflex_superflex", "superflex_robust", "kmeans_surface", "kmeans_volume", "kmeans_full", "kmeans_full_deform",
           "dbscan_full", "scene_kmeans_net", "scene_kmeans_dbscan")
sys.path.insert(0, str(HERE.parents[2] / "papers" / "superflex"))
from superflex.utils.predictions_handler import PredictionHandler  # noqa: E402


def surface_voxels(label):
    """Solid voxels with a non-solid 6-neighbor (the others are hidden anyway)."""
    solid = label >= 0
    P = np.pad(solid, 1)
    inner = solid.copy()
    for ax in range(3):
        for s in (-1, 1):
            inner &= np.roll(P, s, ax)[1:-1, 1:-1, 1:-1]
    return solid & ~inner


def deformed_mesh(z, path, res=20):
    """Primitives (plain SQs: tapering = bending = 0) meshed with SuperFlex's own mesher -> vertices f32, faces u32,
    instance per vertex i16."""
    mesher = PredictionHandler.__new__(PredictionHandler)  # only its mesh methods are used
    verts, faces, inst, n = [], [], [], 0
    for j in range(len(z["trans"])):
        v, f = mesher._superquadric_mesh(z["scale"][j], z["eps"][j], z["rot"][j], z["trans"][j], z["tapering"][j],
                                         z["bending"][j], res)
        verts.append(v), faces.append(f + n), inst.append(np.full(len(v), z["instance"][j]))
        n += len(v)
    verts, faces, inst = np.concatenate(verts), np.concatenate(faces), np.concatenate(inst)
    path.write_bytes(verts.astype("<f4").tobytes() + faces.astype("<u4").tobytes() + inst.astype("<i2").tobytes())
    return [len(verts), len(faces)]


def majority_instance(z, grid):
    """Per (plain) SQ the GT instance with most solid voxels inside it, -1 if none: colors the label-free fits."""
    idx = np.argwhere(grid.label_np >= 0)
    x, lab = grid.lo_np + idx * grid.res, grid.label_np[tuple(idx.T)]
    out = np.full(len(z["trans"]), -1)
    for j, (t, a, (e1, e2), R) in enumerate(zip(z["trans"], z["scale"], z["eps"], z["rot"])):
        near = np.abs(x - t).max(1) <= a.max()
        p = np.abs((x[near] - t) @ R) / a  # local frame: SuperFlex meshes vertices as R v + t
        inside = (p[:, 0] ** (2 / e2) + p[:, 1] ** (2 / e2)) ** (e2 / e1) + p[:, 2] ** (2 / e1) < 1
        if inside.any():
            out[j] = np.bincount(lab[near][inside]).argmax()
    return out


def export(scene):
    pts = data.load_points(scene)
    grid = data.voxelize(pts)
    name = f"{scene}_{data.RES * 100:g}cm"
    out = HERE / "data" / name
    out.mkdir(parents=True, exist_ok=True)
    (out / "points.bin").write_bytes(pts["pos"].astype("<f4").tobytes() + pts["inst"].astype("<i2").tobytes()
                                     + pts["col"].astype("u1").tobytes())
    idx = np.argwhere(surface_voxels(grid.label_np))
    (out / "voxels.bin").write_bytes(idx.astype("<i2").tobytes() + grid.label_np[tuple(idx.T)].astype("<i2").tobytes())
    sqs = {}
    for m in METHODS:
        if (f := data.result_path(scene, m)).exists():
            z = dict(np.load(f))
            if m.startswith("scene_"):
                z["instance"] = majority_instance(z, grid)
            sqs[m] = {"count": len(z["trans"]), "mesh": deformed_mesh(z, out / f"{m}.bin")}
            if "iou" in z:
                sqs[m]["iou"] = float(z["iou"].mean())
            if m.startswith("superflex_"):  # the network's primitives without its tapering / bending
                plain = z | {"tapering": 0 * z["tapering"], "bending": 0 * z["bending"]}
                sqs[m + "_plain"] = {"count": len(z["trans"]), "mesh": deformed_mesh(plain, out / f"{m}_plain.bin")}
    meta = {"points": len(pts["pos"]), "voxels": len(idx), "lo": grid.lo_np.tolist(), "res": grid.res, "sqs": sqs}
    (out / "scene.json").write_text(json.dumps(meta))
    print(f"{name}: {meta['points']} points, {meta['voxels']} voxels, methods {list(sqs)}")


if __name__ == "__main__":
    for scene in data.cli():
        export(scene)
    names = sorted(p.parent.name for p in (HERE / "data").glob("*/scene.json"))
    (HERE / "data" / "index.json").write_text(json.dumps(names))
