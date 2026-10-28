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

METHODS = ("surface", "volume", "volume_overlap", "instances", "adaptive",
           "superflex_superflex", "superflex_unsup_robust", "superflex_robust")
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
    """Tapered / bent primitives (SuperFlex) meshed with SuperFlex's own mesher -> vertices f32, faces u32,
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
            sqs[m] = {"count": len(z["trans"])}
            if "tapering" in z:
                sqs[m]["mesh"] = deformed_mesh(z, out / f"{m}.bin")
            else:
                sqs[m] |= {k: v.round(5).tolist() for k, v in z.items()}
    meta = {"points": len(pts["pos"]), "voxels": len(idx), "lo": grid.lo_np.tolist(), "res": grid.res, "sqs": sqs}
    (out / "scene.json").write_text(json.dumps(meta))
    print(f"{name}: {meta['points']} points, {meta['voxels']} voxels, methods {list(sqs)}")


if __name__ == "__main__":
    for scene in data.cli():
        export(scene)
    names = sorted(p.parent.name for p in (HERE / "data").glob("*/scene.json"))
    (HERE / "data" / "index.json").write_text(json.dumps(names))
