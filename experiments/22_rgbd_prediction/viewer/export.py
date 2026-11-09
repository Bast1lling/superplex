"""Exports GT points, GT voxels, the occupancy the fits see and the target primitives (both occupancy variants) of every scene for index.html into viewer/data/<scene>/
(not versioned). Reuses experiment 16's export (same binary layout).

    uv run python export.py [scene ...]
    python -m http.server -d experiments/22_rgbd_prediction/viewer 8022      -> http://localhost:8022
"""

import json
import sys
from pathlib import Path

import numpy as np
from scipy.ndimage import distance_transform_edt

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1] / "16_superflex_objective" / "viewer"))
sys.path.insert(0, str(HERE.parent / "code"))
from export import gt, gt_grid, instance_data, load_points, mesh, surface_voxels  # noqa: E402
from fit_targets import SCENES  # noqa: E402
from fill_grid import fill  # noqa: E402
from occupancy import load as load_occupancy  # noqa: E402
from sfopt import UNKNOWN  # noqa: E402


RUNS = {"filled": "targets", "solid": "targets_solid", "winding": "targets_winding"}  # viewer name -> results folder
WINDING = 0.5  # UNKNOWN voxels with a per-instance winding number >= this become that instance


def instance_shell(occ, k):
    """Voxels of instance k with a 6-neighbour outside k: grid indices, label k, 1 = filled by the winding number."""
    idx = np.stack(np.unravel_index(occ.occ[k], occ.dims), 1)
    a = idx.min(0) - 1
    box = np.zeros(idx.max(0) - a + 2, bool)
    box[tuple((idx - a).T)] = True
    shell = solid_shell(box)
    keep = shell[tuple((idx - a).T)]
    return idx[keep], np.full(keep.sum(), k), ~np.isin(occ.occ[k][keep], occ.surf[k])


def solid_shell(solid):
    inner = solid.copy()
    for ax in range(3):
        for sh in (-1, 1):
            inner &= np.roll(solid, sh, ax)
    return solid & ~inner


def export(scene):
    run = HERE.parent / "results" / "targets" / scene
    if not (run / "metrics.json").exists():
        return
    g, grid = gt(scene), gt_grid(scene)
    norm = {d["k"]: 2 * np.abs(d["surf"]).max() for d in instance_data(scene, grid)}  # SuperFlex's normalization scale
    out = HERE / "data" / scene
    out.mkdir(parents=True, exist_ok=True)
    col = load_points(scene)["col"]
    (out / "points.bin").write_bytes(g["pos"].astype("<f4").tobytes() + g["inst"].astype("<i2").tobytes()
                                     + col.astype("u1").tobytes())
    idx = np.argwhere(surface_voxels(grid.label_np))
    (out / "voxels.bin").write_bytes(idx.astype("<i2").tobytes() + grid.label_np[tuple(idx.T)].astype("<i2").tobytes())
    # what the fit sees (experiment 16's instance_data): UNKNOWN voxels take the nearest solid voxel's instance
    L = grid.label_np
    _, near = distance_transform_edt(L < 0, return_indices=True)
    filled = np.where(L == UNKNOWN, L[tuple(near)], L)
    grids = {"filled": filled, "grown": fill(L)[0]}  # grown: code/fill_grid.py
    wfile = HERE.parent / "results" / "large_files" / "winding" / f"{scene}.npz"  # code/winding.py
    if wfile.exists():
        w = np.load(wfile)
        ok = w["w_inst"] >= WINDING
        grids["winding"] = L.copy()
        grids["winding"][tuple(w["idx"][ok].T)] = w["label"][ok]
    counts = {}
    ofile = HERE.parent / "results" / "large_files" / "occupancy" / f"{scene}.npz"  # code/occupancy.py
    if ofile.exists():
        occ = load_occupancy(scene)
        parts = [instance_shell(occ, k) for k in occ.occ]
        oidx, olab, oflag = (np.concatenate(x) for x in zip(*parts))
        (out / "voxels_instance.bin").write_bytes(oidx.astype("<i2").tobytes() + olab.astype("<i2").tobytes()
                                                  + oflag.astype("u1").tobytes())
        counts["instance"] = len(oidx)
    for name, F in grids.items():
        fidx = np.argwhere(surface_voxels(F))
        (out / f"voxels_{name}.bin").write_bytes(fidx.astype("<i2").tobytes() + F[tuple(fidx.T)].astype("<i2").tobytes()
                                                 + (L[tuple(fidx.T)] == UNKNOWN).astype("u1").tobytes())
        counts[name] = len(fidx)
    runs = {}
    for name, folder in RUNS.items():
        d = HERE.parent / "results" / folder / scene
        if not (d / "metrics.json").exists():
            continue
        z = dict(np.load(d / "sqs.npz"))
        m = json.loads((d / "metrics.json").read_text())
        nv, nf, count = mesh(z, np.array([norm[k] for k in z["instance"]]), out / f"{name}.bin")
        iou = {int(z["instance"][i]): row["iou"] for i, row in enumerate(m["per_instance"])}
        runs[name] = {"mesh": [nv, nf], "count": count, "summary": m["summary"], "iou": iou}
    meta = {"points": len(g["pos"]), "voxels": len(idx), "occupancy": counts, "lo": grid.lo_np.tolist(), "res": grid.res, "names": g["names"],
            "runs": runs}
    (out / "scene.json").write_text(json.dumps(meta))
    print(scene, {k: round(r["summary"]["viou"], 3) for k, r in runs.items()})


if __name__ == "__main__":
    for scene in sys.argv[1:] or SCENES:
        export(scene)
    names = sorted(p.parent.name for p in (HERE / "data").glob("*/scene.json"))
    (HERE / "data" / "index.json").write_text(json.dumps(names))
