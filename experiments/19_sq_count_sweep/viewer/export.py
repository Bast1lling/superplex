"""Exports the decompositions of every test chair for index.html into viewer/data/ (not versioned).

    uv run python export.py
    python -m http.server -d experiments/19_sq_count_sweep/viewer 8019      -> http://localhost:8019

data/meta.json: chair ids, decompositions (with their mean IoU), per chair and decomposition the IoU and primitive
count, and the faces of one primitive. data/chairs/<id>.bin: 4096 surface points, then the vertices of all primitives
of all decompositions in the order of meta.json (f32). Primitives are meshed with SuperFlex's mesher.
data/rgb links to the dataset's folders for the renderings.
"""

import json
import sys
from multiprocessing import Pool
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[2] / "papers" / "superflex"))
from superflex.utils.predictions_handler import PredictionHandler  # noqa: E402

ROOT = Path("/home/sebastian/data/ShapeNet-Chairs-RGBD")
FITS = ROOT / "sq_fits" / "test"
OUT = HERE / "data"
RES = 16
RESCORED = HERE.parents[1] / "20_thin_structures" / "results" / "rescore.json"  # scores against the fixed occupancy
SUPERFLEX = {"superflex_opt": "SuperFlex optimized", "superflex_pred": "SuperFlex prediction",
             "superflex_opt_plain": "SuperFlex optimized, plain", "superflex_pred_plain": "SuperFlex prediction, plain"}
TITLES = SUPERFLEX | {f.stem: f"k-means K={int(f.stem[1:])}" for f in sorted(FITS.glob("k[0-9][0-9].npz"))}
FITTED = {name: dict(np.load(FITS / f"{name}.npz")) for name in TITLES}
MESHER = PredictionHandler.__new__(PredictionHandler)  # only its mesh methods are used


if RESCORED.exists():
    for name, iou in json.loads(RESCORED.read_text())["scores"].items():
        FITTED[name]["iou"] = np.array(iou)


def exists(z):
    return z["exist"] if "exist" in z else np.ones(z["scale"].shape[:2], bool)


def primitive(z, b, j):
    """Vertices and faces. Plain files and the sweep have no deformations (a plain `bending_k` is an unused leftover)."""
    tapering, bending = np.zeros(2), np.zeros(6)
    if "tapering" in z and z["tapering"][b, j].any():
        tapering = z["tapering"][b, j]
        bending = np.stack([z["bending_k"][b, j], z["bending_alpha"][b, j]], -1).ravel()
    return MESHER._superquadric_mesh(z["scale"][b, j], z["exponents"][b, j], z["rotation"][b, j], z["translation"][b, j],
                                     tapering, bending, RES)


def export_chair(args):
    b, model_id = args
    parts = [np.load(ROOT / "03001627" / model_id / "pointcloud_4096.npz")["points"]]
    for z in FITTED.values():
        parts += [primitive(z, b, j)[0] for j in np.flatnonzero(exists(z)[b])]
    (OUT / "chairs" / f"{model_id}.bin").write_bytes(np.concatenate(parts).astype("<f4").tobytes())


def main():
    ids = [str(i) for i in FITTED["superflex_opt"]["ids"]]
    assert all((z["ids"] == ids).all() for z in FITTED.values())
    (OUT / "chairs").mkdir(parents=True, exist_ok=True)
    (OUT / "rgb").unlink(missing_ok=True)
    (OUT / "rgb").symlink_to(ROOT / "03001627")
    with Pool(12) as pool:
        pool.map(export_chair, list(enumerate(ids)), chunksize=8)
    faces = primitive(FITTED["k01"], 0, 0)[1]
    meta = {"res": RES, "ids": ids, "faces": faces.ravel().tolist(),
            "methods": [{"name": name, "title": title, "mean": float(FITTED[name]["iou"].mean())}
                        for name, title in TITLES.items()],
            "iou": np.round(np.stack([z["iou"] for z in FITTED.values()], 1), 4).tolist(),
            "count": np.stack([exists(z).sum(1) for z in FITTED.values()], 1).tolist()}
    (OUT / "meta.json").write_text(json.dumps(meta))
    print(len(ids), "chairs,", len(TITLES), "decompositions")


if __name__ == "__main__":
    main()
