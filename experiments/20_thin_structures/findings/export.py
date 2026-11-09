"""Data of findings/index.html: every snapshot of code/diagnose.py and the trim test of code/trim_test.py (not versioned).

    python ../code/diagnose.py --out ../results/diagnose.json --fits data/fits
    python ../code/trim_test.py data/fits
    uv run python export.py
    python -m http.server -d experiments/20_thin_structures/findings 8021      -> http://localhost:8021

data/meta.json holds the numbers. data/<fit>.bin: vertices of K primitives for all snapshots (f32), then per snapshot and
surface sample of seat + back whether it is covered (u8). data/samples.bin: those samples (f32). data/back.bin: the
vertices of the full and the trimmed fit, the surface samples lost by trimming, the free query points in the back sheet.
"""

import json
import sys
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "code"))
sys.path.insert(0, str(HERE.parents[1] / "19_sq_count_sweep" / "viewer"))
from diagnose import CONFIGS, regions  # noqa: E402
from export import MESHER  # noqa: E402
from variants import DEV, TARGET, load_chair, scaled_distance  # noqa: E402

OUT = HERE / "data"
FITS = OUT / "fits"
RES = 20
TITLES = {"voxel_radial": "Voxel, radial", "iou_scaled": "IoU, corrected", "iou_radial": "IoU, radial",
          "iou_scaled_uncapped": "IoU, corrected, no cap", "iou_radial_uncapped": "IoU, radial, no cap",
          "voxel_radial_capped": "Voxel, radial, with cap", "iou_scaled_surface10": "IoU, corrected, surface × 10",
          "iou_radial_surface10": "IoU, radial, surface × 10"}


def vertices(z):
    return np.stack([MESHER._superquadric_mesh(z["scale"][j], z["exponents"][j], z["rotation"][j], z["translation"][j],
                                               np.zeros(2), np.zeros(6), RES)[0] for j in range(len(z["scale"]))])


def distances(z, x):
    R, t, s, e = [torch.tensor(z[k], device=DEV, dtype=torch.float32) for k in ("rotation", "translation", "scale", "exponents")]
    return scaled_distance((x[None] - t[:, None]) @ R, s[:, None], e[:, 0, None], e[:, 1, None])


def main():
    chair = load_chair(TARGET)
    q, occ = torch.tensor(chair["q"], device=DEV), torch.tensor(chair["occ"], device=DEV)
    parts = regions()
    seat, back = parts["seat"][::3], parts["back"][::2]
    samples = np.concatenate([seat, back])
    x = torch.tensor(samples, device=DEV)
    (OUT / "samples.bin").write_bytes(samples.astype("<f4").tobytes())
    trace = json.loads((HERE.parent / "results" / "diagnose.json").read_text())
    meta = {"res": RES, "steps": None, "seat": len(seat), "back": len(back), "fits": {}, "trim": json.loads((FITS / "trim.json").read_text())}
    for name, title in TITLES.items():
        z = dict(np.load(FITS / f"{name}_trace.npz"))
        S, K = z["scale"].shape[:2]
        blobs, rows = [], []
        steps = sorted(trace[name]["trace"], key=int)
        for s, step in enumerate(steps):
            state = {key: value[s] for key, value in z.items()}
            blobs.append(vertices(state).astype("<f4").tobytes())
            inside = distances(state, q) < 0
            union = inside.any(0)
            row = trace[name]["trace"][step]
            rows.append({"step": int(step), "half": row["half"], "forces": row["forces"], "query": inside.sum(1).tolist(),
                         "occupied": (inside & occ).sum(1).tolist(), "iou": ((union & occ).sum() / (union | occ).sum()).item(),
                         "precision": ((union & occ).sum() / union.sum().clamp_min(1)).item(), "recall": ((union & occ).sum() / occ.sum()).item()})
        covered = [(distances({key: value[s] for key, value in z.items()}, x).amin(0) < 0.01).cpu().numpy().astype("u1").tobytes() for s in range(S)]
        (OUT / f"{name}.bin").write_bytes(b"".join(blobs + covered))
        meta["fits"][name] = {"title": title, "K": K, "rows": rows, "config": CONFIGS[name][:2]}
        meta["steps"] = [int(step) for step in steps]
    full, cut = dict(np.load(FITS / "voxel_radial.npz")), dict(np.load(FITS / "voxel_radial_trimmed.npz"))
    meta["trim"] |= {"lost": len(cut["lost"]), "air": len(cut["air"]), "K": len(full["scale"])}
    (OUT / "back.bin").write_bytes(b"".join(a.astype("<f4").tobytes() for a in (vertices(full), vertices(cut), cut["lost"], cut["air"])))
    faces = MESHER._superquadric_mesh(np.ones(3), np.ones(2), np.eye(3), np.zeros(3), np.zeros(2), np.zeros(6), RES)[1]
    meta["faces"] = faces.ravel().tolist()
    (OUT / "meta.json").write_text(json.dumps(meta))
    print(len(TITLES), "fits,", len(meta["steps"]), "snapshots")


if __name__ == "__main__":
    main()
