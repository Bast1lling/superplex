"""Every variant of code/variants.py refit on the target chair, next to its decompositions of experiment 19.

    uv run python export.py
    python -m http.server -d experiments/20_thin_structures/viewer 8020      -> http://localhost:8020

Same files as experiment 19's viewer (index.html is a link to it), for one chair and at a higher mesh resolution.
"""

import json
import sys
from pathlib import Path

import numpy as np
import torch
from omegaconf import OmegaConf

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(ROOT / "experiments" / "19_sq_count_sweep" / "viewer"))
sys.path.insert(0, str(HERE.parent / "code"))
import export as base  # noqa: E402
from variants import (CHECKPOINT, DATA, DEV, TARGET, PredictionHandler, SuperFlex, fit, kmeans_params, load_chair,  # noqa: E402
                      predict)

OUT = HERE / "data"
base.RES = 32
STARTS = {"kmeans": "k-means 10", "network": "network"}


def configs():
    for start in STARTS:
        for distance in ("radial", "scaled"):
            yield {"start": start, "loss": "iou", "distance": distance}
            for tau, w_free in ((0.02, 1.0), (0.02, 5.0), (0.005, 1.0), (0.005, 5.0)):
                yield {"start": start, "loss": "vox", "distance": distance, "tau": tau, "w_free": w_free}
        for tau, w_free in ((0.005, 20.0), (0.005, 50.0), (0.002, 20.0), (0.002, 50.0)):
            yield {"start": start, "loss": "vox", "distance": "radial", "tau": tau, "w_free": w_free}


def title(cfg):
    loss = "IoU loss" if cfg["loss"] == "iou" else f"voxel {cfg['tau']} × {cfg['w_free']:.0f}"
    return f"{STARTS[cfg['start']]}, {cfg['distance']}, {loss}"


def main():
    chair = load_chair(TARGET)
    surf = chair["surf"][None]
    model = SuperFlex(OmegaConf.load(CHECKPOINT / "config.yaml").superflex).to(DEV)
    model.load_state_dict(torch.load(CHECKPOINT / "ckpt.pt", map_location=DEV, weights_only=False)["model_state_dict"])
    model.eval()

    b = int(np.flatnonzero(base.FITTED["k01"]["ids"] == TARGET)[0])
    fits = {title_: {key: value[b:b + 1] for key, value in z.items()} for title_, z in zip(base.TITLES.values(), base.FITTED.values())}
    for cfg in configs():
        handler = predict(model, surf, plain=True) if cfg["start"] == "network" else PredictionHandler(kmeans_params([chair], 10))
        fits[title(cfg)] = fit([chair], handler, cfg)
        print(f"{title(cfg)}: IoU {fits[title(cfg)]['iou'][0]:.3f}", flush=True)

    (OUT / "chairs").mkdir(parents=True, exist_ok=True)
    (OUT / "rgb").unlink(missing_ok=True)
    (OUT / "rgb").symlink_to(DATA)
    parts = [np.load(DATA / TARGET / "pointcloud_4096.npz")["points"]]
    for z in fits.values():
        parts += [base.primitive(z, 0, j)[0] for j in np.flatnonzero(base.exists(z)[0])]
    (OUT / "chairs" / f"{TARGET}.bin").write_bytes(np.concatenate(parts).astype("<f4").tobytes())
    meta = {"res": base.RES, "ids": [TARGET], "faces": base.primitive(next(iter(fits.values())), 0, 0)[1].ravel().tolist(),
            "methods": [{"name": name, "title": name, "mean": float(z["iou"][0])} for name, z in fits.items()],
            "iou": [[round(float(z["iou"][0]), 4) for z in fits.values()]],
            "count": [[int(base.exists(z)[0].sum()) for z in fits.values()]]}
    (OUT / "meta.json").write_text(json.dumps(meta))


if __name__ == "__main__":
    main()
