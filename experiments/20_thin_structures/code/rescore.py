"""Scores the saved test decompositions of experiment 19 against the fixed occupancy, without fitting them again.

    python rescore.py --out ../results/rescore.json [--labels occupancies_fusion] [--limit 64]

--labels occupancies_fusion must reproduce the stored IoUs, which checks that the primitives survive the way back in.
One process per kind of fit would be cleaner for `torch.compile`, but scoring is a single forward pass per batch.
"""

import argparse
import json
from pathlib import Path

import numpy as np
from variants import FITS  # noqa: I001, sets the import path
from fit_superflex import PredictionHandler, fit
from fit_sqs import BATCH, DATA

NAMES = ["superflex_opt", "superflex_pred", "superflex_opt_plain", "superflex_pred_plain"] + [f"k{k:02d}" for k in range(1, 10)]


def load(model_id, labels):
    z = np.load(DATA / model_id / "points.npz")
    q = z["points"].astype(np.float32)
    return {"surf": np.load(DATA / model_id / "pointcloud_4096.npz")["points"].astype(np.float32), "q": q,
            "occ": np.unpackbits(z[labels])[:len(q)].astype(bool)}


def handler_of(z, part, surf):
    B, K = z["scale"][part].shape[:2]
    deformed = "bending_k" in z and "plain" not in z["name"]
    tapering = z["tapering"][part] if deformed else np.zeros((B, K, 2))
    bending = np.stack([z["bending_k"][part], z["bending_alpha"][part]], -1).reshape(B, K, 6) if deformed else np.zeros((B, K, 6))
    exist = z["exist"][part] if "exist" in z else np.ones((B, K), bool)
    return PredictionHandler({"names": np.arange(B), "pc": surf, "assign_matrix": np.zeros((B, 1, K)), "scale": z["scale"][part],
                              "translation": z["translation"][part], "rotation": z["rotation"][part], "exponents": z["exponents"][part],
                              "exist": exist.astype(float), "tapering": tapering, "bending": bending}), deformed


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--labels", default="occupancies")
    ap.add_argument("--limit", type=int)
    args = ap.parse_args()
    fits = {name: dict(np.load(FITS / f"{name}.npz")) | {"name": name} for name in NAMES}
    ids = [str(i) for i in fits["k01"]["ids"]][:args.limit]
    scores = {name: [] for name in NAMES}
    for start in range(0, len(ids), BATCH):
        chairs = [load(i, args.labels) for i in ids[start:start + BATCH]]
        surf = np.stack([c["surf"] for c in chairs])
        part = slice(start, start + len(chairs))
        for name, z in fits.items():
            handler, deformed = handler_of(z, part, surf)
            scores[name] += fit(handler, chairs, surf, not deformed, 0)["iou"].tolist()
        print(start + len(chairs), flush=True)
    summary = {name: {"stored": float(fits[name]["iou"][:len(ids)].mean()), "rescored": float(np.mean(scores[name]))} for name in NAMES}
    args.out.write_text(json.dumps({"ids": ids, "summary": summary, "scores": scores}))
    for name, row in summary.items():
        print(f"{name:22s} stored {row['stored']:.3f}  rescored {row['rescored']:.3f}")


if __name__ == "__main__":
    main()
