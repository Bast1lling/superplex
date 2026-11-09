"""How much of the IoU gap between SuperFlex's prediction and its optimization is the thin shell around the surface?

    python metric_gap.py --out ../results/metric_gap.json

Plain fits only (`union_distance` has no tapering or bending), every test chair: IoU, IoU ignoring the shell, surface
recall, see variants.evaluate.
"""

import argparse
import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree
import variants
from variants import DATA, FITS, evaluate, load_chair


def load(model_id):
    chair = load_chair(model_id)
    chair["surface"] = np.load(DATA / model_id / "pointcloud.npz")["points"].astype(np.float32)[::5]
    chair["far"] = cKDTree(chair["surface"]).query(chair["q"])[0] > variants.TOLERANCE
    return chair


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--tolerance", type=float, default=variants.TOLERANCE)
    args = ap.parse_args()
    variants.TOLERANCE = args.tolerance
    fits = {name: dict(np.load(FITS / f"{name}.npz")) for name in ("superflex_pred_plain", "superflex_opt_plain", "k08", "k09")}
    ids = [str(i) for i in fits["k09"]["ids"]]
    for r in fits.values():
        r.setdefault("exist", np.ones(r["scale"].shape[:2], bool))
    rows = {name: [] for name in fits}
    with ThreadPoolExecutor(12) as pool:
        for start in range(0, len(ids), 64):
            chairs = list(pool.map(load, ids[start:start + 64]))
            for name, r in fits.items():
                part = {key: value[start:start + 64] for key, value in r.items() if key != "ids"}
                rows[name] += evaluate(part, chairs)
            print(start + len(chairs), flush=True)
    args.out.write_text(json.dumps({"ids": ids, "rows": rows}))


if __name__ == "__main__":
    main()
