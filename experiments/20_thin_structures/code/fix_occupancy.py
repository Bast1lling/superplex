"""Adds the sheets the depth fusion lost: every query point within TOLERANCE of the 100k surface samples is occupied.
The fusion's labels stay in points.npz as `occupancies_fusion`. Chairs that have them already are skipped.

    python fix_occupancy.py
"""

import json
from concurrent.futures import ThreadPoolExecutor

import numpy as np
from scipy.spatial import cKDTree
from variants import DATA

TOLERANCE = 0.005


def fix(folder):
    path = folder / "points.npz"
    z = dict(np.load(path))
    if "occupancies_fusion" in z:
        return None
    n = len(z["points"])
    fusion = np.unpackbits(z["occupancies"])[:n].astype(bool)
    near = cKDTree(np.load(folder / "pointcloud.npz")["points"].astype(np.float32)).query(z["points"].astype(np.float32))[0]
    fixed = fusion | (near <= TOLERANCE)
    tmp = folder / "points.tmp.npz"
    np.savez_compressed(tmp, **z | {"occupancies": np.packbits(fixed), "occupancies_fusion": z["occupancies"]})
    tmp.rename(path)
    return float(fusion.mean()), float(fixed.mean())


def main():
    folders = [p for p in DATA.iterdir() if p.is_dir()]
    with ThreadPoolExecutor(12) as pool:
        rows = [r for r in pool.map(fix, folders) if r]
    print(len(rows), "chairs fixed of", len(folders))
    if rows:
        print("occupied fraction, mean before / after:", np.mean(rows, 0).round(4))
    (DATA.parent / "occupancy_fix.json").write_text(json.dumps({"tolerance": TOLERANCE, "source": "fusion | near surface samples"}))


if __name__ == "__main__":
    main()
