"""Check every object of the dataset; writes quality.json (per-object stats and the outliers) next to it.

    uv run python experiments/18_chair_rgbd_dataset/code/check_dataset.py /home/sebastian/data/ShapeNet-Chairs-RGBD
"""
import json
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
from PIL import Image

VIEWS, TEXTURES = 8, 4


def check(folder):
    """Fraction of occupied query points, smallest object coverage of any view, and everything that is wrong."""
    problems = []
    try:
        surface = np.load(folder / "pointcloud.npz")
        points = np.load(folder / "points.npz")
        fps = np.load(folder / "pointcloud_4096.npz")
        cameras = np.load(folder / "cameras.npz")
        if len(surface["points"]) != 100_000 or len(points["points"]) != 100_000 or len(fps["points"]) != 4096:
            problems.append("sample count")
        if not (np.isfinite(surface["points"]).all() and np.isfinite(surface["normals"]).all()):
            problems.append("non-finite surface")
        if np.abs(surface["points"]).max() > 0.51:
            problems.append("surface outside the unit cube")
        if len(np.unique(surface["points"], axis=0)) < 20_000:
            problems.append("few distinct surface points")
        occupancy = float(np.unpackbits(points["occupancies"])[:100_000].mean())
        if occupancy == 0:
            problems.append("no occupied point")
        if cameras["c2w"].shape != (VIEWS, 4, 4):
            problems.append("cameras")
        coverage = 1.0
        for v in range(VIEWS):
            mask = np.asarray(Image.open(folder / "depth" / f"v{v:02d}.png")) > 0
            coverage = min(coverage, float(mask.mean()))
            if mask[0].any() or mask[-1].any() or mask[:, 0].any() or mask[:, -1].any():
                problems.append(f"object touches the border in view {v}")
            for t in range(TEXTURES):
                rgb = np.asarray(Image.open(folder / "rgb" / f"t{t:02d}_v{v:02d}.png"))
                if rgb.shape != (512, 512, 3) or (rgb[~mask] != 255).any():
                    problems.append(f"rgb t{t} v{v} does not match the depth mask")
        if coverage < 0.002:
            problems.append("empty view")
        return {"occupancy": occupancy, "coverage": coverage, "problems": problems}
    except Exception as error:
        return {"occupancy": None, "coverage": None, "problems": [f"{type(error).__name__}: {error}"]}


def main():
    root = Path(sys.argv[1])
    expected = sum(((root / "03001627" / f"{s}.lst").read_text().split() for s in ["train", "val", "test"]), [])
    folders = [root / "03001627" / i for i in expected if (root / "03001627" / i).is_dir()]
    with ThreadPoolExecutor(int(sys.argv[2]) if len(sys.argv) > 2 else 8) as pool:
        results = dict(zip((f.name for f in folders), pool.map(check, folders)))
    bad = {k: v["problems"] for k, v in results.items() if v["problems"]}
    occupancy = np.array([v["occupancy"] for v in results.values() if v["occupancy"] is not None])
    summary = {"expected": len(expected), "found": len(folders), "missing": sorted(set(expected) - set(results)),
               "with_problems": len(bad), "occupancy_percentiles_1_50_99": np.percentile(occupancy, [1, 50, 99]).tolist(),
               "thinnest": sorted(results, key=lambda k: results[k]["occupancy"] or 0)[:20]}
    (root / "quality.json").write_text(json.dumps({"summary": summary, "problems": bad, "objects": results}, indent=1))
    print(json.dumps(summary, indent=1))
    print(json.dumps(dict(list(bad.items())[:10]), indent=1))


if __name__ == "__main__":
    main()
