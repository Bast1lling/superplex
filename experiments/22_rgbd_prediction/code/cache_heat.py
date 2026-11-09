"""Alternative existence heatmaps (--heat in train.py) from the cached patch owners (`cache_assign.py`).

    uv run python cache_heat.py [--only names]      -> FRAMES/<scene>/<frame>_heat.npz: linear, geodesic, nearest
                                                       (42 * 75,) float32 each

All keep one positive (heat = 1) per visible SQ at the same peak patch as `dense.heatmap` (its innermost patch) and
the same size scale sigma = max(0.5, sqrt(patches) / 4); they differ in the falloff around it:
- gauss (dense.heatmap, the cached "heat"): exp(-d^2 / 2 sigma^2), d = Euclidean patch distance to the SQ's peak, only
  on the SQ's own patches;
- linear: max(0, 1 - d / (3 sigma)), own patches only (another falloff shape, about the same support);
- geodesic: the Gaussian with d = shortest path within the SQ's own patches (8-neighbourhood, diagonal steps sqrt 2): a
  region split by an occluder, or bent around one, is not "near" across the gap;
- nearest: every patch (also unowned ones and other SQs' patches) gets the Gaussian of the nearest peak of any SQ,
  not clipped to regions (heat as a function of the distance to the closest center patch).
"""

import sys

import numpy as np
from scipy.ndimage import distance_transform_edt
from skimage.graph import MCP_Geometric

import train as T
from model import GH, GW


def peaks_and_sigma(own):
    """-> [(SQ, (row, col) of its innermost patch, sigma, region mask)] for every SQ owning patches."""
    out = []
    for s in np.unique(own[own >= 0]):
        m = own == s
        c = np.unravel_index(np.argmax(distance_transform_edt(np.pad(m, 1))[1:-1, 1:-1]), m.shape)
        out.append((s, c, max(0.5, np.sqrt(m.sum()) / 4), m))
    return out


def heatmaps(owner):
    own = owner.reshape(GH, GW)
    gy, gx = np.indices((GH, GW))
    lin, geo = np.zeros((GH, GW), np.float32), np.zeros((GH, GW), np.float32)
    near_d = np.full((GH, GW), np.inf)
    near = np.zeros((GH, GW), np.float32)
    pk = peaks_and_sigma(own)
    for s, c, sigma, m in pk:
        d = np.sqrt((gy - c[0]) ** 2 + (gx - c[1]) ** 2)
        lin = np.maximum(lin, np.clip(1 - d / (3 * sigma), 0, None) * m)
        cost = np.where(m, 1.0, np.inf)  # paths only through the SQ's own patches
        dg, _ = MCP_Geometric(cost, fully_connected=True).find_costs([c])
        geo = np.maximum(geo, np.where(m & np.isfinite(dg), np.exp(-np.nan_to_num(dg, posinf=1e9) ** 2 / (2 * sigma ** 2)), 0))
        closer = d < near_d
        near_d = np.where(closer, d, near_d)
        near = np.where(closer, np.exp(-d ** 2 / (2 * sigma ** 2)), near)
    for _, c, _, _ in pk:
        lin[c] = geo[c] = near[c] = 1.0
    return {"linear": lin.reshape(-1), "geodesic": geo.reshape(-1), "nearest": near.reshape(-1).astype(np.float32)}


def main():
    items = sorted(p for s in T.TRAIN + T.TEST for p in (T.FRAMES / s).glob("*_assign.npz"))
    if "--only" in sys.argv:
        names = set(sys.argv[sys.argv.index("--only") + 1].split(","))
        items = [p for p in items if f"{p.parent.name}_{p.name[:6]}" in names]
    for i, p in enumerate(items):
        h = heatmaps(np.load(p)["owner"])
        np.savez(str(p).replace("_assign.npz", "_heat.npz"), **h)
        if i % 200 == 0:
            print(i, p.parent.name, p.name[:6], flush=True)
    print(len(items), "frames")


if __name__ == "__main__":
    main()
