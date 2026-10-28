"""Parts from approximate convex decomposition (CoACD, Wei et al. 2022): every instance's voxels are meshed
(marching cubes), decomposed into convex hulls by CoACD (concavity threshold `cx`), and each voxel is assigned to
the hull containing it (else the nearest hull). The number of parts adapts to the shape's concavity, not to how
well an SQ fits; decomp.py then fits one SQ per part (optionally merging first).
"""

from concurrent.futures import ProcessPoolExecutor

import numpy as np
from scipy.spatial import Delaunay, cKDTree


def convex_parts(vox, threshold, min_vox):
    import coacd
    from skimage.measure import marching_cubes
    if len(vox) < 4 * min_vox:
        return [np.arange(len(vox))]
    a = vox.min(0) - 2
    grid = np.zeros(vox.max(0) - a + 3, np.float32)
    grid[tuple((vox - a).T)] = 1
    v, f, _, _ = marching_cubes(grid, 0.5)
    coacd.set_log_level("error")
    hulls = coacd.run_coacd(coacd.Mesh(v, f), threshold=threshold, max_convex_hull=64, preprocess_mode="off")
    if len(hulls) <= 1:
        return [np.arange(len(vox))]
    x = (vox - a).astype(float)
    lab = np.full(len(vox), -1)
    for h, (hv, _) in enumerate(hulls):
        try:
            inside = Delaunay(hv).find_simplex(x) >= 0
        except Exception:
            continue
        lab[(lab < 0) & inside] = h
    cent = np.stack([hv.mean(0) for hv, _ in hulls])
    if (lab < 0).any():
        lab[lab < 0] = cKDTree(cent).query(x[lab < 0])[1]
    out = [np.nonzero(lab == h)[0] for h in np.unique(lab)]
    big = [o for o in out if len(o) >= min_vox] or [np.arange(len(vox))]
    small = [o for o in out if len(o) < min_vox]
    if small and len(big) < len(out):
        bc = np.stack([x[b].mean(0) for b in big])
        for o in small:
            j = int(np.argmin(((bc - x[o].mean(0)) ** 2).sum(1)))
            big[j] = np.concatenate([big[j], o])
    return big


def _job(args):
    return convex_parts(*args)


def decompose(parts, group, threshold, min_vox, workers=12):
    with ProcessPoolExecutor(workers) as ex:
        res = list(ex.map(_job, [(p, threshold, min_vox) for p in parts], chunksize=1))
    out, grp = [], []
    for p, g, r in zip(parts, group, res):
        out += [p[i] for i in r]
        grp += [g] * len(r)
    return out, grp
