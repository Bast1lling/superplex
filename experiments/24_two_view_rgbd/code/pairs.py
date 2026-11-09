"""Pair table and statistics (step 1 of PLAN.md).

    uv run python pairs.py          -> results/pairs.json (overlap of every candidate pair), results/pair_stats.json

Candidates of a cached frame i (every 10th frame): i +- d for d in DELTAS (trajectory frames). Overlap(i, j) = share of
view i's pixels with depth (every 2nd pixel) that are seen by view j: they project into j's image in front of the
camera (z > 0.1 m) and agree with j's depth (|z_j - depth_j| < 3 cm + 3 %). Not symmetric.

Statistics per delta: overlap distribution; for the pairs used in evaluation (overlap closest to 0.5 in [0.2, 0.8]):
targets visible in view 0 / only in view 1, and how much view 1 adds for view 0's partly visible targets (f0 < 0.5):
share also visible in view 1, share with f1 > f0, mean max(f0, f1).
"""

import json
from collections import defaultdict

import numpy as np

import data as D
from common import CX, CY, FX, FY, H, W

DELTAS = (10, 20, 40, 80, 160)


def overlap(vi, vj):
    d = vi["depth"].astype(np.float32)[::2, ::2]
    v, u = np.nonzero(d > 0)
    z = d[v, u]
    v, u = v * 2, u * 2
    x = np.stack([(u - CX) / FX * z, (v - CY) / FY * z, z, np.ones_like(z)], -1)
    Tji = np.linalg.inv(vj["c2w"]) @ vi["c2w"]
    y = x @ Tji.T
    zj = y[:, 2]
    ok = zj > 0.1
    uj = np.round(FX * y[:, 0] / np.maximum(zj, 1e-3) + CX).astype(int)
    vj_ = np.round(FY * y[:, 1] / np.maximum(zj, 1e-3) + CY).astype(int)
    ok &= (uj >= 0) & (uj < W) & (vj_ >= 0) & (vj_ < H)
    dj = np.zeros_like(zj)
    dj[ok] = vj["depth"].astype(np.float32)[vj_[ok], uj[ok]]
    ok &= (dj > 0) & (np.abs(zj - dj) < 0.03 + 0.03 * dj)
    return float(ok.mean()) if len(z) else 0.0


def main():
    table, stats = {}, defaultdict(list)
    for s in D.TRAIN + D.TEST:
        ids = D.frame_ids(s)
        idset = set(ids)
        table[s] = {}
        for f in ids:
            vi = D.load_view(s, f)
            row = {}
            for d in DELTAS:
                for j in (f - d, f + d):
                    if j in idset:
                        o = overlap(vi, D.load_view(s, j))
                        row[j] = o
                        stats[f"overlap_d{d}"].append(o)
            table[s][f] = row
        print(s, len(ids), flush=True)
    D.PAIRS.parent.mkdir(parents=True, exist_ok=True)
    D.PAIRS.write_text(json.dumps(table))
    out = {k: {"mean": float(np.mean(v)), "p10": float(np.percentile(v, 10)), "p50": float(np.median(v)),
               "p90": float(np.percentile(v, 90)), "in_range": float(np.mean([D.OVERLAP[0] <= x <= D.OVERLAP[1] for x in v]))}
           for k, v in stats.items()}
    # visibility gain of the evaluation pairs
    for s in D.TRAIN + D.TEST:
        ds = D.Views([s], views=2, partner="fixed")
        g = defaultdict(list)
        for i in range(len(ds)):
            sc, f, good = ds.items[i]
            j = ds.pick(f, good)
            g["delta"].append(abs(j - f))
            g["overlap"].append(good[j])
            t = ds[i]["targets"]
            vis, fv = t["vis"], t["f"]
            g["targets_union"].append(len(vis))
            g["targets_v0"].append(int(vis[:, 0].sum()))
            g["targets_v1_only"].append(int((~vis[:, 0] & vis[:, 1]).sum()))
            part = vis[:, 0] & (fv[:, 0] < 0.5)
            if part.any():
                g["partial_v0_share"].append(float(part.sum() / max(vis[:, 0].sum(), 1)))
                g["partial_also_v1"].append(float(vis[part, 1].mean()))
                g["partial_f1_gt_f0"].append(float((fv[part, 1] > fv[part, 0]).mean()))
                g["partial_f0"].append(float(fv[part, 0].mean()))
                g["partial_fmax"].append(float(fv[part].max(1).mean()))
        out[f"eval_pairs_{s}"] = {k: float(np.mean(v)) for k, v in g.items()}
        print(s, out[f"eval_pairs_{s}"], flush=True)
    (D.EXP / "results" / "pair_stats.json").write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
