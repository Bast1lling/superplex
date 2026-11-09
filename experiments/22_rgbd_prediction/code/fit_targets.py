"""Target decompositions for the NICE-SLAM scenes: SuperFlex robust per GT instance + its own test-time optimization.

    uv run python fit_targets.py --scene room_0 [--occ filled|solid|winding] [--steps 1000]

Experiment 16's `network / theirs` run with plain SQs (`--plain` there, `../../16_superflex_objective/code/sfopt.py`):
per GT instance 4096 surface points and 100k query points labelled from the 4 cm GT grid, SuperFlex robust as
initialization with its tapering and bending set to 0, then 1000 Adam steps of SuperFlex's objective without them,
best state per instance, evaluated with experiment 12's per-instance vIoU. One process per scene (SuperFlex's torch.compile hits its recompile limit otherwise).
Occupancy of an instance's query points: `filled` (experiment 16) = its solid voxels plus every UNKNOWN voxel (behind a
surface, > 10 cm from it) whose nearest solid voxel is its own, which turns all space behind walls into wall; `solid` =
its solid voxels only, UNKNOWN counts as outside; `winding` = `code/occupancy.py` (voxels with >= 1 of its points plus
those its points enclose, winding number >= 0.5; built first). With `winding` the evaluation uses the same occupancy:
TP = its voxels inside the union, FP = voxels inside the union that no instance occupies (as experiment 12: only free
space counts).
Output: results/targets[_solid|_winding]/<scene>/{sqs.npz, metrics.json}, primitives in the Replica world frame.
"""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

EXP = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(EXP.parent / "16_superflex_objective" / "code"))
from sfopt import BATCH, DEV, evaluate, gt_grid, instance_data, network_init, run_batch, torch  # noqa: E402
from occupancy import load  # noqa: E402

SCENES = ("room_0", "room_1", "room_2", "office_0", "office_1", "office_2", "office_3", "office_4")  # NICE-SLAM's 8


@torch.no_grad()
def evaluate_occ(sq, data, occ, margin=0.5, chunk=1 << 14):
    """sfopt.evaluate on the per-instance occupancy: voxels in the instance's box + margin; TP = its voxels inside the
    union, FP = voxels inside the union that no instance occupies."""
    rows = []
    for b, d in enumerate(data):
        lo = np.floor((d["lo"] - margin - occ.lo) / occ.res).astype(int)
        hi = np.ceil((d["hi"] + margin - occ.lo) / occ.res).astype(int)
        idx = np.stack(np.meshgrid(*[np.arange(x, y + 1) for x, y in zip(lo, hi)], indexing="ij"), -1).reshape(-1, 3)
        x = occ.lo + idx * occ.res
        f = occ.flat(x)
        is_k, free = np.isin(f, occ.occ[d["k"]]), ~np.isin(f, occ.any) | (f < 0)
        keep = is_k | free
        xq = (torch.tensor(x[keep] - d["c"], dtype=torch.float32, device=DEV) - sq.normalization_translation[b]) / sq.normalization_scale[b]
        inside = []
        for a0 in range(0, len(xq), chunk):
            s = sq.sdf_batch(xq[a0:a0 + chunk].T[None].expand(len(data), -1, -1).contiguous())[b]
            s[~sq.exist_mask[b]] = float("inf")
            inside.append(s.min(0).values < 0)
        inside = torch.cat(inside).cpu().numpy()
        k = is_k[keep]
        tp, fp, n = int((inside & k).sum()), int((inside & ~k).sum()), int(k.sum())
        rows.append({"k": d["k"], "gt_vox": n, "tp": tp, "fp": fp, "iou": tp / max(n + fp, 1),
                     "precision": tp / max(tp + fp, 1), "recall": tp / max(n, 1), "K": int(sq.exist_mask[b].sum())})
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scene", choices=SCENES, required=True)
    ap.add_argument("--occ", choices=["filled", "solid", "winding"], default="filled")
    ap.add_argument("--steps", type=int, default=1000)
    a = ap.parse_args()
    t0 = time.time()
    torch.manual_seed(0)
    grid = gt_grid(a.scene)
    data = instance_data(a.scene, grid)
    occ = load(a.scene) if a.occ == "winding" else None
    for d in data:
        if a.occ == "solid":
            d["occ"] = d["lab"] == d["k"]
        elif occ:
            d["occ"] = occ.contains(d["k"], d["q"] + d["c"])
    params = network_init(data)
    params["tapering"], params["bending"] = np.zeros_like(params["tapering"]), np.zeros_like(params["bending"])  # plain SQs
    rows, out = [], {k: [] for k in ("scale", "exponents", "rotation", "translation", "tapering", "bending", "exist")}
    for a0 in range(0, len(data), BATCH):
        sl = slice(a0, a0 + BATCH)
        sq, h = run_batch(data[sl], {k: v[sl].copy() for k, v in params.items()}, "theirs", False, a.steps, deform=False)
        rows += evaluate_occ(sq, data[sl], occ) if occ else evaluate(sq, data[sl], grid)
        for k in out:
            out[k].append(getattr(h, k))
        out["translation"][-1] = out["translation"][-1] + np.stack([d["c"] for d in data[sl]])[:, None]  # un-center
    sqs = {k: np.concatenate(v) for k, v in out.items()} | {"instance": np.array([d["k"] for d in data])}
    summ = {"instances": len(rows), "viou": float(np.mean([r["iou"] for r in rows])),
            "precision": float(np.mean([r["precision"] for r in rows])), "recall": float(np.mean([r["recall"] for r in rows])),
            "primitives": int(sum(r["K"] for r in rows)), "seconds": round(time.time() - t0, 1)}
    d = EXP / "results" / ("targets" + {"filled": "", "solid": "_solid", "winding": "_winding"}[a.occ]) / a.scene
    d.mkdir(parents=True, exist_ok=True)
    np.savez(d / "sqs.npz", **sqs)
    (d / "metrics.json").write_text(json.dumps({"summary": summ, "per_instance": rows, "args": vars(a)}, indent=1))
    print(a.scene, json.dumps(summ), flush=True)


if __name__ == "__main__":
    main()
