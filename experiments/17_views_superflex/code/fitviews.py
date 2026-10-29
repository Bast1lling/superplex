"""SuperFlex's objective (experiment 16) on the view inputs of experiments 12-15 instead of the 3D GT.

    python fitviews.py --setting rgbd_multi|rgb_multi|rgbd_single|rgb_single --scene room_0 [--frame 800]
                       [--init network|kmeans_net]
    python fitviews.py --baseline 12:room0_rgbd ...      (old one-SQ-per-instance fits, re-evaluated as in 16)

Per instance k, from the views only (inputs exactly as in 12-15: same frames, depth, scale alignment, erosion):
- surface points: back-projected pixels labelled k (eroded labels if erode, un-eroded if k loses all pixels), 4096;
- query points: 100k uniform in their box + 10 %, labelled from the carved grid (instance / free / unknown);
- multi-view: occupancy = carved label == k, unknown voxels take the nearest solid voxel's instance (as 16 does with
  the GT grid), SuperFlex's objective unchanged;
- single view: unknown is mostly the unseen back of objects, so unknown query points are left out of the soft IoU
  (`MaskedIoU`), and the bounding-box term (box of the visible points only) is dropped.
Initialization: SuperFlex robust on the (partial) surface points, or k-means with the network's count.
Evaluation: experiment 16's (GT grid, TP / FP within the GT instance box + 0.5 m), on the instances that got an SQ.
"""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from scipy.ndimage import distance_transform_edt, maximum_filter, minimum_filter

EXP = Path(__file__).resolve().parents[1]
ROOT = EXP.parents[1]
sys.path.insert(0, str(ROOT / "experiments" / "16_superflex_objective" / "code"))
sys.path.insert(0, str(ROOT / "experiments" / "12_rgbd_multiview" / "code"))
import sfopt  # noqa: E402
from fit import parse_frames, view_depths  # noqa: E402
from grid import FREE, UNKNOWN, backproject, carve_grid, gt_grid  # noqa: E402
from sq import sdf  # noqa: E402
from views import DEV, gt, load_views  # noqa: E402

SETTINGS = {  # the final configurations of 12-15
    "rgbd_multi": dict(depth="gt", align=0, erode=1, single=False),
    "rgb_multi": dict(depth="depthpro", align=1, erode=1, single=False),
    "rgbd_single": dict(depth="gt", align=0, erode=0, single=True),
    "rgb_single": dict(depth="depthpro", align=0, erode=1, single=True),
}


class MaskedIoU(sfopt.SuperQOptimization):
    """SuperFlex's loss with the soft IoU over the observed (non-unknown) query points only."""

    def _compute_losses(self, sdfs, overlap):
        _, _, Lsdf, L_bbox, L_overlap = super()._compute_losses(sdfs, overlap)
        pred = torch.sigmoid(-sdfs[:, :self.M_points_iou] / 1e-3) * self.known
        occ = self.occupancies.float() * self.known
        iou = (pred * occ).sum(1) / torch.clamp((pred + occ - pred * occ).sum(1), min=1.0)
        loss = -torch.log(iou) + self.w_sdf * Lsdf + self.w_bbox * L_bbox + self.w_overlap * L_overlap
        return loss, iou, Lsdf, L_bbox, L_overlap


def labelled_points(views, depths, erode):
    """Back-projected pixels with their (eroded / un-eroded) instance label."""
    P, lab, lab0 = [], [], []
    for v, d in zip(views, depths):
        ok = d.reshape(-1) > 0
        l0 = v["label"]
        size = 2 * erode + 1
        l = np.where(maximum_filter(l0, size) == minimum_filter(l0, size), l0, -1) if erode else l0
        P.append(backproject(v, d)[ok]), lab.append(l.reshape(-1)[ok]), lab0.append(l0.reshape(-1)[ok])
    return np.concatenate(P), np.concatenate(lab), np.concatenate(lab0)


def view_data(scene, grid, P, lab, lab0, single, seed=0):
    """Per instance seen in the views: as sfopt.instance_data, but surface points and labels from the views."""
    g = gt(scene)
    rng = np.random.default_rng(seed)
    L = grid.label_np
    if not single:
        _, near = distance_transform_edt(L < 0, return_indices=True)
        L = np.where(L == UNKNOWN, L[tuple(near)], L)
    out = []
    for k in range(len(g["ids"])):
        S = P[lab == k] if (lab == k).sum() >= 4 else P[lab0 == k]
        G = g["pos"][g["inst"] == k]
        if len(S) < 4 or len(G) == 0:
            continue
        surf = S[rng.choice(len(S), sfopt.N_SURF, replace=len(S) < sfopt.N_SURF)].astype(np.float64)
        lo, hi = surf.min(0), surf.max(0)
        pad = 0.1 * (hi - lo).max()
        q = lo - pad + rng.random((sfopt.N_IOU, 3)) * (hi - lo + 2 * pad)
        i = np.round((q - grid.lo_np) / grid.res).astype(int)
        inside = ((i >= 0) & (i < L.shape)).all(1)
        i = np.minimum(np.maximum(i, 0), np.array(L.shape) - 1)
        raw = np.where(inside, grid.label_np[tuple(i.T)], FREE)
        occ = np.where(inside, L[tuple(i.T)], FREE) == k
        known = raw != UNKNOWN if single else np.ones(len(q), bool)
        if (occ & known).sum() == 0:  # never solid in the carved grid: no IoU target
            continue
        c = surf.mean(0)
        out.append({"k": k, "c": c, "surf": surf - c, "q": q - c, "lab": raw, "occ": occ, "known": known,
                    "lo": G.min(0), "hi": G.max(0)})  # lo / hi: GT box, for the evaluation region only
    return out


def run(scene, setting, frame, init):
    s = SETTINGS[setting]
    t0 = time.time()
    torch.manual_seed(0)
    cfg = {"scene": scene, "frames": str(frame) if s["single"] else "stride20", "depth": s["depth"], "align": s["align"]}
    views = load_views(scene, parse_frames(cfg["frames"]))
    depths = view_depths(views, cfg)
    n_inst = len(gt(scene)["ids"])
    grid = carve_grid(views, depths, n_inst, erode=s["erode"])
    data = view_data(scene, grid, *labelled_points(views, depths, s["erode"]), s["single"])
    net = sfopt.network_init(data)
    params = net if init == "network" else sfopt.kmeans_init(data, (net["exist"] > 0.5).sum(1).clip(1))
    ref, rows = gt_grid(scene), []
    for a in range(0, len(data), sfopt.BATCH):
        sl = slice(a, a + sfopt.BATCH)
        sq, _ = sfopt.run_batch(data[sl], {k: v[sl].copy() for k, v in params.items()}, "theirs", False, 1000,
                                cls=MaskedIoU if s["single"] else None, w_bbox=0.0 if s["single"] else 1.0)
        rows += sfopt.evaluate(sq, data[sl], ref)
    name = f"{scene.replace('_', '')}_{setting}" + (f"_f{frame}" if s["single"] else "") + ("" if init == "network" else f"_{init}")
    save(name, rows, t0, {"scene": scene, "setting": setting, "frame": frame, "init": init})


def save(name, rows, t0, args):
    summ = {"instances": len(rows), "viou": float(np.mean([r["iou"] for r in rows])),
            "precision": float(np.mean([r["precision"] for r in rows])), "recall": float(np.mean([r["recall"] for r in rows])),
            "primitives": int(sum(r["K"] for r in rows)), "seconds": round(time.time() - t0, 1)}
    d = EXP / "results" / "runs" / name
    d.mkdir(parents=True, exist_ok=True)
    (d / "metrics.json").write_text(json.dumps({"summary": summ, "per_instance": rows, "args": args}, indent=1))
    print(name, json.dumps(summ), flush=True)


@torch.no_grad()
def baseline(spec, margin=0.5, chunk=1 << 15):
    """An old fit of 12-15 (one plain SQ per instance) with experiment 16's evaluation."""
    exp, run_name = spec.split(":")
    src = next((ROOT / "experiments").glob(f"{exp}_*")) / "results" / "runs" / run_name
    t0 = time.time()
    scene = json.loads((src / "config.json").read_text())["scene"]
    z = dict(np.load(src / "sqs.npz"))
    ref, g = gt_grid(scene), gt(scene)
    L, rows = ref.label_np, []
    for k in np.unique(z["instance"]):
        G = g["pos"][g["inst"] == k]
        p = {key: torch.as_tensor(z[key][z["instance"] == k], device=DEV) for key in ("trans", "scale", "eps", "rot")}
        lo = np.floor((G.min(0) - margin - ref.lo_np) / ref.res).astype(int)
        hi = np.ceil((G.max(0) + margin - ref.lo_np) / ref.res).astype(int)
        idx = np.stack(np.meshgrid(*[np.arange(a, b + 1) for a, b in zip(lo, hi)], indexing="ij"), -1).reshape(-1, 3)
        inside = ((idx >= 0) & (idx < L.shape)).all(1)
        lab = np.full(len(idx), FREE)
        lab[inside] = L[tuple(idx[inside].T)]
        keep = (lab == k) | (lab == FREE)
        x = torch.as_tensor(ref.lo_np + idx[keep] * ref.res, dtype=torch.float32, device=DEV)
        occ = torch.cat([(sdf(x[a:a + chunk, None], p) < 0).any(1) for a in range(0, len(x), chunk)]).cpu().numpy()
        is_k = lab[keep] == k
        tp, fp, n = int((occ & is_k).sum()), int((occ & ~is_k).sum()), int(is_k.sum())
        if n == 0:
            continue
        rows.append({"k": int(k), "gt_vox": n, "tp": tp, "fp": fp, "iou": tp / max(n + fp, 1),
                     "precision": tp / max(tp + fp, 1), "recall": tp / max(n, 1), "K": len(p["trans"])})
    save(f"baseline_{exp}_{run_name}", rows, t0, {"baseline": spec})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--setting", choices=list(SETTINGS))
    ap.add_argument("--scene", default="room_0")
    ap.add_argument("--frame", type=int, default=800)
    ap.add_argument("--init", choices=["network", "kmeans_net"], default="network")
    ap.add_argument("--baseline", nargs="*")
    a = ap.parse_args()
    if a.baseline:
        for spec in a.baseline:
            baseline(spec)
    else:
        run(a.scene, a.setting, a.frame, a.init)


if __name__ == "__main__":
    main()
