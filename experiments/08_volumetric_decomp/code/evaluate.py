"""Decomposition metrics and renders.

Per instance (GT): K (number of SQs), volumetric IoU / precision / recall of the SQ union against the instance's
solid voxels (negatives weighted by label as in the fit; reeval.py re-scores with the fixed strict weights), coverage (share of the instance's
points within `cov_tol` of / inside its union). Scene: point -> instance assignment by the most-inside union (mIoU),
as in experiments 05-07. Size bins in 2 cm points.
"""

import numpy as np
import torch

from render import camera, palette, save_grid, splat, sqs_points, view_bounds
from sq import sdf_all

SIZE_BINS = [250, 1250, 5000]  # 2 cm points (~ 1k / 5k / 20k at 1 cm)


def instance_sdf(sqs, x, n_inst, chunk=8192):
    """(N, n_inst) min SDF of points to each instance's union."""
    par = {k: torch.from_numpy(np.asarray(sqs[k])).float() for k in ("trans", "scale", "eps", "rot")}
    inst = torch.from_numpy(np.asarray(sqs["instance"])).long()
    out = []
    for s in range(0, len(x), chunk):
        d = sdf_all(par, torch.from_numpy(x[s:s + chunk]).float())
        u = torch.full((len(d), n_inst), 1e3)
        u = u.scatter_reduce(1, inst[None].expand(len(d), -1), d, reduce="amin")
        out.append(u)
    return torch.cat(out).numpy()


def metrics(cfg, sc, vol, D, sqs, ev, cov_tol=0.03):
    I = len(sc["instance_ids"])
    y = sc["y"]
    U = instance_sdf(sqs, sc["pos"], I)
    own = U[np.arange(len(y)), y]
    pred = U.argmin(1)
    inter = np.bincount(y[pred == y], minlength=I)
    union = np.bincount(y, minlength=I) + np.bincount(pred, minlength=I) - inter
    point_iou = inter / np.maximum(union, 1)
    K = np.bincount(sqs["instance"], minlength=I)
    npts = np.bincount(y, minlength=I)
    tp, fp, fn = ev["tp"].numpy(), ev["fp"].numpy(), ev["fn"].numpy()
    viou = ev["iou"].numpy()
    prec = tp / np.maximum(tp + fp, 1)
    rec = tp / np.maximum(tp + fn, 1)
    cov = np.bincount(y, weights=(own <= cov_tol), minlength=I) / np.maximum(npts, 1)
    per = {"name": sc["class_name"], "instance_id": sc["instance_ids"].tolist(), "points": npts.tolist(),
           "K": K.tolist(), "viou": viou.tolist(), "precision": prec.tolist(), "recall": rec.tolist(),
           "coverage": cov.tolist(), "point_iou": point_iou.tolist()}
    summ = {"instances": I, "sqs": int(K.sum()), "sqs_per_instance": float(K.mean()), "viou": float(viou.mean()),
            "precision": float(prec.mean()), "recall": float(rec.mean()), "coverage": float(cov.mean()),
            "point_miou": float(point_iou.mean()), "point_acc": float((pred == y).mean())}
    b = np.digitize(npts, SIZE_BINS)
    for name, v in (("K", K), ("viou", viou), ("precision", prec), ("coverage", cov)):
        summ[f"{name}_by_size"] = [round(float(v[b == i].mean()), 3) if (b == i).any() else None for i in range(4)]
    summ["instances_by_size"] = np.bincount(b, minlength=4).tolist()
    return {"summary": summ, "per_instance": per}


def render_all(sc, sqs, out, width=800, hide=("ceiling",)):
    I = len(sc["instance_ids"])
    pal = palette(I)
    keep_inst = np.array([n not in hide for n in sc["class_name"]])
    pm = keep_inst[sc["y"]]
    P, Y, Nn = sc["pos"][pm], sc["y"][pm], sc["nrm"][pm]
    sm = keep_inst[sqs["instance"]]
    sub = {k: np.asarray(v)[sm] for k, v in sqs.items() if k != "instance_ids"}
    sq_pts, sq_n, sq_c = sqs_points(sub, pal[sub["instance"]])
    rnd = palette(len(sub["instance"]), seed=11)
    _, _, sq_c2 = sqs_points(sub, rnd)
    imgs, titles = [], []
    for elev, azim in ((55, -60), (89.9, -90)):
        R = camera(elev, azim)
        bnd = view_bounds(P, R)
        imgs.append(splat(P, pal[Y], Nn, R, bnd, width, radius=2))
        titles.append(f"GT instances (points) elev {elev:.0f}")
        imgs.append(splat(sq_pts, sq_c, sq_n, R, bnd, width, radius=2))
        titles.append(f"SQs by instance ({len(sub['instance'])} SQs)")
        imgs.append(splat(sq_pts, sq_c2, sq_n, R, bnd, width, radius=2))
        titles.append("SQs, one color per primitive")
    save_grid(imgs, titles, out / "render.jpg", cols=3)
