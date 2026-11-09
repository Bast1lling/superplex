"""Where and why do fits of the target chair lose its back and seat? Traces both sheets through the optimization.

    python diagnose.py --out ../results/diagnose.json --fits /path/to/folder

Regions: surface samples of the back and seat sheets, picked by normal and position. Per region and snapshot: the share
of samples on an SQ surface (within 0.01), further outside, or deep inside, and how strongly each term of SuperFlex's
loss can still act there: sech^2(d / 0.05) for its truncated surface term, and the share within 0.005 of the
surface in the distance the fit uses, where its occupancy sigmoid (temperature 0.001) has a gradient at all.
"""

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from make_sq import make_sq
from variants import DATA, DEV, TARGET, PredictionHandler, kmeans_params, load_chair, scaled_distance

SNAPSHOTS = (0, 25, 50, 100, 200, 400, 700, 1000)
# name: loss, distance, tau, w_free, cap of the voxel loss, truncation of SuperFlex's surface term, weight of that term
CONFIGS = {"voxel_radial": ("vox", "radial", 0.005, 50.0, None, 0.05, 1.0), "iou_scaled": ("iou", "scaled", 0.01, 1.0, None, 0.05, 1.0),
           "iou_radial": ("iou", "radial", 0.01, 1.0, None, 0.05, 1.0), "iou_scaled_uncapped": ("iou", "scaled", 0.01, 1.0, None, 0.5, 1.0),
           "iou_radial_uncapped": ("iou", "radial", 0.01, 1.0, None, 0.5, 1.0), "voxel_radial_capped": ("vox", "radial", 0.005, 50.0, 0.05, 0.05, 1.0),
           "iou_scaled_surface10": ("iou", "scaled", 0.01, 1.0, None, 0.05, 10.0), "iou_radial_surface10": ("iou", "radial", 0.01, 1.0, None, 0.05, 10.0)}


def regions():
    z = np.load(DATA / TARGET / "pointcloud.npz")
    p, n = z["points"].astype(np.float32), z["normals"].astype(np.float32)
    seat = (np.abs(n[:, 1]) > 0.95) & (np.abs(p[:, 1]) < 0.25) & (np.abs(p[:, 0]) < 0.22)
    back = (np.abs(n[:, 0]) > 0.8) & (p[:, 0] < -0.1) & (np.abs(p[:, 2]) < 0.27) & (p[:, 1] > 0.1)
    return {"seat": p[seat], "back": p[back]}


def distances(sq, x):
    """Scaled distance and the fit's own distance of x (M, 3), normalized frame, to every primitive: (K, M) each."""
    keep = sq.exist_mask[0]
    local = ((x[None] - sq.translation[0][keep, None]) @ sq.rotation()[0][keep])
    e = sq.exponents()[0][keep]
    scaled = scaled_distance(local, sq.scale()[0][keep, None], e[:, 0, None], e[:, 1, None])
    radial = sq.sdf_batch(x.T[None].contiguous())[0][keep]
    return scaled, radial


@torch.no_grad()
def snapshot(sq, parts):
    row = {}
    for name, x in parts.items():
        scaled, radial = distances(sq, x)
        d, owner = scaled.min(0)
        r = radial.amin(0)
        row[name] = {"on": (d.abs() < 0.01).float().mean().item(), "outside": (d >= 0.01).float().mean().item(),
                     "inside": (d <= -0.01).float().mean().item(), "mean_outside": d.clamp_min(0).mean().item(),
                     "surface_pull": (1 / torch.cosh(r / sq.truncation) ** 2).mean().item(),
                     "occupancy_active": (r.abs() < 0.005).float().mean().item(),
                     "owners": torch.bincount(owner, minlength=len(scaled)).tolist()}
    return row


def term(sq, name):
    d = sq.forward()
    if sq.loss_kind == "iou":
        _, iou, surface, bbox, _ = sq._compute_losses(d["sdfs"], d["overlap"])
        return {"iou": -torch.log(iou), "surface": sq.w_sdf * surface, "bbox": bbox}[name]
    x, occupied = d["sdfs"], sq.occupancies.float()
    query, surface = x[:, :sq.M_points_iou], x[:, sq.M_points_iou:]
    if name == "positive":
        return ((occupied * F.softplus(query / sq.tau)).sum(1) + F.softplus(surface / sq.tau).sum(1)) / (occupied.sum(1) + sq.M_points_surf)
    return sq.w_free * ((1 - occupied) * F.softplus(-query / sq.tau)).sum(1) / (1 - occupied).sum(1)


def forces(sq):
    """Per primitive: the growth each loss term asks for, -d term / d log scale, summed over its two largest axes."""
    order = sq.scale().detach()[0].argsort(-1, descending=True)[:, :2]
    out = {}
    for name in (("iou", "surface", "bbox") if sq.loss_kind == "iou" else ("positive", "free")):
        grad = torch.autograd.grad(term(sq, name).sum(), sq.raw_scale)[0]
        out[name] = (-grad[0].gather(1, order).sum(1)).cpu().numpy().round(5).tolist()
    return out


@torch.no_grad()
def state(sq):
    """Primitives in the dataset's frame."""
    t, s = sq.normalization_translation[0], sq.normalization_scale[0]
    return {"scale": (sq.scale()[0] * s).cpu().numpy(), "rotation": sq.rotation()[0].cpu().numpy(),
            "translation": (sq.translation[0] * s + t).cpu().numpy(), "exponents": sq.exponents()[0].cpu().numpy()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--fits", type=Path, required=True)
    args = ap.parse_args()
    args.fits.mkdir(parents=True, exist_ok=True)
    chair = load_chair(TARGET)
    print({"occupied": int(chair["occ"].sum()), "regions": {k: len(v) for k, v in regions().items()}}, flush=True)
    result = {}
    for name, (loss, distance, tau, w_free, cap, truncation, w_sdf) in CONFIGS.items():
        sq = make_sq([chair], PredictionHandler(kmeans_params([chair], 10)), loss, distance, tau, w_free, cap)
        sq.truncation, sq.w_sdf = truncation, w_sdf
        t, s = sq.normalization_translation[0], sq.normalization_scale[0]
        parts = {k: (torch.tensor(v, device=DEV) - t) / s for k, v in regions().items()}
        optimizer = torch.optim.Adam(sq.get_param_groups())
        trace, states = {}, []
        for step in range(SNAPSHOTS[-1] + 1):
            if step in SNAPSHOTS:
                trace[step] = snapshot(sq, parts) | {"forces": forces(sq), "half": sq.scale()[0].detach().cpu().numpy().round(4).tolist()}
                states.append(state(sq))
            if step == SNAPSHOTS[-1]:
                break
            optimizer.zero_grad()
            value, terms = sq.compute_losses(sq.forward())
            value.mean().backward()
            optimizer.step()
        result[name] = {"trace": trace, "normalization": [t.tolist(), s.item()]}
        np.savez(args.fits / f"{name}.npz", **states[-1])
        np.savez(args.fits / f"{name}_trace.npz", **{key: np.stack([st[key] for st in states]) for key in states[0]})
        print(name, "done", flush=True)
    args.out.write_text(json.dumps(result))


if __name__ == "__main__":
    main()
