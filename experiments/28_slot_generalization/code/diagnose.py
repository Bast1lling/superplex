"""Why is the held-out frustum IoU ~0.07? Per matched target (prediction = the union of its slot's active parts,
target = the union of the instance's visible SuperFlex SQs), inside the camera frustum, on training frames vs.
held-out views:

- mask IoU of the matched slot (patch argmax vs. majority label);
- position: distance of the occupied-volume centroids (pred vs. target), and of the slot's anchor (mask centroid)
  to the target's labelled points' centroid;
- size: volume ratio pred / target; extents ratio (sqrt of the occupied volume's covariance eigenvalues, sorted);
- orientation: angle between the main axes (largest eigenvector) of the two volumes;
- IoU after shifting the prediction onto the target centroid (how much is position alone);
- oracle slot: the best IoU over all active slots (is it the matching / mask, or no slot fits at all);
- gtmask_*: the matched slots' features, but the part decoder's patch mask and anchor from the GT patch shares
  (does the geometry transfer once the patches are right).

    uv run python diagnose.py --run F10 [--ckpt model|best] [--n 20]
"""

import argparse
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1] / "27_slot_attention" / "code"))
_s = importlib.util.spec_from_file_location("train28", HERE / "train.py")
T = importlib.util.module_from_spec(_s)
_s.loader.exec_module(T)
T27, D, C = T.T27, T.D, T.C
import fast  # noqa: E402

N_S = 4096


def shift(g, d):
    return {"sq": {**g["sq"], "trans": g["sq"]["trans"] + d[:, None]}, "valid": g["valid"]}


@torch.no_grad()
def stats(x, w, inside):
    """weighted centroid, volume, covariance eigen (vals asc, vecs) of the occupied frustum part -> per row."""
    f = (D.frustum_distance(x) == 0)
    m = (inside & f).float() * w
    vol = m.sum(1)
    c = (m[..., None] * x).sum(1) / vol[:, None].clamp_min(1e-12)
    dx = x - c[:, None]
    cov = torch.einsum("mn,mni,mnj->mij", m, dx, dx) / vol[:, None, None].clamp_min(1e-12)
    ev, evec = torch.linalg.eigh(cov + 1e-9 * torch.eye(3, device=x.device))
    return c, vol, ev.clamp_min(0), evec


@torch.no_grad()
def frame_diag(model, fr):
    pred = T.forward(model, fr)
    sig = T27.match(pred, fr)
    gr = fr["groups"]
    M = fr["M"]
    v = fr["pvalid"]
    am = pred["logits"][v].argmax(1)
    maj = fr["share"][v].argmax(1)
    pex = pred["parts"]["exist_logit"].sigmoid()
    K = len(pex)
    act = pex > 0.5
    act[torch.arange(K), pex.argmax(1)] |= ~act.any(1)
    allP = {k: pred["parts"][k] for k in C.KEYS}
    ga = {"sq": {k: w[sig] for k, w in allP.items()}, "valid": act[sig]}
    x, w = T27.group_box_samples(ga, gr, N_S)
    ip, it = T27.group_inside(x, ga), T27.group_inside(x, gr)
    cp, vp, ep, Vp = stats(x, w, ip)
    ct, vt, et, Vt = stats(x, w, it)
    f = D.frustum_distance(x) == 0
    iou = lambda a, b: ((w * (a & b & f)).sum(1) / (w * ((a | b) & f)).sum(1).clamp_min(1e-12))
    base = iou(ip, it)
    d = (ct - cp)
    gs = shift(ga, d)
    xs, ws = T27.group_box_samples(gs, gr, N_S)
    fs = D.frustum_distance(xs) == 0
    a2, b2 = T27.group_inside(xs, gs), T27.group_inside(xs, gr)
    shifted = (ws * (a2 & b2 & fs)).sum(1) / (ws * ((a2 | b2) & fs)).sum(1).clamp_min(1e-12)
    # oracle: best active slot per target
    ex = pred["exist_logit"].sigmoid() > 0.5
    cand = torch.nonzero(ex)[:, 0]
    best = base.clone()
    for k in cand.tolist():
        gk = {"sq": {kk: ww[k][None].expand(M, *ww[k].shape) for kk, ww in allP.items()}, "valid": act[k][None].expand(M, -1)}
        xk, wk = T27.group_box_samples(gk, gr, 1024)
        fk = D.frustum_distance(xk) == 0
        a3, b3 = T27.group_inside(xk, gk), T27.group_inside(xk, gr)
        best = torch.maximum(best, (wk * (a3 & b3 & fk)).sum(1) / (wk * ((a3 | b3) & fk)).sum(1).clamp_min(1e-12))
    # anchor vs labelled points centroid of the target
    pc = torch.stack([fr["pts"][fr["pts_lab"] == m].mean(0) for m in range(M)])
    anc = pred["anchor"][sig]
    miou = []
    for m in range(M):
        a_, b_ = am == sig[m], maj == m
        miou.append(float((a_ & b_).sum() / (a_ | b_).sum().clamp_min(1)) if b_.any() else np.nan)
    # GT-mask decoder: the matched slots' features, but the part decoder's patch mask and anchor from the GT shares
    share = fr["share"]  # (N, M)
    R = T.input_rotation(fr, T.ARGS, False)
    xyz_in = fr["xyz"] if R is None else fr["xyz"] @ R.T  # the model's input frame
    cg = (share.T @ xyz_in) / share.sum(0)[:, None].clamp_min(1e-6)
    pe = model.pos(fast.M27.fourier3d(xyz_in))
    xt = model.inp(fr["dino"].float()) + pe
    slots, _ = model.sa(xt)
    h = model.out_norm(slots)
    Pg = model.decode_parts(h[sig], {"x": xt, "pe": pe, "content": xt - pe, "xyz": xyz_in}, share, cg)
    if R is not None:
        Pg = {k: v for k, v in Pg.items()}
        Pg["trans"], Pg["rotate"] = Pg["trans"] @ R, R.T @ Pg["rotate"]
    pg_ex = Pg["exist_logit"].sigmoid()
    actg = pg_ex > 0.5
    actg[torch.arange(M), pg_ex.argmax(1)] |= ~actg.any(1)
    gg = {"sq": {k: Pg[k] for k in C.KEYS}, "valid": actg}
    xg, wg = T27.group_box_samples(gg, gr, N_S)
    fg = D.frustum_distance(xg) == 0
    a4, b4 = T27.group_inside(xg, gg), T27.group_inside(xg, gr)
    iou_g = (wg * (a4 & b4 & fg)).sum(1) / (wg * ((a4 | b4) & fg)).sum(1).clamp_min(1e-12)
    cpg, vpg, epg, Vpg = stats(xg, wg, a4)
    ctg, vtg, etg, Vtg = stats(xg, wg, b4)
    ang_g = torch.rad2deg(torch.arccos((Vpg[:, :, 2] * Vtg[:, :, 2]).sum(1).abs().clamp(max=1)))
    ok = (vt > 0) & (vp > 0)
    cos = (Vp[:, :, 2] * Vt[:, :, 2]).sum(1).abs().clamp(max=1)
    ext_ratio = (ep.sqrt() / et.sqrt().clamp_min(1e-6))  # sorted asc: thin, mid, long
    out = {"mask_iou": np.array(miou), "iou": base.cpu().numpy(), "iou_shifted": shifted.cpu().numpy(),
           "iou_oracle_slot": best.cpu().numpy(), "centroid_err": d.norm(dim=1).cpu().numpy(),
           "anchor_err": (anc - pc).norm(dim=1).cpu().numpy(), "vol_ratio": (vp / vt.clamp_min(1e-12)).cpu().numpy(),
           "ext_ratio_long": ext_ratio[:, 2].cpu().numpy(), "ext_ratio_thin": ext_ratio[:, 0].cpu().numpy(),
           "axis_angle_deg": torch.rad2deg(torch.arccos(cos)).cpu().numpy(), "valid": ok.cpu().numpy(),
           "target_size": et[:, 2].sqrt().cpu().numpy(),
           "gtmask_iou": iou_g.cpu().numpy(), "gtmask_centroid_err": (ctg - cpg).norm(dim=1).cpu().numpy(),
           "gtmask_axis_deg": ang_g.cpu().numpy()}
    return out


def summarize(rows):
    keys = [k for k in rows[0] if k != "valid"]
    cat = {k: np.concatenate([r[k][r["valid"]] for r in rows]) for k in keys}
    return {k: (float(np.nanmedian(v)), float(np.nanmean(v))) for k, v in cat.items()}, len(cat["iou"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="F10")
    ap.add_argument("--ckpt", default="model", choices=["model", "best"])
    ap.add_argument("--n", type=int, default=20)
    a = ap.parse_args()
    run = T.OUT / "runs" / a.run
    cfg = json.loads((run / "args.json").read_text())
    A = type("A", (), cfg | {"assign": "gt", "geo_owned": False})()
    T27.A = A
    T.ARGS = A
    model = T.build_model(A)
    model.load_state_dict(torch.load(run / f"{a.ckpt}.pt", map_location=C.DEV))
    model.eval()
    A.n_eval = a.n
    L = T.frame_lists(A)
    ids = {"train": T.spaced(cfg["train_ids"], a.n), "views": L["views"]} | ({"scenes": L["scenes"]} if L["scenes"] else {})
    res = {}
    for split, fs in ids.items():
        rows = [frame_diag(model, T.load(f, A)) for f in fs]
        res[split] = summarize(rows)
    keys = list(res["train"][0])
    print(f"{a.run} ({a.ckpt}.pt): median / mean over matched targets  (" + ", ".join(f"{s} n={r[1]}" for s, r in res.items()) + ")")
    for k in keys:
        print(f"  {k:16s} " + "    ".join(f"{s} {r[0][k][0]:7.3f} / {r[0][k][1]:7.3f}" for s, r in res.items()))


if __name__ == "__main__":
    main()
