"""Train / evaluate the RGB-only single-view SQ predictor (`model.Net`) on exp 22's cached frames.

    uv run python train.py --name f16_pred --setting f16 --steps 3000 [--depth pred|none|gt] [--pose]

Same losses, metrics and settings as exp 24 (1 view), plus the depth head's loss (--depth pred): W["depth"] *
mean |log d_pred - log d_patch| over patches with depth (d_patch = mean depth of the patch's pixels); metrics
depth_absrel / depth_log of the patch depths. The GT depth is used for supervision only (targets, assignment,
depth loss, render depth loss), never as input (except --depth gt, the RGB-D oracle).

Settings: f16 = 16 fixed pairs of the 7 training scenes (train = eval, a multi-frame overfit); frames = train on the
first 75 % of each training trajectory (random partner with overlap in [0.2, 0.8]), evaluate on the last 20 % (fixed
partner, overlap closest to 0.5); scenes = train on the 7 scenes, evaluate on room_0.

Loss (exp 22's top-k loss, DINO only):
- proposals (every patch of every view): CenterNet focal loss on the view's heatmap; for up to 512 positive patches
  per view (owner = the union target it sees, drawn with probability 1 / patches of that target) 10 * Chamfer +
  10 * forced Chamfer (target translation) with 8 x 8 samples, 2 * class CE, 5 * BCE + 5 * Dice of the mask (the
  target instance's pixel share per patch, all views);
- decoder: Hungarian matching to the union targets (cost 10 * Chamfer (4 x 4 samples) + 2 * (1 - p(class)) +
  1 * (1 - exist)), focal existence (alpha 0.25, gamma 2) * 2, Chamfer + forced (16 x 16) * 10, class * 2, mask * 5;
- --render: + projection losses of the final decoder SQs rendered into every view (stride --render_stride):
  instance (-log sum of the weights of the queries matched to the pixel's GT instance), class, depth (relative L1),
  weight 1 each (exp 22's render.py, reimplemented in common.py).
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

import common as C
import data as D
from common import DEV, NP, KEYS
import model as M
from model import Net, masks

OUT = D.EXP / "results" / "large_files" / "runs"
W = {"depth": 1.0, "cd": 10.0, "forced": 10.0, "exist": 2.0, "cls": 2.0, "mask": 5.0, "dice": 5.0, "r_inst": 0.0, "r_cls": 0.0, "r_depth": 0.0}
POS_SAMPLES = 512
COST_SAMPLES = 4
RENDER_STRIDE = 4
RENDER_FRAMES = 2  # projection losses on the first frames of each (shuffled) batch only (speed)


# --------------------------------------------------------------------------------------------------- batch prep

def prepare(batch):
    """Concatenated targets of the batch (`cat`, offsets `off`, frame of each target `tf`) and the mask targets
    `share` (sum T, V * NP): each target's instance pixel share per patch in every view."""
    tg = batch["targets"]
    n = torch.tensor([len(t["scale"]) for t in tg], device=DEV)
    batch["n_t"], batch["off"] = n, torch.cumsum(n, 0) - n
    batch["cat"] = {k: torch.cat([t[k] for t in tg]) for k in tg[0]}
    batch["tf"] = torch.repeat_interleave(torch.arange(len(tg), device=DEV), n)
    shares = []
    for b, t in enumerate(tg):
        lab = batch["label"][b]  # (V, H, W)
        shares.append(torch.cat([C.instance_share(lab[v], t["instance"]) for v in range(lab.shape[0])], 1))
    batch["share"] = torch.cat(shares)
    return batch


def seg_mean(x, fb, B, w=None):
    w = torch.ones_like(x) if w is None else w
    s = x.new_zeros(B).index_add(0, fb, x * w)
    return (s / x.new_zeros(B).index_add(0, fb, w).clamp_min(1)).sum() / B


def pair_losses(pred, batch, fb, qs, tg, n):
    """Losses of the pairs prediction (fb, qs) <-> concatenated target tg."""
    B = len(batch["targets"])
    cat = batch["cat"]
    t = C.tsq(cat, tg)
    tp = C.surf(t, n)
    ps, pe, pr = pred["scale"][fb, qs], pred["shape"][fb, qs], pred["rotate"][fb, qs]
    out = {"cd": seg_mean(C.chamfer(C.surface(ps, pe, pr, pred["trans"][fb, qs], n), tp), fb, B),
           "forced": seg_mean(C.chamfer(C.surface(ps, pe, pr, t["trans"], n), tp), fb, B),
           "cls": seg_mean(F.cross_entropy(pred["cls"][fb, qs], cat["cls"][tg], reduction="none"), fb, B)}
    m_p, m_t = masks(pred, fb, qs), batch["share"][tg]
    out["mask"] = seg_mean(F.binary_cross_entropy_with_logits(m_p, m_t, reduction="none").mean(1), fb, B)
    s = m_p.sigmoid()
    dice = 1 - (2 * (s * m_t).sum(1) + 1) / (s.sum(1) + m_t.sum(1) + 1)
    out["dice"] = seg_mean(dice, fb, B, (m_t.sum(1) > 0).float())
    return out


def proposal_losses(enc, batch):
    B = len(batch["targets"])
    p = enc["exist"].clamp(1e-6, 1 - 1e-6)
    heat = batch["heat"].flatten(1)
    pos = heat >= 1.0
    ex = -torch.where(pos, (1 - p) ** 2 * p.log(), (1 - heat) ** 4 * p ** 2 * (1 - p).log())
    total = {"exist": (ex.sum(1) / pos.sum(1).clamp_min(1)).mean()}
    owner = batch["owner"].flatten(1)
    V = batch["owner"].shape[1]
    fb, qs, tg = [], [], []
    for b in range(B):
        q = torch.nonzero(owner[b] >= 0)[:, 0]
        if len(q):
            o = owner[b, q]
            w = 1.0 / torch.bincount(o)[o].float()
            if len(q) > POS_SAMPLES * V:
                q = q[torch.multinomial(w, POS_SAMPLES * V)]
            fb.append(torch.full_like(q, b)), qs.append(q), tg.append(owner[b, q] + batch["off"][b])
    if fb:
        total |= pair_losses(enc, batch, torch.cat(fb), torch.cat(qs), torch.cat(tg), 8)
    return total


@torch.no_grad()
def match_batch(pred, batch):
    """Hungarian matching per frame -> (fb, qs, tg) over the batch (tg: concatenated target index)."""
    B, Q = pred["exist"].shape
    flat = {k: pred[k].flatten(0, 1) for k in KEYS}
    pp = C.surf(flat, COST_SAMPLES).reshape(B, Q, -1, 3)
    tp = C.surf(C.tsq(batch["cat"]), COST_SAMPLES)
    prob = pred["cls"].softmax(-1)
    fb, qs, tg = [], [], []
    for b, t in enumerate(batch["targets"]):
        nb, o = int(batch["n_t"][b]), int(batch["off"][b])
        if nb == 0:
            continue
        qi, ti = C.match_cost(pp[b], tp[o:o + nb], prob[b][:, t["cls"]], pred["exist"][b])
        fb.append(torch.full((len(qi),), b)), qs.append(torch.as_tensor(qi)), tg.append(torch.as_tensor(ti) + o)
    if not fb:
        e = torch.zeros(0, dtype=torch.long, device=DEV)
        return e, e, e
    return tuple(torch.cat(x).to(DEV) for x in (fb, qs, tg))


def decoder_losses(pred, batch, render=False):
    B = len(batch["targets"])
    fb, qs, tg = match_batch(pred, batch)
    exist_t = torch.zeros_like(pred["exist"])
    exist_t[fb, qs] = 1
    p = pred["exist"].clamp(1e-6, 1 - 1e-6)
    pt = torch.where(exist_t > 0, p, 1 - p)
    alpha = torch.where(exist_t > 0, 0.25, 0.75)
    total = {"exist": (-(alpha * (1 - pt) ** 2 * pt.log()).sum(1) / batch["n_t"].clamp_min(1)).mean()}
    if len(fb):
        total |= pair_losses(pred, batch, fb, qs, tg, 16)
        if render:
            total |= render_losses(pred, batch, fb, qs, tg)
    return total


# ---------------------------------------------------------------------------------------------- render losses

def view_sqs(pred, b, q, T0v):
    """Predicted SQs q of frame b (view-0 frame) in the camera frame of a view (T0v: that view -> view 0)."""
    return C.transform({k: pred[k][b, q] for k in KEYS}, torch.linalg.inv(T0v))


def pixel_targets(batch, b, v, out):
    t = batch["targets"][b]
    lab = batch["label"][b, v][out["v"], out["u"]]
    dep = batch["depth"][b, v][out["v"], out["u"]]
    cls = torch.full_like(lab, -1)
    if len(t["instance"]):
        lut = torch.full((int(max(lab.max(), t["instance"].max())) + 2,), -1, dtype=torch.long, device=DEV)
        lut[t["instance"]] = t["cls"]
        cls = torch.where(lab >= 0, lut[lab.clamp_min(0)], -1)
    return lab, cls, dep


def render_losses(pred, batch, fb, qs, tg):
    B, V = batch["label"].shape[:2]
    tot = {"r_inst": 0.0, "r_cls": 0.0, "r_depth": 0.0}
    EPS = C.EPS
    nb = min(B, RENDER_FRAMES)
    for b in range(nb):
        m = fb == b
        qm, im = qs[m], batch["cat"]["instance"][tg[m]]
        keep = pred["exist"][b] > C.MIN_EXIST
        keep[qm] = True
        q = torch.nonzero(keep)[:, 0]
        q_inst = torch.full((len(q),), -2, dtype=torch.long, device=DEV)
        q_inst[torch.searchsorted(q, qm)] = im
        for v in range(V):
            out = C.render(view_sqs(pred, b, q, batch["T0v"][b, v]), pred["exist"][b, q], RENDER_STRIDE)
            lab, cls, dep = pixel_targets(batch, b, v, out)
            w = out["w"]
            ok = (lab >= 0) & torch.isin(lab, im)
            if ok.any():
                tot["r_inst"] = tot["r_inst"] + (-((w[ok] * (q_inst[None] == lab[ok, None])).sum(1) + EPS).log()).mean() / V
            ok = cls >= 0
            if ok.any():
                pc = (w[ok] * pred["cls"][b, q].softmax(-1)[:, cls[ok]].T).sum(1)
                tot["r_cls"] = tot["r_cls"] + (-(pc + EPS).log()).mean() / V
            ok = (dep > 0) & (out["cover"] > 0.5)
            if ok.any():
                tot["r_depth"] = tot["r_depth"] + ((out["depth"][ok] - dep[ok]).abs() / dep[ok]).mean() / V
    return {k: v / nb for k, v in tot.items()}


def losses(pred, batch, render=False):
    parts = {f"enc_{k}": v for k, v in proposal_losses(pred["enc"], batch).items()}
    for i, out in enumerate(pred["aux"] + [pred]):
        last = i == len(pred["aux"])
        p = decoder_losses(out, batch, render and last)
        if last:
            parts |= p
        else:
            parts |= {f"aux{i}_{k}": v for k, v in p.items()}
    if "logd" in pred:
        _, has = C.patch_centroids(batch["depth"][:, 0])
        c, _ = C.patch_centroids(batch["depth"][:, 0])
        err = (pred["logd"] - c[..., 2].clamp_min(0.05).log()).abs()
        parts["depth"] = (err * has).sum() / has.sum().clamp_min(1)
    loss = sum(W[k.split("_", 1)[1] if k.startswith(("enc_", "aux")) else k] * v for k, v in parts.items())
    return loss, {k: float(v.detach()) if torch.is_tensor(v) else float(v) for k, v in parts.items()}


# ------------------------------------------------------------------------------------------------- evaluation

def group_instances(m, exist, thresh=0.5):
    order = torch.argsort(exist, descending=True)
    order = order[exist[order] >= 0.5]
    ms = (m[order] > 0).cpu().numpy()
    groups = []
    for x in ms:
        if not x.any():
            continue
        for g in groups:
            if (g & x).sum() / (g | x).sum() > thresh:
                g |= x
                break
        else:
            groups.append(x.copy())
    return groups


@torch.no_grad()
def frame_metrics(pred, batch, b, qi, ti, n_occ=32768, extra=True):
    """Metrics of frame b. Targets = the union of the views' visible SQs (view-0 frame); qi, ti: its matching."""
    t = batch["targets"][b]
    ex = pred["exist"][b]
    r = {"targets": len(t["scale"]), "predicted": int((ex > 0.5).sum()), "predicted@0.3": int((ex > 0.3).sum())}
    if len(t["scale"]) == 0:
        return r
    allp = {k: pred[k][b] for k in KEYS}
    tq = C.tsq(t)
    vis, fv = t["vis"], t["f"]
    if len(qi):
        qt, tt = torch.as_tensor(qi, device=DEV), torch.as_tensor(ti, device=DEV)
        iou = C.pair_iou({k: v[qt] for k, v in allp.items()}, {k: v[tt] for k, v in tq.items()})
        r["pair_iou"] = float(iou.mean())
        fmax = fv[tt].max(1).values
        groups = {"seen": fmax >= 0.5, "partial": fmax < 0.5, "v0": vis[tt, 0], "v0_partial": vis[tt, 0] & (fv[tt, 0] < 0.5),
                  "v0_seen": vis[tt, 0] & (fv[tt, 0] >= 0.5)}
        if vis.shape[1] > 1:
            groups["v1only"] = ~vis[tt, 0] & vis[tt, 1]
        groups["hidden"] = ~vis[tt].any(1)  # --targets frustum
        for k, m in groups.items():
            if m.any():
                r[f"pair_iou_{k}"] = float(iou[m].mean())
        r["cd"] = float(C.chamfer(C.surf({k: v[qt] for k, v in allp.items()}, 16, "grid"),
                                  C.surf({k: v[tt] for k, v in tq.items()}, 16, "grid")).mean())
        r["cls_acc"] = float((pred["cls"][b, qt].argmax(-1) == t["cls"][tt]).float().mean())
        for thr, s in ((0.5, ""), (0.3, "@0.3")):
            r["found" + s] = float((ex[qt] > thr).float().mean())
    pts = C.surf(tq, 16, "grid").reshape(-1, 3)
    lo, hi = pts.min(0).values, pts.max(0).values
    x = lo + torch.rand(n_occ, 3, device=DEV) * (hi - lo)
    occ_t = C.inside(x[None], {k: v[None] for k, v in tq.items()})[0].any(0)
    lo_k = ex > 0.3
    ins = C.inside(x[None], {k: v[lo_k][None] for k, v in allp.items()})[0] if lo_k.any() else None
    for thr, s in ((0.5, ""), (0.3, "@0.3")):
        k_ = ex > thr
        occ_p = ins[k_[lo_k]].any(0) if ins is not None and k_.any() else torch.zeros_like(occ_t)
        r["occ_iou" + s] = float((occ_p & occ_t).sum() / (occ_p | occ_t).sum().clamp_min(1))
    iou_m = C.pair_iou_matrix(allp, tq).max(0).values
    r["dec_recall@0.5"] = float((iou_m > 0.5).float().mean())
    if vis.any():
        r["dec_recall_v0@0.5"] = float((iou_m[vis[:, 0]] > 0.5).float().mean()) if vis[:, 0].any() else 0.0
    if extra and "enc" in pred:
        enc = pred["enc"]
        V = batch["owner"].shape[1]
        top = C.peaks(enc["exist"][b:b + 1], V)[0].topk(min(300, V * NP)).indices
        e = C.pair_iou_matrix({k: enc[k][b, top] for k in KEYS}, tq).max(0).values
        r["enc_recall@0.5"] = float((e > 0.5).float().mean())
    # rendering into view 0 (stride 2): coverage, instance / class accuracy of the front query, depth abs rel
    keep = ex > C.MIN_EXIST
    if len(qi):
        keep[torch.as_tensor(qi, device=DEV)] = True
    q = torch.nonzero(keep)[:, 0]
    out = C.render(view_sqs(pred, b, q, batch["T0v"][b, 0]), ex[q], 2, grad=False)
    lab, cls, dep = pixel_targets(batch, b, 0, out)
    q_inst = torch.full((len(q),), -2, dtype=torch.long, device=DEV)
    if len(qi):
        q_inst[torch.searchsorted(q, torch.as_tensor(qi, device=DEV))] = t["instance"][torch.as_tensor(ti, device=DEV)]
    front, cov = out["w"].argmax(1), out["cover"] > 0.5
    if (lab >= 0).any():
        r["r_cover"] = float(cov[lab >= 0].float().mean())
        r["r_inst_acc"] = float(((q_inst[front] == lab) & cov)[lab >= 0].float().mean())
    if ((dep > 0) & cov).any():
        ok = (dep > 0) & cov
        r["r_depth_absrel"] = float(((out["depth"][ok] - dep[ok]).abs() / dep[ok]).mean())
    if extra and "mq" in pred:  # instance masks in view 0
        m = masks(pred, torch.full((len(ex),), b, device=DEV), torch.arange(len(ex), device=DEV))[:, :NP]
        groups = group_instances(m, ex)
        lab0 = batch["label"][b, 0]
        ids = torch.unique(lab0[lab0 >= 0])
        gtm = (C.instance_share(lab0, ids) > 0.5).cpu().numpy()
        gtm = gtm[gtm.sum(1) >= 2]
        if len(gtm):
            best = [max([(g.reshape(-1) & x).sum() / (g.reshape(-1) | x).sum() for g in groups], default=0.0) for x in gtm]
            r["mask_cov"] = float(np.mean(best))
    return r


@torch.no_grad()
def nms(pred, b, thr=0.5):
    """Late fusion: queries (exist > 0.05) suppressed (exist := 0) if their IoU with a more confident one > thr."""
    ex = pred["exist"][b]
    cand = torch.nonzero(ex > 0.05)[:, 0]
    cand = cand[torch.argsort(ex[cand], descending=True)]
    iou = C.pair_iou_matrix({k: pred[k][b, cand] for k in KEYS}, {k: pred[k][b, cand] for k in KEYS})
    keep = torch.ones(len(cand), dtype=torch.bool, device=DEV)
    for i in range(len(cand)):
        if keep[i]:
            sup = iou[i] > thr
            sup[: i + 1] = False
            keep &= ~sup
    out = ex.clone()
    out[cand[~keep]] = 0
    return out


def fused_forward(model, batch):
    """1-view model on each view, view-1 SQs moved into view 0's frame, concatenated, NMS."""
    preds = []
    for v in range(batch["dino"].shape[1]):
        p = model(batch["dino"][:, v:v + 1], batch["depth"][:, v:v + 1], torch.eye(4, device=DEV).expand(len(batch["dino"]), 1, 4, 4))
        s = C.transform({k: p[k] for k in KEYS}, batch["T0v"][:, v, None])
        preds.append(s | {"exist": p["exist"], "cls": p["cls"]})
    pred = {k: torch.cat([p[k] for p in preds], 1) for k in preds[0]}
    pred["exist"] = torch.stack([nms(pred, b) for b in range(len(pred["exist"]))])
    return pred


@torch.no_grad()
def evaluate(model, loader, fusion=False, save=None):
    model.eval()
    rows = []
    for batch in loader:
        batch = prepare(D.to_device(batch))
        pred = model(batch)
        fb, qs, tg = match_batch(pred, batch)
        if "logd" in pred:
            c, has = C.patch_centroids(batch["depth"][:, 0])
            d, dp = c[..., 2], pred["logd"].exp()
        for b in range(len(batch["targets"])):
            m = fb == b
            r = frame_metrics(pred, batch, b, qs[m].cpu().numpy(), (tg[m] - batch["off"][b]).cpu().numpy(), extra=not fusion)
            r["name"] = batch["name"][b]
            if "logd" in pred:
                hb = has[b]
                r["depth_absrel"] = float(((dp[b, hb] - d[b, hb]).abs() / d[b, hb]).mean())
                r["depth_scale"] = float((dp[b, hb] / d[b, hb]).median())
            rows.append(r)
            if save is not None:
                np.savez_compressed(save / f"{r['name']}.npz", **{k: pred[k][b].cpu().numpy() for k in KEYS + ("exist",)},
                                    cls=pred["cls"][b].argmax(-1).cpu().numpy(), match_q=qs[m].cpu().numpy(),
                                    match_t=(tg[m] - batch["off"][b]).cpu().numpy())
    model.train()
    keys = sorted({k for r in rows for k in r if k != "name"})
    return {k: float(np.mean([r[k] for r in rows if k in r])) for k in keys}, rows


# ------------------------------------------------------------------------------------------------------- main

def datasets(a, views):
    if a.setting == "f16":
        ev = D.Views(D.TRAIN, views=views, limit=16)
        return D.Views(D.TRAIN, views=views, limit=16, repeat=64), ev, None
    if a.setting == "frames":
        return (D.Views(D.TRAIN, views=views, part="head", partner="random"), D.Views(D.TRAIN, views=views, part="tail", limit=160),
                D.Views(D.TRAIN, views=views, part="head", limit=80))
    return (D.Views(D.TRAIN, views=views, partner="random"), D.Views(D.TEST, views=views, limit=100),
            D.Views(D.TRAIN, views=views, limit=80))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True)
    ap.add_argument("--views", type=int, default=1)
    ap.add_argument("--setting", choices=["f16", "frames", "scenes"], default="f16")
    ap.add_argument("--steps", type=int, default=6000)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--lr", type=float, default=3e-3)
    ap.add_argument("--warmup", type=float, default=0.05)
    ap.add_argument("--wd", type=float, default=1e-4)
    ap.add_argument("--width", type=int, default=128)
    ap.add_argument("--layers", type=int, default=1)
    ap.add_argument("--dec_layers", type=int, default=1)
    ap.add_argument("--select", type=int, default=300)
    ap.add_argument("--render", action="store_true", help="+ projection losses (instance, class, depth; weight 1 each)")
    ap.add_argument("--render_stride", type=int, default=4)
    ap.add_argument("--render_frames", type=int, default=2)
    ap.add_argument("--sampling", choices=["grid", "cube"], default="grid")
    ap.add_argument("--targets", choices=["visible", "frustum"], default="visible")
    ap.add_argument("--no_view_emb", action="store_true")
    ap.add_argument("--eval_every", type=int, default=0)
    ap.add_argument("--depth", choices=["pred", "none", "gt"], default="pred")
    ap.add_argument("--pose", action="store_true", help="gravity direction + camera height as input")
    ap.add_argument("--w_depth", type=float, default=1.0)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    torch.manual_seed(a.seed), np.random.seed(a.seed)
    C.SAMPLING = D.SAMPLING = a.sampling
    D.TARGETS = a.targets
    global RENDER_STRIDE, RENDER_FRAMES
    RENDER_STRIDE = a.render_stride
    RENDER_FRAMES = a.render_frames
    if a.render:
        W["r_inst"] = W["r_cls"] = W["r_depth"] = 1.0
    classes = json.loads((D.FRAMES / "classes.json").read_text())
    kw = dict(collate_fn=D.collate, num_workers=6, pin_memory=True, persistent_workers=True)
    out = OUT / a.name
    out.mkdir(parents=True, exist_ok=True)
    tr_set, ev_set, tr_ev = datasets(a, a.views)
    train = torch.utils.data.DataLoader(tr_set, batch_size=a.batch, shuffle=True, drop_last=True, **kw)
    ev = torch.utils.data.DataLoader(ev_set, batch_size=a.batch, **kw)
    M.DEPTH = a.depth
    W["depth"] = a.w_depth
    model = Net(len(classes), d=a.width, layers=a.layers, dec_layers=a.dec_layers, k=a.select, pose=a.pose).to(DEV)
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=a.wd)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=a.lr, total_steps=a.steps, pct_start=a.warmup)
    log = open(out / "log.jsonl", "w")
    step, t0 = 0, time.time()
    while step < a.steps:
        for batch in train:
            batch = prepare(D.to_device(batch))
            pred = model(batch)
            loss, parts = losses(pred, batch, render=a.render)
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()
            step += 1
            if step % 100 == 0:
                log.write(json.dumps({"step": step, "loss": float(loss), **parts}) + "\n"), log.flush()
                print(f"step {step} loss {float(loss):.3f} {time.time() - t0:.0f}s", flush=True)
            if a.eval_every and step % a.eval_every == 0 and step < a.steps:
                m, _ = evaluate(model, ev)
                log.write(json.dumps({"step": step, "eval": m}) + "\n"), log.flush()
                print("  eval", json.dumps({k: round(v, 3) for k, v in m.items() if "pair_iou" in k or "occ" in k}), flush=True)
            if step >= a.steps:
                break
    torch.save(model.state_dict(), out / "model.pt")
    pred_dir = out / "predictions"
    pred_dir.mkdir(exist_ok=True)
    m, rows = evaluate(model, ev, save=pred_dir)
    m_tr = evaluate(model, torch.utils.data.DataLoader(tr_ev, batch_size=a.batch, **kw))[0] if tr_ev is not None else None
    print("  test", json.dumps({k: round(v, 3) for k, v in m.items()}), flush=True)
    (D.EXP / "results" / f"{a.name}.json").write_text(json.dumps({"args": vars(a), "minutes": (time.time() - t0) / 60, "test": m,
                                                                  "train_eval": m_tr, "per_frame": rows}, indent=1))


if __name__ == "__main__":
    main()
