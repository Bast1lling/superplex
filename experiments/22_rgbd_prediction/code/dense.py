"""One-to-many training of `model.DenseModel` (--dense in train.py): every patch is assigned the target SQ it sees.

Assignment per frame (visible SQs only; SQs no pixel sees are dropped):
- every labelled pixel belongs to the target SQ of its instance with the smallest SDF at its back-projected point
  (`train.pixel_owner`, as `--mask sq`);
- a patch is positive for the SQ owning >= 50 % of its pixels; every visible SQ without such a patch also gets the
  patch where its share is largest (thin parts); all other patches are negatives.
Loss: existence = CenterNet heatmap (`heatmap`, `center_focal`): one peak per SQ, so that the local maxima of the
existence map are about one proposal per SQ; for up to POS_SAMPLES positives per frame, drawn with probability
1 / (patches of their SQ), so that every SQ counts about equally: 10 * Chamfer + 10 * forced (8 x 8 samples), 2 * class
CE, 5 * BCE + 5 * Dice of the patch's mask against its SQ's instance mask.
Evaluation (visible SQs): the proposals are the top K local maxima (3 x 3) of the existence map (`model.peaks`); recall@0.5 = share of SQs with a proposal of
pair IoU > 0.5, best_iou = mean best pair IoU per SQ; existence precision / recall (> 0.5) against the assignment;
occupancy IoU of the union of the top K and of the patches with exist > 0.5 against the union of the visible SQs;
class accuracy of the assigned patches.
"""

import numpy as np
import torch
import torch.nn.functional as F
from scipy.ndimage import distance_transform_edt

from model import GH, GW, P, peaks
import train as T

POS_SAMPLES = 512
K = 300


@torch.no_grad()
def assign(batch, b, t):
    """-> owner (GH * GW,) target SQ per patch (-1 negative), visible (T,) bool. From the cache (`cache_assign.py`) if the batch has it."""
    if "owner" in batch:
        return batch["owner"][b], t["visible"]
    own = batch["pixown"][b] if "pixown" in batch else T.pixel_owner(batch, b, t)
    n = len(t["scale"])
    share = F.avg_pool2d((own[None] == torch.arange(n, device=own.device)[:, None, None]).float()[:, None], P)[:, 0].flatten(1)  # (T, GH*GW)
    best, owner = share.max(0)
    owner = torch.where(best >= 0.5, owner, -1)
    visible = share.sum(1) > 0
    missing = visible & (torch.bincount(owner[owner >= 0], minlength=n) == 0)
    for s in torch.nonzero(missing)[:, 0]:
        owner[share[s].argmax()] = s
    return owner, visible


@torch.no_grad()
def heatmap(owner):
    """Existence target (GH * GW,), CenterNet style: per SQ a Gaussian over its own patches, 1 at its center = the
    patch deepest inside its region (largest distance to the region's border), sigma = max(0.5, sqrt(patches) / 4)."""
    own = owner.reshape(GH, GW).cpu().numpy()
    heat = np.zeros((GH, GW), np.float32)
    gy, gx = np.indices((GH, GW))
    for s in np.unique(own[own >= 0]):
        m = own == s
        c = np.unravel_index(np.argmax(distance_transform_edt(np.pad(m, 1))[1:-1, 1:-1]), m.shape)
        sigma = max(0.5, np.sqrt(m.sum()) / 4)
        h = np.exp(-((gy - c[0]) ** 2 + (gx - c[1]) ** 2) / (2 * sigma ** 2)) * m
        h[c] = 1.0
        heat = np.maximum(heat, h)
    return torch.tensor(heat.reshape(-1), device=owner.device)


def center_focal(p, heat):
    """CenterNet's penalty-reduced focal loss (alpha 2, beta 4), normalized by the number of centers."""
    p = p.clamp(1e-6, 1 - 1e-6)
    pos = heat >= 1.0
    loss = -torch.where(pos, (1 - p) ** 2 * p.log(), (1 - heat) ** 4 * p ** 2 * (1 - p).log())
    return loss.sum() / pos.sum().clamp_min(1)


def losses(pred, batch, assigned=None):
    """One-to-many loss (module docstring); the geometry of all frames' sampled positives in one pass."""
    B = len(batch["targets"])
    assigned = assigned or [assign(batch, b, t) for b, t in enumerate(batch["targets"])]
    heat = batch["heat"] if "heat" in batch else torch.stack([heatmap(o) for o, _ in assigned])
    p = pred["exist"].clamp(1e-6, 1 - 1e-6)
    pos = heat >= 1.0
    ex = -torch.where(pos, (1 - p) ** 2 * p.log(), (1 - heat) ** 4 * p ** 2 * (1 - p).log())
    if "ignore" in batch:  # --min_patches: patches of dropped SQs are neither positives nor negatives
        ex = ex * ~batch["ignore"]
    total = {k: 0.0 for k in ("cd", "forced", "cls", "mask", "dice")}
    total["exist"] = (ex.sum(1) / pos.sum(1).clamp_min(1)).mean()
    fb, qs, tg, off = [], [], [], 0
    for b, t in enumerate(batch["targets"]):
        owner = assigned[b][0]
        q = torch.nonzero(owner >= 0)[:, 0]
        if len(q):
            w = 1.0 / torch.bincount(owner[q])[owner[q]].float()
            if len(q) > POS_SAMPLES:
                q = q[torch.multinomial(w, POS_SAMPLES)]
            fb.append(torch.full_like(q, b)), qs.append(q), tg.append(owner[q] + off)
        off += len(t["scale"])
    if fb:
        total |= T.pair_losses(pred, batch, torch.cat(fb), torch.cat(qs), torch.cat(tg), 8)
    W = T.W
    return sum(W[k] * v for k, v in total.items()), {k: float(v) for k, v in total.items()}


@torch.no_grad()
def pair_iou_matrix(a, b, M=1024):
    """IoU of every SQ in a (dict of A) with every SQ in b (dict of Bn) -> (A, Bn): `train.pair_iou` per pair, i.e. M
    points uniform in the box of the pair's surface samples. Only pairs whose sample boxes overlap (each widened by
    10 % + 1 cm) are evaluated, all others are 0; the random points are drawn for all pairs as before, so the kept
    pairs get exactly the same value."""
    A, Bn = len(a["scale"]), len(b["scale"])
    box = lambda q: (lambda p: (p.min(1).values, p.max(1).values))(T.surface(*q.values(), 8, kind="grid"))
    (la, ha), (lb, hb) = box(a), box(b)
    pad = lambda lo, hi: ((hi - lo) * 0.1 + 0.01)
    pa, pb = pad(la, ha), pad(lb, hb)
    hit = (((la - pa)[:, None] <= (hb + pb)[None]) & ((lb - pb)[None] <= (ha + pa)[:, None])).all(-1).flatten()  # (A * Bn,)
    lo = torch.minimum(la[:, None], lb[None]).reshape(-1, 3)
    hi = torch.maximum(ha[:, None], hb[None]).reshape(-1, 3)
    ai = torch.arange(A, device=lo.device).repeat_interleave(Bn)
    bi = torch.arange(Bn, device=lo.device).repeat(A)
    out = torch.zeros(A * Bn, device=lo.device)
    for s in range(0, A * Bn, 4096):
        u = torch.rand(min(4096, A * Bn - s), M, 3, device=lo.device)  # the same draws as train.pair_iou per chunk
        k = torch.nonzero(hit[s:s + 4096])[:, 0] + s
        if len(k) == 0:
            continue
        x = lo[k][:, None] + u[k - s] * (hi[k] - lo[k])[:, None]
        ina = T.inside_pairs(x, {n: v[ai[k]] for n, v in a.items()})
        inb = T.inside_pairs(x, {n: v[bi[k]] for n, v in b.items()})
        out[k] = (ina & inb).sum(1) / (ina | inb).sum(1).clamp_min(1)
    return out.reshape(A, Bn)


@torch.no_grad()
def occupancy_ious(pred, b, keeps, tq, x):
    """Occupancy IoU of the union of the predicted SQs in each mask of keeps against the union of the targets tq, at the
    points x: the SDFs of all SQs in any of the masks are evaluated once."""
    occ_t = T.inside(x[None], {k: v[None] for k, v in tq.items()})[0].any(0)
    union = torch.stack(keeps).any(0)
    if not union.any():
        return [0.0] * len(keeps)
    ins = T.inside(x[None], {k: pred[k][b, union][None] for k in ("scale", "shape", "rotate", "trans")})[0]
    out = []
    for keep in keeps:
        occ_p = ins[keep[union]].any(0)
        out.append(float((occ_p & occ_t).sum() / (occ_p | occ_t).sum().clamp_min(1)))
    return out


@torch.no_grad()
def frame_metrics(pred, batch, b, t, owner, visible, n_occ=32768):
    """Proposal metrics of frame b (see the module docstring) -> (row, best proposal per visible SQ, visible SQs)."""
    ex = pred["exist"][b]
    vi = torch.nonzero(visible)[:, 0]
    tq = T.sq_dict(t["scale"][vi], t["exponents"][vi], t["rotation"][vi], t["translation"][vi])
    top = peaks(ex[None])[0].topk(K).indices
    iou = pair_iou_matrix({k: pred[k][b, top] for k in ("scale", "shape", "rotate", "trans")}, tq)  # (K, V)
    pos, on = owner >= 0, ex > 0.5
    surf = T.surface(*tq.values(), 16, kind="grid").reshape(-1, 3)
    lo, hi = surf.min(0).values, surf.max(0).values
    x = lo + torch.rand(n_occ, 3, device=lo.device) * (hi - lo)
    topmask = torch.zeros_like(on)
    topmask[top] = True
    r = {"targets": len(t["scale"]), "visible": len(vi),
         "recall@0.5": float((iou.max(0).values > 0.5).float().mean()), "best_iou": float(iou.max(0).values.mean()),
         "exist_precision": float((on & pos).sum() / on.sum().clamp_min(1)),
         "exist_recall": float((on & pos).sum() / pos.sum().clamp_min(1)), "predicted": int(on.sum()),
         **dict(zip(("occ_iou_topk", "occ_iou"), occupancy_ious(pred, b, [topmask, on], tq, x))),
         "cls_acc": float((pred["cls"][b, pos].argmax(-1) == t["cls"][owner[pos]]).float().mean()) if pos.any() else 0.0}
    return r, top[iou.argmax(0)], vi


def visible_targets(batch, assigned):
    """The batch with each frame's targets restricted to its visible SQs."""
    return batch | {"targets": [{k: v[vis] for k, v in t.items()} for t, (_, vis) in zip(batch["targets"], assigned)]}


def save_frame(save, name, pred, b, qi, ti):
    np.savez_compressed(save / f"{name}.npz", exist=pred["exist"][b].cpu().numpy(),
                        **{k: pred[k][b].cpu().numpy() for k in ("scale", "shape", "rotate", "trans")},
                        cls=pred["cls"][b].argmax(-1).cpu().numpy(), mask=pred["mask"][b].cpu().numpy().astype(np.float16),
                        match_q=np.asarray(qi), match_t=np.asarray(ti))


@torch.no_grad()
def evaluate(model, loader, n_occ=32768, save=None):
    """--dense: proposal metrics (module docstring). With a decoder (--topk): train.py's metrics of the decoder's SQs
    against the visible SQs, dec_recall@0.5 / dec_best_iou (the proposals' recall@0.5 / best_iou over its K SQs), and
    the proposal metrics with the prefix enc_."""
    model.eval()
    rows = []
    for batch in loader:
        batch = T.to_device(batch)
        pred = model(batch["dino"], batch["feats"], batch["idx"], batch["depth"])
        assigned = [assign(batch, b, t) for b, t in enumerate(batch["targets"])]
        vis_batch = visible_targets(batch, assigned)
        for b, t in enumerate(batch["targets"]):
            owner, visible = assigned[b]
            name = batch["name"][b]
            if "enc" in pred:
                r, qi, ti = T.frame_metrics(pred, vis_batch, b, vis_batch["targets"][b], n_occ)
                vt = vis_batch["targets"][b]
                if len(vt["scale"]):  # as the proposals' recall@0.5 / best_iou, over the decoder's K SQs
                    iou = pair_iou_matrix({k: pred[k][b] for k in ("scale", "shape", "rotate", "trans")},
                                          T.sq_dict(vt["scale"], vt["exponents"], vt["rotation"], vt["translation"])).max(0).values
                    r["dec_recall@0.5"], r["dec_best_iou"] = float((iou > 0.5).float().mean()), float(iou.mean())
                # selection recall: visible targets with >= 1 own patch among the K selected (all / owning 1-2 patches)
                own_n = torch.bincount(owner[owner >= 0], minlength=len(visible))[:len(visible)]
                so = owner[pred["select"][b]]
                hit = torch.zeros(len(visible), dtype=torch.bool, device=owner.device)
                hit[so[so >= 0]] = True
                for key, m in (("sel_recall", own_n > 0), ("sel_recall_small", (own_n > 0) & (own_n <= 2))):
                    if m.any():
                        r[key] = float(hit[m].float().mean())
                e, _, _ = frame_metrics(pred["enc"], batch, b, t, owner, visible, n_occ)
                r |= {f"enc_{k}": v for k, v in e.items() if k not in ("targets", "visible")}
                r["visible"] = e["visible"]
                ti = torch.nonzero(visible)[:, 0].cpu().numpy()[ti]  # back to indices into all targets
            else:
                r, qi, ti = frame_metrics(pred, batch, b, t, owner, visible, n_occ)
                qi, ti = qi.cpu().numpy(), ti.cpu().numpy()
            r["name"] = name
            rows.append(r)
            if save is not None:
                save_frame(save, name, pred, b, qi, ti)
    model.train()
    keys = [k for k in rows[0] if k != "name"]
    return {k: float(np.mean([r[k] for r in rows if k in r])) for k in keys}, rows


def topk_losses(pred, batch):
    """--topk: the one-to-many loss on the proposals (logged as enc_*) + train.py's Hungarian loss (focal existence,
    Chamfer + forced, class, instance mask) on every decoder layer against the visible SQs."""
    T.FOCAL, T.COST_SAMPLES = T.EXIST == "focal", 4
    assigned = [assign(batch, b, t) for b, t in enumerate(batch["targets"])]
    loss, parts = losses(pred["enc"], batch, assigned)
    parts = {f"enc_{k}": v for k, v in parts.items()}
    vis_batch = visible_targets(batch, assigned)
    for i, out in enumerate(pred["aux"] + [pred]):
        l, p = T.losses(out, vis_batch, render=i == len(pred["aux"]))
        loss = loss + l
        if i == len(pred["aux"]):
            parts |= p
    if T.UNION != "none":  # on the final decoder SQs
        u, iou = T.union_loss(pred, batch, vis_batch["targets"])
        loss = loss + T.W_UNION * u
        parts |= {"union": float(u), "union_soft_iou": iou}
    return loss, parts
