"""One cached Replica frame (exp 22's cache) -> everything the slot model and its losses need, on the GPU.

Inputs: `data/replica_rgbd_frames/<scene>/<id>_{view.npz,dino.npy}` (DINOv3 ViT-L/16 last layer, 42 x 75 tokens;
depth / GT instance label at 168 x 300; exp 22's target SQs of the frame in the camera frame).

Pixel labels (the mask source, `--mask_source`), each pixel -> one target SQ or -1 (ignore):
- sdf (default): the target SQ with the smallest |SDF| (`common.sdf`, --distance) at the pixel's back-projected point, over all targets (no
  instance labels needed);
- sdf_inst: the same among the target SQs of the pixel's GT instance;
- proj: the first target SQ the pixel's ray enters (z-buffered projection; -1 where it misses all);
- instance: the pixel's GT instance (the rendered label image), a target = all frustum SQs of the instance;
- inst1: the same labels, a target = the one SQ fit to the instance's GT points (`fit_single_sq.py`);
- instance_proj: the instance of the first SuperFlex SQ the pixel's ray enters (the rendered union of the instance's
  SQs as mask), targets as for instance.
Targets = the SQs owning >= MIN_PX labelled pixels (0.5 patch); the pixels of smaller ones become -1.
Cached per frame and source in `data/replica_slot_targets/<scene>/<id>_<source>.npz` (the ray casting is the slow
part).
"""

from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

import common as C
from common import CX, CY, DEV, FX, FY, GH, GW, H, NP, P, W, ZNEAR

FRAMES = Path("/home/sebastian/Documents/Uni/Thesis/data/replica_rgbd_frames")
CACHE = Path("/home/sebastian/Documents/Uni/Thesis/data/replica_slot_targets")
SINGLE = Path("/home/sebastian/Documents/Uni/Thesis/data/replica_single_sq")  # fit_single_sq.py
MIN_PX = P * P // 2  # 0.5 patch
VIS_EPS = 0.03  # m: a point in the frustum is "visible" (in front of / on the observed surface) if z <= depth + VIS_EPS


def backproject(depth):
    """depth (H, W) -> (H * W, 3) camera-frame points."""
    v, u = torch.meshgrid(torch.arange(H, device=depth.device), torch.arange(W, device=depth.device), indexing="ij")
    return torch.stack([(u - CX) / FX * depth, (v - CY) / FY * depth, depth], -1).reshape(-1, 3)


@torch.no_grad()
def pixel_labels(depth, label, t, source):
    """-> (H, W) int64 index into the frame's targets t, -1 = ignore."""
    T = len(t["scale"])
    xyz = backproject(depth)
    valid = depth.reshape(-1) > 0
    s = C.tsq(t)
    if source in ("sdf", "sdf_inst"):
        d = torch.cat([C.sdf(xyz[None, a:a + 8192], {k: v[None] for k, v in s.items()})[0].abs()
                       for a in range(0, len(xyz), 8192)], 1)  # (T, HW)
        if source == "sdf_inst":
            own = t["instance"][:, None] == label.reshape(1, -1)
            d = torch.where(own, d, torch.inf)
        best, arg = d.min(0)
        lab = torch.where(valid & torch.isfinite(best), arg, -1)
    elif source == "proj":
        d, _, _ = C.rays(1, depth.device)
        ri, si, t0, t1 = C._cull(s, d, 1)
        _, hit, te = C._search(s, d, ri, si, t0, t1)
        E = torch.full((len(d), T), torch.inf, device=depth.device)
        E[ri[hit], si[hit]] = te[hit]
        zb, arg = E.min(1)
        lab = torch.where(valid & torch.isfinite(zb), arg, -1)
    else:
        raise ValueError(source)
    return lab.reshape(H, W)


def load_labels(scene, fid, depth, label, t, source):
    p = CACHE / scene / (f"{fid}_{source}_{C.DISTANCE}.npz" if source.startswith("sdf") else f"{fid}_{source}.npz")
    if p.exists():
        return torch.as_tensor(np.load(p)["label"].astype(np.int64), device=DEV)
    lab = pixel_labels(depth, label, t, source)
    p.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(p, label=lab.cpu().numpy().astype(np.int16))
    return lab


N_SURF = 16  # cube samples per target SQ (6 * 7^2 = 294), the pair losses' target side


@torch.no_grad()
def groups(t, members, M):
    """Target groups: members (list of M index tensors into the frame's SQs t) -> padded SQ dict (M, G, ...), valid
    (M, G), cube samples (M, G * S, 3) and keep (M, G * S): samples of valid SQs that do not lie inside another SQ
    of their group (the union's surface)."""
    G = max(len(x) for x in members)
    idx = torch.stack([torch.cat([x, x[:1].repeat(G - len(x))]) for x in members])  # pad with the first member
    valid = torch.stack([torch.arange(G, device=DEV) < len(x) for x in members])
    sq = {k: v[idx] for k, v in C.tsq(t).items()}
    flat = {k: v.flatten(0, 1) for k, v in sq.items()}
    S = C.surf(flat, N_SURF, "cube")  # (M * G, S, 3)
    n = S.shape[1]
    pts = S.reshape(M, G * n, 3)
    keep = valid[:, :, None].expand(M, G, n).reshape(M, G * n).clone()
    if G > 1:
        d = C.sdf(pts, sq)  # (M, G, G * n): SDF of each member at all samples of its group
        own = torch.arange(G, device=DEV).repeat_interleave(n)  # member of each sample
        other = valid[:, :, None] & (torch.arange(G, device=DEV)[None, :, None] != own[None, None])
        keep &= ~((d < -1e-3) & other).any(1)
    return {"sq": sq, "valid": valid, "pts": pts, "keep": keep}


def load_frame(scene, fid, source="sdf", part_targets="all"):
    """-> dict: dino (NP, 1024), depth (H, W), xyz (NP, 3) patch centroids, pts (Q, 3) labelled points, pts_lab (Q,),
    lab (H, W) in 0..M-1 / -1, share (NP, M), pvalid (NP,), M targets, each a group of target SQs ("groups": one SQ
    per target for the SQ-level sources; for source "instance" all frustum SQs of a GT instance), flat (all SQs of
    the groups) and flat_owner (their target).
    part_targets "visible": a target group keeps only its SQs with >= 1 visible cube sample (in the frustum, not behind
    the observed depth + VIS_EPS; the one with most visible samples if none has any); "all": every frustum SQ."""
    stem = FRAMES / scene / fid
    v = np.load(f"{stem}_view.npz")
    dino = torch.as_tensor(np.load(f"{stem}_dino.npy").astype(np.float32), device=DEV).reshape(NP, -1)
    depth = torch.as_tensor(v["depth"].astype(np.float32), device=DEV)
    label = torch.as_tensor(v["label"].astype(np.int64), device=DEV)
    t = {k: torch.as_tensor(v[k], device=DEV) for k in ("scale", "exponents", "rotation", "translation")}
    t = {k: x.float() for k, x in t.items()} | {"instance": torch.as_tensor(v["instance"].astype(np.int64), device=DEV)}
    if source == "inst1":  # GT instance labels, one fitted SQ per instance (`fit_single_sq.py`), moved into the camera
        z = np.load(SINGLE / f"{scene}.npz")
        w2c = torch.as_tensor(np.linalg.inv(v["c2w"]), device=DEV).float()
        R, tt = w2c[:3, :3], w2c[:3, 3]
        t = {"scale": torch.as_tensor(z["scale"], device=DEV).float(), "exponents": torch.as_tensor(z["exponents"], device=DEV).float(),
             "rotation": R @ torch.as_tensor(z["rotation"], device=DEV).float(),
             "translation": torch.as_tensor(z["translation"], device=DEV).float() @ R.T + tt,
             "instance": torch.as_tensor(z["instance"], device=DEV).long()}
    if source == "instance_proj":  # the instance of the first SuperFlex SQ the pixel's ray enters (rendered union)
        lab_sq = load_labels(scene, fid, depth, label, t, "proj")
        label = torch.where(lab_sq >= 0, t["instance"][lab_sq.clamp_min(0)], -1)
    if source in ("instance", "inst1", "instance_proj"):  # the GT instance of the pixel (rendered label = projection of the GT points' labels)
        ids = torch.unique(t["instance"])  # instances with target SQs in the frustum
        ok = (depth > 0) & torch.isin(label, ids)
        raw = torch.where(ok, torch.searchsorted(ids, label.clamp_min(0)), -1)
        cnt = torch.bincount(raw[raw >= 0], minlength=len(ids))
        keep = torch.nonzero(cnt >= MIN_PX)[:, 0]
        members = [torch.nonzero(t["instance"] == ids[k])[:, 0] for k in keep]
        remap = torch.full((len(ids),), -1, dtype=torch.long, device=DEV)
        remap[keep] = torch.arange(len(keep), device=DEV)
        lab = torch.where(raw >= 0, remap[raw.clamp_min(0)], -1)
        inst = ids[keep]
    else:
        lab = load_labels(scene, fid, depth, label, t, source)
        cnt = torch.bincount(lab[lab >= 0], minlength=len(t["scale"]))
        keep = torch.nonzero(cnt >= MIN_PX)[:, 0]
        members = [k[None] for k in keep]
        remap = torch.full((len(t["scale"]),), -1, dtype=torch.long, device=DEV)
        remap[keep] = torch.arange(len(keep), device=DEV)
        lab = torch.where(lab >= 0, remap[lab.clamp_min(0)], -1)
        inst = t["instance"][keep]
    M = len(keep)
    if part_targets == "visible":
        with torch.no_grad():
            S = C.surf(C.tsq(t), N_SURF, "cube")
            nvis = (sample_weights(S, depth, 0.0, 1e-9) > 0.5).sum(1)  # visible samples per frustum SQ
        members = [x[nvis[x] > 0] if (nvis[x] > 0).any() else x[nvis[x].argmax()][None] for x in members]
    onehot = F.one_hot(lab.clamp_min(0), M).float() * (lab >= 0).float()[..., None]  # (H, W, M)
    cnt_p = F.avg_pool2d(onehot.permute(2, 0, 1)[None], P)[0].flatten(1).T * P * P  # (NP, M) pixel counts
    tot = cnt_p.sum(1, keepdim=True)
    share = cnt_p / tot.clamp_min(1)
    xyz_pix = backproject(depth)
    m = lab.reshape(-1) >= 0
    c, _ = C.patch_centroids(depth[None])
    lab_pts = lab.reshape(-1)[m]  # per-target point index table (M, Pmax) for per-step resampling (`obs_cd`)
    cnt_t = torch.bincount(lab_pts, minlength=M)
    order = torch.argsort(lab_pts, stable=True)
    start = torch.cumsum(cnt_t, 0) - cnt_t
    flat_idx = torch.cat(members)
    owner = torch.cat([torch.full((len(x),), i, device=DEV) for i, x in enumerate(members)])
    return {"name": f"{scene}_{fid}", "dino": dino, "depth": depth, "xyz": c[0], "pts": xyz_pix[m],
            "pts_lab": lab.reshape(-1)[m], "lab": lab, "share": share, "pvalid": tot[:, 0] > 0, "M": M,
            "groups": groups(t, members, M), "instance": inst, "flat": C.tsq(t, flat_idx), "flat_idx": flat_idx,
            "flat_owner": owner, "n_all": len(t["scale"]), "rgb": v["rgb"], "source": source,
            "pt_order": order, "pt_start": start, "pt_count": cnt_t, "part_targets": part_targets}


# ------------------------------------------------------------------------------------- visibility weights

_A = {"l": (-0.5 - CX) / FX, "r": (W - 0.5 - CX) / FX, "t": (-0.5 - CY) / FY, "b": (H - 0.5 - CY) / FY}


def frustum_distance(x):
    """x (..., 3) -> (...) approximate metric distance outside the camera frustum (0 inside): the largest violation
    of its 4 side planes (through the camera) and the near plane."""
    X, Y, Z = x.unbind(-1)
    s = [(X - _A["l"] * Z) / (1 + _A["l"] ** 2) ** 0.5, (_A["r"] * Z - X) / (1 + _A["r"] ** 2) ** 0.5,
         (Y - _A["t"] * Z) / (1 + _A["t"] ** 2) ** 0.5, (_A["b"] * Z - Y) / (1 + _A["b"] ** 2) ** 0.5, Z - ZNEAR]
    return torch.relu(-torch.stack(s, -1).amin(-1))


@torch.no_grad()
def sample_weights(x, depth, w_occ, lam):
    """Weights of target / prediction samples x (..., 3): outside the frustum exp(-d / lam), in the frustum 1 if not
    behind the observed surface (z <= depth + VIS_EPS) else w_occ."""
    d = frustum_distance(x)
    z = x[..., 2].clamp_min(ZNEAR)
    u = (FX * x[..., 0] / z + CX).round().long().clamp(0, W - 1)
    v = (FY * x[..., 1] / z + CY).round().long().clamp(0, H - 1)
    obs = depth[v, u]
    vis = (obs <= 0) | (x[..., 2] <= obs + VIS_EPS)
    inside = torch.where(vis, 1.0, w_occ)
    return torch.where(d > 0, torch.exp(-d / lam), inside)
