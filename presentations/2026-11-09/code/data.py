"""One cached Replica frame -> everything the slot model and its losses need, on the GPU (exp 27's data.py, reduced).

Frames: `data/replica_rgbd_frames/<scene>/<id>_{view.npz,dino.npy}` (exp 22's cache: NICE-SLAM's Replica renderings,
every 10th frame, 168 x 300 RGB-D, DINOv3 ViT-L/16 last layer at 42 x 75 patches of 4 x 4 pixels, GT instance labels,
camera pose, and every SuperFlex SQ of exp 22's scene decomposition that reaches into the frustum, in the camera frame).

Targets: one per GT instance with >= half a patch of labelled pixels.
- masks: the instance's share of each patch's pixels (NP, M);
- geometry, `--targets visible` (default): the instance's SuperFlex SQs with >= 1 visible surface sample (in the
  frustum, not behind the observed depth); `single`: one SQ fit per instance (exp 27's fit_single_sq.py; room_2 only).
"""

from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

H, W = 168, 300
FX = FY = 150.0
CX, CY = (599.5 - 2) / 4, (339.5 - 2) / 4 - 1
GH, GW, P = 42, 75, 4
NP = GH * GW
ZNEAR = 0.1
DEV = "cuda"
DATA = Path(__file__).resolve().parents[5] / "data"  # Thesis/data
FRAMES = DATA / "replica_rgbd_frames"
SINGLE = DATA / "replica_single_sq"
MIN_PX = P * P // 2  # half a patch
VIS_EPS = 0.03  # a point is visible if z <= observed depth + 3 cm
N_SURF = 16  # surface samples per target SQ (294)

import sq  # noqa: E402  (after the constants: sq imports them)


def scene_ids(scene):
    return sorted(p.name.split("_")[0] for p in (FRAMES / scene).glob("*_dino.npy"))


def backproject(depth):
    v, u = torch.meshgrid(torch.arange(H, device=depth.device), torch.arange(W, device=depth.device), indexing="ij")
    return torch.stack([(u - CX) / FX * depth, (v - CY) / FY * depth, depth], -1).reshape(-1, 3)


def patch_centroids(depth):
    """depth (H, W) -> (NP, 3) mean of each patch's back-projected pixels (3.5 m along its ray if it has none)."""
    pix = backproject(depth).reshape(GH, P, GW, P, 3).permute(0, 2, 1, 3, 4).reshape(NP, P * P, 3)
    ok = (depth > 0).float().reshape(GH, P, GW, P).permute(0, 2, 1, 3).reshape(NP, P * P, 1)
    n = ok.sum(1)
    gy, gx = torch.meshgrid(torch.arange(GH, device=DEV), torch.arange(GW, device=DEV), indexing="ij")
    ray = torch.stack([((gx + 0.5) * P - 0.5 - CX) / FX, ((gy + 0.5) * P - 0.5 - CY) / FY, torch.ones_like(gx, dtype=torch.float32)], -1)
    return torch.where(n > 0, (pix * ok).sum(1) / n.clamp_min(1), ray.reshape(NP, 3) * 3.5)


def patch_grid():
    """(NP, 2) patch centres in image coordinates normalized to [0, 1] (the no-depth variant's positions)."""
    gy, gx = torch.meshgrid(torch.arange(GH, device=DEV), torch.arange(GW, device=DEV), indexing="ij")
    return torch.stack([(gx.flatten() + 0.5) / GW, (gy.flatten() + 0.5) / GH], -1).float()


def gravity_rotation(c2w):
    """Camera -> gravity-aligned frame (rows: camera forward made horizontal, up x forward, up; Replica's z is up)."""
    up = torch.as_tensor(c2w[:3, :3][2], dtype=torch.float32, device=DEV)
    up = up / up.norm()
    fwd = torch.tensor([0.0, 0.0, 1.0], device=DEV)
    x = fwd - (fwd @ up) * up
    x = x / x.norm()
    return torch.stack([x, torch.linalg.cross(up, x), up])


def frustum_distance(x):
    """x (..., 3) -> approximate metric distance outside the camera frustum (0 inside)."""
    X, Y, Z = x.unbind(-1)
    l, r, t, b = (-0.5 - CX) / FX, (W - 0.5 - CX) / FX, (-0.5 - CY) / FY, (H - 0.5 - CY) / FY
    s = [(X - l * Z) / (1 + l * l) ** 0.5, (r * Z - X) / (1 + r * r) ** 0.5, (Y - t * Z) / (1 + t * t) ** 0.5,
         (b * Z - Y) / (1 + b * b) ** 0.5, Z - ZNEAR]
    return torch.relu(-torch.stack(s, -1).amin(-1))


@torch.no_grad()
def sample_weights(x, depth, w_occ=1.0, lam=0.1):
    """Weights of samples x (..., 3) in the Chamfer terms: outside the frustum exp(-distance / lam); inside 1 if not
    behind the observed surface, else w_occ."""
    z = x[..., 2].clamp_min(ZNEAR)
    u = (FX * x[..., 0] / z + CX).round().long().clamp(0, W - 1)
    v = (FY * x[..., 1] / z + CY).round().long().clamp(0, H - 1)
    obs = depth[v, u]
    inside = torch.where((obs <= 0) | (x[..., 2] <= obs + VIS_EPS), 1.0, w_occ)
    d = frustum_distance(x)
    return torch.where(d > 0, torch.exp(-d / lam), inside)


@torch.no_grad()
def groups(s, members):
    """Target groups: members (list of M index tensors into the SQs s) -> padded SQs (M, G, ...), valid (M, G),
    surface samples pts (M, G * 294, 3), keep (M, G * 294): samples not inside another SQ of the group (union surface)."""
    M, G = len(members), max(len(x) for x in members)
    idx = torch.stack([torch.cat([x, x[:1].repeat(G - len(x))]) for x in members])
    valid = torch.stack([torch.arange(G, device=DEV) < len(x) for x in members])
    g = {k: v[idx] for k, v in s.items()}
    S = sq.surf({k: v.flatten(0, 1) for k, v in g.items()}, N_SURF)
    n = S.shape[1]
    pts = S.reshape(M, G * n, 3)
    keep = valid[:, :, None].expand(M, G, n).reshape(M, G * n).clone()
    if G > 1:
        d = sq.sdf(pts, g)  # (M, G, G n)
        own = torch.arange(G, device=DEV).repeat_interleave(n)
        other = valid[:, :, None] & (torch.arange(G, device=DEV)[None, :, None] != own[None, None])
        keep &= ~((d < -1e-3) & other).any(1)
    return {"sq": g, "valid": valid, "pts": pts, "keep": keep}


def load_frame(scene, fid, targets="visible"):
    v = np.load(FRAMES / scene / f"{fid}_view.npz")
    dino = torch.as_tensor(np.load(FRAMES / scene / f"{fid}_dino.npy"), device=DEV).reshape(NP, -1)  # fp16 (as cached)
    depth = torch.as_tensor(v["depth"].astype(np.float32), device=DEV)
    label = torch.as_tensor(v["label"].astype(np.int64), device=DEV)
    t = {"scale": v["scale"], "shape": v["exponents"], "rotate": v["rotation"], "trans": v["translation"]}
    s = {k: torch.as_tensor(x, device=DEV).float() for k, x in t.items()}
    sq_inst = torch.as_tensor(v["instance"].astype(np.int64), device=DEV)
    if targets == "single":  # one fitted SQ per instance, world -> camera
        z = np.load(SINGLE / f"{scene}.npz")
        w2c = torch.as_tensor(np.linalg.inv(v["c2w"]), device=DEV).float()
        R = w2c[:3, :3]
        s = {"scale": torch.as_tensor(z["scale"], device=DEV).float(), "shape": torch.as_tensor(z["exponents"], device=DEV).float(),
             "rotate": R @ torch.as_tensor(z["rotation"], device=DEV).float(),
             "trans": torch.as_tensor(z["translation"], device=DEV).float() @ R.T + w2c[:3, 3]}
        sq_inst = torch.as_tensor(z["instance"].astype(np.int64), device=DEV)
    # instances with SQs and >= half a patch of labelled pixels -> pixel labels 0..M-1 (-1: ignored)
    ids = torch.unique(sq_inst)
    raw = torch.where((depth > 0) & torch.isin(label, ids), torch.searchsorted(ids, label.clamp_min(0)), -1)
    cnt = torch.bincount(raw[raw >= 0], minlength=len(ids))
    keep = torch.nonzero(cnt >= MIN_PX)[:, 0]
    remap = torch.full((len(ids),), -1, dtype=torch.long, device=DEV)
    remap[keep] = torch.arange(len(keep), device=DEV)
    lab = torch.where(raw >= 0, remap[raw.clamp_min(0)], -1)
    M = len(keep)
    members = [torch.nonzero(sq_inst == ids[k])[:, 0] for k in keep]
    if targets == "visible":  # SQs with >= 1 visible surface sample (the most visible one if none)
        with torch.no_grad():
            nvis = (sample_weights(sq.surf(s, N_SURF), depth, 0.0, 1e-9) > 0.5).sum(1)
        members = [x[nvis[x] > 0] if (nvis[x] > 0).any() else x[nvis[x].argmax()][None] for x in members]
    onehot = F.one_hot(lab.clamp_min(0), max(M, 1)).float() * (lab >= 0).float()[..., None]
    cnt_p = F.avg_pool2d(onehot.permute(2, 0, 1)[None], P)[0].flatten(1).T * P * P  # (NP, M) pixels per patch
    tot = cnt_p.sum(1, keepdim=True)
    pix = backproject(depth)
    m = lab.reshape(-1) >= 0
    pts_lab = lab.reshape(-1)[m]
    cnt_t = torch.bincount(pts_lab, minlength=M)
    return {"name": f"{scene}/{fid}", "dino": dino, "depth": depth, "rgb": v["rgb"], "lab": lab, "M": M,
            "instance": ids[keep], "xyz": patch_centroids(depth), "uv": patch_grid(),
            "share": cnt_p / tot.clamp_min(1), "pvalid": tot[:, 0] > 0,
            "pts": pix[m], "pts_lab": pts_lab, "pt_order": torch.argsort(pts_lab, stable=True),
            "pt_start": torch.cumsum(cnt_t, 0) - cnt_t, "pt_count": cnt_t,
            "groups": groups(s, members), "R_g": gravity_rotation(v["c2w"])}
