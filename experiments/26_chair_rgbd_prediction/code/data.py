"""Chairs of exp 18 (`~/data/ShapeNet-Chairs-RGBD`) as single RGB-D views, with exp 19's SuperFlex fits as targets.

Item = (chair, view, texture): train = a random view of 8 and a random texture of 4 (t00 = original materials,
t01-t03 random procedural textures); evaluation = view (index % 8), texture t00.
- rgb (512, 512, 3) uint8 (DINOv3 is run on the GPU in train.py), depth (128, 128) = full-resolution pixels 4i + 1;
- targets: the chair's SuperFlex fit (`sq_fits/<split>/superflex_opt_plain.npz`: SuperFlex's network on the full point
  cloud, exist > 0.5, then 1000 steps of its objective, plain SQs; ~8 SQs per chair) moved into the camera frame
  (OpenCV axes), instance = SQ index, cls = 0;
- pixel owner: the target SQ with the smallest |SDF| at the back-projected point (label (128, 128), -1 background),
  vis = SQs owning >= 1 pixel; per patch the SQ owning >= 50 % of its pixels (+ each visible SQ without such a patch
  gets its best patch), heatmap: exp 22's CenterNet Gaussian (sigma = max(0.5, sqrt(patches) / 4), 1 at the
  patch deepest inside the SQ's region);
- occupancy: 8192 of the chair's 100k labelled query points (exp 18 / 20: depth fusion + surface margin), camera frame.
The decoder is trained against **all** SQs of the chair (amodal), the proposals only on visible ones.
"""

from functools import lru_cache
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from scipy.ndimage import distance_transform_edt

from common import CX, CY, FX, FY, GH, GW, H, NP, P, W

ROOT = Path("/home/sebastian/data/ShapeNet-Chairs-RGBD")
DATA = ROOT / "03001627"
FITS = ROOT / "sq_fits"
EXP = Path(__file__).resolve().parents[1]
N_OCC = 8192


@lru_cache(maxsize=None)
def fits(split):
    z = np.load(FITS / split / "superflex_opt_plain.npz")
    return {k: z[k] for k in z.files}, {m: i for i, m in enumerate(z["ids"])}


def sdf_np(x, scale, shape, rot, trans):
    """Radial SDF (log space) of SQs at points x (M, 3) -> (S, M)."""
    X = np.einsum("sji,smj->smi", rot, x[None] - trans[:, None])
    l = np.log(np.maximum(np.abs(X), 1e-6) / scale[:, None])
    e1, e2 = shape[:, :1], shape[:, 1:]
    lxy = np.logaddexp(2 / e2 * l[..., 0], 2 / e2 * l[..., 1]) * (e2 / e1)
    lF = np.logaddexp(lxy, 2 / e1 * l[..., 2])
    return np.linalg.norm(X, axis=-1) * (1 - np.exp(-e1 / 2 * lF))


def heatmap(owner):
    own = owner.reshape(GH, GW)
    heat = np.zeros((GH, GW), np.float32)
    gy, gx = np.indices((GH, GW))
    for s in np.unique(own[own >= 0]):
        m = own == s
        c = np.unravel_index(np.argmax(distance_transform_edt(np.pad(m, 1))[1:-1, 1:-1]), m.shape)
        sigma = max(0.5, np.sqrt(m.sum()) / 4)
        h = np.exp(-((gy - c[0]) ** 2 + (gx - c[1]) ** 2) / (2 * sigma ** 2)) * m
        h[c] = 1.0
        heat = np.maximum(heat, h)
    return heat.reshape(-1)


class Chairs(torch.utils.data.Dataset):
    def __init__(self, split, train=True, limit=None, views_per_chair=1, fit_split=None):
        self.split, self.train = split, train
        fz, idx = fits(fit_split or split)
        ids = [m for m in (DATA / f"{split}.lst").read_text().split() if m in idx]
        if limit:
            ids = ids[:: max(1, len(ids) // limit)][:limit]
        self.items = [(m, v) for m in ids for v in range(views_per_chair)]
        self.fit_split = fit_split or split

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        m, k = self.items[i]
        v = np.random.randint(8) if self.train else (i * 3 + k) % 8
        t = np.random.randint(4) if self.train else 0
        return self.make(m, v, t, None if self.train else i)

    def make(self, m, v, t, seed=None):
        d = DATA / m
        cam = np.load(d / "cameras.npz")
        rgb = np.asarray(Image.open(d / "rgb" / f"t{t:02d}_v{v:02d}.png").convert("RGB"))
        depth = np.asarray(Image.open(d / "depth" / f"v{v:02d}.png")).astype(np.float32)[1::4, 1::4] * float(cam["depth_unit"])
        w2c = np.linalg.inv(cam["c2w"][v])
        fz, idx = fits(self.fit_split)
        j = idx[m]
        e = fz["exist"][j] > 0.5
        R, tt = w2c[:3, :3], w2c[:3, 3]
        tg = {"scale": fz["scale"][j][e].astype(np.float32), "exponents": fz["exponents"][j][e].astype(np.float32),
              "rotation": (R @ fz["rotation"][j][e]).astype(np.float32),
              "translation": (fz["translation"][j][e] @ R.T + tt).astype(np.float32)}
        T = len(tg["scale"])
        # pixel owner
        yy, xx = np.nonzero(depth > 0)
        z = depth[yy, xx]
        x = np.stack([(xx - CX) / FX * z, (yy - CY) / FY * z, z], -1)
        label = np.full((H, W), -1, np.int64)
        if T and len(z):
            label[yy, xx] = np.abs(sdf_np(x, tg["scale"], tg["exponents"], tg["rotation"], tg["translation"])).argmin(0)
        share = np.stack([(label == s).reshape(GH, P, GW, P).mean((1, 3)).reshape(-1) for s in range(T)]) if T else np.zeros((0, NP))
        best = share.max(0) if T else np.zeros(NP)
        owner = np.where(best >= 0.5, share.argmax(0) if T else -1, -1).astype(np.int64)
        vis = share.sum(1) > 0 if T else np.zeros(0, bool)
        for s in np.flatnonzero(vis & (np.bincount(owner[owner >= 0], minlength=T) == 0)):
            owner[share[s].argmax()] = s
        tg |= {"instance": np.arange(T, dtype=np.int64), "cls": np.zeros(T, np.int64), "gid": np.arange(T, dtype=np.int64),
               "vis": vis[:, None], "f": vis[:, None].astype(np.float32)}
        q = np.load(d / "points.npz")
        pts = q["points"].astype(np.float32)
        occ = np.unpackbits(q["occupancies"])[:len(pts)].astype(bool)
        sel = np.random.RandomState(seed).choice(len(pts), N_OCC, replace=False)
        return {"name": f"{m}_v{v}_t{t}", "rgb": rgb[None], "depth": depth[None], "label": label[None],
                "T0v": np.eye(4, dtype=np.float32)[None], "owner": owner[None], "heat": heatmap(owner)[None],
                "occ_x": (pts[sel] @ R.T + tt).astype(np.float32), "occ": occ[sel], "targets": tg}


def collate(batch):
    out = {k: torch.tensor(np.stack([b[k] for b in batch])) for k in ("rgb", "depth", "label", "T0v", "owner", "heat", "occ_x", "occ")}
    out["name"] = [b["name"] for b in batch]
    out["targets"] = [{k: torch.tensor(v) for k, v in b["targets"].items()} for b in batch]
    return out


def to_device(batch, dev="cuda"):
    out = {k: v.to(dev, non_blocking=True) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
    out["targets"] = [{k: v.to(dev) for k, v in t.items()} for t in batch["targets"]]
    return out


class Dino:
    """Frozen DINOv3 ViT-L/16 on the GPU: rgb (B, 1, 512, 512, 3) uint8 -> (B, 1, 32, 32, 1024) float."""

    def __init__(self):
        from transformers import AutoModel
        self.net = AutoModel.from_pretrained("facebook/dinov3-vitl16-pretrain-lvd1689m", dtype=torch.bfloat16).cuda().eval()
        self.skip = 1 + self.net.config.num_register_tokens
        self.mean = torch.tensor([0.485, 0.456, 0.406], device="cuda").view(1, 3, 1, 1)
        self.std = torch.tensor([0.229, 0.224, 0.225], device="cuda").view(1, 3, 1, 1)

    @torch.no_grad()
    def __call__(self, rgb):
        B = rgb.shape[0]
        x = ((rgb[:, 0].permute(0, 3, 1, 2).float() / 255 - self.mean) / self.std).to(torch.bfloat16)
        tok = self.net(pixel_values=x).last_hidden_state[:, self.skip:]
        return tok.float().reshape(B, 1, GH, GW, -1)
