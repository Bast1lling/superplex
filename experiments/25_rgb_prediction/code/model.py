"""Single-view SQ predictor without 3D input: DINOv3 tokens of the RGB image only (+ optionally the camera pose).

Tokens: per patch, LayerNorm(DINOv3 token) -> Linear 1024 -> 256 -> GELU -> d, plus a Linear of the Fourier encoding
of the patch's 2D grid position; with --pose also a Linear of the Fourier encoding of the gravity direction in the
camera frame (3) and the camera height above the scene's floor (1), added to every token. Encoder: `layers` pre-norm
self-attention layers.

3D comes from a per-patch depth head (DEPTH = "pred", default): log depth of the patch (trained against the mean
depth of its pixels, L1 in log space), anchor = ray through the patch center * predicted depth (detached). After the
encoder the tokens get a Linear of the Fourier encoding of their (predicted) 3D anchor, then exp 22's proposal heads
(translation = anchor + offset) and top-k decoder follow unchanged.
DEPTH = "none": no depth head, anchor = 3 m along the ray, no 3D encoding (everything from the offsets).
DEPTH = "gt": anchor = the patch's GT centroid (RGB-D oracle, = exp 22 / 24's input positions).
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from common import GH, GW, NP, fourier, patch_centroids, patch_rays, peaks, quat2mat

DEPTH = "pred"


class Heads(nn.Module):
    def __init__(self, d, n_classes):
        super().__init__()
        self.scale, self.shape, self.rot, self.trans = nn.Linear(d, 3), nn.Linear(d, 2), nn.Linear(d, 4), nn.Linear(d, 3)
        self.exist, self.cls = nn.Linear(d, 1), nn.Linear(d, n_classes)
        self.mask_q = nn.Sequential(nn.Linear(d, d), nn.GELU(), nn.Linear(d, d))
        nn.init.constant_(self.scale.bias, math.log(math.e ** 0.2 - 1))
        nn.init.zeros_(self.trans.weight), nn.init.zeros_(self.trans.bias)

    def forward(self, q, ref):
        return {"scale": F.softplus(self.scale(q)) + 0.005, "shape": 0.1 + 1.8 * torch.sigmoid(self.shape(q)),
                "rotate": quat2mat(self.rot(q)), "trans": ref + self.trans(q),
                "exist": torch.sigmoid(self.exist(q))[..., 0], "cls": self.cls(q), "mq": self.mask_q(q)}


class RefineHeads(Heads):
    def __init__(self, d, n_classes):
        super().__init__(d, n_classes)
        for lin in (self.scale, self.shape):
            nn.init.zeros_(lin.weight), nn.init.zeros_(lin.bias)

    def forward(self, q, ref):
        out = super().forward(q, ref["trans"])
        u = ((ref["shape"] - 0.1) / 1.8).clamp(1e-4, 1 - 1e-4)
        out["scale"] = ref["scale"] * torch.exp(self.scale(q))
        out["shape"] = 0.1 + 1.8 * torch.sigmoid(torch.logit(u) + self.shape(q))
        return out


def layer(d, kind):
    cls = nn.TransformerEncoderLayer if kind == "enc" else nn.TransformerDecoderLayer
    return cls(d, 8, 512, dropout=0.0, activation="gelu", batch_first=True, norm_first=True)


class Net(nn.Module):
    def __init__(self, n_classes, d=128, layers=1, dec_layers=1, k=300, pose=False):
        super().__init__()
        self.k, self.pose = k, pose
        self.inp = nn.Sequential(nn.LayerNorm(1024), nn.Linear(1024, 256), nn.GELU(), nn.Linear(256, d))
        self.pos2d = nn.Linear(2 * 2 * 6, d)
        self.pose_emb = nn.Linear(2 * 4 * 6, d) if pose else None
        self.encoder = nn.TransformerEncoder(layer(d, "enc"), layers, norm=nn.LayerNorm(d), enable_nested_tensor=False)
        self.depth = nn.Linear(d, 1)
        nn.init.zeros_(self.depth.weight), nn.init.constant_(self.depth.bias, math.log(2.5))
        self.pos3d = nn.Linear(48, d)
        self.heads = Heads(d, n_classes)
        self.mask_p = nn.Linear(d, d)
        self.dec = nn.ModuleList(layer(d, "dec") for _ in range(dec_layers))
        self.dec_norm = nn.ModuleList(nn.LayerNorm(d) for _ in range(dec_layers))
        self.dec_pos = nn.Linear(48, d)
        self.dec_heads = nn.ModuleList(RefineHeads(d, n_classes) for _ in range(dec_layers))

    def forward(self, batch):
        dino = batch["dino"][:, 0]  # (B, GH, GW, 1024), view 0 only
        B = dino.shape[0]
        dev = dino.device
        gy, gx = torch.meshgrid(torch.linspace(0, 1, GH, device=dev), torch.linspace(0, 1, GW, device=dev), indexing="ij")
        grid = torch.stack([gx, gy], -1).reshape(1, NP, 2)
        tok = self.inp(dino.reshape(B, NP, -1)) + self.pos2d(fourier(grid, 6, 2.0))
        if self.pose:
            g = batch["c2w"][:, 0, 2, :3]  # world up (z) in the camera frame... row 2 of R = up expressed in camera axes
            pose = torch.cat([g, batch["height"][:, None]], -1)  # (B, 4)
            tok = tok + self.pose_emb(fourier(pose, 6, 4.0))[:, None]
        mem = self.encoder(tok)
        ray = patch_rays(dev)[None]
        out = {}
        if DEPTH == "gt":
            anchor, _ = patch_centroids(batch["depth"][:, 0])
        elif DEPTH == "pred":
            logd = self.depth(mem)[..., 0]
            out["logd"] = logd
            anchor = ray * logd.detach().exp()[..., None]
        else:
            anchor = (ray * 3.0).expand(B, -1, -1)
        if DEPTH != "none":
            mem = mem + self.pos3d(fourier(anchor, 8, 16.0))
        enc = self.heads(mem, anchor)
        mp = self.mask_p(mem)
        enc["mp"] = mp
        top = peaks(enc["exist"].detach(), 1).topk(self.k, dim=1).indices
        pick = lambda x: torch.gather(x, 1, top.reshape(*top.shape, *[1] * (x.dim() - 2)).expand(-1, -1, *x.shape[2:]))
        q = pick(mem)
        ref = {k: pick(enc[k]).detach() for k in ("scale", "shape", "trans")}
        outs = []
        for lay, norm, head in zip(self.dec, self.dec_norm, self.dec_heads):
            q = lay(q + self.dec_pos(fourier(ref["trans"], 8, 16.0)), mem)
            outs.append(head(norm(q), ref) | {"mp": mp})
            ref = {k: outs[-1][k].detach() for k in ("scale", "shape", "trans")}
        return outs[-1] | out | {"enc": enc, "aux": outs[:-1], "select": top}


def masks(pred, fb, qs):
    out = pred["mp"].new_zeros(len(fb), pred["mp"].shape[1])
    for b in torch.unique(fb).tolist():
        m = torch.nonzero(fb == b)[:, 0]
        out = out.index_put((m,), pred["mq"][b, qs[m]] @ pred["mp"][b].T)
    return out
