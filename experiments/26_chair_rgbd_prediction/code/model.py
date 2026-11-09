"""Two-view (or one-view) SQ predictor: exp 22's top-k design (DINO only), rewritten for V views in a common frame.

Tokens: per patch of each view, LayerNorm(DINOv3 token) -> Linear 1024 -> 256 -> GELU -> d, plus a Linear of the
Fourier encoding of the patch's 3D centroid (from the view's depth) **in view 0's camera frame**, plus a learned view
embedding. Encoder: `layers` pre-norm self-attention layers over all V * 3150 tokens (the views exchange information
through their 3D-consistent positions). Per token (proposal) heads: plain SQ (translation = anchor + offset),
existence (CenterNet heatmap), class, mask embedding (mask logits = embedding . patch embedding over all views).
Decoder: the K best 3 x 3 local maxima of the existence map of each view (pooled over views) become queries (their
encoder token, reference SQ detached), `dec_layers` pre-norm decoder layers (self-attention, cross-attention to all
tokens), refinement heads (scale * exp(delta), exponent logits + delta, translation + delta).
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from common import NP, POS_SCALE, fourier, patch_centroids, peaks, quat2mat


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
    def __init__(self, n_classes, d=128, layers=1, dec_layers=1, k=300, views=2, view_emb=True):
        super().__init__()
        self.k, self.views = k, views
        self.inp = nn.Sequential(nn.LayerNorm(1024), nn.Linear(1024, 256), nn.GELU(), nn.Linear(256, d))
        self.pos = nn.Linear(48, d)
        self.view_emb = nn.Embedding(2, d) if view_emb else None
        self.encoder = nn.TransformerEncoder(layer(d, "enc"), layers, norm=nn.LayerNorm(d), enable_nested_tensor=False)
        self.heads = Heads(d, n_classes)
        self.mask_p = nn.Linear(d, d)
        self.dec = nn.ModuleList(layer(d, "dec") for _ in range(dec_layers))
        self.dec_norm = nn.ModuleList(nn.LayerNorm(d) for _ in range(dec_layers))
        self.dec_pos = nn.Linear(48, d)
        self.dec_heads = nn.ModuleList(RefineHeads(d, n_classes) for _ in range(dec_layers))

    def anchors(self, depth, T0v):
        """depth (B, V, H, W), T0v (B, V, 4, 4) -> patch centroids in view 0's frame (B, V * NP, 3)."""
        B, V = depth.shape[:2]
        c, _ = patch_centroids(depth.flatten(0, 1))
        c = c.reshape(B, V, NP, 3)
        c = (T0v[:, :, None, :3, :3] @ c[..., None])[..., 0] + T0v[:, :, None, :3, 3]
        return c.reshape(B, V * NP, 3)

    def tokens(self, dino, depth, T0v):
        B, V = dino.shape[:2]
        anchor = self.anchors(depth, T0v)
        tok = self.inp(dino.reshape(B, V * NP, -1)) + self.pos(fourier(anchor, 8, POS_SCALE))
        if self.view_emb is not None:
            tok = tok + self.view_emb.weight[:V].repeat_interleave(NP, 0)[None]
        return tok, anchor

    def forward(self, dino, depth, T0v):
        B, V = dino.shape[:2]
        tok, anchor = self.tokens(dino, depth, T0v)
        mem = self.encoder(tok)
        enc = self.heads(mem, anchor)
        mp = self.mask_p(mem)
        enc["mp"] = mp
        top = peaks(enc["exist"].detach(), V).topk(min(self.k, V * NP), dim=1).indices
        pick = lambda x: torch.gather(x, 1, top.reshape(*top.shape, *[1] * (x.dim() - 2)).expand(-1, -1, *x.shape[2:]))
        q = pick(mem)
        ref = {k: pick(enc[k]).detach() for k in ("scale", "shape", "trans")}
        outs = []
        for lay, norm, head in zip(self.dec, self.dec_norm, self.dec_heads):
            q = lay(q + self.dec_pos(fourier(ref["trans"], 8, POS_SCALE)), mem)
            outs.append(head(norm(q), ref) | {"mp": mp})
            ref = {k: outs[-1][k].detach() for k in ("scale", "shape", "trans")}
        return outs[-1] | {"enc": enc, "aux": outs[:-1], "select": top}


def masks(pred, fb, qs):
    """Mask logits (n, V * NP) of the queries (fb, qs)."""
    out = pred["mp"].new_zeros(len(fb), pred["mp"].shape[1])
    for b in torch.unique(fb).tolist():
        m = torch.nonzero(fb == b)[:, 0]
        out = out.index_put((m,), pred["mq"][b, qs[m]] @ pred["mp"][b].T)
    return out
