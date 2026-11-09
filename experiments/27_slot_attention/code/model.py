"""Slot model: DINO patch tokens + 3D position -> Slot Attention -> one plain SQ + existence per slot.

tokens = MLP(LN(DINO 1024)) + Linear(Fourier(patch centroid xyz, camera frame, wavelengths 4 cm .. 20 m))   (d)
Slot Attention (Locatello et al. 2020), K slots with a learned init (BO-QSA, Jia et al. 2023: the slots are detached
before the last iteration, with a straight-through gradient to the init), T iterations, softmax over the slots.
The attention of the last iteration (N x K) is the mask; its logits feed the mask loss.
Optional `sa_layers` self-attention layers among the slots. Heads per slot:
scale 0.005 + softplus, shape 0.1 + 1.8 sigmoid, rotation from a quaternion (init identity), translation = anchor +
offset (anchor = soft 3D centroid of the slot's mask, detached; --anchor none: a learned vector, init 2 m in front of
the camera), existence logit (--exist_detach: on detached slot features, so it does not train the slots).
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from common import quat2mat


def fourier3d(x, n=10, lo=0.04, hi=20.0):
    """x (..., 3) metres -> (..., 6 n): sin / cos at n wavelengths log-spaced in [lo, hi]."""
    wl = torch.logspace(math.log10(lo), math.log10(hi), n, device=x.device)
    a = x[..., None] * (2 * math.pi / wl)
    return torch.cat([a.sin(), a.cos()], -1).flatten(-2)


class SlotAttention(nn.Module):
    def __init__(self, d, K, T, eps=1e-8):
        super().__init__()
        self.K, self.T, self.eps, self.scale = K, T, eps, d ** -0.5
        self.init = nn.Parameter(torch.randn(K, d) * d ** -0.5)
        self.norm_in, self.norm_s, self.norm_m = nn.LayerNorm(d), nn.LayerNorm(d), nn.LayerNorm(d)
        self.q, self.k, self.v = nn.Linear(d, d, bias=False), nn.Linear(d, d, bias=False), nn.Linear(d, d, bias=False)
        self.gru = nn.GRUCell(d, d)
        self.mlp = nn.Sequential(nn.Linear(d, 2 * d), nn.ReLU(), nn.Linear(2 * d, d))

    def forward(self, x):
        """x (N, d) -> slots (K, d), logits (N, K) of the last iteration."""
        x = self.norm_in(x)
        k, v = self.k(x), self.v(x)
        slots = self.init
        for it in range(self.T):
            if it == self.T - 1:  # BO-QSA
                slots = slots.detach() + self.init - self.init.detach()
            logits = k @ self.q(self.norm_s(slots)).T * self.scale  # (N, K)
            attn = logits.softmax(1) + self.eps
            upd = (attn / attn.sum(0, keepdim=True)).T @ v  # (K, d) weighted mean
            slots = self.gru(upd, slots)
            slots = slots + self.mlp(self.norm_m(slots))
        return slots, logits


class M2FLayer(nn.Module):
    """Mask2Former decoder layer: masked cross-attention (queries -> tokens), self-attention, FFN (post-norm)."""

    def __init__(self, d, heads=8):
        super().__init__()
        self.cross = nn.MultiheadAttention(d, heads, batch_first=True)
        self.self_ = nn.MultiheadAttention(d, heads, batch_first=True)
        self.ffn = nn.Sequential(nn.Linear(d, 4 * d), nn.GELU(), nn.Linear(4 * d, d))
        self.n1, self.n2, self.n3 = nn.LayerNorm(d), nn.LayerNorm(d), nn.LayerNorm(d)

    def forward(self, q, qpos, x, xpos, block):
        """q, qpos (K, d); x, xpos (N, d); block (K, N) bool, True = not attended."""
        h = self.cross(q[None] + qpos[None], x[None] + xpos[None], x[None], attn_mask=block)[0][0]
        q = self.n1(q + h)
        h = self.self_(q[None] + qpos[None], q[None] + qpos[None], q[None])[0][0]
        q = self.n2(q + h)
        return self.n3(q + self.ffn(q))


class M2FDecoder(nn.Module):
    """Mask2Former-style queries (--arch m2f): K learned queries (content + positional), L layers of masked
    cross-attention; masks = sigmoid(mask_embed(q) . mask_feature(token)), independent per query (no competition).
    A query attends only to the patches its previous mask covers (sigmoid > 0.5; all patches if none)."""

    def __init__(self, d, K, L=3):
        super().__init__()
        self.q = nn.Parameter(torch.randn(K, d) * d ** -0.5)
        self.qpos = nn.Parameter(torch.randn(K, d) * d ** -0.5)
        self.layers = nn.ModuleList(M2FLayer(d) for _ in range(L))
        self.norm = nn.LayerNorm(d)
        self.mask_embed = nn.Sequential(nn.Linear(d, d), nn.ReLU(), nn.Linear(d, d))
        self.mask_feat = nn.Linear(d, d)

    def masks(self, q, mf):
        return mf @ self.mask_embed(self.norm(q)).T  # (N, K) logits

    def forward(self, x, xpos):
        mf = self.mask_feat(x)
        q = self.q
        for lay in self.layers:
            with torch.no_grad():
                block = (self.masks(q, mf).T.sigmoid() < 0.5)  # (K, N)
                block[block.all(1)] = False
            q = lay(q, self.qpos, x, xpos, block)
        return self.norm(q), self.masks(q, mf)


class PartDecoder(nn.Module):
    """--parts G: a set of G part SQs per slot. Part queries = slot feature + G learned part embeddings; L layers of
    self-attention among a slot's parts, cross-attention to the tokens restricted to the slot's patches (mask > 0.5;
    all if none), FFN (post-norm). Per part: plain SQ (translation = the slot's anchor + offset; --anchor none: predicted
    directly in the camera frame, init 2 m ahead) + existence logit."""

    def __init__(self, d, G=16, L=2, heads=8):
        super().__init__()
        self.emb = nn.Parameter(torch.randn(G, d) * d ** -0.5)
        self.self_ = nn.ModuleList(nn.MultiheadAttention(d, heads, batch_first=True) for _ in range(L))
        self.cross = nn.ModuleList(nn.MultiheadAttention(d, heads, batch_first=True) for _ in range(L))
        self.ffn = nn.ModuleList(nn.Sequential(nn.Linear(d, 4 * d), nn.GELU(), nn.Linear(4 * d, d)) for _ in range(L))
        self.norms = nn.ModuleList(nn.LayerNorm(d) for _ in range(3 * L))
        self.scale, self.shape, self.rot = nn.Linear(d, 3), nn.Linear(d, 2), nn.Linear(d, 4)
        self.trans, self.exist = nn.Linear(d, 3), nn.Linear(d, 1)
        nn.init.constant_(self.scale.bias, math.log(math.e ** 0.1 - 1))  # 0.1 m half-axes
        nn.init.zeros_(self.rot.weight), nn.init.constant_(self.rot.bias, 0.0), self.rot.bias.data.__setitem__(0, 1.0)
        nn.init.normal_(self.trans.weight, std=1e-3), nn.init.zeros_(self.trans.bias)

    def forward(self, h, x, xpos, mask, anchor):
        """h (K, d) slots, x / xpos (N, d) tokens, mask (N, K) slot mask probabilities, anchor (K, 3) -> dict (K, G, ...)."""
        K = len(h)
        q = h[:, None] + self.emb[None]  # (K, G, d)
        kpm = mask.T < 0.5  # (K, N), True = not attended
        kpm[kpm.all(1)] = False
        mem, val = (x + xpos)[None].expand(K, -1, -1), x[None].expand(K, -1, -1)
        for i, (sa, ca, ff) in enumerate(zip(self.self_, self.cross, self.ffn)):
            n1, n2, n3 = self.norms[3 * i: 3 * i + 3]
            q = n1(q + sa(q, q, q)[0])
            q = n2(q + ca(q, mem, val, key_padding_mask=kpm)[0])
            q = n3(q + ff(q))
        return {"scale": F.softplus(self.scale(q)) + 0.005, "shape": 0.1 + 1.8 * torch.sigmoid(self.shape(q)),
                "rotate": quat2mat(self.rot(q)), "trans": anchor[:, None] + self.trans(q), "exist_logit": self.exist(q)[..., 0]}


class Net(nn.Module):
    def __init__(self, d=128, K=128, T=3, sa_layers=0, anchor="centroid", exist_detach=False, arch="slot", parts=0):
        super().__init__()
        self.anchor, self.exist_detach, self.arch = anchor, exist_detach, arch
        self.part_dec = PartDecoder(d, parts) if parts else None
        if parts and anchor == "none":  # parts predict their translation directly (camera frame), init 2 m ahead
            self.part_dec.trans.bias.data[2] = 2.0
        self.inp = nn.Sequential(nn.LayerNorm(1024), nn.Linear(1024, 256), nn.GELU(), nn.Linear(256, d))
        self.pos = nn.Linear(60, d)
        self.sa = SlotAttention(d, K, T) if arch == "slot" else None
        self.dec = M2FDecoder(d, K) if arch == "m2f" else None
        self.layers = nn.ModuleList(nn.TransformerEncoderLayer(d, 8, 4 * d, dropout=0.0, activation="gelu",
                                                               batch_first=True, norm_first=True) for _ in range(sa_layers))
        self.out_norm = nn.LayerNorm(d)
        self.scale, self.shape, self.rot = nn.Linear(d, 3), nn.Linear(d, 2), nn.Linear(d, 4)
        self.trans, self.exist = nn.Linear(d, 3), nn.Linear(d, 1)
        nn.init.constant_(self.scale.bias, math.log(math.e ** 0.2 - 1))  # 0.2 m half-axes
        nn.init.zeros_(self.rot.weight), nn.init.constant_(self.rot.bias, 0.0), self.rot.bias.data.__setitem__(0, 1.0)
        nn.init.zeros_(self.trans.weight), nn.init.zeros_(self.trans.bias)
        if anchor == "none":
            self.trans.bias.data[2] = 2.0

    def forward(self, dino, xyz, gt_share=None):
        """dino (N, 1024), xyz (N, 3) -> dict of per-slot SQ (K, ...), exist logit (K,), mask logits (N, K).
        gt_share (N, M) (--gt_masks diagnostic): no Slot Attention; slot m = the tokens averaged with target m's GT
        patch shares, mask logits = log(share) (K = M slots, slot m = target m)."""
        pe = self.pos(fourier3d(xyz))
        x = self.inp(dino) + pe
        if self.arch == "m2f" and gt_share is None:
            slots, logits = self.dec(x, pe)
        elif gt_share is not None:
            slots = (gt_share.T @ x) / gt_share.sum(0)[:, None].clamp_min(1e-6)
            logits = (gt_share + 1e-4).log()
        else:
            slots, logits = self.sa(x)
        for lay in self.layers:
            slots = lay(slots[None])[0]
        h = self.out_norm(slots)
        out = {"scale": F.softplus(self.scale(h)) + 0.005, "shape": 0.1 + 1.8 * torch.sigmoid(self.shape(h)),
               "rotate": quat2mat(self.rot(h)), "exist_logit": self.exist(h.detach() if self.exist_detach else h)[:, 0], "logits": logits,
               "sigmoid": self.arch == "m2f" and gt_share is None}
        if self.anchor == "centroid":
            a = (logits.detach().sigmoid() if out["sigmoid"] else logits.detach().softmax(1))  # (N, K)
            c = (a.T @ xyz) / a.sum(0)[:, None].clamp_min(1e-6)
            out["anchor"] = c
            out["trans"] = c + self.trans(h)
        else:
            out["trans"] = self.trans(h)
        if self.part_dec is not None:  # parts: queries see the slot's patches; anchored at the slot's mask centroid
            a = (logits.detach().sigmoid() if out["sigmoid"] else logits.detach().softmax(1))
            if self.anchor == "centroid":
                c = out["anchor"]
            else:  # --anchor none: no anchor for the parts either
                c = torch.zeros(len(h), 3, device=h.device)
            out["parts"] = self.part_dec(h, x, pe, a, c)
        return out
