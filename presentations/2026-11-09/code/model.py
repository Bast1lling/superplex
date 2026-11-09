"""The slot model: DINO patches (+ 3D patch positions) -> Slot Attention -> instance masks + superquadrics.

tokens = MLP(LN(DINO 1024)) + Linear(Fourier(patch position))                                               (N, d)
  position: the patch's 3D centroid from depth (--depth), else its 2D image position (no depth input).
Slot Attention (Locatello et al. 2020; learned init, BO-QSA, Jia et al. 2023): K slots, T iterations, softmax over
  the slots; the attention of the last iteration is the instance mask (N, K).
Per slot h (K, d), one of two geometry decoders:
- `head`: one SQ per slot: scale 0.005 + softplus, shape 0.1 + 1.8 sigmoid, rotation from a quaternion, translation =
  anchor (the slot mask's 3D centroid; without depth: none, the translation is regressed, init 2 m ahead) + offset.
- `parts` (exp 28's relative part decoder): 16 part queries per slot (slot feature + learned embeddings), 2 layers of
  self-attention among them, cross-attention to the slot's patches (mask > 0.5) and FFN. The patches are seen as
  DINO content + Fourier(position - anchor), no absolute position. Part centre = a softmax pointer over the slot's
  patches times their 3D centroids + a small offset; scale / shape / rotation regressed; existence logit.
Plus an existence logit per slot. Parameter names as in exp 27 / 28 (their checkpoints load into Net).
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from sq import fourier, quat2mat


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
            if it == self.T - 1:  # BO-QSA: detach before the last iteration, straight-through to the init
                slots = slots.detach() + self.init - self.init.detach()
            logits = k @ self.q(self.norm_s(slots)).T * self.scale  # (N, K)
            attn = logits.softmax(1) + self.eps  # softmax over the slots: the slots compete for each patch
            slots = self.gru((attn / attn.sum(0, keepdim=True)).T @ v, slots)  # weighted mean of the values
            slots = slots + self.mlp(self.norm_m(slots))
        return slots, logits


def sq_heads(mod, q):
    return {"scale": F.softplus(mod.scale(q)) + 0.005, "shape": 0.1 + 1.8 * torch.sigmoid(mod.shape(q)),
            "rotate": quat2mat(mod.rot(q)), "exist_logit": mod.exist(q)[..., 0]}


def init_heads(mod, half_axes, trans_std):
    nn.init.constant_(mod.scale.bias, math.log(math.e ** half_axes - 1))
    nn.init.zeros_(mod.rot.weight), nn.init.zeros_(mod.rot.bias), mod.rot.bias.data.__setitem__(0, 1.0)  # identity
    nn.init.normal_(mod.trans.weight, std=trans_std), nn.init.zeros_(mod.trans.bias)


class PartDecoder(nn.Module):
    def __init__(self, d, G=16, L=2, heads=8):
        super().__init__()
        self.emb = nn.Parameter(torch.randn(G, d) * d ** -0.5)
        self.self_ = nn.ModuleList(nn.MultiheadAttention(d, heads, batch_first=True) for _ in range(L))
        self.cross = nn.ModuleList(nn.MultiheadAttention(d, heads, batch_first=True) for _ in range(L))
        self.ffn = nn.ModuleList(nn.Sequential(nn.Linear(d, 4 * d), nn.GELU(), nn.Linear(4 * d, d)) for _ in range(L))
        self.norms = nn.ModuleList(nn.LayerNorm(d) for _ in range(3 * L))
        self.scale, self.shape, self.rot = nn.Linear(d, 3), nn.Linear(d, 2), nn.Linear(d, 4)
        self.trans, self.exist = nn.Linear(d, 3), nn.Linear(d, 1)
        self.posrel = nn.Linear(60, d)  # Fourier(patch centroid - slot anchor)
        self.ptr_q, self.ptr_k = nn.Linear(d, d), nn.Linear(d, d)  # the pointer's query / key projections
        init_heads(self, 0.1, 1e-3)

    def cross_attn(self, ca, q, mem, block):
        """ca's multi-head attention, one memory per slot: q (S, G, d), mem (S, N, d), block (S, N) True = masked."""
        S, G, d = q.shape
        h = ca.num_heads
        Wq, Wk, Wv = ca.in_proj_weight.chunk(3)
        bq, bk, bv = ca.in_proj_bias.chunk(3)
        Q = F.linear(q, Wq, bq).reshape(S, G, h, d // h).transpose(1, 2)
        Kx = F.linear(mem, Wk, bk).reshape(S, -1, h, d // h).transpose(1, 2)
        Vx = F.linear(mem, Wv, bv).reshape(S, -1, h, d // h).transpose(1, 2)
        o = F.scaled_dot_product_attention(Q, Kx, Vx, attn_mask=~block[:, None, None, :])
        return ca.out_proj(o.transpose(1, 2).reshape(S, G, d))

    def forward(self, h, content, xyz, mask, anchor):
        """h (S, d) slots, content (N, d) patch tokens without position, xyz (N, 3), mask (N, S) slot mask
        probabilities, anchor (S, 3) -> parts dict (S, G, ...) and the pointer (S, G, N)."""
        block = mask.T < 0.5  # a slot's parts see only its patches (all if it owns none)
        block[block.all(1)] = False
        mem = content[None] + self.posrel(fourier(xyz[None] - anchor[:, None]))  # (S, N, d)
        q = h[:, None] + self.emb[None]
        for i, (sa, ca, ff) in enumerate(zip(self.self_, self.cross, self.ffn)):
            n1, n2, n3 = self.norms[3 * i: 3 * i + 3]
            q = n1(q + sa(q, q, q)[0])
            q = n2(q + self.cross_attn(ca, q, mem, block))
            q = n3(q + ff(q))
        lg = self.ptr_q(q) @ self.ptr_k(mem).transpose(1, 2) / math.sqrt(q.shape[-1])  # (S, G, N)
        ptr = lg.masked_fill(block[:, None], float("-inf")).softmax(-1)
        return sq_heads(self, q) | {"trans": ptr @ xyz + self.trans(q)}, ptr


class Net(nn.Module):
    def __init__(self, d=128, K=64, T=3, depth=True, decoder="parts", parts=16):
        super().__init__()
        self.depth, self.decoder = depth, decoder
        self.inp = nn.Sequential(nn.LayerNorm(1024), nn.Linear(1024, 256), nn.GELU(), nn.Linear(256, d))
        self.pos = nn.Linear(60 if depth else 40, d)
        self.sa = SlotAttention(d, K, T)
        self.out_norm = nn.LayerNorm(d)
        self.scale, self.shape, self.rot = nn.Linear(d, 3), nn.Linear(d, 2), nn.Linear(d, 4)
        self.trans, self.exist = nn.Linear(d, 3), nn.Linear(d, 1)
        init_heads(self, 0.2, 0.0)
        if not depth:
            self.trans.bias.data[2] = 2.0  # no anchor: start 2 m ahead
        self.part_dec = PartDecoder(d, parts) if decoder == "parts" else None

    def forward(self, dino, xyz, uv=None, match=None):
        """dino (N, 1024), xyz (N, 3) patch centroids (ignored without depth), uv (N, 2) -> dict: per slot SQ (K, ...),
        exist_logit (K,), logits (N, K), anchor (K, 3); parts (K, G, ...) and pointer (K, G, N) for the part decoder.
        match: function out -> slot per target; then the parts are decoded only for the matched slots (training: the
        others get no geometry loss), out["sig"] = the matching."""
        pe = self.pos(fourier(xyz) if self.depth else fourier(uv, 10, 0.02, 2.0))
        x = self.inp(dino.float()) + pe
        s, logits = self.sa(x)
        h = self.out_norm(s)
        out = sq_heads(self, h) | {"logits": logits}
        a = logits.detach().softmax(1)
        if self.depth:
            c = out["anchor"] = (a.T @ xyz) / a.sum(0)[:, None].clamp_min(1e-6)
            out["trans"] = c + self.trans(h)
        else:
            out["trans"] = self.trans(h)
        if self.part_dec is None:
            return out
        slots = torch.arange(len(h), device=h.device)
        if match is not None:
            slots = out["sig"] = match(out)
        p, ptr = self.part_dec(h[slots], x - pe, xyz, a[:, slots], out["anchor"][slots])
        out["parts"] = {k: v.new_zeros((len(h),) + v.shape[1:]).index_copy(0, slots, v) for k, v in p.items()}
        out["pointer"] = ptr.new_zeros((len(h),) + ptr.shape[1:]).index_copy(0, slots, ptr)
        return out
