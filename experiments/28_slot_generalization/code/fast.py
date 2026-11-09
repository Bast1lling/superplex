"""Faster, numerically equivalent versions of exp 27's part decoder (same parameters, state dicts interchangeable).

1. Shared key / value projections: exp 27's PartDecoder expands the token memory to (K, N, d) and lets
   nn.MultiheadAttention project it once per slot; here keys / values are projected once per layer (N, d) and
   attended by all slots' part queries with F.scaled_dot_product_attention (per-slot patch mask as attn_mask).
2. `slots`: the part decoder runs only for the given slots (training: the matched ones; unmatched slots' parts get
   no loss, so the gradients are the same). The outputs are scattered back to (K, G, ...) (zeros elsewhere).
Checked against exp 27's forward in `check()` (python fast.py).

RelNet (--part_dec rel, exp 28's "option 1": geometry relative to the slot): the part decoder's memory per slot is the
patch content (DINO projection, no absolute position) + an encoding of the patch position relative to the slot's
anchor (Fourier(xyz - anchor)); a part's translation = an attention-weighted average of the slot's patch centroids
(pointer over the patches the slot owns, its own query / key projections) + a small offset. So the part positions
move with the patches by construction instead of being regressed in absolute coordinates.
"""

import sys
from pathlib import Path

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "27_slot_attention" / "code"))
import model as M27  # noqa: E402
from common import quat2mat  # noqa: E402


def _cross(ca, q, mem, val, block):
    """nn.MultiheadAttention ca (batch_first) on queries q (S, G, d), shared memory mem / val (N, d), block (S, N)
    True = not attended -> (S, G, d)."""
    S, G, d = q.shape
    h = ca.num_heads
    Wq, Wk, Wv = ca.in_proj_weight.chunk(3)
    bq, bk, bv = ca.in_proj_bias.chunk(3)
    Q = F.linear(q, Wq, bq).reshape(S, G, h, d // h).transpose(1, 2)  # (S, h, G, dh)
    Kx = F.linear(mem, Wk, bk).reshape(-1, h, d // h).transpose(0, 1)[None].expand(S, -1, -1, -1)  # (S, h, N, dh)
    Vx = F.linear(val, Wv, bv).reshape(-1, h, d // h).transpose(0, 1)[None].expand(S, -1, -1, -1)
    o = F.scaled_dot_product_attention(Q, Kx, Vx, attn_mask=~block[:, None, None, :])  # True = attend
    return ca.out_proj(o.transpose(1, 2).reshape(S, G, d))


def part_forward(pd, h, x, xpos, mask, anchor):
    """exp 27's PartDecoder.forward with shared projections (same signature / outputs)."""
    q = h[:, None] + pd.emb[None]
    block = mask.T < 0.5
    block[block.all(1)] = False
    mem, val = x + xpos, x
    for i, (sa, ca, ff) in enumerate(zip(pd.self_, pd.cross, pd.ffn)):
        n1, n2, n3 = pd.norms[3 * i: 3 * i + 3]
        q = n1(q + sa(q, q, q)[0])
        q = n2(q + _cross(ca, q, mem, val, block))
        q = n3(q + ff(q))
    return {"scale": F.softplus(pd.scale(q)) + 0.005, "shape": 0.1 + 1.8 * torch.sigmoid(pd.shape(q)),
            "rotate": quat2mat(pd.rot(q)), "trans": anchor[:, None] + pd.trans(q), "exist_logit": pd.exist(q)[..., 0]}


class Net(M27.Net):
    """exp 27's Net (Slot Attention only); forward(..., slots=None): parts only for those slot indices (scattered
    into (K, G, ...), zeros elsewhere)."""

    def forward(self, dino, xyz, slots=None, match=None):
        """match: optional function out -> slot index per target (the Hungarian matching on the masks); then the
        parts are decoded for the matched slots only and out["sig"] holds the matching."""
        assert self.arch == "slot"
        pe = self.pos(M27.fourier3d(xyz))
        x = self.inp(dino) + pe
        s, logits = self.sa(x)
        for lay in self.layers:
            s = lay(s[None])[0]
        h = self.out_norm(s)
        out = {"scale": F.softplus(self.scale(h)) + 0.005, "shape": 0.1 + 1.8 * torch.sigmoid(self.shape(h)),
               "rotate": self.slot_rotation(h), "exist_logit": self.exist(h.detach() if self.exist_detach else h)[:, 0],
               "logits": logits, "sigmoid": False}
        a = logits.detach().softmax(1)
        if self.anchor == "centroid":
            c = (a.T @ xyz) / a.sum(0)[:, None].clamp_min(1e-6)
            out["anchor"], out["trans"] = c, c + self.trans(h)
        else:
            c = torch.zeros(len(h), 3, device=h.device)
            out["trans"] = self.trans(h)
        if self.part_dec is None:
            return out
        if match is not None:
            slots = out["sig"] = match(out)
        ctx = {"x": x, "pe": pe, "content": x - pe, "xyz": xyz}
        Rs = out["rotate"] if self.part_frame else None  # parts in the slot's frame (BoxRelNet)
        if slots is None:
            out["parts"] = self.decode_parts(h, ctx, a, c, Rs)
        else:
            p = self.decode_parts(h[slots], ctx, a[:, slots], c[slots], None if Rs is None else Rs[slots])
            out["parts"] = {k: v.new_zeros((len(h),) + v.shape[1:]).index_copy(0, slots, v) for k, v in p.items()}
        return out

    part_frame = False

    def slot_rotation(self, h):
        return quat2mat(self.rot(h))

    def decode_parts(self, h, ctx, mask, anchor, R=None):
        """h (S, d) slot features, mask (N, S) their patch probabilities, anchor (S, 3) -> parts dict (S, G, ...)."""
        return part_forward(self.part_dec, h, ctx["x"], ctx["pe"], mask, anchor)


def _cross_per(ca, q, mem, block):
    """As _cross, but a memory per slot: mem (S, N, d) (keys = values)."""
    S, G, d = q.shape
    h = ca.num_heads
    Wq, Wk, Wv = ca.in_proj_weight.chunk(3)
    bq, bk, bv = ca.in_proj_bias.chunk(3)
    Q = F.linear(q, Wq, bq).reshape(S, G, h, d // h).transpose(1, 2)
    Kx = F.linear(mem, Wk, bk).reshape(S, -1, h, d // h).transpose(1, 2)  # (S, h, N, dh)
    Vx = F.linear(mem, Wv, bv).reshape(S, -1, h, d // h).transpose(1, 2)
    o = F.scaled_dot_product_attention(Q, Kx, Vx, attn_mask=~block[:, None, None, :])
    return ca.out_proj(o.transpose(1, 2).reshape(S, G, d))


class PartDecoderRel(M27.PartDecoder):
    def __init__(self, d, G=16, L=2, heads=8):
        super().__init__(d, G, L, heads)
        self.posrel = nn.Linear(60, d)
        self.ptr_q, self.ptr_k = nn.Linear(d, d), nn.Linear(d, d)


def rel_forward(pd, h, content, xyz, mask, anchor, R=None):
    """content (N, d) patch features without position, xyz (N, 3) patch centroids -> parts dict (S, G, ...).
    R (S, 3, 3): optional slot frame; the parts' offsets and rotations are then expressed in it."""
    block = mask.T < 0.5
    block[block.all(1)] = False
    mem = content[None] + pd.posrel(M27.fourier3d(xyz[None] - anchor[:, None]))  # (S, N, d)
    q = h[:, None] + pd.emb[None]
    for i, (sa, ca, ff) in enumerate(zip(pd.self_, pd.cross, pd.ffn)):
        n1, n2, n3 = pd.norms[3 * i: 3 * i + 3]
        q = n1(q + sa(q, q, q)[0])
        q = n2(q + _cross_per(ca, q, mem, block))
        q = n3(q + ff(q))
    lg = pd.ptr_q(q) @ pd.ptr_k(mem).transpose(1, 2) / math.sqrt(q.shape[-1])  # (S, G, N)
    w = lg.masked_fill(block[:, None], float("-inf")).softmax(-1)
    centre = w @ xyz  # (S, G, 3)
    off, rot = pd.trans(q), quat2mat(pd.rot(q))
    if R is not None:
        off, rot = torch.einsum("sij,sgj->sgi", R, off), R[:, None] @ rot
    return {"scale": F.softplus(pd.scale(q)) + 0.005, "shape": 0.1 + 1.8 * torch.sigmoid(pd.shape(q)),
            "rotate": rot, "trans": centre + off, "exist_logit": pd.exist(q)[..., 0],
            "pointer": w}


class RelNet(Net):
    """Net with the relative part decoder (PartDecoderRel, rel_forward)."""

    def __init__(self, d=128, K=128, T=3, sa_layers=0, anchor="centroid", exist_detach=False, arch="slot", parts=16):
        super().__init__(d, K, T, sa_layers, anchor, exist_detach, arch, parts)
        self.part_dec = PartDecoderRel(d, parts)

    def decode_parts(self, h, ctx, mask, anchor, R=None):
        p = rel_forward(self.part_dec, h, ctx["content"], ctx["xyz"], mask, anchor, R)
        p.pop("pointer")
        return p


def yaw_matrix(cs):
    """(S, 2) unnormalized (cos, sin) -> (S, 3, 3) rotations about z."""
    cs = cs / cs.norm(dim=-1, keepdim=True).clamp_min(1e-6)
    c, s = cs[:, 0], cs[:, 1]
    z, o = torch.zeros_like(c), torch.ones_like(c)
    return torch.stack([c, -s, z, s, c, z, z, z, o], -1).reshape(-1, 3, 3)


class BoxRelNet(RelNet):
    """RelNet whose slot is a gravity-aligned box: centre = anchor + trans head, half extents = scale head, rotation =
    yaw about the input frame's z (needs --frame gravity) from a (cos, sin) head; the parts' offsets / rotations are
    expressed in that box frame (--slot_frame yaw). The box can be supervised with the amodal instance box (--w_box)."""

    part_frame = True

    def __init__(self, *a, **kw):
        super().__init__(*a, **kw)
        self.yaw = nn.Linear(self.out_norm.normalized_shape[0], 2)

    def slot_rotation(self, h):
        return yaw_matrix(self.yaw(h))


@torch.no_grad()
def check():
    import data as D
    torch.manual_seed(0)
    fr = D.load_frame("room_2", "000500", "instance", "visible")
    ref = M27.Net(128, 64, 3, 0, "centroid", False, "slot", 16).cuda().eval()
    new = Net(128, 64, 3, 0, "centroid", False, "slot", 16).cuda().eval()
    new.load_state_dict(ref.state_dict())
    for p in ref.parameters():  # random non-trivial weights
        p.add_(torch.randn_like(p) * 0.02)
    new.load_state_dict(ref.state_dict())
    a, b = ref(fr["dino"], fr["xyz"]), new(fr["dino"], fr["xyz"])
    for k in a["parts"]:
        print(k, float((a["parts"][k] - b["parts"][k]).abs().max()))
    idx = torch.arange(0, 64, 3, device="cuda")
    c = new(fr["dino"], fr["xyz"], slots=idx)
    print("subset", max(float((a["parts"][k][idx] - c["parts"][k][idx]).abs().max()) for k in a["parts"]))


if __name__ == "__main__":
    check()
