"""SQ decomposition from one RGB-D frame: frozen DINOv3 + frozen Sonata features -> patch tokens -> SuperFlex's decoder.

Patch tokens (42 x 75 patches of 16 px, 4 x 4 pixels of the /4 view each):
- image: the DINOv3 token, Linear 1024 -> 256;
- geometry: per pixel its Sonata feature (3 coarsest stages concatenated, 1088) through a tiny MLP (-> 128 -> 64; its
  first layer is applied to the stage tokens and then gathered per pixel, which is the same function), plus its
  camera-frame position relative to the patch's centroid; a shared MLP per pixel, max- and mean-pooled over the
  patch's valid pixels (-> 256);
- both + the share of valid pixels -> MLP -> D; plus a Fourier encoding of the patch centroid (camera frame) and of
  the patch's grid position.
Decoder: SuperFlex's TransformerDecoder (3 layers sharing one DecoderLayer, as there, 1 head, feed-forward 512,
sinusoidal query encodings), Q queries. With patch_queries every patch token is a query (through the same layers) and
its SQ translation = the patch's 3D centroid + a predicted offset (initialized to 0). With two_stage (Deformable DETR /
DINO style) the patch-query pass is the encoder and its predictions are proposals; the `select` best proposals by
existence become the queries (content: their encoder token; reference: their translation, detached) of a second
decoder (3 layers sharing one new DecoderLayer, cross-attending to the encoder tokens), whose own heads predict the
SQs, translation = reference + offset. Output: the decoder's predictions, the proposals under "enc". Heads per query: plain SQ (scale > 0 in m, exponents 0.1 .. 1.9, quaternion
rotation, translation in the camera frame), existence, class logits, mask embedding (mask = embedding . patch token).
"""

import math
import sys
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "papers" / "superflex"))
from superflex.models.decoder import TransformerDecoder  # noqa: E402
from superflex.models.decoder_layer import DecoderLayer  # noqa: E402
from superflex.models.heads import SuperFlexHead  # noqa: E402

INPUTS = "all"  # --inputs: "dino" / "sonata" zeroes the other feature (ablation)
DROPOUT = 0.0  # --dropout: in the encoder / decoder layers of DenseModel / TopKModel (read at construction)
FEAT_DROP = 0.0  # --feat_drop: dropout on the input DINO / Sonata features (training only)
FX = FY = 150.0  # /4 view, rows 4..676 of the full image (see cache_frames.py)
CX, CY = (599.5 - 2) / 4, (339.5 - 2) / 4 - 1
GH, GW, P = 42, 75, 4  # patch grid, pixels per patch side


def mlp(*dims):
    layers = []
    for a, b in zip(dims[:-1], dims[1:]):
        layers += [nn.Linear(a, b), nn.GELU()]
    return nn.Sequential(*layers[:-1])


def peaks(exist):
    """exist (B, GH * GW) -> the same with every patch that is not a 3 x 3 local maximum set to -1 (CenterNet's NMS)."""
    e = exist.reshape(-1, 1, GH, GW)
    keep = e == F.max_pool2d(e, 3, 1, 1)
    return torch.where(keep, e, -1.0).reshape(exist.shape)


def pool_tokens(f, idx, valid):
    """Max of the stage tokens f (N + 1, C) over each patch's pixels with depth: idx (B, H, W) token per pixel, valid
    (B, H, W) -> (B, GH, GW, C), 0 for patches without depth. Reduces over the unique (patch, token) pairs only (a
    patch sees 1-3 tokens of a stage), instead of gathering C channels for every pixel."""
    B, H, W = idx.shape
    v, u = torch.meshgrid(torch.arange(H, device=idx.device), torch.arange(W, device=idx.device), indexing="ij")
    patch = (torch.arange(B, device=idx.device)[:, None, None] * GH + v // P) * GW + u // P
    key = torch.unique(patch[valid] * len(f) + idx[valid])
    out = f.new_zeros(B * GH * GW, f.shape[1])
    out.index_reduce_(0, key // len(f), f[key % len(f)], "amax", include_self=False)
    return out.reshape(B, GH, GW, -1)


def fourier(x, n, scale):
    """x (..., c) -> (..., 2 c n), frequencies 2^i / scale."""
    f = (2.0 ** torch.arange(n, device=x.device)) * math.pi / scale
    a = x[..., None] * f
    return torch.cat([a.sin(), a.cos()], -1).flatten(-2)


class Tokenizer(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.img = nn.Linear(1024, 256)
        self.sonata_in = nn.ModuleList([nn.Linear(192, 128), nn.Linear(384, 128, bias=False), nn.Linear(512, 128, bias=False)])
        self.sonata_out = nn.Sequential(nn.GELU(), nn.Linear(128, 64))
        self.point = mlp(64 + 3, 128, 128)
        self.fuse = mlp(256 + 256 + 1, 256, d)
        self.pos = nn.Linear(2 * 3 * 8 + 2 * 2 * 6, d)

    def forward(self, dino, feats, idx, depth):
        """dino (B, GH, GW, 1024); feats: Sonata stage tokens of the batch [(n2, 192), (n3, 384), (n4, 512)], last row 0;
        idx: per stage (B, H, W) token of every pixel (the zero row without depth); depth (B, H, W)
        -> tokens (B, GH * GW, d), anchors (B, GH * GW, 3)."""
        B, H, W = depth.shape
        sonata = self.sonata_out(sum(lin(f)[i] for lin, f, i in zip(self.sonata_in, feats, idx)))  # (B, H, W, 64)
        v, u = torch.meshgrid(torch.arange(H, device=depth.device), torch.arange(W, device=depth.device), indexing="ij")
        xyz = torch.stack([(u - CX) / FX * depth, (v - CY) / FY * depth, depth], -1)
        group = lambda x: x.reshape(B, GH, P, GW, P, -1).permute(0, 1, 3, 2, 4, 5).reshape(B, GH, GW, P * P, -1)
        valid = group((depth > 0).float()[..., None])
        xyz = group(xyz)
        n = valid.sum(3).clamp_min(1)
        centroid = (xyz * valid).sum(3) / n  # (B, GH, GW, 3)
        f = torch.cat([group(sonata), (xyz - centroid[..., None, :]) * valid], -1)
        f = self.point(f)
        pooled = torch.cat([f.masked_fill(valid == 0, -1e4).amax(3), (f * valid).sum(3) / n], -1)
        pooled = pooled * (valid.sum(3) > 0)
        tok = self.fuse(torch.cat([self.img(dino), pooled, valid.mean(3)], -1))
        gy, gx = torch.meshgrid(torch.linspace(0, 1, GH, device=depth.device), torch.linspace(0, 1, GW, device=depth.device),
                                indexing="ij")
        grid = torch.stack([gx, gy], -1).expand(B, -1, -1, -1)
        tok = tok + self.pos(torch.cat([fourier(centroid, 8, 16.0), fourier(grid, 6, 2.0)], -1))
        # anchor of patches without depth: 3 m along the ray through the patch center
        ray = torch.stack([((gx * (GW - 1) + 0.5) * P - 0.5 - CX) / FX, ((gy * (GH - 1) + 0.5) * P - 0.5 - CY) / FY,
                           torch.ones_like(gx)], -1) * 3.0
        anchor = torch.where(valid.sum(3) > 0, centroid, ray)
        return tok.reshape(B, GH * GW, -1), anchor.reshape(B, GH * GW, 3)


class Model(nn.Module):
    def __init__(self, n_classes, queries=200, d=128, layers=3, patch_queries=False, two_stage=False, select=300):
        super().__init__()
        self.patch_queries = patch_queries
        self.tokenizer = Tokenizer(d)
        layer = DecoderLayer(d_model=d, nhead=1, dim_feedforward=512, batch_first=True)
        self.decoder = TransformerDecoder(layer, n_layers=layers, max_len=queries, masked_attention=False,
                                          pos_encoding_type="sinusoidal")
        self.decoder.project_queries = mlp(d, d, d)  # SuperFlex's (assignment matrix, unused here)
        self.register_buffer("init_queries", torch.zeros(queries + 1, d))
        self.heads = Heads(d, n_classes, 0.0 if patch_queries else 3.0)
        self.mask_p = nn.Linear(d, d)
        self.select = select
        if two_stage:
            assert patch_queries
            self.dec2 = DecoderLayer(d_model=d, nhead=1, dim_feedforward=512, batch_first=True)
            self.heads2 = Heads(d, n_classes, 0.0)
        self.two_stage = two_stage
        self.n_layers = layers

    def forward(self, dino, feats, idx, depth):
        tok, anchor = self.tokenizer(dino, feats, idx, depth)
        mp = self.mask_p(tok)
        if not self.patch_queries:
            q = self.decoder(self.init_queries, tok)[0][-1][:, :-1]  # last layer, without SuperFlex's extra query
            return self.heads(q, mp, 0.0)
        q = tok
        for layer in self.decoder.layers:
            q = layer(q, tok)
        out = self.heads(q, mp, anchor)
        if not self.two_stage:
            return out
        top = out["exist"].detach().topk(self.select, dim=1).indices  # (B, K)
        pick = lambda x: torch.gather(x, 1, top[..., None].expand(-1, -1, x.shape[-1]))
        mem, q2, ref = q, pick(q), pick(out["trans"]).detach()
        for _ in range(self.n_layers):
            q2 = self.dec2(q2, mem)
        res = self.heads2(q2, mp, ref)
        res["enc"], res["select"] = out, top
        return res


class Heads(nn.Module):
    """Per query: plain SQ (translation = ref + offset), existence, class logits, mask (embedding . patch embedding)."""

    def __init__(self, d, n_classes, z0):
        super().__init__()
        self.scale, self.shape, self.rot, self.trans = nn.Linear(d, 3), nn.Linear(d, 2), nn.Linear(d, 4), nn.Linear(d, 3)
        self.exist, self.cls = nn.Linear(d, 1), nn.Linear(d, n_classes)
        self.mask_q = mlp(d, d, d)
        nn.init.constant_(self.scale.bias, math.log(math.e ** 0.2 - 1))  # softplus -> 0.2 m
        nn.init.zeros_(self.trans.weight)
        self.trans.bias.data = torch.tensor([0.0, 0.0, z0])

    def forward(self, q, mp, ref):
        return {"scale": F.softplus(self.scale(q)) + 0.005, "shape": 0.1 + 1.8 * torch.sigmoid(self.shape(q)),
                "rotate": SuperFlexHead.quat2mat(F.normalize(self.rot(q), dim=-1)), "trans": ref + self.trans(q),
                "exist": torch.sigmoid(self.exist(q))[..., 0], "cls": self.cls(q),
                "mask": torch.einsum("bqd,bpd->bqp", self.mask_q(q), mp).reshape(*q.shape[:2], GH, GW)}


class DenseModel(nn.Module):
    """--dense: one SQ per patch, trained one-to-many (`dense.py`), D = 128.

    Tokenizer: per patch, LayerNorm([DINO token 1024 || Sonata 1088 max-pooled over the patch's pixels with depth])
    -> Linear 2112 -> 256 -> GELU -> 128, + Linear(Fourier of the patch's 3D centroid). Encoder: `layers` standard
    pre-norm encoder layers (self-attention, 8 heads, feed-forward 512, no dropout) + LayerNorm. Heads as `Heads`,
    translation = anchor (centroid, or 3 m along the ray without depth) + offset.
    """

    def __init__(self, n_classes, d=128, layers=1):
        super().__init__()
        self.inp = nn.Sequential(nn.LayerNorm(1024 + 1088), nn.Linear(1024 + 1088, 256), nn.GELU(), nn.Linear(256, d))
        self.pos = nn.Linear(2 * 3 * 8, d)
        layer = nn.TransformerEncoderLayer(d, 8, 512, dropout=DROPOUT, activation="gelu", batch_first=True, norm_first=True)
        self.encoder = nn.TransformerEncoder(layer, layers, norm=nn.LayerNorm(d), enable_nested_tensor=False)
        self.heads = Heads(d, n_classes, 0.0)
        self.mask_p = nn.Linear(d, d)

    def encode(self, dino, feats, idx, depth):
        """-> per-patch predictions, encoder output tokens (B, GH * GW, d)."""
        B, H, W = depth.shape
        group = lambda x: x.reshape(B, GH, P, GW, P, -1).permute(0, 1, 3, 2, 4, 5).reshape(B, GH, GW, P * P, -1)
        valid = group((depth > 0)[..., None])  # (B, GH, GW, 16, 1)
        has = valid.any(3)  # (B, GH, GW, 1)
        sonata = torch.cat([pool_tokens(f, i, depth > 0) for f, i in zip(feats, idx)], -1)
        v, u = torch.meshgrid(torch.arange(H, device=depth.device), torch.arange(W, device=depth.device), indexing="ij")
        xyz = group(torch.stack([(u - CX) / FX * depth, (v - CY) / FY * depth, depth], -1))
        centroid = (xyz * valid).sum(3) / valid.sum(3).clamp_min(1)
        gy, gx = torch.meshgrid(torch.arange(GH, device=depth.device), torch.arange(GW, device=depth.device), indexing="ij")
        ray = torch.stack([((gx + 0.5) * P - 0.5 - CX) / FX, ((gy + 0.5) * P - 0.5 - CY) / FY, torch.ones_like(gx, dtype=depth.dtype)], -1) * 3.0
        anchor = torch.where(has, centroid, ray).reshape(B, GH * GW, 3)
        if INPUTS == "dino":
            sonata = torch.zeros_like(sonata)
        elif INPUTS == "sonata":
            dino = torch.zeros_like(dino)
        if FEAT_DROP and self.training:
            dino, sonata = F.dropout(dino, FEAT_DROP), F.dropout(sonata, FEAT_DROP)
        tok = self.inp(torch.cat([dino, sonata], -1)).reshape(B, GH * GW, -1) + self.pos(fourier(anchor, 8, 16.0))
        q = self.encoder(tok)
        return self.heads(q, self.mask_p(q), anchor), q

    def forward(self, dino, feats, idx, depth):
        return self.encode(dino, feats, idx, depth)[0]


class RefineHeads(Heads):
    """Heads that refine a reference SQ: scale = ref * exp(delta), exponents = 0.1 + 1.8 sigmoid(logit(ref) + delta),
    translation = ref + delta (all deltas start at 0); rotation, existence, class and mask are predicted anew."""

    def __init__(self, d, n_classes):
        super().__init__(d, n_classes, 0.0)
        for lin in (self.scale, self.shape):
            nn.init.zeros_(lin.weight), nn.init.zeros_(lin.bias)

    def forward(self, q, mp, ref):
        out = super().forward(q, mp, ref["trans"])
        u = ((ref["shape"] - 0.1) / 1.8).clamp(1e-4, 1 - 1e-4)
        out["scale"] = ref["scale"] * torch.exp(self.scale(q))
        out["shape"] = 0.1 + 1.8 * torch.sigmoid(torch.logit(u) + self.shape(q))
        return out


class TopKModel(DenseModel):
    """--topk: `DenseModel` (the encoder; its per-patch SQs are the proposals) + a decoder over the K proposals with the
    highest existence among the 3 x 3 local maxima of the existence map (`peaks`). Query = the proposal's encoder output token (gradients flow into the encoder); reference = its
    SQ (detached); per decoder layer (pre-norm: self-attention over the K queries, cross-attention to all encoder
    tokens, feed-forward; 8 heads, no dropout) a Fourier encoding of the current reference translation is added to the
    query, and `RefineHeads` refine the reference, which (detached) is the next layer's reference.
    Output: the last layer's SQs, plus "enc" (the proposals), "aux" (earlier layers) and "select" (chosen patches)."""

    def __init__(self, n_classes, d=128, layers=1, dec_layers=1, k=300):
        super().__init__(n_classes, d, layers)
        self.k = k
        self.dec = nn.ModuleList(nn.TransformerDecoderLayer(d, 8, 512, dropout=DROPOUT, activation="gelu", batch_first=True,
                                                            norm_first=True) for _ in range(dec_layers))
        self.dec_norm = nn.ModuleList(nn.LayerNorm(d) for _ in range(dec_layers))
        self.dec_pos = nn.Linear(2 * 3 * 8, d)
        self.dec_heads = nn.ModuleList(RefineHeads(d, n_classes) for _ in range(dec_layers))

    def forward(self, dino, feats, idx, depth):
        enc, mem = self.encode(dino, feats, idx, depth)
        top = peaks(enc["exist"].detach()).topk(self.k, dim=1).indices  # (B, K): local maxima first
        pick = lambda x: torch.gather(x, 1, top.reshape(*top.shape, *[1] * (x.dim() - 2)).expand(-1, -1, *x.shape[2:]))
        q = pick(mem)
        ref = {k: pick(enc[k]).detach() for k in ("scale", "shape", "trans")}
        mp, outs = self.mask_p(mem), []
        for layer, norm, head in zip(self.dec, self.dec_norm, self.dec_heads):
            q = layer(q + self.dec_pos(fourier(ref["trans"], 8, 16.0)), mem)
            outs.append(head(norm(q), mp, ref))
            ref = {k: outs[-1][k].detach() for k in ("scale", "shape", "trans")}
        return outs[-1] | {"enc": enc, "aux": outs[:-1], "select": top}
