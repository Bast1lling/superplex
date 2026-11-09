"""Train the single-frame SQ predictor (`model.py`) on the cached frames: 7 scenes for training, room_0 for testing.

    uv run python train.py --name base [--epochs 60] [--patch_queries] [--overfit room_2_000500]
    uv run python dashboard.py                                  -> http://127.0.0.1:7863

Targets per frame (`cache_frames.py`): the target SQs intersecting the camera frustum, in the camera frame.
Loss (PLAN.md, first setup), per frame after Hungarian matching of the targets to the Q queries:
- matching cost: 10 * Chamfer (6 x 6 surface samples per SQ, m) + 2 * (1 - p(target class)) + 1 * (1 - exist);
- 10 * Chamfer (16 x 16 samples) of matched pairs + 10 * the same with the target's translation ("forced", as
  SuperFlex: rotation, scale and shape learn while the position is still wrong);
- 2 * existence BCE over all queries (matched = 1);
- 2 * class cross entropy over the matched queries;
- instance mask of the matched query = visible mask of its target's instance (share of its pixels per patch):
  5 * BCE + 5 * Dice (Dice only where the instance is visible).
SQ surfaces are sampled on a regular (eta, omega) grid (`surface`, the same points as SuperFlex's
`parametric_to_points_extended` for plain SQs). The losses run as one geometric pass over all frames of the batch
(`losses`; only the matching is per frame); `--loss vol / both` and `--mask sq / sq_amodal` use the per-frame loop
`losses_loop`.
Evaluation on all test frames (and 100 training frames): see `evaluate`.
--patch_queries: every patch is a query, translation = its 3D centroid + offset (`model.py`); 4 x 4 samples in the
matching cost (3150 queries), existence by sigmoid focal loss (alpha 0.25, gamma 2, as Deformable DETR), since all
but ~100 of the queries are negatives.
--loss vol: instead of the two Chamfer terms, a volumetric pair loss: |clamp(sdf_pred) - clamp(sdf_target)| / tau
(tau 0.1 m, SuperFlex's radial SDF) at 1024 points per pair, half uniform in the target's box (+20 %), half near its
surface (samples + N(0, 3 cm)); 5 x it + 5 x the same with the target's translation ("forced").
--loss both: the Chamfer terms and the volumetric terms (Chamfer gives the gradient far from the target, where the
truncated SDF has none).
--two_stage (with --patch_queries): the patch predictions become proposals, the --select (300) best by existence
are decoded again (`model.py`); the same loss on the decoder's output and on the proposals (own Hungarian matching
each, weight 1 each; focal existence on both).
--dense: `model.DenseModel` (simplified tokenizer, --layers encoder layers, an SQ per patch), trained one-to-many
without matching (`dense.py`: each patch -> the visible target SQ it sees), evaluated on its top 300 patches.
--topk: `model.TopKModel` = --dense's encoder (its patch SQs are proposals) + a decoder over the --select (300) best
local maxima of its existence map, refining their SQs; loss = the one-to-many proposal loss + the Hungarian loss on
the decoder's SQs against the visible SQs (`dense.topk_losses`).
--union target / observed (with --topk): + W_UNION (1) * -log soft IoU (SuperFlex's IoU loss) of the union of the
final decoder SQs, p(x) = 1 - prod_i (1 - exist_i sigmoid(-sdf_i(x) / 1 cm)), on labelled points per frame
(`union_points`): target = inside the union of the visible target SQs; observed = from the depth map only (free in
front of the surface, occupied 0.5-3 cm behind it, the rest unlabelled).
--frames N: train and evaluate on the same N training frames (evenly spaced), a multi-frame overfit.
(An IoU-aware existence target with NMS was tried on one frame and dropped, see README.)
--mask sq: the matched query's mask target is its own SQ's visible part instead of its instance: every labelled pixel
belongs to the target SQ of its instance with the smallest SDF at its back-projected point; share per patch.
--mask sq_amodal: the patches the whole SQ projects onto (32 x 32 surface samples, z > 0.1 m), hidden parts included.
Both report sq_mask_iou (matched query's mask > 0 against its target > 0.5, targets with a non-empty mask).
"""

import argparse
import json
import os
import sys
import time
from functools import lru_cache
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment

EXP = Path(__file__).resolve().parents[1]
os.environ.setdefault("TRACKIO_DIR", str(EXP / "results" / "large_files" / "trackio"))
import trackio  # noqa: E402

from model import CX, CY, FX, FY, GH, GW, P, Model  # noqa: E402
from model import DenseModel as dense_model  # noqa: E402
import types  # noqa: E402
sys.modules["superflex.loss.sampler"] = types.SimpleNamespace(EqualDistanceSamplerSQ=None)  # its Cython sampler is unbuilt, unused here
from superflex.loss.utils import sdf_batch  # noqa: E402

FRAMES = Path("/home/sebastian/Documents/Uni/Thesis/data/replica_rgbd_frames")
TRAIN = ("room_1", "room_2", "office_0", "office_1", "office_2", "office_3", "office_4")
TEST = ("room_0",)
OUT = EXP / "results" / "large_files" / "runs"
PROJECT = "rgbd_prediction"
W = {"cd": 10.0, "forced": 10.0, "vol": 5.0, "vol_forced": 5.0, "exist": 2.0, "cls": 2.0, "mask": 5.0, "dice": 5.0,
     "r_inst": 0.0, "r_cls": 0.0, "r_depth": 0.0}  # r_*: projection losses (`render.py`, --render_*), final decoder only
C = {"cd": 10.0, "cls": 2.0, "exist": 1.0}
DEV = "cuda"
FOCAL = False  # set by --patch_queries
COST_SAMPLES = 6
LOSS = "cd"  # --loss
TAU = 0.1
MASK = "instance"  # --mask
EXIST = "focal"  # --exist: existence loss of the --topk decoder (sigmoid focal, or plain BCE)
UNION = "none"  # --union: soft-IoU loss of the decoder's SQ union, against the target union or the observed depth
W_UNION = 1.0
UNION_TAU = 0.01  # m, sigmoid temperature of the soft occupancy
UNION_POINTS = 2048  # per frame and kind (target: uniform in box / near surfaces; observed: free / occupied)
PARTIAL = "full"  # --partial clip: clipped Chamfer of partially visible targets (see `target_visibility`)
VIS_X = 0.5  # --vis_x: targets with visible fraction f < VIS_X are "partially visible"
VIS_EPS = 0.03  # m, depth tolerance of the occlusion test
CLIP_OCCLUDED = True  # --clip_what all: partial = hidden or outside the image; trunc: only outside the image
FORCED_WEIGHT = True  # --forced_weight f: forced Chamfer of partial targets weighted by min(f / VIS_X, 1)
MIN_PATCHES = 0  # --min_patches (training frames): visible SQs owning fewer patches are dropped, their patches ignored
W_DEC_CD = 1.0  # --w_dec_cd: factor on the Hungarian (decoder) Chamfer terms cd / forced (0: projection losses only)
HEAT = "gauss"  # --heat: existence heatmap of the proposals (`cache_heat.py`; gauss = dense.heatmap, cached as "heat")
SAMPLING = "grid"  # --sampling: surface samples of the losses, the matching cost and the visibility (`surface`)
OWNER_RULE = "zbuffer"  # `pixel_owner`: front SQ of the pixel's instance along its ray; "sdf": the old rule (smallest SDF)
VIS_EPS_Z = 0.005  # m, tolerance of `target_visibility`'s test against the pixel owner's surface (`pixel_owner`)
FAST_SDF = True  # `inside` compiled with torch.compile (fuses SuperFlex's ~30 elementwise passes; see results/speedups.md)


SELECTION = EXP / "results" / "frame_selection.json"  # `select_frames.py`: --select diverse


@lru_cache(maxsize=1)
def sonata_dims():
    """Channels of the cached Sonata stages (f2, f3, f4), for frames cached without Sonata (`select_frames.py`)."""
    s = np.load(next((FRAMES / TEST[0]).glob("*_sonata.npz")))
    return [s[k].shape[1] for k in ("f2", "f3", "f4")]


class Frames(torch.utils.data.Dataset):
    """Cached frames of the scenes: every 10th frame of the trajectory (`cache_frames.py`), or with select = {scene:
    [frame ids]} those frames (`select_frames.py`; frames without Sonata get empty Sonata inputs, for --inputs dino).
    min_patches: see MIN_PATCHES."""

    def __init__(self, scenes, limit=None, only=None, repeat=1, part=None, flip=0.0, select=None, min_patches=0):
        self.flip = flip  # --flip: probability of a horizontal mirror (training only)
        self.min_patches = min_patches
        if select is None:
            self.items = sorted(p for s in scenes for p in (FRAMES / s).glob("*_view.npz") if int(p.name[:6]) % 10 == 0)
        else:
            self.items = [FRAMES / s / f"{f:06d}_view.npz" for s in scenes for f in sorted(select[s])]
        if part:  # --split frames: per scene the first 75 % of the trajectory ("head") or the last 20 % ("tail"), 5 % gap
            per = {}
            for p in self.items:
                per.setdefault(p.parent.name, []).append(p)
            lo, hi = (0.0, 0.75) if part == "head" else (0.8, 1.0)
            self.items = [p for ps in per.values() for i, p in enumerate(ps) if lo <= i / len(ps) < hi]
        if only:
            self.items = [p for p in self.items if f"{p.parent.name}_{p.name[:6]}" == only] * repeat
        if limit:
            self.items = self.items[:: max(1, len(self.items) // limit)][:limit]

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        p = self.items[i]
        stem = str(p)[: -len("_view.npz")]
        v = np.load(p)
        if os.path.exists(stem + "_sonata.npz"):
            s = np.load(stem + "_sonata.npz")
        else:  # no Sonata cache: one zero token per stage, no pixel has a token
            s = {k: np.zeros((0, c), np.float16) for k, c in zip(("f2", "f3", "f4"), sonata_dims())}
            s["idx"] = np.full((3, *v["depth"].shape), -1, np.int32)
        extra = {}
        if os.path.exists(stem + "_assign.npz"):  # cache_assign.py; otherwise dense.assign computes it per step
            a = np.load(stem + "_assign.npz")
            extra = {"owner": a["owner"].astype(np.int64), "heat": a["heat"].astype(np.float32), "visible": a["visible"]}
            if HEAT != "gauss":  # an alternative heatmap of the same owners
                extra["heat"] = np.load(stem + "_heat.npz")[HEAT].astype(np.float32)
            fk = "f_cube" if SAMPLING == "cube" else "f"  # f of the --sampling's samples
            if fk in a:
                extra["f"] = a[fk].astype(np.float32)
            if "pixown" in a:
                extra["pixown"] = a["pixown"].astype(np.int64)
            if "ownz" in a:  # cached with the z-buffer owner rule
                extra["ownz"] = a["ownz"].astype(np.float32)
            if self.min_patches:
                own, vis = extra["owner"], extra["visible"]
                drop = vis & (np.bincount(own[own >= 0], minlength=len(vis)) < self.min_patches)
                gone = (own >= 0) & drop[np.maximum(own, 0)]
                extra |= {"owner": np.where(gone, -1, own), "heat": np.where(gone, 0.0, extra["heat"]).astype(np.float32),
                          "visible": vis & ~drop, "ignore": gone}
        item = extra | {"name": f"{p.parent.name}_{p.name[:6]}", "dino": np.load(stem + "_dino.npy"), "depth": v["depth"].astype(np.float32),
                        "label": v["label"].astype(np.int64), **{k: s[k] for k in ("f2", "f3", "f4", "idx")},
                        **{k: v[k] for k in ("scale", "exponents", "rotation", "translation", "instance", "cls")}}
        return mirror(item) if self.flip and np.random.rand() < self.flip else item


def mirror(item):
    """Horizontal mirror of a frame: image-space maps flipped along the width (the DINO map too, an approximation:
    features of the mirrored image would differ), SQs reflected at the camera's x = 0 plane (x -> -x; rotation
    S R S, S = diag(-1, 1, 1), the SQ is symmetric in its own axes). CX is 0.25 px off the image center (ignored)."""
    out = dict(item)
    for k in ("dino", "depth", "label", "pixown", "ownz"):
        if k in item:
            out[k] = np.ascontiguousarray(item[k][:, ::-1])
    out["idx"] = np.ascontiguousarray(item["idx"][..., ::-1])
    for k in ("owner", "heat", "ignore"):
        if k in item:
            out[k] = np.ascontiguousarray(item[k].reshape(GH, GW)[:, ::-1].reshape(-1))
    S = np.diag([-1.0, 1.0, 1.0]).astype(item["rotation"].dtype)
    out["rotation"] = S @ item["rotation"] @ S
    out["translation"] = item["translation"] * np.array([-1.0, 1.0, 1.0], item["translation"].dtype)
    return out


def collate(batch):
    out = {"name": [b["name"] for b in batch], "dino": torch.tensor(np.stack([b["dino"] for b in batch])).float(),
           "depth": torch.tensor(np.stack([b["depth"] for b in batch])), "label": torch.tensor(np.stack([b["label"] for b in batch]))}
    feats, idx = [], []
    for s, k in enumerate(("f2", "f3", "f4")):
        off = np.cumsum([0] + [len(b[k]) for b in batch])
        f = np.concatenate([b[k] for b in batch] + [np.zeros((1, batch[0][k].shape[1]), np.float16)])  # last row: no depth
        feats.append(torch.tensor(f.astype(np.float32)))
        idx.append(torch.tensor(np.stack([np.where(b["idx"][s] >= 0, b["idx"][s] + o, len(f) - 1) for b, o in zip(batch, off)])))
    out["sonata"], out["sonata_idx"] = feats, idx
    out["targets"] = [{k: torch.tensor(b[k].astype(np.float32 if k not in ("instance", "cls") else np.int64))
                       for k in ("scale", "exponents", "rotation", "translation", "instance", "cls")} for b in batch]
    if "owner" in batch[0]:
        out["owner"] = torch.tensor(np.stack([b["owner"] for b in batch]))
        out["heat"] = torch.tensor(np.stack([b["heat"] for b in batch]))
        for t, b in zip(out["targets"], batch):
            t["visible"] = torch.tensor(b["visible"])
    for k in ("pixown", "ownz"):
        if all(k in b for b in batch):
            out[k] = torch.tensor(np.stack([b[k] for b in batch]))
    if "ignore" in batch[0]:
        out["ignore"] = torch.tensor(np.stack([b["ignore"] for b in batch]))
    if all("f" in b for b in batch):
        for t, b in zip(out["targets"], batch):
            t["f"] = torch.tensor(b["f"])
    return out


def to_device(batch):
    b = {k: v.to(DEV, non_blocking=True) for k, v in batch.items() if isinstance(v, torch.Tensor)}
    b |= {"feats": [f.to(DEV, non_blocking=True) for f in batch["sonata"]],
          "idx": [i.to(DEV, non_blocking=True) for i in batch["sonata_idx"]], "name": batch["name"],
          "targets": [{k: v.to(DEV) for k, v in t.items()} for t in batch["targets"]]}
    for t in b["targets"]:  # slot: index among the frame's targets (kept by `dense.visible_targets`), for pixown
        t["slot"] = torch.arange(len(t["scale"]), device=DEV)
    if "pixown" not in b or ("ownz" not in b and OWNER_RULE == "zbuffer"):  # target SQ per pixel, unless cached
        po = [pixel_owner(b, i, t, with_depth=True) for i, t in enumerate(b["targets"])]
        b["pixown"] = torch.stack([o for o, _ in po])
        if OWNER_RULE == "zbuffer":
            b["ownz"] = torch.stack([z for _, z in po])
    for i, t in enumerate(b["targets"]):  # visible fraction f, unless cached
        if "f" not in t:
            t["f"] = visible_fraction(t, b["pixown"][i:i + 1], b["depth"][i:i + 1], ownz=zslice(b, i))
    return b


def angles(n, like):
    eta = (torch.arange(n, device=like.device) + 0.5) / n * torch.pi - torch.pi / 2
    omega = (torch.arange(n, device=like.device) + 0.5) / n * 2 * torch.pi - torch.pi
    e, o = torch.meshgrid(eta, omega, indexing="ij")
    return e.reshape(-1), o.reshape(-1)


def spow(x, p):
    return torch.sign(x) * x.abs().clamp_min(1e-6) ** p


def surface(scale, shape, rot, trans, n, kind=None):
    """Points on plain SQs: (S, 3) ... -> (S, N, 3). kind (default SAMPLING, --sampling): "grid" = the regular
    (eta, omega) grid, n x n (as SuperFlex's parametric_to_points_extended; leaves the inside of the pole faces of
    boxy SQs unsampled, see README); "cube" = a cube grid projected radially onto the SQ, 6 x m x m with
    m = `cube_m`(n) (about the same number of samples)."""
    if (kind or SAMPLING) == "cube":
        return surface_cube(scale, shape, rot, trans, cube_m(n))[0]
    return surface_grid(scale, shape, rot, trans, n)


def surface_grid(scale, shape, rot, trans, n):
    """The (eta, omega) grid: (S, n * n, 3)."""
    e, o = angles(n, scale)
    eps1, eps2 = shape[:, :1], shape[:, 1:]  # exponent along eta (latitude), omega (longitude)
    ce = spow(e.cos(), eps1)
    p = torch.stack([scale[:, :1] * ce * spow(o.cos(), eps2), scale[:, 1:2] * ce * spow(o.sin(), eps2),
                     scale[:, 2:] * spow(e.sin(), eps1)], -1)
    return p @ rot.transpose(1, 2) + trans[:, None]


def cube_m(n):
    """Cube grid side for the budget of an n x n grid: 6 m^2 ~ n^2."""
    return max(2, round(n / 6 ** 0.5))


@lru_cache(maxsize=16)
def cube_dirs(m, device):
    """Cell centers of an m x m grid on each face of the cube [-1, 1]^3 -> (6, m, m, 3); faces +x, -x, +y, -y, +z,
    -z."""
    g = (torch.arange(m, device=device) + 0.5) / m * 2 - 1
    a, b = torch.meshgrid(g, g, indexing="ij")
    one = torch.ones_like(a)
    return torch.stack([torch.stack(f, -1) for f in ((one, a, b), (-one, b, a), (b, one, a), (a, -one, b),
                                                     (a, b, one), (b, a, -one))])


def surface_cube(scale, shape, rot, trans, m):
    """Radial projection of the cube grid onto plain SQs: direction u on the cube, point s * u * F(u)^(-e1 / 2) with
    F the inside-outside function, F(u) = (|ux|^(2/e2) + |uy|^(2/e2))^(e2/e1) + |uz|^(2/e1) (F(l u) = l^(2/e1) F(u)).
    For boxy SQs (exponents -> 0) the samples tend to a uniform grid on each box face; for ellipsoids it is the
    cube-sphere mapping. -> world points (S, 6 m^2, 3), local points (S, 6, m, m, 3)."""
    u = cube_dirs(m, scale.device)
    e1, e2 = shape[:, 0, None, None, None], shape[:, 1, None, None, None]
    ax = u.abs().clamp_min(1e-9)
    G = (ax[..., 0] ** (2 / e2) + ax[..., 1] ** (2 / e2)).clamp_min(1e-30)  # 0 on the z axis: no NaN gradient of pow
    F = G ** (e2 / e1) + ax[..., 2] ** (2 / e1)  # (S, 6, m, m), >= 1
    loc = u * F[..., None] ** (-e1[..., None] / 2) * scale[:, None, None, None]
    return loc.reshape(len(scale), -1, 3) @ rot.transpose(1, 2) + trans[:, None], loc


def surface_normals(scale, shape, rot, n, kind=None):
    """Outward normals (not normalized) at `surface`'s samples, in the frame of rot: the gradient of the SQ's
    inside-outside function. grid: (cos^(2-e1) eta cos^(2-e2) omega / a, cos^(2-e1) eta sin^(2-e2) omega / b,
    sin^(2-e1) eta / c); cube: from the local points, in log space (only the direction counts)."""
    if (kind or SAMPLING) == "cube":
        loc = surface_cube(scale, shape, torch.eye(3, device=scale.device).expand(len(scale), 3, 3),
                           torch.zeros_like(scale), cube_m(n))[1].reshape(len(scale), -1, 3)
        e1, e2 = shape[:, :1], shape[:, 1:]  # (S, 1)
        lq = (loc / scale[:, None]).abs().clamp_min(1e-12).log()
        lG = torch.logaddexp(2 / e2 * lq[..., 0], 2 / e2 * lq[..., 1])  # (S, N)
        lxy = ((e2 / e1 - 1) * lG)[..., None] + (2 / e2 - 1)[..., None] * lq[..., :2] - scale[:, None, :2].log()
        lz = (2 / e1 - 1) * lq[..., 2] - scale[:, None, 2].log()
        lm = torch.cat([lxy, lz[..., None]], -1)
        nl = torch.sign(loc) * (lm - lm.amax(-1, keepdim=True)).exp()
        return nl @ rot.transpose(1, 2)
    e, o = angles(n, scale)
    eps1, eps2 = shape[:, :1], shape[:, 1:]
    ce = spow(e.cos(), 2 - eps1)
    nl = torch.stack([ce * spow(o.cos(), 2 - eps2) / scale[:, :1], ce * spow(o.sin(), 2 - eps2) / scale[:, 1:2],
                      spow(e.sin(), 2 - eps1) / scale[:, 2:]], -1)
    return nl @ rot.transpose(1, 2)


def sample_areas(scale, shape, n, kind=None):
    """Surface area per sample of `surface` (finite differences of the sample grid; grid: the cell areas exclude
    the unsampled pole caps) -> (S, N)."""
    S = len(scale)
    eye, zero = torch.eye(3, device=scale.device).expand(S, 3, 3), torch.zeros_like(scale)
    if (kind or SAMPLING) == "cube":
        loc = surface_cube(scale, shape, eye, zero, cube_m(n))[1]  # (S, 6, m, m, 3)
        da, db = torch.gradient(loc, dim=2)[0], torch.gradient(loc, dim=3)[0]
        return torch.cross(da, db, dim=-1).norm(dim=-1).reshape(S, -1)
    g = surface_grid(scale, shape, eye, zero, n).reshape(S, n, n, 3)  # (S, eta, omega, 3)
    de = torch.gradient(g, dim=1)[0]
    do = (torch.roll(g, -1, 2) - torch.roll(g, 1, 2)) / 2
    return torch.cross(de, do, dim=-1).norm(dim=-1).reshape(S, -1)


def zslice(batch, i):
    """ownz of frame i (as a 1-frame batch) or None."""
    return batch["ownz"][i:i + 1] if "ownz" in batch else None


def zpair(batch):
    return batch.get("ownz")


@torch.no_grad()
def target_zbuffer(t, H, W):
    """Hard ray casting of the frame's target SQs t (`render.entry_depths`, one ray per pixel) -> zbuf (H, W) depth of
    the first target surface (inf: none), zown (H, W) its SQ (-1: none), E (H * W, T) entry depth per SQ."""
    import render as R
    E = R.entry_depths(sq_dict(t["scale"], t["exponents"], t["rotation"], t["translation"]), H, W)[0]
    if E.shape[1] == 0:
        return torch.full((H, W), torch.inf, device=E.device), torch.full((H, W), -1, device=E.device), E
    zb, zo = E.min(1)
    return zb.reshape(H, W), torch.where(torch.isinf(zb), -1, zo).reshape(H, W), E


@torch.no_grad()
def target_visibility(t, fb, pixown, depth, n, occlusion=True, ownz=None, others=False, kind=None):
    """Visibility of the surface samples (`surface`, n x n) of the target SQs t (dict, camera frame, with slot = index
    among its frame's targets) of frames fb (index into pixown / depth (B, H, W)): a sample is visible if it projects
    into the image, faces the camera and is not hidden: z <= observed depth + VIS_EPS, or its pixel is owned by this
    SQ (`pixel_owner`; the fit can lie a bit behind its own surface); with ownz (B, H, W) (`pixel_owner`'s entry
    depth of the owner) also not behind the surface of its pixel's owner SQ, if that is another SQ: z <= ownz +
    VIS_EPS_Z (an SQ inside / behind a bigger SQ of the same wall passes the depth test, not this one; SQs of other
    instances do not hide: the GT label says what the pixel shows, a target fit of another instance that sticks out
    in front of the observed surface is no occluder); with others (t = all targets of its frames, with instance) also
    not inside another target SQ of its instance (exact where the owner test is within its pixel-rounding noise of a
    few mm) -> visible, facing (S, n * n), samples."""
    p = surface(t["scale"], t["exponents"], t["rotation"], t["translation"], n, kind)
    facing = (surface_normals(t["scale"], t["exponents"], t["rotation"], n, kind) * p).sum(-1) < 0
    z = p[..., 2]
    H, W = depth.shape[1:]
    u = (FX * p[..., 0] / z.clamp_min(1e-3) + CX).round().long()
    v = (FY * p[..., 1] / z.clamp_min(1e-3) + CY).round().long()
    inimg = (z > 0.1) & (u >= 0) & (u < W) & (v >= 0) & (v < H)
    f = fb[:, None].expand_as(u)
    u, v = u.clamp(0, W - 1), v.clamp(0, H - 1)
    d, own = depth[f, v, u], pixown[f, v, u]
    seen = ((d > 0) & (z <= d + VIS_EPS)) | (own == t["slot"][:, None]) if occlusion else True
    if occlusion and ownz is not None:
        seen = seen & ((own < 0) | (own == t["slot"][:, None]) | (z <= ownz[f, v, u] + VIS_EPS_Z))
    if occlusion and others and len(p) > 1:
        ins = inside(p.reshape(1, -1, 3), {"scale": t["scale"][None], "shape": t["exponents"][None],
                                           "rotate": t["rotation"][None], "trans": t["translation"][None]})[0]  # (S, S * n * n)
        ins = ins.reshape(len(p), len(p), -1)  # [other SQ, SQ, sample]
        ins &= ((fb[:, None] == fb[None, :]) & (t["instance"][:, None] == t["instance"][None, :]))[:, :, None]
        ins &= ~torch.eye(len(p), dtype=torch.bool, device=p.device)[:, :, None]
        seen = seen & ~ins.any(0)
    return inimg & facing & seen, facing, p


@torch.no_grad()
def visible_fraction(t, pixown, depth, n=16, ownz=None, kind=None):
    """f per target SQ of one frame: visible surface area / camera-facing surface area (`sample_areas`; at most
    ~half of a surface faces the camera, so f = 1 means seen as fully as a single view allows)."""
    if len(t["scale"]) == 0:
        return torch.zeros(0, device=depth.device)
    vis, facing, p = target_visibility(t, torch.zeros(len(t["scale"]), dtype=torch.long, device=depth.device),
                                       pixown, depth, n, ownz=ownz, others=True, kind=kind)
    a = sample_areas(t["scale"], t["exponents"], n, kind)
    return (a * vis).sum(1) / (a * facing).sum(1).clamp_min(1e-12)


@torch.no_grad()
def clip_keep(t, fb, batch, n):
    """--partial clip: the target samples that count in the target -> prediction Chamfer direction (S, n * n): all
    of a well-seen target (f >= VIS_X), only the visible ones of a partially visible target; and the f used
    (--clip_what trunc: f = in-image share of the camera-facing samples, occlusion ignored). None, None otherwise."""
    if PARTIAL != "clip":
        return None, None
    vis, facing, _ = target_visibility(t, fb, batch["pixown"], batch["depth"], n, CLIP_OCCLUDED, zpair(batch))
    f = t["f"] if CLIP_OCCLUDED else vis.sum(1) / facing.sum(1).clamp_min(1)
    return vis | (f >= VIS_X)[:, None], f


def chamfer(a, b, keep=None):
    """Pairwise: a (A, n, 3), b (A, n, 3) -> (A,) mean nearest distance both ways. keep (A, n): b's samples that count
    in the b -> a direction (`clip_keep`); a -> b is always against all of b."""
    d = torch.cdist(a, b)
    ba = d.min(1).values
    if keep is not None:
        k = keep.float()
        return d.min(2).values.mean(1) + (ba * k).sum(1) / k.sum(1).clamp_min(1)
    return d.min(2).values.mean(1) + ba.mean(1)


def instance_masks(label, instances):
    """Share of each instance's pixels per patch: label (H, W), instances (T,) -> (T, GH, GW)."""
    m = (label[None] == instances[:, None, None]).float()
    return F.avg_pool2d(m[:, None], P)[:, 0]


def sdf_pairs(x, sq):
    """x (S, M, 3), sq: dict of S plain SQs (scale, shape, rotate, trans) -> SDF (S, M) of SQ s at its own points."""
    return sdf_batch(x, {k: v[:, None] for k, v in sq.items()})[:, 0]


def _inside(x, sq):
    return sdf_batch(x, sq) < 0


_inside_compiled = torch.compile(_inside, dynamic=True)


@torch.no_grad()
def inside(x, sq):
    """sdf_batch(x, sq) < 0: x (B, M, 3), sq dict of (B, N, ...) -> (B, N, M) bool. With FAST_SDF compiled: 4-5x faster,
    the same decisions up to float rounding at the surface."""
    return (_inside_compiled if FAST_SDF else _inside)(x, sq)


def inside_pairs(x, sq):
    """x (S, M, 3), dict of S SQs -> (S, M): whether x[s] lies inside SQ s."""
    return inside(x, {k: v[:, None] for k, v in sq.items()})[:, 0]


def sq_dict(scale, shape, rotate, trans):
    return {"scale": scale, "shape": shape, "rotate": rotate, "trans": trans}


@torch.no_grad()
def pair_iou(a, b, M=2048):
    """Volumetric IoU of SQ pairs a[s], b[s] (dicts of S SQs): points uniform in the box around both."""
    pa, pb = surface(*a.values(), 8, kind="grid"), surface(*b.values(), 8, kind="grid")
    both = torch.cat([pa, pb], 1)
    lo, hi = both.min(1).values, both.max(1).values
    x = lo[:, None] + torch.rand(len(lo), M, 3, device=lo.device) * (hi - lo)[:, None]
    ia, ib = inside_pairs(x, a), inside_pairs(x, b)
    return (ia & ib).sum(1) / (ia | ib).sum(1).clamp_min(1)


def vol_loss(p, g, M=1024, reduce=True):
    """Truncated SDF L1 between predicted p and target g SQs (dicts of S SQs), at points around g -> mean (or (S,))."""
    S = len(g["scale"])
    surf = surface(*g.values(), 16)
    pick = surf[:, torch.randint(surf.shape[1], (M // 2,), device=surf.device)]
    near = pick + 0.03 * torch.randn_like(pick)
    lo, hi = surf.min(1).values, surf.max(1).values
    c, h = (lo + hi) / 2, (hi - lo) / 2 * 1.2
    box = c[:, None] + (torch.rand(S, M // 2, 3, device=c.device) * 2 - 1) * h[:, None]
    x = torch.cat([near, box], 1)
    with torch.no_grad():
        st = sdf_pairs(x, g).clamp(-TAU, TAU)
    per = (sdf_pairs(x, p).clamp(-TAU, TAU) - st).abs().mean(1) / TAU
    return per.mean() if reduce else per


@torch.no_grad()
def mask_targets(batch, b, t, ti):
    """(len(ti), GH, GW) mask targets of the targets ti of frame b (see --mask)."""
    if MASK == "instance":
        return instance_masks(batch["label"][b], t["instance"][ti])
    if MASK == "sq_amodal":
        x = surface(t["scale"][ti], t["exponents"][ti], t["rotation"][ti], t["translation"][ti], 32)
        z = x[..., 2]
        u = (FX * x[..., 0] / z.clamp_min(1e-3) + CX) / P
        v = (FY * x[..., 1] / z.clamp_min(1e-3) + CY) / P
        ok = (z > 0.1) & (u >= 0) & (u < GW) & (v >= 0) & (v < GH)
        m = torch.zeros(len(ti), GH * GW, device=x.device)
        s_idx = torch.arange(len(ti), device=x.device)[:, None].expand_as(z)[ok]
        m[s_idx, v[ok].long() * GW + u[ok].long()] = 1
        return m.reshape(-1, GH, GW)
    m = (pixel_owner(batch, b, t)[None] == ti[:, None, None]).float()
    return F.avg_pool2d(m[:, None], P)[:, 0]


@torch.no_grad()
def pixel_owner(batch, b, t, with_depth=False):
    """(H, W) target SQ of every pixel of frame b, -1 for unlabelled pixels and instances without target SQs.
    OWNER_RULE zbuffer: among the SQs of the pixel's instance, the first one its ray enters (`target_zbuffer`); if the
    ray misses them all, the one with the smallest |SDF| at the back-projected point. sdf (old): the SQ of its
    instance with the smallest SDF at the back-projected point (= the one the point lies deepest in: an SQ hidden
    inside a bigger one of the same wall took the pixels).
    with_depth: also (H, W) the owner's entry depth along the pixel ray (inf: fallback / no owner; zbuffer rule)."""
    depth, lab = batch["depth"][b], batch["label"][b]
    H, W = depth.shape
    if len(t["scale"]) == 0:
        o = torch.full((H, W), -1, device=DEV)
        return (o, torch.full((H, W), torch.inf, device=DEV)) if with_depth else o
    vv, uu = torch.meshgrid(torch.arange(H, device=DEV), torch.arange(W, device=DEV), indexing="ij")
    xyz = torch.stack([(uu - CX) / FX * depth, (vv - CY) / FY * depth, depth], -1).reshape(-1, 3)
    sdf = sdf_batch(xyz[None], {"scale": t["scale"][None], "shape": t["exponents"][None], "rotate": t["rotation"][None],
                                "trans": t["translation"][None]})[0]  # (T, H * W)
    own = t["instance"][:, None] == lab.reshape(1, -1)
    if OWNER_RULE == "zbuffer":
        E = target_zbuffer(t, H, W)[2]  # (H * W, T)
        zmin, zarg = torch.where(own.T, E, torch.inf).min(1)
        owner = torch.where(torch.isfinite(zmin), zarg, torch.where(own, sdf.abs(), torch.inf).argmin(0))
        owner = torch.where(own.any(0), owner, -1).reshape(H, W)
        return (owner, zmin.reshape(H, W)) if with_depth else owner
    owner = torch.where(own.any(0), torch.where(own, sdf, torch.inf).argmin(0), -1).reshape(H, W)
    return (owner, torch.full((H, W), torch.inf, device=DEV)) if with_depth else owner


@torch.no_grad()
def match_cost(pp, tp, prob, exist, keep=None):
    """Hungarian matching from surface samples: pp (Q, n, 3) queries, tp (T, n, 3) targets, prob (Q, T) of the targets'
    classes, exist (Q,), keep (T, n) the target samples of the target -> query Chamfer direction (`clip_keep`)
    -> (query index, target index)."""
    Q, n = pp.shape[:2]
    T = len(tp)
    tp = tp.reshape(-1, 3)
    cd = []
    for a in range(0, Q, 512):  # chunks of queries: (512 * n, T * n) distances at a time
        d = torch.cdist(pp[a:a + 512].reshape(-1, 3), tp).reshape(-1, n, T, n)
        tq = d.min(1).values  # (q, T, n): target sample -> nearest query sample
        tq = tq.mean(2) if keep is None else (tq * keep).sum(2) / keep.sum(1).clamp_min(1)
        cd.append(d.min(3).values.mean(1) + tq)
    cost = C["cd"] * torch.cat(cd) + C["cls"] * (1 - prob) + C["exist"] * (1 - exist)[:, None]
    return linear_sum_assignment(cost.cpu().numpy())


@torch.no_grad()
def match(pred, b, t, batch=None):
    """Hungarian matching of frame b's targets t to the queries: (query index, target index). With the batch:
    --partial clip's cost."""
    if len(t["scale"]) == 0:
        return np.zeros(0, int), np.zeros(0, int)
    pp = surface(pred["scale"][b], pred["shape"][b], pred["rotate"][b], pred["trans"][b], COST_SAMPLES)
    tp = surface(t["scale"], t["exponents"], t["rotation"], t["translation"], COST_SAMPLES)
    keep = None if batch is None else clip_keep(t, torch.full((len(tp),), b, dtype=torch.long, device=DEV), batch,
                                                COST_SAMPLES)[0]
    return match_cost(pp, tp, pred["cls"][b].softmax(-1)[:, t["cls"]], pred["exist"][b],
                      None if keep is None else keep.float())


def losses_loop(pred, batch):
    """`losses` frame by frame, with the volumetric loss and the SQ mask targets."""
    total = {k: 0.0 for k in W}
    n_frames = len(batch["targets"])
    for b, t in enumerate(batch["targets"]):
        qi, ti = match(pred, b, t)
        exist_t = torch.zeros_like(pred["exist"][b])
        exist_t[qi] = 1
        if FOCAL:
            p = pred["exist"][b].clamp(1e-6, 1 - 1e-6)
            pt = torch.where(exist_t > 0, p, 1 - p)
            alpha = torch.where(exist_t > 0, 0.25, 0.75)
            total["exist"] = total["exist"] - (alpha * (1 - pt) ** 2 * pt.log()).sum() / max(len(t["scale"]), 1)
        else:
            total["exist"] = total["exist"] + F.binary_cross_entropy(pred["exist"][b], exist_t)
        if len(qi) == 0:
            continue
        qi, ti = torch.as_tensor(qi, device=DEV), torch.as_tensor(ti, device=DEV)
        tgt = surface(t["scale"][ti], t["exponents"][ti], t["rotation"][ti], t["translation"][ti], 16)
        ps, pe, pr = pred["scale"][b, qi], pred["shape"][b, qi], pred["rotate"][b, qi]
        if LOSS in ("cd", "both"):
            total["cd"] = total["cd"] + chamfer(surface(ps, pe, pr, pred["trans"][b, qi], 16), tgt).mean()
            total["forced"] = total["forced"] + chamfer(surface(ps, pe, pr, t["translation"][ti], 16), tgt).mean()
        if LOSS in ("vol", "both"):
            g = sq_dict(t["scale"][ti], t["exponents"][ti], t["rotation"][ti], t["translation"][ti])
            total["vol"] = total["vol"] + vol_loss(sq_dict(ps, pe, pr, pred["trans"][b, qi]), g)
            total["vol_forced"] = total["vol_forced"] + vol_loss(sq_dict(ps, pe, pr, t["translation"][ti]), g)
        total["cls"] = total["cls"] + F.cross_entropy(pred["cls"][b, qi], t["cls"][ti])
        m_t = mask_targets(batch, b, t, ti)
        m_p = pred["mask"][b, qi]
        total["mask"] = total["mask"] + F.binary_cross_entropy_with_logits(m_p, m_t)
        vis = m_t.flatten(1).sum(1) > 0
        if vis.any():
            s, g = m_p[vis].sigmoid().flatten(1), m_t[vis].flatten(1)
            total["dice"] = total["dice"] + (1 - (2 * (s * g).sum(1) + 1) / (s.sum(1) + g.sum(1) + 1)).mean()
    total = {k: v / n_frames for k, v in total.items()}
    return sum(W[k] * v for k, v in total.items()), {k: float(v) for k, v in total.items()}


def seg_mean(x, fb, B, w=None):
    """Mean of x over the entries of each frame (fb: frame of each entry), then summed / B (frames without entries 0)."""
    w = torch.ones_like(x) if w is None else w
    s = x.new_zeros(B).index_add(0, fb, x * w)
    return (s / x.new_zeros(B).index_add(0, fb, w).clamp_min(1)).sum() / B


def pair_losses(pred, batch, fb, qs, tg, n_samples):
    """Losses of matched / assigned pairs over the whole batch at once: prediction (fb, qs) against the concatenated
    target tg -> cd, forced, cls, mask, dice (means per frame, averaged over the B frames)."""
    B = len(batch["targets"])
    cat = {k: torch.cat([t[k] for t in batch["targets"]]) for k in ("scale", "exponents", "rotation", "translation", "instance", "cls", "f", "slot")}
    tgt = surface(cat["scale"][tg], cat["exponents"][tg], cat["rotation"][tg], cat["translation"][tg], n_samples)
    ps, pe, pr = pred["scale"][fb, qs], pred["shape"][fb, qs], pred["rotate"][fb, qs]
    out = {"cls": seg_mean(F.cross_entropy(pred["cls"][fb, qs], cat["cls"][tg], reduction="none"), fb, B)}
    if LOSS in ("cd", "both"):
        keep, f = clip_keep({k: v[tg] for k, v in cat.items()}, fb, batch, n_samples)
        w_forced = (f / VIS_X).clamp(max=1.0) if keep is not None and FORCED_WEIGHT else 1.0  # --partial clip
        out["cd"] = seg_mean(chamfer(surface(ps, pe, pr, pred["trans"][fb, qs], n_samples), tgt, keep), fb, B)
        out["forced"] = seg_mean(w_forced * chamfer(surface(ps, pe, pr, cat["translation"][tg], n_samples), tgt, keep), fb, B)
    if LOSS in ("vol", "both"):
        g = sq_dict(cat["scale"][tg], cat["exponents"][tg], cat["rotation"][tg], cat["translation"][tg])
        out["vol"] = seg_mean(vol_loss(sq_dict(ps, pe, pr, pred["trans"][fb, qs]), g, reduce=False), fb, B)
        out["vol_forced"] = seg_mean(vol_loss(sq_dict(ps, pe, pr, cat["translation"][tg]), g, reduce=False), fb, B)
    with torch.no_grad():
        m_t = F.avg_pool2d((batch["label"][fb] == cat["instance"][tg][:, None, None]).float()[:, None], P)[:, 0]
    m_p = pred["mask"][fb, qs]
    out["mask"] = seg_mean(F.binary_cross_entropy_with_logits(m_p, m_t, reduction="none").flatten(1).mean(1), fb, B)
    s, g = m_p.sigmoid().flatten(1), m_t.flatten(1)
    dice = 1 - (2 * (s * g).sum(1) + 1) / (s.sum(1) + g.sum(1) + 1)
    out["dice"] = seg_mean(dice, fb, B, (g.sum(1) > 0).float())
    return out


def losses(pred, batch, render=False):
    """Hungarian loss (module docstring): surfaces of all frames sampled at once, matching per frame (--loss cd / vol /
    both; --mask sq / sq_amodal use `losses_loop`). render: + the projection losses of `render.py` (weights W["r_*"])."""
    if MASK != "instance":
        return losses_loop(pred, batch)
    tgts = batch["targets"]
    B, Q = pred["exist"].shape
    n_t = torch.tensor([len(t["scale"]) for t in tgts], device=DEV)
    off = torch.cumsum(n_t, 0) - n_t
    with torch.no_grad():
        flat = {k: pred[k].flatten(0, 1) for k in ("scale", "shape", "rotate", "trans")}
        pp = surface(flat["scale"], flat["shape"], flat["rotate"], flat["trans"], COST_SAMPLES).reshape(B, Q, -1, 3)
        tc = {k: torch.cat([t[k] for t in tgts]) for k in ("scale", "exponents", "rotation", "translation", "instance", "f", "slot")}
        tp = surface(tc["scale"], tc["exponents"], tc["rotation"], tc["translation"], COST_SAMPLES)
        keep = clip_keep(tc, torch.repeat_interleave(torch.arange(B, device=DEV), n_t), batch, COST_SAMPLES)[0]
        keep = None if keep is None else keep.float()
        prob = pred["cls"].softmax(-1)
    fb, qs, tg = [], [], []
    for b, t in enumerate(tgts):
        if n_t[b] == 0:
            continue
        sl = slice(int(off[b]), int(off[b] + n_t[b]))
        qi, ti = match_cost(pp[b], tp[sl], prob[b][:, t["cls"]], pred["exist"][b], None if keep is None else keep[sl])
        fb.append(torch.full((len(qi),), b)), qs.append(torch.as_tensor(qi)), tg.append(torch.as_tensor(ti) + int(off[b]))
    exist_t = torch.zeros_like(pred["exist"])
    total = {k: 0.0 for k in W}
    if fb:
        fb, qs, tg = (torch.cat(x).to(DEV) for x in (fb, qs, tg))
        exist_t[fb, qs] = 1
        total |= pair_losses(pred, batch, fb, qs, tg, 16)
        total["cd"], total["forced"] = W_DEC_CD * total["cd"], W_DEC_CD * total["forced"]
        if render and any(W[k] > 0 for k in ("r_inst", "r_cls", "r_depth")):
            import render as R
            total |= R.render_losses(pred, batch, fb, qs, tg, tc | {"cls": torch.cat([t["cls"] for t in tgts])})
    if FOCAL:
        p = pred["exist"].clamp(1e-6, 1 - 1e-6)
        pt = torch.where(exist_t > 0, p, 1 - p)
        alpha = torch.where(exist_t > 0, 0.25, 0.75)
        total["exist"] = (-(alpha * (1 - pt) ** 2 * pt.log()).sum(1) / n_t.clamp_min(1)).mean()
    else:
        total["exist"] = F.binary_cross_entropy(pred["exist"], exist_t)
    return sum(W[k] * v for k, v in total.items()), {k: float(v) for k, v in total.items()}


@torch.no_grad()
def union_points(batch, b, t):
    """Labelled points of frame b for `union_loss` -> x (M, 3), occupied (M,) float.
    --union target: UNION_POINTS uniform in the box of the targets' surface samples + UNION_POINTS near their surfaces
    (sample + N(0, 3 cm)), occupied = inside any target SQ (t: the visible targets).
    --union observed: from the depth map only, for UNION_POINTS random pixels with depth z each a free point on the
    ray in front of the surface (z' uniform in [0.1 m, z - 3 cm]) and an occupied one behind it (z' = z + U(0.5, 3) cm);
    all other space is unlabelled."""
    n = UNION_POINTS
    if UNION == "target":
        surf = surface(t["scale"], t["exponents"], t["rotation"], t["translation"], 16, kind="grid").reshape(-1, 3)
        lo, hi = surf.min(0).values, surf.max(0).values
        x = torch.cat([lo + torch.rand(n, 3, device=DEV) * (hi - lo),
                       surf[torch.randint(len(surf), (n,), device=DEV)] + 0.03 * torch.randn(n, 3, device=DEV)])
        occ = inside(x[None], {"scale": t["scale"][None], "shape": t["exponents"][None], "rotate": t["rotation"][None],
                               "trans": t["translation"][None]})[0].any(0)
        return x, occ.float()
    d = batch["depth"][b]
    v, u = torch.nonzero(d > 0.2, as_tuple=True)
    i = torch.randint(len(v), (n,), device=DEV)
    v, u, z = v[i], u[i], d[v[i], u[i]]
    ray = torch.stack([(u - CX) / FX, (v - CY) / FY, torch.ones_like(z)], -1)
    z_free = 0.1 + torch.rand(n, device=DEV) * (z - 0.13).clamp_min(0)
    z_occ = z + 0.005 + 0.025 * torch.rand(n, device=DEV)
    x = torch.cat([ray * z_free[:, None], ray * z_occ[:, None]])
    return x, torch.cat([torch.zeros(n, device=DEV), torch.ones(n, device=DEV)])


def union_loss(pred, batch, targets):
    """-log soft IoU per frame (SuperFlex's IoU loss): p(x) = 1 - prod_i (1 - exist_i * sigmoid(-sdf_i(x) / UNION_TAU))
    over all queries, against the labelled points of `union_points`; mean over frames. -> loss, mean soft IoU."""
    losses, ious = [], []
    for b, t in enumerate(targets):
        if UNION == "target" and len(t["scale"]) == 0:
            continue
        x, occ = union_points(batch, b, t)
        sdf = sdf_batch(x[None], {k: pred[k][b][None] for k in ("scale", "shape", "rotate", "trans")}, use_floor=True)[0]
        p = (pred["exist"][b][:, None] * torch.sigmoid(-sdf / UNION_TAU)).clamp(max=0.9999)
        occ_p = 1 - torch.exp(torch.log1p(-p).sum(0))
        iou = ((occ_p * occ).sum() + 1e-6) / ((occ_p + occ - occ_p * occ).sum().clamp_min(1) + 1e-6)
        losses.append(-torch.log(iou))
        ious.append(float(iou))
    if not losses:
        return pred["exist"].sum() * 0, 0.0
    return torch.stack(losses).mean(), float(np.mean(ious))


def group_instances(masks, exist, thresh=0.5):
    """Queries with exist > 0.5, merged greedily (most confident first) when their binary masks overlap (IoU > 0.5)."""
    order = torch.argsort(exist, descending=True)
    order = order[exist[order] >= 0.5]
    ms = (masks[order] > 0).cpu().numpy()  # (n, GH, GW), on the CPU: tiny, and no GPU sync per comparison
    groups = []
    for m in ms:
        if not m.any():
            continue
        for g in groups:
            if (g & m).sum() / (g | m).sum() > thresh:
                g |= m
                break
        else:
            groups.append(m.copy())
    return [torch.from_numpy(g).to(masks.device) for g in groups]


@torch.no_grad()
def frame_metrics(pred, batch, b, t, n_occ=32768):
    """Metrics of frame b's predictions against its targets t (see `evaluate`) -> (row, matched queries, targets)."""
    qi, ti = match(pred, b, t, batch)
    ex = pred["exist"][b]
    keep = ex > 0.5
    r = {"name": batch["name"][b], "targets": len(t["scale"]), "predicted": int(keep.sum())}
    if len(qi):
        qt, tt = torch.as_tensor(qi, device=DEV), torch.as_tensor(ti, device=DEV)
        pp = surface(pred["scale"][b, qt], pred["shape"][b, qt], pred["rotate"][b, qt], pred["trans"][b, qt], 16, kind="grid")
        tp = surface(t["scale"][tt], t["exponents"][tt], t["rotation"][tt], t["translation"][tt], 16, kind="grid")
        r["cd"] = float(chamfer(pp, tp).mean())
        iou = pair_iou(sq_dict(pred["scale"][b, qt], pred["shape"][b, qt], pred["rotate"][b, qt], pred["trans"][b, qt]),
                       sq_dict(t["scale"][tt], t["exponents"][tt], t["rotation"][tt], t["translation"][tt]))
        r["pair_iou"] = float(iou.mean())
        # by visibility: well seen (f >= VIS_X) / partially visible targets; Chamfer clipped as --partial clip
        tsel = {k: v[tt] for k, v in t.items() if k in ("scale", "exponents", "rotation", "translation", "instance", "f", "slot")}
        seen = tsel["f"] >= VIS_X
        keep = target_visibility(tsel, torch.full((len(tt),), b, dtype=torch.long, device=DEV), batch["pixown"],
                                 batch["depth"], 16, ownz=zpair(batch), kind="grid")[0] | seen[:, None]
        r["cd_clip"] = float(chamfer(pp, tp, keep).mean())
        if seen.any():
            r["pair_iou_seen"] = float(iou[seen].mean())
        if (~seen).any():
            r["pair_iou_partial"] = float(iou[~seen].mean())
            r["cd_clip_partial"] = float(chamfer(pp[~seen], tp[~seen], keep[~seen]).mean())
        r["partial_share"] = float((~seen).float().mean())
        r["cls_acc"] = float((pred["cls"][b, qt].argmax(-1) == t["cls"][tt]).float().mean())
        import render as R
        r |= R.render_metrics(pred, batch, b, t, qi, ti)
        if MASK != "instance":
            mt = mask_targets(batch, b, t, tt) > 0.5
            mp = pred["mask"][b, qt] > 0
            ne = mt.flatten(1).any(1)
            if ne.any():
                r["sq_mask_iou"] = float(((mp & mt).flatten(1).sum(1) / (mp | mt).flatten(1).sum(1).clamp_min(1))[ne].mean())
        corners = tp.reshape(-1, 3)
        lo, hi = corners.min(0).values, corners.max(0).values
        x = lo + torch.rand(n_occ, 3, device=DEV) * (hi - lo)
        occ_t = inside(x[None], {"scale": t["scale"][None], "shape": t["exponents"][None],
                                 "rotate": t["rotation"][None], "trans": t["translation"][None]})[0].any(0)
        lo_k = ex > 0.3  # inside tests of all SQs above the lower threshold, once
        ins = inside(x[None], {k: pred[k][b, lo_k][None] for k in ("scale", "shape", "rotate", "trans")})[0] \
            if lo_k.any() else None
        for thr, suffix in ((0.5, ""), (0.3, "@0.3")):
            k_ = ex > thr
            occ_p = ins[k_[lo_k]].any(0) if ins is not None else torch.zeros_like(occ_t)
            r["occ_iou" + suffix] = float((occ_p & occ_t).sum() / (occ_p | occ_t).sum().clamp_min(1))
            r["found" + suffix] = float((ex[qt] > thr).float().mean())
            r["predicted" + suffix] = int(k_.sum())
    groups = group_instances(pred["mask"][b], ex)
    lab = batch["label"][b]
    ids = torch.unique(lab[lab >= 0])
    gtm = instance_masks(lab, ids) > 0.5
    gtm = gtm[gtm.flatten(1).sum(1) >= 2]
    if len(gtm):
        if groups:
            g, m = torch.stack(groups).flatten(1).float(), gtm.flatten(1).float()
            inter = g @ m.T
            best = (inter / (g.sum(1)[:, None] + m.sum(1)[None] - inter)).max(0).values.tolist()
        else:
            best = [0.0] * len(gtm)
        r["mask_cov"] = float(np.mean(best))
    r["instances_pred"], r["instances_gt"] = len(groups), len(gtm)
    return r, qi, ti


@torch.no_grad()
def evaluate(model, loader, n_occ=32768, save=None):
    """Per frame: matched Chamfer (m), share of targets whose query exists (exist > 0.5, and > 0.3), predicted / target count, class accuracy,
    occupancy IoU of the predicted union (exist > 0.5) against the target union on points uniform in the targets'
    box, and mask coverage = mean over visible instances (>= 2 patches) of the best IoU of a predicted instance
    (`group_instances`) at patch level (instance share > 0.5)."""
    model.eval()
    rows = []
    for batch in loader:
        batch = to_device(batch)
        pred = model(batch["dino"], batch["feats"], batch["idx"], batch["depth"])
        for b, t in enumerate(batch["targets"]):
            r, qi, ti = frame_metrics(pred, batch, b, t, n_occ)
            ex = pred["exist"][b]
            rows.append(r)
            if save is not None:
                np.savez_compressed(save / f"{r['name']}.npz", exist=ex.cpu().numpy(),
                                    **{k: pred[k][b].cpu().numpy() for k in ("scale", "shape", "rotate", "trans")},
                                    cls=pred["cls"][b].argmax(-1).cpu().numpy(), mask=pred["mask"][b].cpu().numpy().astype(np.float16),
                                    match_q=qi, match_t=ti)
    model.train()
    keys = [k for k in rows[0] if k != "name"]
    return {k: float(np.mean([r[k] for r in rows if k in r])) for k in keys}, rows


MAIN = ("pair_iou", "found", "predicted", "mask_cov")  # test metrics in the dashboard's top group, next to the train loss


def group(a):
    """Dashboard group of a run (trackio `group`)."""
    return "overfit 1 frame" if a.overfit else f"overfit {a.frames} frames" if a.frames else "full"


def train_log(loss, parts, lr, loss_kind):
    """Dashboard names (groups = prefixes, in the order of dashboard.py's PLOT_ORDER): main/train_loss; loss/<term>
    (unweighted, only the active ones); proposals/<term> (two-stage encoder); optim/lr."""
    off = {"cd": ("vol", "vol_forced"), "vol": ("cd", "forced"), "both": ()}[loss_kind]  # inactive terms
    d = {"main/train_loss": float(loss), "optim/lr": lr}
    for k, v in parts.items():
        enc = k.startswith("enc_")
        k = k[4:] if enc else k
        if k not in off:
            d[f"{'proposals' if enc else 'loss'}/{k}"] = v
    return d


DASH_NAME = {"enc_recall@0.5": "recall@0.5"}  # --topk's proposal recall = --dense's recall@0.5: one plot for both


def eval_log(m_test, m_train=None):
    """main/<MAIN> and test/<rest> on the test frames; train_frames/<all> on the training frames, if different."""
    name = lambda k: DASH_NAME.get(k, k)
    d = {f"{'main' if k in MAIN else 'test'}/{name(k)}": v for k, v in m_test.items()}
    return d | {f"train_frames/{name(k)}": v for k, v in (m_train or {}).items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True)
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--warmup", type=float, default=0.3, help="OneCycle: share of the steps spent warming up")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--queries", type=int, default=200)
    ap.add_argument("--eval_every", type=int, default=5)
    ap.add_argument("--eval_subsample", type=int, default=2,
                    help="periodic evaluations on every n-th test frame and 100 / n training frames (the final one on all)")
    ap.add_argument("--patch_queries", action="store_true")
    ap.add_argument("--loss", choices=["cd", "vol", "both"], default="cd")
    ap.add_argument("--union", choices=["none", "target", "observed"], default="none",
                    help="--topk: + W_UNION * -log soft IoU of the decoder's SQ union (see union_loss)")
    ap.add_argument("--w_union", type=float, default=1.0)
    ap.add_argument("--w_mask", type=float, default=5.0, help="weight of the mask BCE and Dice losses (0: no mask loss)")
    ap.add_argument("--exist", choices=["focal", "bce"], default="focal", help="--topk: decoder existence loss")
    ap.add_argument("--width", type=int, default=128, help="token width d of --dense / --topk")
    ap.add_argument("--mask", choices=["instance", "sq", "sq_amodal"], default="instance")
    ap.add_argument("--two_stage", action="store_true")
    ap.add_argument("--select", type=int, default=300)
    ap.add_argument("--dense", action="store_true")
    ap.add_argument("--layers", type=int, default=1, help="encoder layers of --dense / --topk")
    ap.add_argument("--topk", action="store_true")
    ap.add_argument("--dec_layers", type=int, default=1, help="decoder layers of --topk")
    ap.add_argument("--frames", type=int, help="train and test on this many training frames")
    ap.add_argument("--split", choices=["scenes", "frames"], default="scenes",
                    help="frames: train on the first 75 %% of each training scene's frames, test on its last 20 %%")
    ap.add_argument("--flip", type=float, default=0.0, help="probability of a horizontal mirror of a training frame")
    ap.add_argument("--inputs", choices=["all", "dino", "sonata"], default="all")
    ap.add_argument("--partial", choices=["full", "clip"], default="full",
                    help="clip: Chamfer of partially visible targets (f < --vis_x) clipped to their visible samples")
    ap.add_argument("--vis_x", type=float, default=0.5)
    ap.add_argument("--dropout", type=float, default=0.0, help="--dense / --topk: dropout in the transformer layers")
    ap.add_argument("--feat_drop", type=float, default=0.0, help="dropout on the input features")
    ap.add_argument("--wd", type=float, default=1e-4, help="AdamW weight decay")
    ap.add_argument("--train_scenes", type=int, help="only the first N training scenes (learning curve over scenes)")
    ap.add_argument("--train_limit", type=int, help="training frames: this many, evenly spaced (learning curve)")
    ap.add_argument("--clip_what", choices=["all", "trunc"], default="all")
    ap.add_argument("--forced_weight", choices=["f", "none"], default="f")
    ap.add_argument("--min_patches", type=int, default=0, help="training frames: drop visible SQs owning fewer patches")
    ap.add_argument("--train_frames", choices=["stride", "diverse"], default="stride",
                    help="training frames: every 10th, or the diverse selection of select_frames.py")
    ap.add_argument("--render_inst", type=float, default=0.0, help="weight of the rendered instance loss (render.py)")
    ap.add_argument("--render_cls", type=float, default=0.0, help="weight of the rendered class loss")
    ap.add_argument("--render_depth", type=float, default=0.0, help="weight of the rendered depth loss")
    ap.add_argument("--w_dec_cd", type=float, default=1.0, help="factor on the decoder's Chamfer terms")
    ap.add_argument("--heat", choices=["gauss", "linear", "geodesic", "nearest"], default="gauss",
                    help="existence heatmap of the proposals (cache_heat.py)")
    ap.add_argument("--sampling", choices=["grid", "cube"], default="grid",
                    help="surface samples of the losses / matching / visibility (metrics: always the grid)")
    ap.add_argument("--owner_rule", choices=["zbuffer", "sdf"], default="zbuffer",
                    help="pixel owner / visibility rule where not cached (sdf: the rule of the caches before 2026-11-07)")
    ap.add_argument("--overfit", help="train and test on this one frame, e.g. room_2_000500")
    a = ap.parse_args()
    import train as module  # this file runs as __main__; dense.py uses the module copy "train", configure both
    for m in {sys.modules[__name__], module}:
        m.FOCAL, m.COST_SAMPLES, m.LOSS, m.MASK = a.patch_queries, 4 if a.patch_queries else 6, a.loss, a.mask
        m.UNION, m.W_UNION, m.EXIST = a.union, a.w_union, a.exist
        m.W["mask"] = m.W["dice"] = a.w_mask
        m.PARTIAL, m.VIS_X = a.partial, a.vis_x
        m.CLIP_OCCLUDED, m.FORCED_WEIGHT = a.clip_what == "all", a.forced_weight == "f"
        m.W["r_inst"], m.W["r_cls"], m.W["r_depth"], m.W_DEC_CD = a.render_inst, a.render_cls, a.render_depth, a.w_dec_cd
        m.OWNER_RULE, m.SAMPLING, m.HEAT = a.owner_rule, a.sampling, a.heat
    import model as model_module
    model_module.INPUTS, model_module.DROPOUT, model_module.FEAT_DROP = a.inputs, a.dropout, a.feat_drop
    torch.manual_seed(a.seed)
    classes = json.loads((FRAMES / "classes.json").read_text())
    kw = dict(collate_fn=collate, num_workers=6, pin_memory=True, persistent_workers=True)
    scenes = TRAIN + TEST
    select = json.loads(SELECTION.read_text())["frames"] if a.train_frames == "diverse" else None
    if a.overfit:
        train_set, test_set = Frames(scenes, only=a.overfit, repeat=a.batch * 8), Frames(scenes, only=a.overfit)
    elif a.frames:
        test_set = Frames(TRAIN[:a.train_scenes], limit=a.frames)
        train_set = torch.utils.data.ConcatDataset([Frames(TRAIN[:a.train_scenes], limit=a.frames, min_patches=a.min_patches)]
                                                   * max(1, a.batch * 8 // a.frames))
    elif a.split == "frames":
        train_set = Frames(TRAIN, part="head", flip=a.flip, min_patches=a.min_patches)
        test_set = Frames(TRAIN, part="tail")
    else:
        train_set = Frames(TRAIN[:a.train_scenes], flip=a.flip, select=select, min_patches=a.min_patches, limit=a.train_limit)
        test_set = Frames(TEST)
    train = torch.utils.data.DataLoader(train_set, batch_size=a.batch, shuffle=True, drop_last=True, **kw)
    test = torch.utils.data.DataLoader(test_set, batch_size=a.batch, **kw)
    train_eval_set = test_set if a.overfit or a.frames else \
        Frames(TRAIN[:a.train_scenes], limit=100, part="head" if a.split == "frames" else None, select=select)
    train_eval = torch.utils.data.DataLoader(train_eval_set, batch_size=a.batch, **kw)
    n = a.eval_subsample  # periodic evaluations: every n-th frame of both sets
    sub = lambda d: torch.utils.data.Subset(d, range(0, len(d), n))
    test_p = torch.utils.data.DataLoader(sub(test_set), batch_size=a.batch, **kw) if n > 1 else test
    train_eval_p = torch.utils.data.DataLoader(sub(train_eval_set), batch_size=a.batch, **kw) if n > 1 else train_eval
    global MAIN, evaluate, losses
    if a.topk:
        import dense
        from model import TopKModel
        model = TopKModel(len(classes), d=a.width, layers=a.layers, dec_layers=a.dec_layers, k=a.select).to(DEV)
        MAIN, evaluate, losses = ("pair_iou", "found", "predicted", "mask_cov", "enc_recall@0.5"), dense.evaluate, dense.topk_losses
    elif a.dense:
        import dense
        model = dense_model(len(classes), d=a.width, layers=a.layers).to(DEV)
        MAIN, evaluate, losses = ("best_iou", "recall@0.5", "predicted", "occ_iou_topk"), dense.evaluate, dense.losses
    else:
        model = Model(len(classes), a.queries, patch_queries=a.patch_queries, two_stage=a.two_stage, select=a.select).to(DEV)
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=a.wd)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=a.lr, epochs=a.epochs, steps_per_epoch=len(train),
                                                pct_start=a.warmup)
    trackio.init(project=PROJECT, name=a.name, config=vars(a) | {"weights": W, "costs": C, "train_frames": len(train.dataset),
                                                                 "test_frames": len(test.dataset),
                                                                 "params": sum(p.numel() for p in model.parameters())},
                 group=group(a))
    same = bool(a.overfit or a.frames)  # train_eval = the test frames
    out = OUT / a.name
    out.mkdir(parents=True, exist_ok=True)
    step, t0 = 0, time.time()
    for epoch in range(a.epochs):
        for batch in train:
            batch = to_device(batch)
            pred = model(batch["dino"], batch["feats"], batch["idx"], batch["depth"])
            loss, parts = losses(pred, batch)
            if "enc" in pred and not a.topk:  # --two_stage (--topk's loss covers its proposals itself)
                loss_e, parts_e = losses(pred["enc"], batch)
                loss, parts = loss + loss_e, parts | {f"enc_{k}": v for k, v in parts_e.items()}
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()
            step += 1
            if step % 20 == 0:
                trackio.log(train_log(loss, parts, sched.get_last_lr()[0], LOSS), step=step)
        print(f"epoch {epoch} step {step} loss {float(loss):.3f} {time.time() - t0:.0f}s", flush=True)
        if (epoch + 1) % a.eval_every == 0 and epoch < a.epochs - 1:  # the last epoch: the final evaluation below
            m_test, _ = evaluate(model, test_p)
            m_train = None if same else evaluate(model, train_eval_p)[0]
            trackio.log(eval_log(m_test, m_train), step=step)
            print("  test", json.dumps({k: round(v, 3) for k, v in m_test.items()}), flush=True)
            torch.save(model.state_dict(), out / "model.pt")
    torch.save(model.state_dict(), out / "model.pt")
    pred_dir = out / "predictions"
    pred_dir.mkdir(exist_ok=True)
    m_test, rows = evaluate(model, test, save=pred_dir)
    m_train = m_test if same else evaluate(model, train_eval)[0]
    trackio.log(eval_log(m_test, None if same else m_train), step=step)
    print("  test", json.dumps({k: round(v, 3) for k, v in m_test.items()}), flush=True)
    (EXP / "results" / f"{a.name}.json").write_text(json.dumps({"args": vars(a), "test": m_test, "train_eval": m_train,
                                                                "per_frame": rows}, indent=1))
    trackio.finish()


if __name__ == "__main__":
    main()
