"""Frames and frame pairs of exp 22's cache (`Thesis/data/replica_rgbd_frames`, every 10th frame of the 8 NICE-SLAM
Replica trajectories): DINOv3 tokens (42 x 75 x 1024), /4 view (depth, GT instance label, c2w), the target SQs in the
frustum (camera frame), and the cached assignment (zbuffer pixel owner rule): owner patch -> target, CenterNet
heatmap, visible targets, visible fraction f (grid / cube samples).

Targets across views are identified by their global index in the scene's target list (exp 22's
`results/targets_winding/<scene>/sqs.npz`, exist > 0.5), found by matching world-frame translations.

An item of `Views` holds V = 1 or 2 views; everything is expressed in the camera frame of view 0:
- dino (V, GH, GW, 1024), depth (V, H, W), label (V, H, W), T0v (V, 4, 4) camera v -> camera 0, c2w (V, 4, 4);
- targets = union over the views of their visible SQs (TARGETS = "visible") or of all frustum SQs ("frustum"):
  scale, exponents, rotation, translation (view-0 frame), instance, cls, gid, vis (T, V) visible in view v, f (T, V);
- owner (V, NP) union target per patch (-1 none), heat (V, NP) the CenterNet heatmap of each view.
"""

import json
from functools import lru_cache
from pathlib import Path

import numpy as np
import torch

from common import NP

DATA = Path("/home/sebastian/Documents/Uni/Thesis/data")
FRAMES = DATA / "replica_rgbd_frames"
TARGETS_DIR = Path(__file__).resolve().parents[2] / "22_rgbd_prediction" / "results" / "targets_winding"
TRAIN = ("room_1", "room_2", "office_0", "office_1", "office_2", "office_3", "office_4")
TEST = ("room_0",)
EXP = Path(__file__).resolve().parents[1]
PAIRS = EXP / "results" / "pairs.json"
TARGETS = "visible"
SAMPLING = "grid"
OVERLAP = (0.2, 0.8)


def frame_ids(scene):
    return sorted(int(p.name[:6]) for p in (FRAMES / scene).glob("*_view.npz") if int(p.name[:6]) % 10 == 0)


@lru_cache(maxsize=None)
def global_table(scene):
    z = np.load(TARGETS_DIR / scene / "sqs.npz")
    rows = [(b, j) for b in range(len(z["instance"])) for j in np.flatnonzero(z["exist"][b] > 0.5)]
    return np.array([z["translation"][b, j] for b, j in rows], np.float64), np.array([z["instance"][b] for b, _ in rows])


def gids(scene, v):
    """Global index of each target of a view (by world translation + instance)."""
    tw, inst = global_table(scene)
    R, t = v["c2w"][:3, :3], v["c2w"][:3, 3]
    w = v["translation"].astype(np.float64) @ R.T + t
    d = np.linalg.norm(w[:, None] - tw[None], axis=-1) + 1e3 * (v["instance"][:, None] != inst[None])
    g = d.argmin(1)
    assert len(g) == 0 or d[np.arange(len(g)), g].max() < 1e-3, (scene, d.min(1).max())
    return g


@lru_cache(maxsize=4096)
def load_view(scene, f):
    stem = FRAMES / scene / f"{f:06d}"
    v = dict(np.load(f"{stem}_view.npz"))
    a = np.load(f"{stem}_assign.npz")
    v["owner"], v["heat"], v["visible"] = a["owner"].astype(np.int64), a["heat"].astype(np.float32), a["visible"]
    v["f"] = a["f_cube" if SAMPLING == "cube" else "f"].astype(np.float32)
    v["gid"] = gids(scene, v)
    del v["rgb"]
    return v


def load_dino(scene, f):
    return np.load(FRAMES / scene / f"{f:06d}_dino.npy")


class Views(torch.utils.data.Dataset):
    """Items = (scene, view-0 frame); partner chosen per item: `fixed` = the candidate with overlap closest to 0.5
    (evaluation), `random` = a random candidate with overlap in OVERLAP (training). views = 1: view 0 only."""

    def __init__(self, scenes, views=2, partner="fixed", limit=None, part=None, repeat=1):
        self.views, self.partner = views, partner
        table = json.loads(PAIRS.read_text())
        self.items = []
        for s in scenes:
            ids = frame_ids(s)
            if part:
                lo, hi = (0.0, 0.75) if part == "head" else (0.8, 1.0)
                ids = [f for i, f in enumerate(ids) if lo <= i / len(ids) < hi]
            idset = set(ids)
            for f in ids:
                cand = {int(j): o for j, o in table[s][str(f)].items() if int(j) in idset}
                good = {j: o for j, o in cand.items() if OVERLAP[0] <= o <= OVERLAP[1]}
                if not good:  # fall back to the most overlapping neighbour
                    good = dict(sorted(cand.items(), key=lambda x: -x[1])[:1])
                self.items.append((s, f, good))
        if limit:
            self.items = self.items[:: max(1, len(self.items) // limit)][:limit]
        self.items = self.items * repeat

    def __len__(self):
        return len(self.items)

    def pick(self, f, good):
        if self.partner == "random":
            return int(np.random.choice(list(good)))
        return min(good, key=lambda j: (abs(good[j] - 0.5), abs(j - f)))

    def __getitem__(self, i):
        s, f, good = self.items[i]
        fr = [f] if self.views == 1 else [f, self.pick(f, good)]
        vs = [load_view(s, x) for x in fr]
        c2w = np.stack([v["c2w"] for v in vs]).astype(np.float32)
        T0v = np.linalg.inv(c2w[0])[None] @ c2w  # camera v -> camera 0
        # union targets, first occurrence wins (view 0 parameters for targets it holds)
        order, seen = [], {}
        for vi, v in enumerate(vs):
            keep = v["visible"] if TARGETS == "visible" else np.ones(len(v["gid"]), bool)
            for ti in np.flatnonzero(keep):
                g = int(v["gid"][ti])
                if g not in seen:
                    seen[g] = len(order)
                    order.append((vi, ti))
        V, T = len(vs), len(order)
        tg = {k: np.zeros((T, *sh), np.float32) for k, sh in (("scale", (3,)), ("exponents", (2,)), ("rotation", (3, 3)),
                                                               ("translation", (3,)))}
        tg |= {"instance": np.zeros(T, np.int64), "cls": np.zeros(T, np.int64), "gid": np.zeros(T, np.int64),
               "vis": np.zeros((T, V), bool), "f": np.zeros((T, V), np.float32)}
        for u, (vi, ti) in enumerate(order):
            v = vs[vi]
            R, t = T0v[vi][:3, :3], T0v[vi][:3, 3]
            tg["scale"][u], tg["exponents"][u] = v["scale"][ti], v["exponents"][ti]
            tg["rotation"][u], tg["translation"][u] = R @ v["rotation"][ti], R @ v["translation"][ti] + t
            tg["instance"][u], tg["cls"][u], tg["gid"][u] = v["instance"][ti], v["cls"][ti], v["gid"][ti]
        owner = np.full((V, NP), -1, np.int64)
        for vi, v in enumerate(vs):
            loc = np.array([seen.get(int(g), -1) for g in v["gid"]], np.int64)
            ok = v["visible"] & (loc >= 0)
            tg["vis"][loc[ok], vi] = True
            tg["f"][loc[ok], vi] = v["f"][ok]
            o = v["owner"]
            owner[vi] = np.where(o >= 0, loc[np.maximum(o, 0)], -1)
        return {"name": f"{s}_{f:06d}" + ("" if V == 1 else f"_{fr[1]:06d}"), "scene": s,
                "dino": np.stack([load_dino(s, x) for x in fr]), "depth": np.stack([v["depth"] for v in vs]).astype(np.float32),
                "label": np.stack([v["label"] for v in vs]).astype(np.int64), "T0v": T0v.astype(np.float32), "c2w": c2w,
                "owner": owner, "heat": np.stack([v["heat"] for v in vs]), "targets": tg}


def collate(batch):
    out = {k: torch.tensor(np.stack([b[k] for b in batch])) for k in ("depth", "label", "T0v", "c2w", "owner", "heat")}
    out["dino"] = torch.tensor(np.stack([b["dino"] for b in batch])).float()
    out["name"] = [b["name"] for b in batch]
    out["targets"] = [{k: torch.tensor(v) for k, v in b["targets"].items()} for b in batch]
    return out


def to_device(batch, dev="cuda"):
    out = {k: v.to(dev, non_blocking=True) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
    out["targets"] = [{k: v.to(dev) for k, v in t.items()} for t in batch["targets"]]
    return out
