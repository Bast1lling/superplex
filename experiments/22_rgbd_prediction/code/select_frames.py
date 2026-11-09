"""Diverse training frames: per training scene, POOL_STRIDE-spaced trajectory frames scored by what they show, then
FRAMES_PER_SCENE picked greedily for diversity of the visible SQs (instead of every 10th frame).

    uv run python select_frames.py      -> results/frame_selection.json {"frames": {scene: [ids]}, "stats": ...},
                                           results/large_files/frame_scores/<scene>.npz (the scores, reused if present)
    uv run python cache_frames.py --select && uv run python cache_assign.py --select   (DINO, view, assign; no Sonata)
    uv run python train.py ... --train_frames diverse

Unit = (target SQ, octant of the viewing direction in the SQ's own frame: signs of R^T (camera - center)), so an SQ
seen from a new side counts as new. A frame shows unit u with value v = f (`train.visible_fraction`) of the SQ if
the SQ owns at least one pixel (`train.pixel_owner`), else 0. Greedy: the frame maximizing
sum_u sqrt(n_u + v_u) - sqrt(n_u), n_u = sum of v_u over the frames picked so far (diminishing returns: rarely seen
units pull frames in, often seen ones stop counting). The test scene keeps every 10th frame.
"""

import json
import sys
from pathlib import Path

import numpy as np
import torch

import train as T
from cache_frames import DOWN, RENDERS, ROW0, ROWS, frustum, load_view, sq_meshes

POOL_STRIDE = 2
FRAMES_PER_SCENE = 200
SCORES = T.EXP / "results" / "large_files" / "frame_scores"


@torch.no_grad()
def score_scene(scene):
    """-> frame ids (F,), V (F, units) float32: the value of every unit per pool frame."""
    path = SCORES / f"{scene}.npz"
    if path.exists():
        z = np.load(path)
        return z["frames"], z["V"]
    verts, rows = sq_meshes(scene)
    n = len(list((RENDERS / scene.replace("_", "") / "results").glob("frame*.jpg")))
    ids = np.arange(0, n, POOL_STRIDE)
    V = np.zeros((len(ids), len(rows) * 8), np.float32)
    r = slice(ROW0 // DOWN, (ROW0 + ROWS) // DOWN)
    for i, f in enumerate(ids):
        v = load_view(scene, int(f), DOWN)
        w2c = np.linalg.inv(v["c2w"])
        sel = np.flatnonzero(frustum(verts, w2c))
        if len(sel) == 0:
            continue
        Rw, cam = w2c[:3, :3], v["c2w"][:3, 3]
        dev = lambda x, dt=torch.float32: torch.tensor(np.asarray(x), dtype=dt, device=T.DEV)
        t = {"scale": dev([rows[j][0] for j in sel]), "exponents": dev([rows[j][1] for j in sel]),
             "rotation": dev([Rw @ rows[j][2] for j in sel]), "translation": dev([Rw @ rows[j][3] + w2c[:3, 3] for j in sel]),
             "instance": dev([rows[j][4] for j in sel], torch.long)}
        t["slot"] = torch.arange(len(sel), device=T.DEV)
        b = {"depth": dev(v["depth"][r])[None], "label": dev(v["label"][r], torch.long)[None]}
        pixown = T.pixel_owner(b, 0, t)
        owns = torch.bincount(pixown[pixown >= 0], minlength=len(sel)) > 0
        fv = (T.visible_fraction(t, pixown[None], b["depth"]) * owns).cpu().numpy()
        local = np.stack([rows[j][2].T @ (cam - rows[j][3]) for j in sel])  # camera in the SQ's frame
        octant = (local[:, 0] > 0) * 4 + (local[:, 1] > 0) * 2 + (local[:, 2] > 0)
        V[i, sel * 8 + octant] = fv
        if i % 100 == 0:
            print(scene, f, f"{len(sel)} SQs in view, {int(owns.sum())} owning pixels", flush=True)
    SCORES.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, frames=ids, V=V)
    return ids, V


def greedy(V, k):
    Vt = torch.tensor(V, device=T.DEV)
    n = torch.zeros(V.shape[1], device=T.DEV)
    picked = []
    for _ in range(k):
        gain = (torch.sqrt(n + Vt) - torch.sqrt(n)).sum(1)
        gain[picked] = -1
        i = int(gain.argmax())
        picked.append(i)
        n += Vt[i]
    return sorted(picked)


def stats(V, rows):
    """Coverage of a frame subset: units / SQs seen (v > 0), mean octants per seen SQ, mean views per seen unit, and
    the share of the scene's total unit mass on the 10 % most seen units (concentration)."""
    m = V[rows]
    seen = (m > 0).sum(0)
    per_sq = (seen.reshape(-1, 8) > 0)
    tot = m.sum(0)
    top = np.sort(tot)[::-1]
    return {"units": int((seen > 0).sum()), "sqs": int(per_sq.any(1).sum()),
            "octants_per_sq": float(per_sq.sum(1)[per_sq.any(1)].mean()),
            "views_per_unit": float(seen[seen > 0].mean()),
            "top10%_mass": float(top[: max(1, len(top[top > 0]) // 10)].sum() / max(top.sum(), 1e-9))}


def main():
    out = {"frames": {}, "stats": {}, "pool_stride": POOL_STRIDE}
    for scene in sys.argv[1:] or T.TRAIN:
        ids, V = score_scene(scene)
        pick = greedy(V, FRAMES_PER_SCENE)
        stride = [i for i, f in enumerate(ids) if f % 10 == 0]
        out["frames"][scene] = [int(ids[i]) for i in pick]
        out["stats"][scene] = {"diverse": stats(V, pick), "stride10": stats(V, stride), "pool": stats(V, list(range(len(ids)))),
                               "overlap_with_stride10": int(sum(ids[i] % 10 == 0 for i in pick))}
        print(scene, json.dumps(out["stats"][scene]), flush=True)
    T.SELECTION.write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
