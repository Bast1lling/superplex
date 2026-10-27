"""Precompute frozen Sonata stage-4 tokens for all Replica scenes, in several augmented versions.

Version v = (rotation about z by 45° * (v % 8), mirror x if v >= 8). Per version the 1 cm cloud is mapped by
    x_aug = R @ x + t,   t chosen so the scene's xy-center is at 0 and its floor at z = 0,
then grid-sampled (2 cm) and encoded; grid sampling and Sonata's order shuffling are seeded with v.
Tokens are stored in the augmented frame; R and t map the original 1 cm points (and GT labels) into it.

Output: Thesis/data/replica_v1_sonata/stage4/<scene>.npz with
    coord (T, 3) f32, feat (T, 512) f16, n_points (T,) i32   all versions concatenated
    offsets (V + 1,) i64   tokens of version v are offsets[v]:offsets[v+1]
    R (V, 3, 3) f32, t (V, 3) f32, angle_deg (V,), mirror (V,)

    cd experiments/04_ptv3_replica && uv run python code/precompute.py
"""

import math
from pathlib import Path
import time

import numpy as np
import torch

from encoder import list_scenes, load_model, load_points, sonata

OUT = Path("/home/sebastian/Documents/Uni/Thesis/data/replica_v1_sonata/stage4")
N_ROT, MIRROR = 8, (False, True)
GRID = 0.02

TRANSFORM = sonata.transform.Compose([  # Sonata's default pipeline without CenterShift (done explicitly below)
    dict(type="GridSample", grid_size=GRID, hash_type="fnv", mode="train", return_grid_coord=True, return_inverse=True),
    dict(type="NormalizeColor"),
    dict(type="ToTensor"),
    dict(type="Collect", keys=("coord", "grid_coord", "color", "inverse"), feat_keys=("coord", "color", "normal")),
])


def augmentation(pos: np.ndarray, angle: float, mirror: bool) -> tuple[np.ndarray, np.ndarray]:
    c, s = math.cos(angle), math.sin(angle)
    R = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]], np.float32)
    if mirror:
        R = R @ np.diag([-1, 1, 1]).astype(np.float32)
    rot = pos @ R.T
    lo, hi = rot.min(0), rot.max(0)
    t = -np.array([(lo[0] + hi[0]) / 2, (lo[1] + hi[1]) / 2, lo[2]], np.float32)
    return R, t


@torch.no_grad()
def encode_version(model, p: dict, R: np.ndarray, t: np.ndarray, seed: int):
    np.random.seed(seed)
    torch.manual_seed(seed)
    point = {"coord": (p["pos"] @ R.T + t).astype(np.float32), "color": p["col"].astype(np.float32),
             "normal": (p["nrm"].astype(np.float32) @ R.T).astype(np.float32)}
    point = {k: v.cuda(non_blocking=True) if isinstance(v, torch.Tensor) else v for k, v in TRANSFORM(point).items()}
    with torch.autocast("cuda", dtype=torch.bfloat16):
        out = model(point)
    # stage-4 tokens: number of 2 cm input points per token via the pooling chain
    chain, assign = [out], None
    while "pooling_parent" in chain[-1].keys():
        chain.append(chain[-1]["pooling_parent"])
    assign = torch.arange(len(chain[-1].feat), device="cuda")
    for c in reversed(chain[:-1]):
        assign = c["pooling_inverse"][assign]
    n_points = torch.bincount(assign, minlength=len(out.feat))
    return out.coord.float().cpu().numpy(), out.feat.float().half().cpu().numpy(), n_points.int().cpu().numpy()


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    model = load_model(pretrained=True).cuda().eval()
    versions = [(k * 2 * math.pi / N_ROT, m) for m in MIRROR for k in range(N_ROT)]
    t0 = time.perf_counter()
    for scene in list_scenes():
        ts = time.perf_counter()
        p = load_points(scene)
        coords, feats, counts, offsets, Rs, tsh = [], [], [], [0], [], []
        for v, (angle, mirror) in enumerate(versions):
            R, t = augmentation(p["pos"], angle, mirror)
            c, f, n = encode_version(model, p, R, t, seed=v)
            coords.append(c); feats.append(f); counts.append(n)
            offsets.append(offsets[-1] + len(c)); Rs.append(R); tsh.append(t)
        np.savez(OUT / f"{scene}.npz", coord=np.concatenate(coords), feat=np.concatenate(feats),
                 n_points=np.concatenate(counts), offsets=np.array(offsets, np.int64), R=np.stack(Rs), t=np.stack(tsh),
                 angle_deg=np.array([math.degrees(a) for a, _ in versions], np.float32),
                 mirror=np.array([m for _, m in versions]))
        print(f"{scene:16s} {len(versions)} versions, {offsets[1]:,} tokens (v0), {time.perf_counter() - ts:5.1f} s", flush=True)
    print(f"total {time.perf_counter() - t0:.0f} s")


if __name__ == "__main__":
    main()
