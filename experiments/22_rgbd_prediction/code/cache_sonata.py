"""Per-frame training cache, part 2: frozen Sonata features of each frame's own depth (after `cache_frames.py`).

    cd experiments/04_ptv3_replica && uv run python ../22_rgbd_prediction/code/cache_sonata.py [scene ...]
    -> FRAMES/<scene>/<frame>_sonata.npz

Input: the valid pixels of the /4 view (168 x 300), back-projected with their depth and moved to the world frame (z up,
as Sonata was trained), color, normal from the depth map (cross product of central differences, towards the camera).
Sonata as in experiment 04 (2 cm grid, flash attention off). Kept: its 3 coarsest stages, whose features at a point
concatenated are Sonata's recommended per-point feature (experiment 04's "full"): f2, f3, f4 (tokens x channels,
float16) and idx (3, 168, 300) int32 = token of every pixel per stage, -1 without depth.
"""

import sys
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1] / "04_ptv3_replica" / "code"))
from encoder import load_model, sonata  # noqa: E402

FRAMES = Path("/home/sebastian/Documents/Uni/Thesis/data/replica_rgbd_frames")
FX = FY = 600 / 4
CX, CY = (599.5 - 2) / 4, (339.5 - 2) / 4 - 1  # /4 pixel centers (views.intrinsics), minus the 1 cropped row
SCENES = ("room_0", "room_1", "room_2", "office_0", "office_1", "office_2", "office_3", "office_4")
TRANSFORM = sonata.transform.Compose([
    dict(type="CenterShift", apply_z=True),
    dict(type="GridSample", grid_size=0.02, hash_type="fnv", mode="train", return_grid_coord=True, return_inverse=True),
    dict(type="NormalizeColor"),
    dict(type="ToTensor"),
    dict(type="Collect", keys=("coord", "grid_coord", "color", "inverse"), feat_keys=("coord", "color", "normal")),
])


def camera_points(depth):
    v, u = np.indices(depth.shape)
    return np.stack([(u - CX) / FX * depth, (v - CY) / FY * depth, depth], -1)


def normals(P):
    dx = np.zeros_like(P)
    dy = np.zeros_like(P)
    dx[:, 1:-1] = P[:, 2:] - P[:, :-2]
    dy[1:-1] = P[2:] - P[:-2]
    n = np.cross(dx, dy)
    n /= np.linalg.norm(n, axis=-1, keepdims=True) + 1e-9
    return np.where((n * P).sum(-1, keepdims=True) > 0, -n, n)  # towards the camera


@torch.no_grad()
def encode(model, view):
    depth = view["depth"].astype(np.float32)
    valid = depth > 0
    P = camera_points(depth)
    N = normals(P)
    R, t = view["c2w"][:3, :3], view["c2w"][:3, 3]
    point = {"coord": (P[valid] @ R.T + t).astype(np.float32), "normal": (N[valid] @ R.T).astype(np.float32),
             "color": view["rgb"][valid].astype(np.float32)}
    inp = {k: v.cuda() if isinstance(v, torch.Tensor) else v for k, v in TRANSFORM(point).items()}
    torch.manual_seed(0)
    with torch.autocast("cuda", dtype=torch.bfloat16):
        out = model(dict(inp))
    chain = [out]
    while "pooling_parent" in chain[-1].keys():
        chain.append(chain[-1]["pooling_parent"])
    chain = chain[::-1]  # finest first
    assign = [inp["inverse"]]  # input point -> stage-0 token
    for s in range(1, len(chain)):
        assign.append(chain[s]["pooling_inverse"][assign[-1]])
    n = len(chain)
    feats, idx = {}, np.full((3,) + depth.shape, -1, np.int32)
    for i, s in enumerate((n - 3, n - 2, n - 1)):
        feats[f"f{i + 2}"] = chain[s].feat.float().cpu().numpy().astype(np.float16)
        idx[i][valid] = assign[s].cpu().numpy()
    return feats, idx


def main():
    model = load_model().cuda().eval()
    for scene in sys.argv[1:] or SCENES:
        for p in sorted((FRAMES / scene).glob("*_view.npz")):
            out = p.with_name(p.name.replace("_view.npz", "_sonata.npz"))
            if out.exists():
                continue
            feats, idx = encode(model, dict(np.load(p)))
            np.savez(out, idx=idx, **feats)
            print(scene, p.name[:6], {k: v.shape for k, v in feats.items()}, flush=True)


if __name__ == "__main__":
    main()
