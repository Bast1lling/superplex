"""Monocular metric depth for the RGB settings (experiments 13 / 15), cached in Thesis/data/replica_mono_depth/.

Models (Hugging Face, weights in Thesis/data/models/hf):
- depthpro:          Apple Depth Pro, metric depth from the known focal length (600 px): it predicts canonical
                     inverse depth + a field of view; depth is rescaled from its own focal estimate to the true one.
- depthpro_ownfocal: Depth Pro with its own focal estimate (intrinsics unknown).
- dav2:              Depth Anything V2 Metric Indoor Large (Hypersim, ~60 deg FoV; ours is 90 deg: ~25 % too far).
Checked on room_0 frames 0 / 800 / 1600 (AbsRel vs. rendered depth): depthpro 0.024-0.046, depthpro_ownfocal
0.031-0.068, dav2 0.31-0.43 (0.05-0.06 after a per-image scale).
"""

from functools import lru_cache

import numpy as np
import torch
from PIL import Image

from views import CAM, DATA, DOWN, RENDERS

CACHE = DATA / "replica_mono_depth"
HF = {"depthpro": "apple/DepthPro-hf", "depthpro_ownfocal": "apple/DepthPro-hf",
      "dav2": "depth-anything/Depth-Anything-V2-Metric-Indoor-Large-hf"}


@lru_cache(maxsize=1)
def model(hf_name):
    from transformers import AutoImageProcessor, AutoModelForDepthEstimation
    cache = str(DATA / "models" / "hf")
    proc = AutoImageProcessor.from_pretrained(hf_name, cache_dir=cache)
    net = AutoModelForDepthEstimation.from_pretrained(hf_name, cache_dir=cache, dtype=torch.float16).cuda().eval()
    return proc, net


@torch.no_grad()
def full_res(scene, frame, name):
    proc, net = model(HF[name])
    img = Image.open(RENDERS / scene.replace("_", "") / "results" / f"frame{frame:06d}.jpg")
    size = (img.height, img.width)
    out = net(**proc(images=img, return_tensors="pt").to("cuda", torch.float16))
    if name.startswith("depthpro"):
        post = proc.post_process_depth_estimation(out, target_sizes=[size])[0]
        depth = post["predicted_depth"].float()
        if name == "depthpro":
            depth = depth * CAM["fx"] / float(post["focal_length"])
    else:
        depth = torch.nn.functional.interpolate(out.predicted_depth[:, None].float(), size=size, mode="bilinear")[0, 0]
    return depth.cpu().numpy()


def predict(scene, view, name, down=DOWN):
    """Predicted depth on the view's (downsampled) pixel grid."""
    f = CACHE / name / scene / f"{view['frame']:06d}.npy"
    if not f.exists():
        f.parent.mkdir(parents=True, exist_ok=True)
        o = down // 2
        np.save(f, full_res(scene, view["frame"], name)[o::down, o::down].astype(np.float32))
    return np.load(f)


@torch.no_grad()
def align_scales(views, depths, neighbors=4, iters=10, tol=0.15):
    """One scale per view so that the views agree (multi-view only; the poses are metric, so this also fixes a global
    scale error). First a global scale (grid search: share of reprojected points agreeing within 5 %), then Jacobi
    iterations: the points of the `neighbors` views before / after view i (scaled with their
    current scale) are projected into i; where the instance label matches and the depth agrees within `tol`
    (relative, against occlusions), s_i = median(projected z / predicted depth of i)."""
    from grid import backproject
    from views import DEV, intrinsics
    fx, fy, cx, cy = intrinsics()
    D = [torch.as_tensor(d, device=DEV) for d in depths]
    L = [torch.as_tensor(v["label"], device=DEV) for v in views]
    P = [torch.as_tensor(backproject(v, np.ones_like(d)), dtype=torch.float32, device=DEV) for v, d in zip(views, depths)]
    C = [torch.as_tensor(v["c2w"][:3, 3], dtype=torch.float32, device=DEV) for v in views]
    W2C = [torch.as_tensor(np.linalg.inv(v["c2w"]), dtype=torch.float32, device=DEV) for v in views]
    def ratios(i, s, tol):
        """Depth ratios (projected z / predicted depth of i) of the neighbors' points that agree with view i."""
        r, n = [], 0
        for j in range(max(0, i - neighbors), min(len(views), i + neighbors + 1)):
            if j == i:
                continue
            x = C[j] + (s[j] * D[j].reshape(-1, 1)) * (P[j] - C[j])  # camera center + s_j * depth * unit-depth ray
            c = x @ W2C[i][:3, :3].T + W2C[i][:3, 3]
            z = c[:, 2].clamp_min(1e-6)
            u, v = torch.round(fx * c[:, 0] / z + cx).long(), torch.round(fy * c[:, 1] / z + cy).long()
            h, w = D[i].shape
            ok = (c[:, 2] > 0.1) & (u >= 0) & (u < w) & (v >= 0) & (v < h) & (L[j].reshape(-1) >= 0)
            u, v, z, lab = u[ok], v[ok], z[ok], L[j].reshape(-1)[ok]
            ratio = z / D[i][v, u].clamp_min(1e-6)
            good = (L[i][v, u] == lab) & ((ratio / s[i] - 1).abs() < tol)
            r.append(ratio[good])
            n += len(z)
        return torch.cat(r), n

    # global scale: the one under which most reprojected points agree within 5 % (the baselines are metric)
    def agreement(g):
        s = torch.full((len(views),), g, device=DEV)
        return np.mean([(lambda r, n: len(r) / max(n, 1))(*ratios(i, s, 0.05)) for i in range(0, len(views), 5)])

    grid = np.geomspace(0.5, 2.0, 41)
    s = torch.full((len(views),), float(grid[int(np.argmax([agreement(g) for g in grid]))]), device=DEV)
    for _ in range(iters):  # per-view scales
        new = s.clone()
        for i in range(len(views)):
            r, _ = ratios(i, s, tol)
            if len(r) > 100:
                new[i] = r.median()
        s = new
    print(f"  scales: median {s.median().item():.3f}, range {s.min().item():.3f}-{s.max().item():.3f}", flush=True)
    return [d * float(si) for d, si in zip(depths, s)]
