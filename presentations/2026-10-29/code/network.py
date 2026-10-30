"""network: SuperFlex (ECCV 2026, papers/superflex) run on every GT instance separately, results merged.

    python network.py [--res 0.02] [scene ...]      -> superflex_<checkpoint> for every checkpoint in CHECKPOINTS

Per instance: 4096 of its points (with repetition if it has fewer), centered and scaled to unit size (SuperFlex's
normalize_points), rotated from z-up to the network's y-up (as in its scene pipeline); all instances in one batch.
Primitives with existence > 0.5 are mapped back to the scene. Saved: the SQ parameters plus SuperFlex's tapering
(kx, ky) and bending (k, alpha per axis); the viewer meshes the deformed primitives with SuperFlex's own mesher.
"""

import sys
from pathlib import Path

import numpy as np
import torch
import torch.utils.cpp_extension as cpp
from omegaconf import OmegaConf

from data import DATA, DEV, cli, load_points, result_path

# SuperFlex compiles its PVCNN ops with -std=c++17, torch >= 2.14 headers need C++20 (submodule left untouched)
_load = cpp.load
cpp.load = lambda *a, extra_cflags=(), **kw: _load(*a, extra_cflags=[f.replace("c++17", "c++20") for f in extra_cflags], **kw)
sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "papers" / "superflex"))
from superflex.superflex import SuperFlex  # noqa: E402
from superflex.utils.predictions_handler import extend_dict  # noqa: E402

CHECKPOINTS = ("superflex", "unsup_robust", "robust")  # in Thesis/data/models/superflex/
N_POINTS = 4096
ROT_X_90 = np.array([[1.0, 0, 0], [0, 0, -1], [0, 1, 0]])  # y-up (network) -> z-up (Replica), column vectors


def load_model(name):
    folder = DATA / "models" / "superflex" / name
    model = SuperFlex(OmegaConf.load(folder / "config.yaml").superflex).to(DEV)
    model.load_state_dict(torch.load(folder / "ckpt.pt", map_location=DEV, weights_only=False)["model_state_dict"])
    return model.eval()


def instance_batch(pts, ks=None, seed=0):
    """(I, N_POINTS, 3) normalized, y-up points per instance (all, or those in ks), and the centers / scales to undo it."""
    rng = np.random.default_rng(seed)
    batch, centers, scales = [], [], []
    for k in range(pts["inst"].max() + 1) if ks is None else ks:
        p = pts["pos"][pts["inst"] == k].astype(np.float64)
        p = p[rng.choice(len(p), N_POINTS, replace=len(p) < N_POINTS)]
        c = p.mean(0)
        s = max(2 * np.abs(p - c).max(), 1e-4)
        batch.append((p - c) / s @ ROT_X_90)  # row vectors: q = R^T p
        centers.append(c), scales.append(s)
    return torch.tensor(np.stack(batch), dtype=torch.float32, device=DEV), np.array(centers), np.array(scales)


@torch.no_grad()
def predict(model, batch, centers, scales):
    out = model(batch)
    if "bending_k" in out:
        out["bending_k"] = out["bending_k"] / torch.tensor(scales, device=DEV, dtype=torch.float32)[:, None, None]
    out = {k: v.cpu().numpy() for k, v in extend_dict(out).items() if isinstance(v, torch.Tensor)}
    exist = out["exist"].reshape(out["scale"].shape[:2]) > 0.5
    inst, prim = np.nonzero(exist)
    s = scales[inst][:, None]
    return {"trans": s * (out["trans"][inst, prim] @ ROT_X_90.T) + centers[inst],
            "scale": s * out["scale"][inst, prim], "eps": out["shape"][inst, prim],
            "rot": ROT_X_90 @ out["rotate"][inst, prim], "instance": inst,
            "tapering": out["tapering"][inst, prim], "bending": out["bending"][inst, prim]}


def main(scene):
    pts = load_points(scene)
    batch, centers, scales = instance_batch(pts)
    for name in CHECKPOINTS:
        sqs = predict(load_model(name), batch, centers, scales)
        out = result_path(scene, f"superflex_{name}")
        out.parent.mkdir(parents=True, exist_ok=True)
        np.savez(out, **sqs)
        print(f"{scene} superflex_{name}: {len(sqs['trans'])} SQs for {len(scales)} instances -> {out}")


if __name__ == "__main__":
    for scene in cli():
        main(scene)
