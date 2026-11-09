"""SuperFlex's network on a split, as predicted and after its test-time optimization.

    python fit_superflex.py --split test --out /home/sebastian/data/ShapeNet-Chairs-RGBD/sq_fits [--plain]

`superflex` checkpoint, primitives with existence > 0.5, 1000 steps of fit_sqs.optimize with SuperFlex's defaults.
By default with tapering and bending, as SuperFlex. With --plain they are dropped from the prediction and from the
optimization, comparable to the K sweep.

Output `<out>/<split>/superflex_{pred,opt}[_plain].npz`, in the dataset's frame: fit_sqs' fields for all 16 slots,
plus exist (N, 16) and tapering (N, 16, 2), bending_k and bending_alpha (N, 16, 3).
"""

import argparse
from pathlib import Path

import numpy as np
import torch
import torch.utils.cpp_extension as cpp
from fit_sqs import BATCH, DATA, DEV, STEPS, load_chair, optimize
from omegaconf import OmegaConf

# SuperFlex compiles its PVCNN ops with -std=c++17, torch >= 2.14 headers need C++20 (as in experiment 16)
_load = cpp.load
cpp.load = lambda *a, extra_cflags=(), **kw: _load(*a, extra_cflags=[f.replace("c++17", "c++20") for f in extra_cflags], **kw)
from superflex.optimization import SuperQOptimization  # noqa: E402
from superflex.superflex import SuperFlex  # noqa: E402
from superflex.utils.predictions_handler import PredictionHandler, extend_dict  # noqa: E402

CHECKPOINT = Path("/home/sebastian/Documents/Uni/Thesis/data/models/superflex/superflex")


@torch.no_grad()
def predict(model, surf, plain):
    out = extend_dict(model(torch.tensor(surf, device=DEV)))
    out = {key: value.cpu().numpy() for key, value in out.items() if isinstance(value, torch.Tensor)}
    B, N = out["scale"].shape[:2]
    return PredictionHandler({
        "names": np.arange(B), "pc": surf, "assign_matrix": np.zeros((B, 1, N)), "scale": out["scale"],
        "translation": out["trans"], "rotation": out["rotate"], "exponents": out["shape"],
        "exist": out["exist"].reshape(B, N),
        "tapering": np.zeros_like(out["tapering"]) if plain else out["tapering"],
        "bending": np.zeros_like(out["bending"]) if plain else out["bending"]})


def fit(handler, chairs, surf, plain, steps):
    external = {"points": torch.tensor(surf, device=DEV),
                "points_iou": torch.tensor(np.stack([c["q"] for c in chairs]), device=DEV),
                "occupancies": torch.tensor(np.stack([c["occ"] for c in chairs]), device=DEV)}
    cfg = OmegaConf.create({"tapering": not plain, "bending": not plain, "pruning": False})
    sq = SuperQOptimization(pred_handler=handler, indices=list(range(len(chairs))), device=DEV, external_data=external, cfg=cfg)
    result = optimize(sq, steps)
    with torch.no_grad():
        kb, alpha = sq.bending()
        return result | {"exist": sq.exist_mask.cpu().numpy(), "tapering": sq.tapering().cpu().numpy(),
                         "bending_k": (kb / sq.normalization_scale[:, None, None]).cpu().numpy(),
                         "bending_alpha": alpha.cpu().numpy()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="test")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--plain", action="store_true")
    ap.add_argument("--limit", type=int)
    args = ap.parse_args()

    model = SuperFlex(OmegaConf.load(CHECKPOINT / "config.yaml").superflex).to(DEV)
    model.load_state_dict(torch.load(CHECKPOINT / "ckpt.pt", map_location=DEV, weights_only=False)["model_state_dict"])
    model.eval()

    ids = (DATA / f"{args.split}.lst").read_text().split()[:args.limit]
    results = {"pred": [], "opt": []}
    for start in range(0, len(ids), BATCH):
        chairs = [load_chair(i) for i in ids[start:start + BATCH]]
        surf = np.stack([c["surf"] for c in chairs])
        handler = predict(model, surf, args.plain)
        results["pred"].append(fit(handler, chairs, surf, args.plain, 0))
        results["opt"].append(fit(handler, chairs, surf, args.plain, STEPS))
        print(f"{start + len(chairs)}/{len(ids)}  iou {results['pred'][-1]['iou'].mean():.3f} -> "
              f"{results['opt'][-1]['iou'].mean():.3f}", flush=True)

    suffix = "_plain" if args.plain else ""
    (args.out / args.split).mkdir(parents=True, exist_ok=True)
    for name, parts in results.items():
        np.savez(args.out / args.split / f"superflex_{name}{suffix}.npz", ids=np.array(ids),
                 **{key: np.concatenate([p[key] for p in parts]) for key in parts[0]})


if __name__ == "__main__":
    main()
