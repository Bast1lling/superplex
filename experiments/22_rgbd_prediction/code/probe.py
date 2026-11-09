"""Linear probes on the frozen inputs: how much of the targets is linearly predictable from a patch's DINO token (+ its
3D centroid), trained on the training frames, tested on room_0? Compares to what the trained model reaches.

    uv run python probe.py      -> results/probe.json

Per patch owned by a visible target SQ (`cache_assign.py`): class (logistic regression, accuracy), and the target's
center offset from the patch's 3D centroid and its log scale (ridge regression, R^2 and median error). Every 2nd
training frame, every test frame.
"""

import json

import numpy as np
import torch
import torch.nn.functional as F

import train as T
from model import CX, CY, FX, FY, GH, GW, P


def patches(scenes, step):
    ds = T.Frames(scenes)
    X, C, Y_off, Y_scale = [], [], [], []
    for i in range(0, len(ds), step):
        it = ds[i]
        own = it["owner"].reshape(-1)
        m = own >= 0
        d = torch.tensor(it["depth"])
        v, u = torch.meshgrid(torch.arange(d.shape[0]), torch.arange(d.shape[1]), indexing="ij")
        xyz = torch.stack([(u - CX) / FX * d, (v - CY) / FY * d, d], -1)
        valid = (d > 0).float()[..., None]
        cen = F.avg_pool2d((xyz * valid).permute(2, 0, 1)[None], P)[0] / F.avg_pool2d(valid.permute(2, 0, 1)[None], P)[0].clamp_min(1e-6)
        cen = cen.permute(1, 2, 0).reshape(-1, 3).numpy()
        tok = it["dino"].reshape(GH * GW, -1).astype(np.float32)
        o = own[m]
        X.append(np.concatenate([tok[m], cen[m]], 1)), C.append(it["cls"][o])
        Y_off.append(it["translation"][o] - cen[m]), Y_scale.append(np.log(it["scale"][o]))
    return [np.concatenate(z) for z in (X, C, Y_off, Y_scale)]


def main():
    tr, te = patches(T.TRAIN, 2), patches(T.TEST, 1)
    rng = np.random.default_rng(0)  # subsample (the ridge normal equations of ~2M x 1027 do not fit the GPU)
    tr = [z[k] for k in [rng.choice(len(tr[0]), min(len(tr[0]), 300000), replace=False)] for z in tr]
    te = [z[k] for k in [rng.choice(len(te[0]), min(len(te[0]), 100000), replace=False)] for z in te]
    mu, sd = tr[0].mean(0), tr[0].std(0) + 1e-6
    Xtr, Xte = (torch.tensor((x[0] - mu) / sd, device=T.DEV) for x in (tr, te))
    out = {"train_patches": len(Xtr), "test_patches": len(Xte)}
    n_cls = int(max(tr[1].max(), te[1].max())) + 1
    for wd in (1e-4, 1e-2):  # logistic regression, full batch LBFGS-free Adam
        lin = torch.nn.Linear(Xtr.shape[1], n_cls).to(T.DEV)
        opt = torch.optim.Adam(lin.parameters(), 1e-3, weight_decay=wd)
        ytr, yte = torch.tensor(tr[1], device=T.DEV).long(), torch.tensor(te[1], device=T.DEV).long()
        for _ in range(1500):
            idx = torch.randint(len(Xtr), (8192,), device=T.DEV)
            loss = F.cross_entropy(lin(Xtr[idx]), ytr[idx])
            opt.zero_grad(), loss.backward(), opt.step()
        with torch.no_grad():
            out[f"cls_acc_wd{wd}"] = {"train": float((lin(Xtr).argmax(1) == ytr).float().mean()),
                                      "test": float((lin(Xte).argmax(1) == yte).float().mean())}
    for name, k in (("center_offset", 2), ("log_scale", 3)):  # ridge regression, closed form
        A = torch.cat([Xtr, torch.ones(len(Xtr), 1, device=T.DEV)], 1).double()
        At = torch.cat([Xte, torch.ones(len(Xte), 1, device=T.DEV)], 1).double()
        y, yt = torch.tensor(tr[k], device=T.DEV).double(), torch.tensor(te[k], device=T.DEV).double()
        for lam in (1e1, 1e3, 1e5):
            w = torch.linalg.solve(A.T @ A + lam * torch.eye(A.shape[1], device=T.DEV, dtype=A.dtype), A.T @ y)
            r2 = lambda X, Y: float(1 - ((X @ w - Y) ** 2).sum() / ((Y - Y.mean(0)) ** 2).sum())
            err = lambda X, Y: float((X @ w - Y).norm(dim=1).median())
            out[f"{name}_lam{lam:g}"] = {"r2_train": r2(A, y), "r2_test": r2(At, yt), "median_err_train": err(A, y),
                                         "median_err_test": err(At, yt), "median_err_zero_test": float(yt.norm(dim=1).median())
                                         if name == "center_offset" else float((yt - y.mean(0)).norm(dim=1).median())}
    (T.EXP / "results" / "probe.json").write_text(json.dumps(out, indent=1))
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
