"""Time the parts of one training step (train.py's), synchronized per section; plus the eval cost.

    uv run python profile_step.py [--batch 4] [--n_train 10] [--steps 60]
"""

import argparse
import time
from collections import defaultdict
from contextlib import contextmanager

import numpy as np
import torch

import importlib.util
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1] / "27_slot_attention" / "code"))  # "train" = exp 27's
_s = importlib.util.spec_from_file_location("train28", HERE / "train.py")
T = importlib.util.module_from_spec(_s)
_s.loader.exec_module(T)
from model import fourier3d  # noqa: E402

T27, D, C, Net = T.T27, T.D, T.C, T.Net
TIMES = defaultdict(list)


@contextmanager
def tick(name):
    torch.cuda.synchronize()
    t = time.perf_counter()
    yield
    torch.cuda.synchronize()
    TIMES[name].append(time.perf_counter() - t)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--n_train", type=int, default=10)
    ap.add_argument("--steps", type=int, default=60)
    ap.add_argument("--K", type=int, default=64)
    ap.add_argument("--trace", default="")
    ap.add_argument("--fast", action="store_true")
    a = ap.parse_args()
    A = type("A", (), {})()  # train.py's defaults
    defaults = dict(d=128, K=a.K, T=3, parts=16, anchor="centroid", losses="mask,dice,exist,part_cd,part_exist",
                    mask_source="instance", part_targets="visible", exist_smooth=0.01, adam_eps=1e-6,
                    part_match_exist=True, w_occ=1.0, lam=0.1, assign="gt", geo_owned=False, lr=3e-5, lr_heads=1e-3)
    for k, v in defaults.items():
        setattr(A, k, v)
    T27.A = A
    ids = T.spaced([f"room_2/{i}" for i in T.block("room_2", "train")], a.n_train)
    frames = [T.load(f, A) for f in ids]
    model = (T.fast.Net if a.fast else Net)(A.d, A.K, A.T, 0, A.anchor, False, "slot", A.parts).to(C.DEV)
    opt = torch.optim.Adam(model.parameters(), lr=1e-4, eps=A.adam_eps)
    use = set(A.losses.split(","))
    for step in range(a.steps):
        if step == 10:
            TIMES.clear()
        batch = [frames[i] for i in np.random.choice(len(frames), min(a.batch, len(frames)), replace=False)]
        with tick("step total"):
            loss = 0.0
            for fr in batch:
                with tick("forward + match"):
                    if a.fast:
                        pred = model(fr["dino"], fr["xyz"], match=lambda o: T27.match(o, fr))
                        sig = pred["sig"]
                    else:
                        pred = model(fr["dino"], fr["xyz"])
                        sig = T27.match(pred, fr)
                with tick("losses"):
                    lp = T27.losses(pred, fr, sig)
                loss = loss + sum(lp.values()) / len(batch)
                with tick("float(loss parts)"):
                    _ = {k: float(v) for k, v in lp.items()}
            with tick("backward"):
                opt.zero_grad(set_to_none=True)
                loss.backward()
            with tick("clip + step"):
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step()
    # sub-parts of the forward / losses for one frame
    fr = frames[0]
    for _ in range(30):
        with tick("  fwd: input + Slot Attention"):
            pe = model.pos(fourier3d(fr["xyz"]))
            x = model.inp(fr["dino"]) + pe
            slots, logits = model.sa(x)
        with tick("  fwd: part decoder (all slots)"):
            h = model.out_norm(slots)
            att = logits.detach().softmax(1)
            c = (att.T @ fr["xyz"]) / att.sum(0)[:, None].clamp_min(1e-6)
            (T.fast.part_forward(model.part_dec, h, x, pe, att, c) if a.fast else model.part_dec(h, x, pe, att, c))
        pred = model(fr["dino"], fr["xyz"])
        sig = T27.match(pred, fr)
        Pm = {k: pred["parts"][k][sig] for k in C.KEYS}
        with tick("  loss: part_match (Hungarian per target)"):
            mi, pi, ti = T27.part_match({k: v.detach() for k, v in Pm.items()}, fr["groups"], pred["parts"]["exist_logit"][sig].detach())
        with tick("  loss: part_cd"):
            pp = C.surf({k: v[mi, pi] for k, v in Pm.items()}, 16, "cube")
            Gt = fr["groups"]["valid"].shape[1]
            tp = fr["groups"]["pts"].reshape(fr["M"], Gt, -1, 3)[mi, ti]
            wp = D.sample_weights(pp.detach(), fr["depth"], 1.0, 0.1)
            wt = D.sample_weights(tp, fr["depth"], 1.0, 0.1)
            T27.weighted_chamfer(pp, tp, wp, wt).mean()
    with tick("eval: metrics per frame"):
        with torch.no_grad():
            T27.metrics(model(fr["dino"], fr["xyz"]), fr)
    for _ in range(3):
        with tick("eval: metrics per frame"):
            with torch.no_grad():
                T27.metrics(model(fr["dino"], fr["xyz"]), fr)
    print(f"batch {a.batch}, K {a.K}, {len(frames)} frames, M = {[f['M'] for f in frames]}")
    for k, v in TIMES.items():
        print(f"{k:45s} {1000 * np.median(v):8.2f} ms  (n {len(v)})")
    if a.trace:
        from torch.profiler import ProfilerActivity, profile
        with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
            for _ in range(5):
                if a.fast:
                    pred = model(fr["dino"], fr["xyz"], match=lambda o: T27.match(o, fr))
                    lp = T27.losses(pred, fr, pred["sig"])
                else:
                    pred = model(fr["dino"], fr["xyz"])
                    lp = T27.losses(pred, fr, T27.match(pred, fr))
                sum(lp.values()).backward()
        print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=22))


if __name__ == "__main__":
    main()
