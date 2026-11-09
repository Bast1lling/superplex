"""Latency of every part of a training step and of the evaluation, per mode (--dense, --topk, --patch_queries).

    uv run python latency_breakdown.py [--steps 60] [--modes dense topk patch]   -> results/latency.md (appends a table per mode)

Wraps the functions of train.py / dense.py / model.py with CUDA-synchronized timers (inclusive and self time, i.e.
without the wrapped functions it calls), runs --steps real training steps on the full training set (after 10 warm-up
steps) and evaluate() on --eval_batches test batches. Data loading = the time the loop waits for the DataLoader.
The syncs serialize CPU and GPU, so the sum is somewhat above the plain step time (also reported, measured unwrapped).
"""

import argparse
import functools
import importlib
import itertools
import json
import time
from collections import defaultdict

import torch

import train as T
import dense as D
import model as M

INC, SELF, CALLS = defaultdict(float), defaultdict(float), defaultdict(int)
STACK = []


def timed(name, fn):
    @functools.wraps(fn)
    def wrap(*a, **k):
        torch.cuda.synchronize()
        STACK.append(0.0)
        t0 = time.perf_counter()
        out = fn(*a, **k)
        torch.cuda.synchronize()
        dt = time.perf_counter() - t0
        child = STACK.pop()
        INC[name] += dt
        SELF[name] += dt - child
        CALLS[name] += 1
        if STACK:
            STACK[-1] += dt
        return out
    return wrap


class Block:
    """Timer for a code block, same bookkeeping as `timed`."""

    def __init__(self, name):
        self.name = name

    def __enter__(self):
        torch.cuda.synchronize()
        STACK.append(0.0)
        self.t0 = time.perf_counter()

    def __exit__(self, *_):
        torch.cuda.synchronize()
        dt = time.perf_counter() - self.t0
        child = STACK.pop()
        INC[self.name] += dt
        SELF[self.name] += dt - child
        CALLS[self.name] += 1
        if STACK:
            STACK[-1] += dt


def patch(model):
    for mod, names in ((T, ("pixel_owner", "match", "linear_sum_assignment", "surface", "chamfer", "instance_masks",
                            "losses", "frame_metrics", "pair_iou", "group_instances", "sdf_batch", "mask_targets",
                            "losses_loop", "pair_losses", "match_cost")),
                       (D, ("assign", "heatmap", "center_focal", "losses", "topk_losses", "pair_iou_matrix",
                            "occupancy_ious", "frame_metrics", "visible_targets"))):
        for n in names:
            setattr(mod, n, timed(f"{mod.__name__}.{n}", getattr(mod, n)))
    M.peaks = D.peaks = timed("model.peaks", M.peaks)
    if hasattr(model, "encode"):
        model.encode = timed("model.encode", model.encode)
        model.inp.forward = timed("model.tokenizer_mlp", model.inp.forward)
        model.encoder.forward = timed("model.encoder", model.encoder.forward)
        model.heads.forward = timed("model.enc_heads", model.heads.forward)
    if hasattr(model, "dec"):
        for l in model.dec:
            l.forward = timed("model.decoder_layer", l.forward)
        for h in model.dec_heads:
            h.forward = timed("model.dec_heads", h.forward)
    if hasattr(model, "tokenizer"):
        model.tokenizer.forward = timed("model.tokenizer", model.tokenizer.forward)
        model.decoder.layers[0].forward = timed("model.decoder_layer", model.decoder.layers[0].forward)
        model.heads.forward = timed("model.heads", model.heads.forward)


def build(mode, n_classes):
    T.FOCAL, T.COST_SAMPLES, T.LOSS, T.MASK = mode == "patch", 4 if mode == "patch" else 6, "cd", "instance"
    if mode == "dense":
        return M.DenseModel(n_classes).to(T.DEV), D.losses, D.evaluate
    if mode == "topk":
        return M.TopKModel(n_classes).to(T.DEV), D.topk_losses, D.evaluate
    return M.Model(n_classes, patch_queries=True).to(T.DEV), T.losses, T.evaluate


def step(model, opt, losses, batch):
    with Block("step.to_device"):
        batch = T.to_device(batch)
    with Block("step.forward"):
        pred = model(batch["dino"], batch["feats"], batch["idx"], batch["depth"])
    with Block("step.loss"):
        loss, _ = losses(pred, batch)
    with Block("step.backward"):
        opt.zero_grad()
        loss.backward()
    with Block("step.optimizer"):
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()


def table(title, n):
    lines = [title, "| part | calls | inclusive ms | self ms |", "|---|---|---|---|"]
    for k in sorted(INC, key=lambda k: -INC[k]):
        lines.append(f"| {k} | {CALLS[k] / n:.1f} | {INC[k] / n * 1e3:.1f} | {SELF[k] / n * 1e3:.1f} |")
    INC.clear(), SELF.clear(), CALLS.clear()
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=60)
    ap.add_argument("--eval_batches", type=int, default=5)
    ap.add_argument("--modes", nargs="+", default=["dense", "topk", "patch"])
    a = ap.parse_args()
    torch.manual_seed(0)
    n_classes = len(json.loads((T.FRAMES / "classes.json").read_text()))
    ds = T.Frames(T.TRAIN)
    t0 = time.perf_counter()
    for i in range(8):
        T.collate([ds[i * 100]])
    load_one = (time.perf_counter() - t0) / 8
    loader = torch.utils.data.DataLoader(ds, batch_size=8, shuffle=True, drop_last=True, collate_fn=T.collate,
                                         num_workers=6, pin_memory=True, persistent_workers=True)
    test = [T.collate([T.Frames(T.TEST)[i] for i in range(j * 8, j * 8 + 8)]) for j in range(a.eval_batches)]
    out = [f"\n# Latency {time.strftime('%Y-%m-%d %H:%M')}, {torch.cuda.get_device_name()}, batch 8, {a.steps} steps"]
    for mode in a.modes:
        model, losses, evaluate = build(mode, n_classes)
        opt = torch.optim.AdamW(model.parameters(), lr=1e-4)
        it = itertools.chain.from_iterable(itertools.repeat(loader))
        for _ in range(10):  # warm-up
            step(model, opt, losses, next(it))
        INC.clear(), SELF.clear(), CALLS.clear()
        # plain step time and loader wait, no wrappers
        torch.cuda.synchronize()
        wait, t0 = 0.0, time.perf_counter()
        for _ in range(a.steps):
            tw = time.perf_counter()
            batch = next(it)
            wait += time.perf_counter() - tw
            batch = T.to_device(batch)
            pred = model(batch["dino"], batch["feats"], batch["idx"], batch["depth"])
            loss, _ = losses(pred, batch)
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
        torch.cuda.synchronize()
        plain = (time.perf_counter() - t0) / a.steps
        INC.clear(), SELF.clear(), CALLS.clear()
        batches = list(itertools.islice(it, a.steps))
        patch(model)
        for b in batches:
            step(model, opt, losses, b)
        res = [f"\n## {mode}\n", f"plain step (unwrapped, no syncs): {plain * 1e3:.0f} ms; DataLoader wait per step (6 "
               f"workers): {wait / a.steps * 1e3:.1f} ms; one frame loaded + collated in a single process: {load_one * 1e3:.0f} ms\n",
               table("per training step:", a.steps)]
        with Block("eval.total"):
            evaluate(model, test)
        res.append(table(f"\nper test batch of 8 (evaluate):", a.eval_batches))
        out.append("\n".join(res))
        print(out[-1], flush=True)
        for mod in (M, T, D):  # undo the wrappers
            importlib.reload(mod)
    with open(T.EXP / "results" / "latency.md", "a") as f:
        f.write("\n".join(out) + "\n")


if __name__ == "__main__":
    main()
