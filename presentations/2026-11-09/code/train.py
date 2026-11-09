"""Train the slot model on one or more Replica frames.

    uv run python train.py --name parts --variant parts                 # overfit room_2/000500 (viewer 2)
    uv run python train.py --name F150 --frames room_2 --frame gravity --yaw_aug --steps 30000

--variant (overfit runs; what changes, everything else is the standard pipeline):
  render      no depth input (2D patch positions, no anchor), one SQ per slot, loss: the rendered SQs vs. the
              instance masks (no depth target)
  render_obs  depth input, one SQ per slot, render (instance mask + depth) + observed-point Chamfer (no target SQs)
  single      depth input, one SQ per slot, Chamfer to one SQ fit per instance
  parts       depth input, relative part decoder, part Chamfer + part existence vs. the visible SuperFlex SQs (standard)
Always: mask + dice + exist. Adam (eps 1e-6), lr 3e-5 for the input projection + Slot Attention, 1e-3 for the heads
and the part decoder, gradient clipping 1, K = 64 slots, T = 3. --frames: comma list of scenes (their training block:
the first 75 % of the trajectory) or scene/id frames. --frame gravity: inputs rotated into a z-up frame (camera
forward made horizontal), --yaw_aug: plus a random rotation about z in training (predictions rotated back to the
camera frame before losses and metrics). Results: results/runs/<name>.json (args, eval history), checkpoints in
results/large/<name>/ (not versioned).
"""

import argparse
import json
import math
import time
from pathlib import Path

import numpy as np
import torch

import data as D
import sq
from data import DEV
from losses import losses, match
from metrics import metrics
from model import Net

RESULTS = Path(__file__).resolve().parents[1] / "results"
VARIANTS = {"render": dict(depth=False, decoder="head", targets="visible", losses="render"),
            "render_obs": dict(depth=True, decoder="head", targets="visible", losses="render,r_depth,obs_cd"),
            "single": dict(depth=True, decoder="head", targets="single", losses="pair_cd"),
            "parts": dict(depth=True, decoder="parts", targets="visible", losses="part_cd")}


def train_block(scene):
    ids = D.scene_ids(scene)
    return ids[:int(0.75 * len(ids))]


def frame_ids(spec):
    out = []
    for s in spec.split(","):
        out += [s] if "/" in s else [f"{s}/{i}" for i in train_block(s)]
    return out


def input_rotation(fr, A, train):
    R = fr["R_g"] if A.frame == "gravity" else None
    if train and A.yaw_aug:
        a = 2 * math.pi * torch.rand(()).item()
        Rz = torch.tensor([[math.cos(a), -math.sin(a), 0], [math.sin(a), math.cos(a), 0], [0, 0, 1]], device=DEV)
        R = Rz @ (R if R is not None else torch.eye(3, device=DEV))
    return R


def forward(model, fr, A, train=False, match_fn=None):
    """The model in the run's input frame; all predictions returned in the camera frame."""
    R = input_rotation(fr, A, train)
    out = model(fr["dino"], fr["xyz"] if R is None else fr["xyz"] @ R.T, fr["uv"], match=match_fn)
    if R is not None:  # x_in = R x_cam  ->  back: positions @ R, rotations R^T @
        for d in [out] + ([out["parts"]] if "parts" in out else []):
            d["trans"], d["rotate"] = d["trans"] @ R, R.T @ d["rotate"]
        out["anchor"] = out["anchor"] @ R
    return out


def build(A):
    return Net(A.d, A.K, A.T, A.depth, A.decoder).to(DEV)


def evaluate(model, frames, A):
    with torch.no_grad():
        ms = [metrics(forward(model, fr, A), fr) for fr in frames]
    return {k: float(np.nanmean([m.get(k, np.nan) for m in ms])) for k in ms[0]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True)
    ap.add_argument("--variant", default="parts", choices=list(VARIANTS))
    ap.add_argument("--frames", default="room_2/000500")
    ap.add_argument("--frame", default="camera", choices=["camera", "gravity"])
    ap.add_argument("--yaw_aug", action="store_true")
    ap.add_argument("--steps", type=int, default=10000)
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--eval_every", type=int, default=1000)
    ap.add_argument("--lr", type=float, default=3e-5)
    ap.add_argument("--lr_heads", type=float, default=1e-3)
    ap.add_argument("--d", type=int, default=128)
    ap.add_argument("--K", type=int, default=64)
    ap.add_argument("--T", type=int, default=3)
    ap.add_argument("--seed", type=int, default=0)
    A = ap.parse_args()
    for k, v in VARIANTS[A.variant].items():
        setattr(A, k, v)
    use = set(A.losses.split(","))
    torch.manual_seed(A.seed), np.random.seed(A.seed)
    frames = [fr for fr in (D.load_frame(*f.split("/"), A.targets) for f in frame_ids(A.frames)) if fr["M"] > 0]
    ev_frames = [frames[i] for i in np.linspace(0, len(frames) - 1, min(20, len(frames))).round().astype(int)]
    print(f"{len(frames)} training frames", flush=True)
    model = build(A)
    heads = [p for n, p in model.named_parameters() if n.split(".")[0] in ("scale", "shape", "rot", "trans", "exist", "out_norm", "part_dec")]
    hid = {id(p) for p in heads}
    opt = torch.optim.Adam([{"params": [p for p in model.parameters() if id(p) not in hid], "lr": A.lr},
                            {"params": heads, "lr": A.lr_heads}], eps=1e-6)
    out = RESULTS / "large" / A.name
    out.mkdir(parents=True, exist_ok=True)
    hist, t0, order, pos = [], time.time(), np.random.permutation(len(frames)), 0
    for step in range(1, A.steps + 1):
        batch = []
        while len(batch) < min(A.batch, len(frames)):
            if pos == len(order):
                order, pos = np.random.permutation(len(frames)), 0
            batch.append(frames[order[pos]])
            pos += 1
        loss = 0.0
        for fr in batch:
            pred = forward(model, fr, A, train=True, match_fn=lambda o, fr=fr: match(o, fr))
            sig = pred["sig"] if "sig" in pred else match(pred, fr)
            loss = loss + sum(losses(pred, fr, sig, use).values()) / len(batch)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        gn = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        if torch.isfinite(gn):  # skip non-finite steps (a NaN would poison Adam's state)
            opt.step()
        if step % A.eval_every == 0 or step == A.steps:
            model.eval()
            m = evaluate(model, ev_frames, A)
            model.train()
            hist.append({"step": step, "loss": float(loss), **m})
            print(f"step {step} loss {float(loss):.3f} " + " ".join(f"{k} {v:.3f}" for k, v in m.items()) + f" {time.time() - t0:.0f}s", flush=True)
    torch.save(model.state_dict(), out / "model.pt")
    res = {"name": A.name, "args": vars(A), "history": hist, "seconds": time.time() - t0}
    (RESULTS / "runs").mkdir(parents=True, exist_ok=True)
    (RESULTS / "runs" / f"{A.name}.json").write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
