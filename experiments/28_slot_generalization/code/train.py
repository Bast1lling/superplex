"""Exp 27's final model trained on several Replica frames; evaluated on its training frames, on held-out views of the
same scenes and (later) on held-out scenes.

    uv run python train.py --name F10 --train room_2 --n_train 10
    uv run python dashboard.py                                          -> http://127.0.0.1:7828

Model, losses, matching and metrics are exp 27's (imported from ../../27_slot_attention/code; its README section 5):
Slot Attention (K = 64 [exp 27: 128], T = 3, learned init), set part decoder (16 parts per slot, translation = slot mask
centroid + offset), `instance` labels (GT instance masks, target = the instance's visible SuperFlex SQs), losses
mask + dice + exist + part_cd + part_exist (weights 1), exist smoothing 0.01, Adam eps 1e-6, existence-aware part
matching, Adam lr 3e-5 (input projection + Slot Attention) / 1e-3 (heads + part decoder), gradient clip 1.

Frames (exp 22's cache: 200 per scene, every 10th of the trajectory). Per scene, by trajectory position: the first
75 % = training block, the last 20 % = held-out views (the 5 % between them unused, so neighbouring near-duplicates
do not leak). --train: comma list of scenes (their training block) or single frames `scene/id`; --n_train N: N
frames evenly spaced over that list (0 = all). Each step: --batch frames drawn without replacement (one epoch =
one pass), the losses averaged over them (each frame has its own Hungarian matching).
Eval every --eval_every steps, metrics averaged over frames: `train` (up to --n_eval of the training frames, evenly
spaced), `views` (up to --n_eval held-out views of the training scenes), `scenes` (--eval_scenes: up to --n_eval
frames of each held-out scene, the whole trajectory). best.pt = the best views/pair_iou_fr.
--part_dec rel: fast.RelNet (geometry relative to the slot's patches); --frame gravity: the model works in a
gravity-aligned frame (z up), --yaw_aug: random yaw about it during training (all predictions rotated back to the
camera frame before losses / metrics).
--rot_aug: rotation augmentation (training only; the model sees x R^T, its SQs are rotated back, targets / depth /
visibility untouched).
--slot_frame yaw / --w_box / --w_hidden (`box.py`, `fast.BoxRelNet`): slots as gravity-aligned boxes with the parts in
the box frame; box loss against the amodal instance box (from all of the instance's frustum SQs); hidden SQs as
low-weight part targets. Metrics stay against the visible targets; `am_*` = against all frustum SQs.
Viewer: per eval, predictions of --n_show frames per split as exp 27-style single-frame runs
(results/large_files/views/<run>__<split>__<scene>_<id>/).
"""

import argparse
import json
import math
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch

EXP = Path(__file__).resolve().parents[1]
E27 = EXP.parent / "27_slot_attention" / "code"
os.environ.setdefault("TRACKIO_DIR", str(EXP / "results" / "large_files" / "trackio"))
sys.path.insert(0, str(E27))
sys.path.insert(1, str(EXP / "code"))  # fast.py (after exp 27's, whose train.py is "train")
import trackio  # noqa: E402

import common as C  # noqa: E402
import data as D  # noqa: E402
import train as T27  # noqa: E402  (losses / metrics read their options from T27.A)
from common import DEV  # noqa: E402
from model import Net  # noqa: E402
import fast  # noqa: E402  (exp 28: equivalent faster part decoder)
import box  # noqa: E402  (exp 28: amodal box / hidden-part targets)

PROJECT = "slot_generalization"
OUT = EXP / "results" / "large_files"
MAIN = ("mask_miou", "pair_iou_fr", "part_iou_fr")
TRAIN_FRAC, TEST_FRAC = 0.75, 0.20


def scene_ids(scene):
    return sorted(p.name.split("_")[0] for p in (D.FRAMES / scene).glob("*_dino.npy"))


def block(scene, which):
    ids = scene_ids(scene)
    n = len(ids)
    if which == "train":
        return ids[:int(TRAIN_FRAC * n)]
    if which == "views":
        return ids[n - int(TEST_FRAC * n):]
    return ids


def spaced(xs, n):
    if n <= 0 or n >= len(xs):
        return list(xs)
    return [xs[int(i)] for i in np.linspace(0, len(xs) - 1, n).round()]


def frame_lists(A):
    train = []
    for s in A.train.split(","):
        train += [s] if "/" in s else [f"{s}/{i}" for i in block(s, "train")]
    train = spaced(train, A.n_train)
    scenes = sorted({f.split("/")[0] for f in train})
    views = spaced([f"{s}/{i}" for s in scenes for i in block(s, "views")], A.n_eval)
    other = [f"{s}/{i}" for s in A.eval_scenes.split(",") if s for i in spaced(block(s, "all"), A.n_eval)]
    return {"train": train, "train_eval": spaced(train, A.n_eval), "views": views, "scenes": other}


def gravity_rotation(c2w):
    """Camera -> gravity-aligned frame (rows of R: x = the camera's forward direction made horizontal, y = z cross x,
    z = up; Replica's world z is up). x_g = R x_cam."""
    up = torch.as_tensor(c2w[:3, :3][2], dtype=torch.float32, device=DEV)  # world z in camera coordinates
    up = up / up.norm()
    fwd = torch.tensor([0.0, 0.0, 1.0], device=DEV)
    x = fwd - (fwd @ up) * up
    x = x / x.norm()
    return torch.stack([x, torch.linalg.cross(up, x), up])


def load(fid, A):
    scene, i = fid.split("/")
    fr = D.load_frame(scene, i, A.mask_source, A.part_targets)
    if fr["M"] == 0:
        return None
    fr["R_g"] = gravity_rotation(np.load(D.FRAMES / scene / f"{i}_view.npz")["c2w"])
    fr["dino"] = fr["dino"].half()  # the cache is fp16 anyway (lossless); halves GPU memory for the 7-scene rung
    if amodal(A):  # all of each instance's frustum SQs (same target order): box / hidden-part targets, am_ metrics
        fa = D.load_frame(scene, i, A.mask_source, "all")
        assert fa["M"] == fr["M"]
        fr["groups_all"] = fa["groups"]
        fr["vis_all"] = box.visible_members(fa["groups"], fr["depth"])
        fr["box"] = box.gt_box(fa["groups"], fr["R_g"])
    return fr


def amodal(A):
    return getattr(A, "slot_frame", "none") != "none" or getattr(A, "w_box", 0) > 0 or getattr(A, "w_hidden", 0) > 0


def random_rotation(max_deg):
    """Uniform on SO(3) for max_deg >= 180, else a uniform axis and an angle uniform in [0, max_deg]."""
    if max_deg >= 180:
        q = torch.randn(4, device=DEV)
        return C.quat2mat(q[None] / q.norm())[0]
    ax = torch.randn(3, device=DEV)
    ax = ax / ax.norm()
    a = math.radians(max_deg) * torch.rand((), device=DEV)
    Kx = torch.tensor([[0, -ax[2], ax[1]], [ax[2], 0, -ax[0]], [-ax[1], ax[0], 0]], device=DEV)
    return torch.eye(3, device=DEV) + torch.sin(a) * Kx + (1 - torch.cos(a)) * Kx @ Kx


def unrotate(pred, R):
    """Predictions made on positions x R^T (rotated about the camera centre) -> back in the camera frame."""
    for d in [pred] + ([pred["parts"]] if "parts" in pred else []):
        d["trans"] = d["trans"] @ R
        d["rotate"] = R.T @ d["rotate"]
    if "anchor" in pred:
        pred["anchor"] = pred["anchor"] @ R
    return pred


ARGS = None  # main()'s args (or the run's, set by the viewer / diagnose.py)


def input_rotation(fr, A, train):
    """The rotation R of the model's input positions (x R^T; its SQs are rotated back), or None: --frame gravity
    (R_g), times a random yaw about the gravity axis (--yaw_aug, training only), times a random rotation (--rot_aug,
    training only)."""
    R = fr["R_g"] if getattr(A, "frame", "camera") == "gravity" else None
    if train and getattr(A, "yaw_aug", False):
        a = 2 * math.pi * torch.rand((), device=DEV)
        c, s_ = torch.cos(a), torch.sin(a)
        z, o = torch.zeros((), device=DEV), torch.ones((), device=DEV)
        Rz = torch.stack([torch.stack([c, -s_, z]), torch.stack([s_, c, z]), torch.stack([z, z, o])])
        R = Rz @ (R if R is not None else torch.eye(3, device=DEV))
    if train and getattr(A, "rot_aug", 0) > 0:
        R = random_rotation(A.rot_aug) @ (R if R is not None else torch.eye(3, device=DEV))
    return R


def forward(model, fr, A=None, train=False, match=None):
    """The model on frame fr, in the input frame of the run's args (camera / gravity / augmented); predictions in
    the camera frame. match: fast.Net's matched-slots-only part decoding."""
    A = A or ARGS
    R = input_rotation(fr, A, train) if A is not None else None
    kw = {"match": match} if match is not None else {}
    out = model(fr["dino"].float(), fr["xyz"] if R is None else fr["xyz"] @ R.T, **kw)
    return out if R is None else unrotate(out, R)


def build_model(A):
    if getattr(A, "slot_frame", "none") == "yaw":
        assert A.part_dec == "rel" and A.frame == "gravity"
    cls = fast.BoxRelNet if getattr(A, "slot_frame", "none") == "yaw" else fast.RelNet if getattr(A, "part_dec", "abs") == "rel" else (fast.Net if getattr(A, "fast", True) else Net)
    net = cls(A.d, A.K, A.T, 0, A.anchor, False, "slot", A.parts)
    L = getattr(A, "dec_layers", 2)
    if L != 2:  # exp 27's constructors build the part decoder with 2 layers
        net.part_dec = (fast.PartDecoderRel if getattr(A, "part_dec", "abs") == "rel" else fast.M27.PartDecoder)(A.d, A.parts, L)
    return net.to(DEV)


def lr_factor(step, A):
    """linear warmup over --warmup steps, then constant or cosine decay to 0 at --steps (both lr groups)."""
    w = min(1.0, step / A.warmup) if getattr(A, "warmup", 0) > 0 else 1.0
    if getattr(A, "lr_sched", "const") == "cosine":
        w *= 0.5 * (1 + math.cos(math.pi * min(1.0, step / A.steps)))
    return w


@torch.no_grad()
def evaluate(model, frames):
    ms = []
    for fr in frames:
        pred = forward(model, fr)
        m = T27.metrics(pred, fr)
        if "groups_all" in fr:  # against all of the instance's frustum SQs (amodal targets)
            am = T27.metrics(pred, {**fr, "groups": fr["groups_all"]})
            m |= {f"am_{k}": am[k] for k in ("pair_iou_fr", "part_iou_fr") if k in am}
        if getattr(model, "part_frame", False):
            m |= box.box_metrics(pred, fr)
        ms.append(m)
    return {k: float(np.nanmean([m[k] for m in ms])) for k in ms[0]} if ms else {}


def view_dir(A, split, fr):
    return OUT / "views" / f"{A.name}__{split}__{fr['name']}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True)
    ap.add_argument("--group", default="generalization")
    ap.add_argument("--train", default="room_2", help="comma list of scenes (training block) or scene/id frames")
    ap.add_argument("--n_train", type=int, default=0, help="frames evenly spaced over the list (0 = all)")
    ap.add_argument("--eval_scenes", default="", help="held-out scenes (comma list)")
    ap.add_argument("--n_eval", type=int, default=20)
    ap.add_argument("--n_show", type=int, default=3, help="frames per split saved for the viewer")
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--steps", type=int, default=10000)
    ap.add_argument("--eval_every", type=int, default=1000)
    ap.add_argument("--lr", type=float, default=3e-5)
    ap.add_argument("--lr_heads", type=float, default=1e-3)
    ap.add_argument("--d", type=int, default=128)
    ap.add_argument("--K", type=int, default=64)
    ap.add_argument("--T", type=int, default=3)
    ap.add_argument("--parts", type=int, default=16)
    ap.add_argument("--anchor", default="centroid", choices=["centroid", "none"])
    ap.add_argument("--losses", default="mask,dice,exist,part_cd,part_exist")
    ap.add_argument("--mask_source", default="instance")
    ap.add_argument("--part_targets", default="visible", choices=["all", "visible"])
    ap.add_argument("--exist_smooth", type=float, default=0.01)
    ap.add_argument("--adam_eps", type=float, default=1e-6)
    ap.add_argument("--part_match_exist", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--w_occ", type=float, default=1.0)
    ap.add_argument("--lam", type=float, default=0.1)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--part_dec", default="abs", choices=["abs", "rel"],
                    help="abs: exp 27's part decoder; rel: fast.RelNet (positions relative to the slot anchor, part "
                         "centres = pointer over the slot's patch centroids + offset)")
    ap.add_argument("--frame", default="camera", choices=["camera", "gravity"],
                    help="the model's input frame (gravity: rotated so z is up, x = camera forward made horizontal)")
    ap.add_argument("--dec_layers", type=int, default=2, help="part decoder layers (exp 27: 2)")
    ap.add_argument("--lr_sched", default="const", choices=["const", "cosine"], help="cosine: decay to 0 at --steps")
    ap.add_argument("--warmup", type=int, default=0, help="linear lr warmup steps (0 = none)")
    ap.add_argument("--slot_frame", default="none", choices=["none", "yaw"],
                    help="yaw: fast.BoxRelNet, each slot a gravity-aligned box (yaw head), its parts in that frame")
    ap.add_argument("--w_box", type=float, default=0.0, help="weight of the slot box loss vs. the amodal instance box")
    ap.add_argument("--w_hidden", type=float, default=0.0,
                    help="> 0: part targets = all of the instance's frustum SQs, hidden ones weighted this (0: visible only)")
    ap.add_argument("--yaw_aug", action="store_true", help="training: random yaw about the gravity axis")
    ap.add_argument("--rot_aug", type=float, default=0.0,
                    help="training: patch positions rotated about the camera centre by a random rotation (max angle in "
                         "degrees, >= 180: uniform on SO(3)), predictions rotated back before the losses; 0 = off")
    ap.add_argument("--fast", action=argparse.BooleanOptionalAction, default=True,
                    help="fast.Net: shared K/V projections, parts of matched slots only in training (same results)")
    A = ap.parse_args()
    A.assign, A.geo_owned = "gt", False  # exp 27 options its losses read (unused by this loss set)
    T27.A = A
    torch.manual_seed(A.seed), np.random.seed(A.seed)
    lists = frame_lists(A)
    t0 = time.time()
    cache = {}
    sets = {}
    for k, ids in lists.items():
        sets[k] = []
        for f in ids:
            if f not in cache:
                cache[f] = load(f, A)
            if cache[f] is not None:
                sets[k].append(cache[f])
    print({k: len(v) for k, v in sets.items()}, f"frames loaded in {time.time() - t0:.0f}s;",
          "targets per training frame: max", max(fr["M"] for fr in sets["train"]), flush=True)
    global ARGS
    ARGS = A
    model = build_model(A)
    heads = [p for n, p in model.named_parameters() if n.split(".")[0] in
             ("scale", "shape", "rot", "yaw", "trans", "exist", "out_norm", "layers", "part_dec")]
    hid = {id(p) for p in heads}
    rest = [p for p in model.parameters() if id(p) not in hid]
    opt = torch.optim.Adam([{"params": rest, "lr": A.lr}, {"params": heads, "lr": A.lr_heads}], eps=A.adam_eps)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda i: lr_factor(i, A))
    cfg = vars(A) | {"frames": {k: len(v) for k, v in sets.items()}, "train_ids": lists["train"]}
    trackio.init(project=PROJECT, name=A.name, group=A.group, config=cfg)
    (OUT / "runs" / A.name).mkdir(parents=True, exist_ok=True)
    (OUT / "runs" / A.name / "args.json").write_text(json.dumps(cfg))
    show = {"train": spaced(sets["train_eval"], A.n_show), "views": spaced(sets["views"], A.n_show),
            "scenes": spaced(sets["scenes"], A.n_show)}
    for split, frs in show.items():  # exp 27 viewer runs, one per shown frame
        for fr in frs:
            d = view_dir(A, split, fr)
            d.mkdir(parents=True, exist_ok=True)
            (d / "args.json").write_text(json.dumps(vars(A) | {"name": d.name, "group": f"{A.name} ({split})",
                                                               "frame": "/".join(fr["name"].rsplit("_", 1))}))
    train = sets["train"]
    order, pos = np.random.permutation(len(train)), 0
    hist, skipped, best = [], 0, (-1.0, 0, None)
    t0 = time.time()
    for step in range(1, A.steps + 1):
        batch = []
        while len(batch) < min(A.batch, len(train)):
            if pos == len(order):
                order, pos = np.random.permutation(len(train)), 0
            batch.append(train[order[pos]])
            pos += 1
        parts = {}
        loss = 0.0
        for fr in batch:
            if A.fast or A.part_dec == "rel":
                pred = forward(model, fr, A, train=True, match=lambda o, fr=fr: T27.match(o, fr))
                sig = pred["sig"]
            else:
                pred = forward(model, fr, A, train=True)
                sig = T27.match(pred, fr)
            lp = T27.losses(pred, fr, sig)
            if A.w_hidden > 0:
                lp |= box.part_losses_all(pred, fr, sig, A.w_hidden)
            if A.w_box > 0:
                lp["box"] = A.w_box * box.box_loss(pred["trans"][sig], pred["rotate"][sig], pred["scale"][sig], fr["box"]).mean()
            loss = loss + sum(lp.values()) / len(batch)
            for k, v in lp.items():
                parts[k] = parts.get(k, 0.0) + v.detach() / len(batch)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        gn = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        if not torch.isfinite(gn):
            skipped += 1
            opt.zero_grad(set_to_none=True)
            print(f"step {step}: non-finite gradient, skipped ({skipped} so far)", flush=True)
            continue
        opt.step()
        sched.step()
        log = {"main/train_loss": float(loss.detach()), "other/grad_norm": float(gn),
               **{f"loss/{k}": float(v) for k, v in parts.items()}}
        if step % A.eval_every == 0 or step == A.steps:
            model.eval()
            ev = {}
            with torch.no_grad():
                for split, key in (("train", "train_eval"), ("views", "views"), ("scenes", "scenes")):
                    if sets[key]:
                        ev[split] = evaluate(model, sets[key])
                for split, frs in show.items():
                    for fr in frs:
                        pe = forward(model, fr)
                        T27.save(pe, fr, T27.match(pe, fr), view_dir(A, split, fr) / "steps" / f"{step:06d}.npz")
            model.train()
            for split, m in ev.items():
                log |= {f"{'main' if k in MAIN else 'other'}/{split}_{k}": v for k, v in m.items()}
            hist.append({"step": step, **ev})
            score = ev.get("views", {}).get("pair_iou_fr", float("nan"))
            if score == score and score > best[0]:
                best = (score, step, {k: v.detach().clone() for k, v in model.state_dict().items()})
            print(f"step {step} loss {float(loss):.3f} " + " | ".join(
                f"{s}: miou {m['mask_miou']:.3f} iou_fr {m['pair_iou_fr']:.3f} part {m.get('part_iou_fr', float('nan')):.3f}"
                for s, m in ev.items()) + "".join(f" | {s} am {m['am_pair_iou_fr']:.3f}" + (f" box {m['box_corner_err']:.3f}m yaw {m['box_yaw_err']:.0f}" if "box_corner_err" in m else "")
                                                  for s, m in ev.items() if "am_pair_iou_fr" in m) + f" {time.time() - t0:.0f}s", flush=True)
        trackio.log(log, step=step)
    trackio.finish()
    torch.save(model.state_dict(), OUT / "runs" / A.name / "model.pt")
    if best[2] is not None:
        torch.save(best[2], OUT / "runs" / A.name / "best.pt")
    res = {"name": A.name, "group": A.group, "args": cfg, "final": hist[-1] if hist else None, "history": hist,
           "seconds": time.time() - t0, "skipped_steps": skipped, "best": {"step": best[1], "views_pair_iou_fr": best[0]}}
    (EXP / "results" / "runs").mkdir(parents=True, exist_ok=True)
    (EXP / "results" / "runs" / f"{A.name}.json").write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
