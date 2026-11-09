"""Exports the runs for the static viewers into viewer/data/ (not versioned).

    uv run python export.py                       # every run in RUNS whose checkpoint exists
    python -m http.server -d presentations/2026-11-09/viewer 8030      -> http://localhost:8030

Overfit runs (viewer 2): results/large/overfit_<variant>/model.pt (code/train.py) on room_2/000500.
Generalization runs (viewer 3): exp 28's checkpoints and eval histories, copied once into results/large/<run>/ and
results/runs/<run>.json by `--copy_exp28` (they load unchanged into code/model.py's Net), on 12 of their training
frames, the held-out views of the training scenes and up to 20 frames of their held-out scene.
data/index.json: the runs; data/frames/<frame>.json: per frame RGB, GT labels, points, targets (shared by the runs);
data/runs/<run>/run.json: args, curves, frames with split + metrics; data/runs/<run>/<frame>.json: the prediction.
"""

import argparse
import base64
import colorsys
import io
import json
import shutil
import sys
import types
from pathlib import Path

import numpy as np
import torch
from PIL import Image

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "code"))
import data as D  # noqa: E402
import sq  # noqa: E402
import train as T  # noqa: E402
from losses import match  # noqa: E402
from metrics import active_parts, metrics  # noqa: E402

RES = HERE.parent / "results"
OUT = HERE / "data"
EXP28 = HERE.parents[2].parent / "superplex" / "experiments" / "28_slot_generalization" / "results"  # main checkout
OVERFIT = {"overfit_render": "a) RGB only, rendered instance masks",
           "overfit_render_obs": "b) RGB-D, render + observed points",
           "overfit_single": "c) RGB-D, one target SQ per instance",
           "overfit_parts": "d) RGB-D, part decoder + SuperFlex SQs (standard)"}
GENERAL = {"F150_rel_30k": "a) room_2, 150 frames, camera frame",
           "F150_rel_grav_30k": "b) room_2, 150 frames, gravity frame + yaw augmentation",
           "S7_rel_grav_30k": "c) 7 scenes (1050 frames) → room_0 held out, gravity frame + yaw"}
# fixed frames shown for a run (else: 12 training frames + all held-out views): the 7-scene run, one held-out view of
# 3 training scenes + one frame of the held-out scene, each the upper-median frustum IoU of 8 candidates of its scene;
# plus the middle training frame of 4 training scenes
SHOW = {"S7_rel_grav_30k": [("office_0/000750", "train"), ("office_3/000750", "train"), ("room_1/000750", "train"),
                            ("room_2/000750", "train"), ("office_0/001990", "views"), ("office_3/001880", "views"), ("room_1/001660", "views"),
                            ("room_0/001990", "scene")]}
N_PTS = 25000


def hue(i):
    """Target colors, as in the viewer: HSL((0.618 i) mod 1, 0.65, 0.55)."""
    r, g, b = colorsys.hls_to_rgb((i * 0.618034) % 1, 0.55, 0.65)
    return np.array([r, g, b]) * 255


def png(a):
    b = io.BytesIO()
    Image.fromarray(a.astype(np.uint8)).save(b, format="png")
    return base64.b64encode(b.getvalue()).decode()


def b64(a):
    return base64.b64encode(np.ascontiguousarray(a).tobytes()).decode()


def colorize(lab, M):
    img = np.full(lab.shape + (3,), 25, np.float32)
    for m in range(M):
        img[lab == m] = hue(m)
    return img


def sq_json(s, i):
    return {"scale": s["scale"][i].tolist(), "shape": s["shape"][i].tolist(), "rot": s["rotate"][i].reshape(-1).tolist(),
            "trans": s["trans"][i].tolist()}


def np_sq(s):
    return {k: v.detach().cpu().numpy().astype(np.float64).round(4) for k, v in s.items() if k in sq.KEYS}


@torch.no_grad()
def export_frame(fr, path):
    if path.exists():
        return
    lab = fr["lab"].cpu().numpy()
    xyz = D.backproject(fr["depth"]).cpu().numpy()
    ok = np.flatnonzero(fr["depth"].reshape(-1).cpu().numpy() > 0)
    sel = ok[np.random.RandomState(0).permutation(len(ok))[:N_PTS]]
    gr = fr["groups"]
    g = np_sq({k: v.flatten(0, 1) for k, v in gr["sq"].items()})
    valid = gr["valid"].reshape(-1).cpu().numpy()
    G = gr["valid"].shape[1]
    tsq = [sq_json(g, j) | {"t": int(j // G)} for j in np.flatnonzero(valid)]
    targets = [{"id": m, "instance": int(fr["instance"][m]), "px": int((lab == m).sum()), "n_sq": int(gr["valid"][m].sum())}
               for m in range(fr["M"])]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"frame": fr["name"], "M": fr["M"], "rgb": png(fr["rgb"]), "labels": png(colorize(lab, fr["M"])),
                                "points": {"xyz": b64(xyz[sel].astype(np.float32)), "rgb": b64(fr["rgb"].reshape(-1, 3)[sel].astype(np.uint8)),
                                           "label": b64(lab.reshape(-1)[sel].astype(np.int16))},
                                "targets": targets, "target_sqs": tsq}))


@torch.no_grad()
def export_pred(model, fr, A, path):
    pred = T.forward(model, fr, A)
    m = metrics(pred, fr, per_target=True)
    sig = match(pred, fr)
    K = pred["logits"].shape[1]
    tgt_of = np.full(K, -1)
    tgt_of[sig.cpu().numpy()] = np.arange(fr["M"])
    ex = pred["exist_logit"].sigmoid().cpu().numpy()
    attn = pred["logits"].softmax(1)  # (N, K)
    am = pred["logits"].argmax(1).cpu().numpy()
    # predicted masks: patch -> its argmax slot, colored as the slot's target (gray: unmatched)
    col = np.array([hue(t) if t >= 0 else np.array([110, 110, 110]) for t in tgt_of])
    pm = col[am].reshape(D.GH, D.GW, 3).repeat(D.P, 0).repeat(D.P, 1)
    pm[~fr["pvalid"].cpu().numpy().reshape(D.GH, D.GW).repeat(D.P, 0).repeat(D.P, 1)] = 25
    sqs, owner = [], []  # the SQs shown: matched slots' (active parts), unmatched slots' with exist > 0.3
    if "parts" in pred:
        pex = pred["parts"]["exist_logit"].sigmoid()
        act = pex > 0.5
        act[sig] = active_parts(pred, sig)
        P = np_sq(pred["parts"])
        pe = pex.cpu().numpy()
        for k in range(K):
            if tgt_of[k] < 0 and ex[k] <= 0.3:
                continue
            for g_ in np.flatnonzero(act[k].cpu().numpy()):
                sqs.append({k_: (v[k, g_].reshape(-1) if k_ == "rotate" else v[k, g_]).tolist() for k_, v in P.items()}
                           | {"exist": float(ex[k]), "pexist": float(pe[k, g_]), "slot": k, "part": int(g_), "t": int(tgt_of[k])})
                owner.append(k)
    else:
        S = np_sq(pred)
        for k in range(K):
            if tgt_of[k] < 0 and ex[k] <= 0.3:
                continue
            sqs.append({k_: (v[k].reshape(-1) if k_ == "rotate" else v[k]).tolist() for k_, v in S.items()}
                       | {"exist": float(ex[k]), "slot": k, "t": int(tgt_of[k])})
            owner.append(k)
    for s in sqs:
        s["rot"] = s.pop("rotate")
    # rendered predicted SQs (one color per target, gray: unmatched)
    if sqs:
        rs = {k_: torch.tensor(np.array([s[k_ if k_ != "rotate" else "rot"] for s in sqs]), device=D.DEV).float() for k_ in sq.KEYS}
        rs["rotate"] = rs["rotate"].reshape(-1, 3, 3)
        r = sq.render(rs, stride=2)  # 84 x 150, shown upscaled
        w = r["w"].cpu().numpy()
        who = w.argmax(1)
        tcol = np.array([hue(s["t"]) if s["t"] >= 0 else np.array([110, 110, 110]) for s in sqs])
        img = np.where((r["cover"].cpu().numpy() > 0.5)[:, None], tcol[who], 25).reshape(D.H // 2, D.W // 2, 3).repeat(2, 0).repeat(2, 1)
    else:
        img = np.full((D.H, D.W, 3), 25)
    pt = m.pop("per_target")
    per = [{"id": i, "slot": int(sig[i]), "exist": float(ex[int(sig[i])]), "mask_iou": pt["mask_iou"][i], "iou_fr": pt["iou_fr"][i],
            "patches": int((am == int(sig[i])).sum()), "n_parts": sum(1 for s in sqs if s["t"] == i)} for i in range(fr["M"])]
    att = (attn[:, sig].T.cpu().numpy() * 255).round().astype(np.uint8)  # (M, N) attention of each target's slot
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"metrics": m, "sqs": sqs, "per_target": per, "pred_masks": png(pm), "pred_render": png(img),
                                "attn": [b64(a) for a in att]}, allow_nan=True).replace("NaN", "null"))
    return m


def curves(hist):
    """train.py history (flat) or exp 28's ({split: metrics}) -> {split: {metric: [[step, value]]}}."""
    out = {}
    for h in hist:
        splits = {k: v for k, v in h.items() if isinstance(v, dict)} or {"train": h}
        for s, mm in splits.items():
            for k in ("mask_miou", "pair_iou_fr", "part_iou_fr", "ari"):
                if k in mm and mm[k] == mm[k]:
                    out.setdefault(s, {}).setdefault(k, []).append([h["step"], mm[k]])
    return out


def copy_exp28(runs=GENERAL):
    for run in runs:
        (RES / "large" / run).mkdir(parents=True, exist_ok=True)
        shutil.copy(EXP28 / "large_files" / "runs" / run / "model.pt", RES / "large" / run / "model.pt")
        r = json.loads((EXP28 / "runs" / f"{run}.json").read_text())
        (RES / "runs").mkdir(parents=True, exist_ok=True)
        (RES / "runs" / f"{run}.json").write_text(json.dumps({"name": run, "args": r["args"], "history": r["history"],
                                                              "seconds": r["seconds"], "source": "exp 28"}, indent=1))


def run_args(r):
    a = r["args"]
    return types.SimpleNamespace(d=a.get("d", 128), K=a["K"], T=a.get("T", 3), depth=a.get("depth", True),
                                 decoder=a.get("decoder", "parts"), frame=a.get("frame", "camera"), yaw_aug=False,
                                 targets=a.get("targets", "visible"))


def frames_of(r):
    """[(frame id, split)] shown for a run."""
    a = r["args"]
    if r["name"] in SHOW:
        return SHOW[r["name"]]
    if "train_ids" not in a:  # overfit
        return [(a["frames"], "train")]
    tr = a["train_ids"]
    out = [(tr[int(i)], "train") for i in np.linspace(0, len(tr) - 1, 12).round()]
    for s in sorted({f.split("/")[0] for f in tr}):
        ids = D.scene_ids(s)
        out += [(f"{s}/{i}", "views") for i in ids[len(ids) - int(0.2 * len(ids)):]]
    for s in [x for x in a.get("eval_scenes", "").split(",") if x]:
        ids = D.scene_ids(s)
        out += [(f"{s}/{ids[int(i)]}", "scene") for i in np.linspace(0, len(ids) - 1, 20).round()]
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--copy_exp28", default="", help="comma list of exp 28 runs to copy (all: GENERAL)")
    ap.add_argument("--runs", default="", help="comma list (default: all available)")
    a = ap.parse_args()
    if a.copy_exp28:
        copy_exp28(list(GENERAL) if a.copy_exp28 == "all" else a.copy_exp28.split(","))
    index = {"overfit": [], "general": []}
    for kind, runs in (("overfit", OVERFIT), ("general", GENERAL)):
        for run, title in runs.items():
            ck, rj = RES / "large" / run / "model.pt", RES / "runs" / f"{run}.json"
            if not (ck.exists() and rj.exists()):
                print("skip", run)
                continue
            r = json.loads(rj.read_text())
            if a.runs and run not in a.runs.split(","):
                index[kind].append(json.loads((OUT / "runs" / run / "run.json").read_text())["summary"])
                continue
            A = run_args(r)
            model = T.build(A)
            model.load_state_dict(torch.load(ck, map_location=D.DEV))
            model.eval()
            fl = []
            for fid, split in frames_of(r):
                fr = D.load_frame(*fid.split("/"), A.targets)
                if fr["M"] == 0:
                    continue
                tag = fid.replace("/", "_")
                export_frame(fr, OUT / "frames" / f"{tag}_{A.targets}.json")
                m = export_pred(model, fr, A, OUT / "runs" / run / f"{tag}.json")
                fl.append({"frame": fid, "tag": tag, "split": split, "targets": A.targets,
                           **{k: m[k] for k in ("mask_miou", "pair_iou_fr", "part_iou_fr", "ari") if k in m}})
                print(run, fid, split, {k: round(v, 3) for k, v in m.items() if isinstance(v, float)}, flush=True)
            mean = {s: {k: float(np.nanmean([f[k] for f in fl if f["split"] == s and k in f]))
                        for k in ("mask_miou", "pair_iou_fr", "part_iou_fr")} for s in sorted({f["split"] for f in fl})}
            summary = {"name": run, "title": title, "kind": kind, "mean": mean}
            (OUT / "runs" / run).mkdir(parents=True, exist_ok=True)
            (OUT / "runs" / run / "run.json").write_text(json.dumps(
                {"summary": summary, "args": {k: v for k, v in r["args"].items() if k != "train_ids"},
                 "curves": curves(r["history"]), "frames": fl}, allow_nan=True).replace("NaN", "null"))
            index[kind].append(summary)
    (OUT / "index.json").write_text(json.dumps(index, allow_nan=True).replace("NaN", "null"))


if __name__ == "__main__":
    main()
