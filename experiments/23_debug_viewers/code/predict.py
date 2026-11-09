"""Runs exp 22's trained dense / top-k checkpoints on a fixed set of frames and exports everything the results viewer
(`viewer/results.html`) shows.

    uv run python experiments/23_debug_viewers/code/predict.py [--runs of_topk ...]

Frames: the single overfit frame (room_2_000500), the 16 frames of the --frames 16 runs (training scenes) and every
10th frame of room_0 (test, 20 frames). Every run is evaluated on all of them; the viewer marks which ones it was
trained on.
Output, results/large_files/ (not versioned):
- gt/<frame>.json: the cached targets (camera frame) with `visible`, the patch owner (`dense.assign`) and the heatmap
  target (`dense.heatmap`), both 42 x 75;
- predictions/<run>/<frame>.json: the existence map (42 x 75), the 300 proposals (top 300 of the 3 x 3 local
  maxima: patch, peak flag, SQ, existence, class, best visible target and its pair IoU) and, for top-k, the decoder's
  300 SQs (same fields, plus the Hungarian match of `train.frame_metrics`); the frame's metrics.
results/predict_summary.json: the metrics per run and frame, and their means over seen / unseen frames.
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

EXP = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(EXP.parent / "22_rgbd_prediction" / "code"))
import train as T  # noqa: E402
import dense as D  # noqa: E402
from model import DenseModel, TopKModel, peaks  # noqa: E402

OUT = EXP / "results" / "large_files"
RUNS = {"of_dense_heat": "dense", "of_topk": "topk", "f16_dense": "dense", "f16_topk": "topk",
        "full_dense": "dense", "full_topk": "topk"}
OVERFIT = "room_2_000500"
SQ = ("scale", "shape", "rotate", "trans")


def frame_sets():
    """-> {frame name: path}, and the frame names each training regime saw."""
    f16 = T.Frames(T.TRAIN, limit=16).items
    test = T.Frames(T.TEST).items[::10]
    over = [p for p in T.Frames(T.TRAIN).items if f"{p.parent.name}_{p.name[:6]}" == OVERFIT]
    name = lambda p: f"{p.parent.name}_{p.name[:6]}"
    items = {name(p): p for p in over + f16 + test}
    seen = {"of": {OVERFIT}, "f16": {name(p) for p in f16},
            "full": {n for n in items if not n.startswith(T.TEST)}}
    return items, seen


def sq_rows(pred, b, idx, iou_best, iou_arg, vis_idx, classes):
    """List of dicts (one per query in idx): SQ parameters, existence, class, best visible target + IoU."""
    rows = []
    for j, q in enumerate(idx.tolist()):
        rows.append({"q": q, "scale": pred["scale"][b, q].tolist(), "shape": pred["shape"][b, q].tolist(),
                     "rot": pred["rotate"][b, q].flatten().tolist(), "trans": pred["trans"][b, q].tolist(),
                     "exist": float(pred["exist"][b, q]), "cls": int(pred["cls"][b, q].argmax()),
                     "best_t": int(vis_idx[iou_arg[j]]) if len(vis_idx) else -1,
                     "best_iou": float(iou_best[j]) if len(vis_idx) else 0.0})
    return rows


def round_floats(x, n=4):
    if isinstance(x, float):
        return round(x, n)
    if isinstance(x, list):
        return [round_floats(v, n) for v in x]
    if isinstance(x, dict):
        return {k: round_floats(v, n) for k, v in x.items()}
    return x


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="+", default=list(RUNS))
    a = ap.parse_args()
    T.FOCAL, T.COST_SAMPLES, T.LOSS, T.MASK = True, 4, "cd", "instance"
    torch.manual_seed(0)
    classes = json.loads((T.FRAMES / "classes.json").read_text())
    items, seen = frame_sets()
    ds = T.Frames(T.TRAIN)
    ds.items = list(items.values())
    batches = [T.collate([ds[i] for i in range(s, min(s + 8, len(ds)))]) for s in range(0, len(ds), 8)]
    (OUT / "gt").mkdir(parents=True, exist_ok=True)
    summary = json.loads((EXP / "results" / "predict_summary.json").read_text()) if (EXP / "results" / "predict_summary.json").exists() else {}
    gt_done = False
    for run in a.runs:
        kind = RUNS[run]
        ckpt = T.OUT / run / "model.pt"
        model = (TopKModel(len(classes)) if kind == "topk" else DenseModel(len(classes))).to(T.DEV)
        model.load_state_dict(torch.load(ckpt, map_location=T.DEV))
        model.eval()
        regime = run.split("_")[0]
        (OUT / "predictions" / run).mkdir(parents=True, exist_ok=True)
        rows = {}
        with torch.no_grad():
            for batch in batches:
                batch = T.to_device(batch)
                pred = model(batch["dino"], batch["feats"], batch["idx"], batch["depth"])
                enc = pred["enc"] if kind == "topk" else pred
                assigned = [D.assign(batch, b, t) for b, t in enumerate(batch["targets"])]
                vis_batch = D.visible_targets(batch, assigned)
                for b, t in enumerate(batch["targets"]):
                    name = batch["name"][b]
                    owner, visible = assigned[b]
                    vi = torch.nonzero(visible)[:, 0]
                    tq = T.sq_dict(t["scale"][vi], t["exponents"][vi], t["rotation"][vi], t["translation"][vi])
                    if not gt_done:
                        g = {"name": name, "scale": t["scale"].tolist(), "shape": t["exponents"].tolist(),
                             "rot": t["rotation"].reshape(-1, 9).tolist(), "trans": t["translation"].tolist(),
                             "instance": t["instance"].tolist(), "cls": t["cls"].tolist(), "visible": visible.tolist(),
                             "owner": owner.tolist(), "heat": D.heatmap(owner).tolist()}
                        (OUT / "gt" / f"{name}.json").write_text(json.dumps(round_floats(g)))
                    m, _, _ = D.frame_metrics(enc, batch, b, t, owner, visible)
                    pk = peaks(enc["exist"][b][None])[0]
                    top = pk.topk(D.K).indices
                    iou = D.pair_iou_matrix({k: enc[k][b, top] for k in SQ}, tq) if len(vi) else None
                    best, arg = (iou.max(1) if iou is not None else (None, None))
                    props = sq_rows(enc, b, top, best, arg, vi.tolist(), classes)
                    for r in props:
                        r["peak"] = bool(pk[r["q"]] >= 0)
                    out = {"name": name, "kind": kind, "seen": name in seen[regime], "exist": enc["exist"][b].tolist(),
                           "proposals": props}
                    if kind == "topk":
                        r_dec, qi, ti = T.frame_metrics(pred, vis_batch, b, vis_batch["targets"][b])
                        r_dec.pop("name")
                        m = r_dec | {f"enc_{k}": v for k, v in m.items() if k not in ("targets", "visible")} | {"visible": m["visible"]}
                        allq = torch.arange(pred["exist"].shape[1], device=T.DEV)
                        iou_d = D.pair_iou_matrix({k: pred[k][b] for k in SQ}, tq) if len(vi) else None
                        bd, ad = (iou_d.max(1) if iou_d is not None else (None, None))
                        dec = sq_rows(pred, b, allq, bd, ad, vi.tolist(), classes)
                        match = dict(zip(np.asarray(qi).tolist(), np.asarray(ti).tolist()))
                        sel = pred["select"][b].tolist()
                        for r in dec:
                            r["patch"] = sel[r["q"]]
                            if r["q"] in match:
                                tl = match[r["q"]]
                                r["match_t"] = int(vi[tl])
                                r["match_iou"] = float(iou_d[r["q"], tl])
                        out["decoder"] = dec
                    out["metrics"] = m
                    rows[name] = m | {"seen": out["seen"]}
                    (OUT / "predictions" / run / f"{name}.json").write_text(json.dumps(round_floats(out)))
        gt_done = True
        mean = lambda fl: {k: float(np.mean([r[k] for r in fl if k in r])) for k in fl[0] if k != "seen"} if fl else {}
        summary[run] = {"kind": kind, "seen": mean([r for r in rows.values() if r["seen"]]),
                        "unseen": mean([r for r in rows.values() if not r["seen"]]), "frames": rows}
        print(run, "seen", {k: round(v, 3) for k, v in summary[run]["seen"].items()}, flush=True)
        print(run, "unseen", {k: round(v, 3) for k, v in summary[run]["unseen"].items()}, flush=True)
    (EXP / "results" / "predict_summary.json").write_text(json.dumps(round_floats(summary), indent=1))
    (EXP / "results" / "frames.json").write_text(json.dumps({"overfit": OVERFIT, "seen": {k: sorted(v) for k, v in seen.items()},
                                                            "frames": list(items)}, indent=1))


if __name__ == "__main__":
    main()
