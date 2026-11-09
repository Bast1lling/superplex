"""Where does a trained --topk model fail? Error decomposition on the test frames vs training frames.

    uv run python diagnose.py <run> [--inputs dino]   -> results/diagnose_<run>.json (+ printed table)

Per visible target (`dense.visible_targets`), on every 2nd test frame and 100 training frames:
- decoder, Hungarian matched (eval matching): pair IoU as predicted; with the target's translation (prediction's
  scale / shape / rotation kept: "shape" error); with the target's scale + shape + rotation at the predicted
  translation ("position" error); translation error in m and relative to the target's largest half-axis.
- proposals: the best IoU of the proposals of the target's own patches (owner == target: oracle selection, pure
  regression quality) and whether any of its patches is among the decoder's selected 300 (selection recall).
- existence: the proposal existence at the target's center patch (heat == 1) vs the 300th-best peak.
- class: accuracy of matched queries; share of test targets whose class never appears in the training frames.
"""

import json
import sys

import numpy as np
import torch

import dense as D
import model as M
import train as T
from model import TopKModel


@torch.no_grad()
def run(model, loader):
    rows = []
    for batch in loader:
        batch = T.to_device(batch)
        pred = model(batch["dino"], batch["feats"], batch["idx"], batch["depth"])
        enc = pred["enc"]
        assigned = [D.assign(batch, b, t) for b, t in enumerate(batch["targets"])]
        vb = D.visible_targets(batch, assigned)
        for b, t in enumerate(vb["targets"]):
            if len(t["scale"]) == 0:
                continue
            owner = assigned[b][0]
            G = T.sq_dict(t["scale"], t["exponents"], t["rotation"], t["translation"])
            qi, ti = T.match(pred, b, t, vb)
            qi, ti = torch.as_tensor(qi, device=T.DEV), torch.as_tensor(ti, device=T.DEV)
            P = T.sq_dict(pred["scale"][b, qi], pred["shape"][b, qi], pred["rotate"][b, qi], pred["trans"][b, qi])
            g = {k: v[ti] for k, v in G.items()}
            iou = T.pair_iou(P, g)
            iou_t = T.pair_iou(P | {"trans": g["trans"]}, g)
            iou_s = T.pair_iou(g | {"trans": P["trans"]}, g)
            dt = (P["trans"] - g["trans"]).norm(dim=1)
            # proposals of each target's own patches (slot = index among all frame targets = owner's index)
            sel = torch.zeros(T.GH * T.GW, dtype=torch.bool, device=T.DEV)
            sel[pred["select"][b]] = True
            heat = batch["heat"][b]
            kth = torch.sort(enc["exist"][b][pred["select"][b]]).values[0]
            best_prop, in_sel, ex_c = [], [], []
            for j, slot in enumerate(t["slot"].tolist()):
                pt = torch.nonzero(owner == slot)[:, 0]
                if len(pt) == 0:
                    best_prop.append(np.nan), in_sel.append(np.nan), ex_c.append(np.nan)
                    continue
                Pp = T.sq_dict(enc["scale"][b, pt], enc["shape"][b, pt], enc["rotate"][b, pt], enc["trans"][b, pt])
                gj = {k: v[j:j + 1].expand(len(pt), *v.shape[1:]) for k, v in G.items()}
                best_prop.append(float(T.pair_iou(Pp, gj).max()))
                in_sel.append(float(sel[pt].any()))
                c = pt[heat[pt].argmax()]
                ex_c.append(float(enc["exist"][b, c] >= kth))
            for k_, j in enumerate(ti.tolist()):
                rows.append({"iou": float(iou[k_]), "iou_gt_trans": float(iou_t[k_]), "iou_gt_shape": float(iou_s[k_]),
                             "dt": float(dt[k_]), "dt_rel": float(dt[k_] / g["scale"][k_].max()),
                             "cls_ok": float(pred["cls"][b, qi[k_]].argmax() == t["cls"][j]),
                             "f": float(t["f"][j]), "cls": int(t["cls"][j]), "patches": int((owner == t["slot"][j]).sum()),
                             "best_proposal": best_prop[j], "in_select": in_sel[j], "center_above_kth": ex_c[j]})
    return rows


def summary(rows, train_classes=None):
    a = {k: np.array([r[k] for r in rows], float) for k in rows[0]}
    s = {k: float(np.nanmean(a[k])) for k in ("iou", "iou_gt_trans", "iou_gt_shape", "dt_rel", "cls_ok", "best_proposal",
                                                "in_select", "center_above_kth")}
    s["dt_median_m"] = float(np.median(a["dt"]))
    s["proposal_recall@0.5_oracle"] = float(np.nanmean(a["best_proposal"] >= 0.5))
    for lo, hi in ((1, 2), (3, 16), (17, 10 ** 6)):
        m = (a["patches"] >= lo) & (a["patches"] <= hi)
        s[f"iou_patches_{lo}-{hi}"] = float(a["iou"][m].mean()) if m.any() else None
    for lo, hi in ((0, 0.25), (0.25, 0.5), (0.5, 0.75), (0.75, 1.01)):
        m = (a["f"] >= lo) & (a["f"] < hi)
        if m.any():
            s[f"f_{lo}-{min(hi, 1)}: share / iou / iou_gt_trans / dt_rel / in_select"] = [
                round(float(m.mean()), 3), round(float(a["iou"][m].mean()), 3), round(float(a["iou_gt_trans"][m].mean()), 3),
                round(float(np.median(a["dt_rel"][m])), 3), round(float(np.nanmean(a["in_select"][m])), 3)]
    for lo, hi in ((1, 2), (3, 16), (17, 10 ** 6)):
        m = (a["patches"] >= lo) & (a["patches"] <= hi)
        s[f"patches_{lo}-{hi}: share / in_select / best_proposal"] = [
            round(float(m.mean()), 3), round(float(np.nanmean(a["in_select"][m])), 3), round(float(np.nanmean(a["best_proposal"][m])), 3)]
    if train_classes is not None:
        unseen = ~np.isin(a["cls"], list(train_classes))
        s["unseen_class_share"] = float(unseen.mean())
        s["iou_unseen_class"] = float(a["iou"][unseen].mean()) if unseen.any() else None
    return s


def main():
    name = sys.argv[1]
    M.INPUTS = "dino" if "dino" in sys.argv else "all"
    T.FOCAL, T.COST_SAMPLES = True, 4
    classes = json.loads((T.FRAMES / "classes.json").read_text())
    args = json.loads((T.EXP / "results" / f"{name}.json").read_text())["args"]
    model = TopKModel(len(classes), d=args["width"], layers=args["layers"], dec_layers=args["dec_layers"],
                      k=args["select"]).to(T.DEV).eval()
    model.load_state_dict(torch.load(T.OUT / name / "model.pt", map_location=T.DEV))
    kw = dict(batch_size=8, collate_fn=T.collate, num_workers=4)
    test = T.Frames(T.TEST)
    test = torch.utils.data.Subset(test, range(0, len(test), 2))
    train = T.Frames(T.TRAIN, limit=100)
    r_tr = run(model, torch.utils.data.DataLoader(train, **kw))
    r_te = run(model, torch.utils.data.DataLoader(test, **kw))
    train_cls = {r["cls"] for r in r_tr} | {int(c) for p in T.Frames(T.TRAIN).items for c in np.load(p)["cls"]}
    out = {"train": summary(r_tr), "test": summary(r_te, train_cls)}
    (T.EXP / "results" / f"diagnose_{name}.json").write_text(json.dumps(out, indent=1))
    for k in out["test"]:
        print(f"{k:60s} train {out['train'].get(k, '-')!s:32.32} test {out['test'][k]!s:32.32}")


if __name__ == "__main__":
    main()
