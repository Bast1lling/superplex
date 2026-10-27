"""Score SQ sets from other experiments (06 snapshots) with the metrics of this experiment.

    python eval_external.py <snapshots.npz> <run name> [--sqs-per-instance K]

06 snapshots are in the frame of the Sonata token version (pos_v = pos @ R.T + t); converted back here.
Instances of 06 without points >= 20 / unlabeled are dropped consistently with fit.py (min_points = 20).
"""

import argparse
import json

import numpy as np
import torch

from evaluate import metrics, render_all
from fit import CFG, RUNS, prepare, evaluate
from sq import SQSet, matrix_to_quat
from volume import load_scene


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("snapshots")
    ap.add_argument("name")
    ap.add_argument("--sqs-per-instance", type=int, default=6)
    args = ap.parse_args()
    cfg = dict(CFG)
    sc = load_scene(cfg["scene"], min_points=cfg["min_points"])
    vol = dict(np.load(f"/tmp/claude-0/volcache/{cfg['scene']}_{cfg['res']}_{cfg['min_points']}_instance_v2.npz"))
    vol["res"] = float(vol["res"])
    s = np.load(args.snapshots)
    R, t = s["R"], s["t"]
    last = -1
    K6 = args.sqs_per_instance
    act = s["active"][last].astype(bool) if "active" in s.files else np.ones(s["scale"].shape[1], bool)
    if "existence" in s.files:
        act &= s["existence"][last] > 0.5
    ids6 = s["instance_ids"]
    inst_id = ids6[np.arange(len(act)) // K6]
    id2k = {v: k for k, v in enumerate(sc["instance_ids"])}
    keep = act & np.array([i in id2k for i in inst_id])
    sqs = {"trans": (s["trans"][last][keep] - t) @ R, "rot": np.einsum("ij,mjk->mik", R.T, s["rot"][last][keep]),
           "scale": s["scale"][last][keep], "eps": s["eps"][last][keep],
           "instance": np.array([id2k[i] for i in inst_id[keep]])}
    D = prepare(cfg, sc, vol)
    I = D["I"]
    K = int(np.bincount(sqs["instance"], minlength=I).max())
    model = SQSet(I, K)
    slot = np.zeros(I, int)
    with torch.no_grad():
        for m in range(len(sqs["instance"])):
            i = sqs["instance"][m]
            model.trans[i, slot[i]] = torch.tensor(sqs["trans"][m]).float()
            model.log_scale[i, slot[i]] = torch.tensor(sqs["scale"][m]).float().clamp(0.01, 10).log()
            e = (torch.tensor(sqs["eps"][m]).float() - model.eps_min) / (model.eps_max - model.eps_min)
            model.eps_raw[i, slot[i]] = torch.logit(e.clamp(1e-4, 1 - 1e-4))
            model.quat[i, slot[i]] = matrix_to_quat(torch.tensor(sqs["rot"][m]).float()[None])[0]
            model.active[i, slot[i]] = True
            slot[i] += 1
    ev = evaluate(model, D)
    M = metrics(cfg, sc, vol, D, sqs, ev)
    out = RUNS / args.name
    out.mkdir(parents=True, exist_ok=True)
    (out / "metrics.json").write_text(json.dumps({"name": args.name, "source": args.snapshots,
                                                  "summary": M["summary"], "per_instance": M["per_instance"]}, indent=1))
    render_all(sc, sqs, out)
    print(json.dumps({k: v for k, v in M["summary"].items()}, indent=None))


if __name__ == "__main__":
    main()
