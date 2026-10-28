"""Re-score a run's SQs with the default metric weights (fit.CFG: w_free 1, w_other 0.3, w_unknown 0.1), so runs
fitted with other loss weights are comparable. Writes results/runs/<run>/metrics_std.json.

    python reeval.py <run> [<run> ...]
"""

import json
import sys

import numpy as np
import torch

from evaluate import metrics
from fit import CFG, RUNS, cached_volume, evaluate, prepare
from sq import SQSet, matrix_to_quat
from volume import load_scene


def reeval(name):
    run = RUNS / name
    rcfg = json.loads((run / "config.json").read_text())
    cfg = dict(CFG) | {k: rcfg[k] for k in ("scene", "labels", "res", "min_points") if k in rcfg}
    sc = load_scene(cfg["scene"], min_points=cfg["min_points"], labels=cfg["labels"])
    vol = cached_volume(cfg, sc, "clean")
    D = prepare(cfg, sc, vol)
    s = dict(np.load(run / "sqs.npz"))
    I = D["I"]
    K = int(np.bincount(s["instance"], minlength=I).max())
    model = SQSet(I, K, eps_range=(0.1, 1.9))
    slot = np.zeros(I, int)
    with torch.no_grad():
        for m, i in enumerate(s["instance"]):
            j = slot[i]
            model.trans[i, j] = torch.tensor(s["trans"][m])
            model.log_scale[i, j] = torch.tensor(s["scale"][m]).log()
            e = (torch.tensor(s["eps"][m]) - model.eps_min) / (model.eps_max - model.eps_min)
            model.eps_raw[i, j] = torch.logit(e.clamp(1e-5, 1 - 1e-5))
            model.quat[i, j] = matrix_to_quat(torch.tensor(s["rot"][m])[None])[0]
            model.active[i, j] = True
            slot[i] += 1
    ev = evaluate(model, D)
    M = metrics(cfg, sc, vol, D, s, ev)
    (run / "metrics_std.json").write_text(json.dumps({"name": name, "summary": M["summary"],
                                                      "per_instance": M["per_instance"]}, indent=1))
    print(name, {k: round(v, 3) for k, v in M["summary"].items() if isinstance(v, float)})


if __name__ == "__main__":
    for n in sys.argv[1:]:
        reeval(n)
