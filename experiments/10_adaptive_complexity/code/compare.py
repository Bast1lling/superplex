"""Per-instance comparison of runs (K / strict vIoU), grouped by class, plus summaries.

    python compare.py <run> [<run> ...]      run = name in results/runs, or <exp>:<name> (e.g. 08:office0_v9_exterior)
Runs of other experiments are re-scored with this experiment's scorer (decomp.score) so all numbers are comparable.
"""

import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

import decomp
import partfit as pf

ROOT = Path(__file__).resolve().parents[2]
EXPS = {p.name[:2]: p for p in ROOT.glob("[0-9][0-9]_*")}


def load(run):
    exp, name = run.split(":") if ":" in run else ("10", run)
    d = EXPS[exp] / "results" / "runs" / name
    m = json.loads((d / "metrics.json").read_text())
    if exp in ("09", "10"):
        return m["summary"], m["per_instance"]
    cfg = json.loads((d / "config.json").read_text())
    sqs = dict(np.load(d / "sqs.npz"))
    S = pf.Scene(cfg["scene"], planar=False)
    return decomp.summarize(S, sqs, decomp.score(cfg["scene"], sqs, S.n_inst))


def main(runs):
    res = [load(r) for r in runs]
    print("summary")
    for r, (s, _) in zip(runs, res):
        print(f"  {r:40s} sqs {s['sqs']:5d}  vIoU {s['viou']:.3f}  prec {s['precision']:.3f}  rec {s['recall']:.3f}  "
              f"plausible {s['plausible']:.2f} (over {s['over']:.2f})")
    by = defaultdict(lambda: [[] for _ in runs])
    for k, (_, p) in enumerate(res):
        for n, K, v in zip(p["name"], p["K"], p["viou"]):
            by[n][k].append((K, v))
    print("\nper class: mean K / mean vIoU (n instances)")
    for n in sorted(by, key=lambda n: -max(np.mean([x[0] for x in by[n][k]]) for k in range(len(runs)))):
        cols = [f"{np.mean([x[0] for x in c]):5.1f} / {np.mean([x[1] for x in c]):.2f}" for c in by[n][:]]
        print(f"  {n:16s} ({len(by[n][0]):2d})  " + "   ".join(cols))


if __name__ == "__main__":
    main(sys.argv[1:])
