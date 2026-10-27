"""Export runs of this experiment to the snapshot format of experiment 06's viewer.

    python export_viewer.py [run ...]      (default: all runs)
Writes results/viewer_runs/<run>/{config.yaml, metrics.json, snapshots.npz} (one snapshot, in the frame of Sonata
token version 0 like 05/06). Serve with: python viewer/server.py  (wraps 06's server, port 8008).
"""

import json
import sys
from pathlib import Path

import numpy as np
import yaml

EXP = Path(__file__).resolve().parents[1]
ROOT = EXP.parents[1]
OUT = EXP / "results" / "viewer_runs"


def export(name):
    run = EXP / "results" / "runs" / name
    m = json.loads((run / "metrics.json").read_text())
    cfg = json.loads((run / "config.json").read_text()) if (run / "config.json").exists() else {"scene": "room_0"}
    s = np.load(run / "sqs.npz")
    tok = np.load(ROOT / "data" / "replica_v1_sonata" / "stage4" / f"{cfg['scene']}.npz")
    R, t = tok["R"][0], tok["t"][0]
    v06 = yaml.safe_load((ROOT / "experiments" / "06_sq_gt_instance_fit" / "config.yaml").read_text())
    v06["data"]["scene"] = cfg["scene"]
    v06["data"]["version"] = 0
    v06["exp08"] = cfg
    n = len(s["instance"])
    ids = s["instance_ids"]
    o = OUT / name
    o.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(o / "snapshots.npz", step=np.array([0]), scale=s["scale"][None], eps=s["eps"][None],
                        rot=np.einsum("ij,mjk->mik", R, s["rot"])[None], trans=(s["trans"] @ R.T + t)[None],
                        existence=np.ones((1, n)), sq_label=s["instance"][None], purity=np.ones((1, n)),
                        active=np.ones((1, n), bool), owned=np.zeros((1, n), int), instance_ids=ids, R=R, t=t)
    (o / "config.yaml").write_text(yaml.safe_dump(v06))
    summ = m["summary"]
    final = {k: v for k, v in summ.items() if not isinstance(v, list)}
    (o / "metrics.json").write_text(json.dumps({"name": name, "seconds": 0, "final": final, "history": [final],
                                                "final_iou": m["per_instance"]["viou"]}))


if __name__ == "__main__":
    names = sys.argv[1:] or [p.name for p in (EXP / "results" / "runs").iterdir() if (p / "sqs.npz").exists()]
    for nm in names:
        export(nm)
        print("exported", nm)
