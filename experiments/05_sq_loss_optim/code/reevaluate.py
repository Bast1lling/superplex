"""Re-evaluate the final snapshot of existing runs with the current metrics (updates metrics.json["final_reeval"]).

    uv run python experiments/05_sq_loss_optim/code/reevaluate.py run1 run2 ...
"""

import json
import sys

import numpy as np
import torch
import yaml

from optimize import DEVICE, RESULTS, DistanceGrid, evaluate, load_scene

for name in sys.argv[1:]:
    d = RESULTS / "runs" / name
    cfg = yaml.safe_load((d / "config.yaml").read_text())
    snap = d / "snapshots.npz"
    if not snap.exists():
        snap = RESULTS / "large_files" / "runs" / name / "snapshots.npz"
    if not snap.exists():
        print(f"skip {name}: snapshots.npz not available")
        continue
    z = np.load(snap)
    sq = {k: torch.from_numpy(z[k][-1]).float().to(DEVICE) for k in ("scale", "eps", "rot", "trans", "existence")}
    scene = load_scene(cfg)
    rc = cfg["loss"].get("reverse_chamfer", {"grid_res": 0.02, "grid_pad": 0.5})
    ev = evaluate(cfg, sq, scene, DistanceGrid(scene["pos"], rc.get("grid_res", 0.02), rc.get("grid_pad", 0.5)))
    m = json.loads((d / "metrics.json").read_text())
    m["final_reeval"] = ev["metrics"]
    (d / "metrics.json").write_text(json.dumps(m, indent=1))
    f = ev["metrics"]
    print(f"{name:34s} mIoU {f['eval/mean_iou']:.3f}  enclosed_correct {f['eval/enclosed_correct']:.3f} "
          f"(per instance {f['eval/enclosed_correct_inst']:.3f})  outside_owner {f['eval/outside_owner']:.3f}  "
          f"in_air {f['eval/surface_in_air']:.3f}  swallowed {f['eval/swallowed']:.3f}  active {f['eval/active_sqs']}", flush=True)
