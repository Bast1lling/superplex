"""Export runs of this experiment to the snapshot format of experiment 06's viewer.

    python export_viewer.py [--exp NN] [run ...]      (default: all runs of experiment 08)
With --exp NN, runs of experiment NN (same sqs.npz format, e.g. 09 / 10) are exported into that experiment's own
results/viewer_runs (served by its viewer/server.py, a wrapper of ours).
Writes results/viewer_runs/<run>/{config.yaml, metrics.json, snapshots.npz} (one snapshot, in the frame of Sonata
token version 0 like 05/06). Serve with: python viewer/server.py  (wraps 06's server, port 8008).
06's viewer colors points by their index in its own instance list (all labeled instances of the 1 cm points), so
sq_label is remapped from this experiment's list (>= 20 points at 2 cm) into that index space.
"""

import json
import sys
from pathlib import Path

import numpy as np
import yaml

EXP = Path(__file__).resolve().parents[1]
ROOT = EXP.parents[1]
sys.path.insert(0, str(ROOT / "experiments" / "01_scene_viewer" / "code"))
from scenes import load_points, load_semantics  # noqa: E402


def viewer_instance_ids(scene):
    """Instance ids in the order of 06's loader (drop_unlabeled=True), i.e. the viewer's point labels."""
    inst = load_points(scene)["instance"].astype(np.int64)
    l2 = np.array(load_semantics(scene)["id_to_label"])
    return np.unique(inst[l2[np.minimum(inst, len(l2) - 1)] > 0])


def export(name, exp=EXP):
    run = exp / "results" / "runs" / name
    m = json.loads((run / "metrics.json").read_text())
    cfg = json.loads((run / "config.json").read_text()) if (run / "config.json").exists() else {"scene": "room_0"}
    s = np.load(run / "sqs.npz")
    tok = np.load(ROOT / "data" / "replica_v1_sonata" / "stage4" / f"{cfg['scene']}.npz")
    R, t = tok["R"][0], tok["t"][0]
    v06 = yaml.safe_load((ROOT / "experiments" / "06_sq_gt_instance_fit" / "config.yaml").read_text())
    v06["data"]["scene"] = cfg["scene"]
    v06["data"]["version"] = 0
    if "planar" in cfg:  # experiments 09 / 10: key names of the voxel layer (viewer/server.py)
        cfg = cfg | {"fill_planar": cfg["planar"], "planar_thick": 0.2, "planar_min": 0.5}
    v06["exp08"] = cfg
    n = len(s["instance"])
    ids, lab, iou = s["instance_ids"], s["instance"], np.array(m["per_instance"]["viou"])
    if cfg.get("labels", "instance") == "instance":  # semantic-label runs have component ids, colors stay arbitrary
        v_ids = viewer_instance_ids(cfg["scene"])
        assert np.isin(ids, v_ids).all(), f"{name}: instances missing from the viewer's point labels"
        pos = np.searchsorted(v_ids, ids)
        lab, ids = pos[lab], v_ids
        iou = np.zeros(len(v_ids)); iou[pos] = m["per_instance"]["viou"]  # 0 for instances this experiment skipped
    o = exp / "results" / "viewer_runs" / name
    pur = s["purity"] if "purity" in s else np.ones(n)  # 09: share of the SQ's voxels in its assigned instance
    o.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(o / "snapshots.npz", step=np.array([0]), scale=s["scale"][None], eps=s["eps"][None],
                        rot=np.einsum("ij,mjk->mik", R, s["rot"])[None], trans=(s["trans"] @ R.T + t)[None],
                        existence=np.ones((1, n)), sq_label=lab[None], purity=pur[None],
                        active=np.ones((1, n), bool), owned=np.zeros((1, n), int), instance_ids=ids, R=R, t=t)
    (o / "config.yaml").write_text(yaml.safe_dump(v06))
    summ = m["summary"]
    final = {k: v for k, v in summ.items() if not isinstance(v, list)}
    final["eval/mean_iou"] = summ["viou"]  # 06's run list labels runs with this key (here: mean instance vIoU)
    (o / "metrics.json").write_text(json.dumps({"name": name, "seconds": 0, "final": final, "history": [{"step": 0, **final}],
                                                "final_iou": iou.tolist()}))


if __name__ == "__main__":
    args = sys.argv[1:]
    exp = EXP
    if args[:1] == ["--exp"]:
        exp, args = next(ROOT.glob(f"experiments/{args[1]}_*")), args[2:]
    names = args or [p.name for p in (exp / "results" / "runs").iterdir() if (p / "sqs.npz").exists()]
    for nm in names:
        export(nm, exp)
        print("exported", nm)
