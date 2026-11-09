"""Exports a training run's test-frame predictions for frames.html into viewer/data/frames/<run>/ (not versioned).

    uv run python export_frames.py base
    python -m http.server -d experiments/22_rgbd_prediction/viewer 8022      -> http://localhost:8022/frames.html

Per frame (camera frame: x right, y down, z forward): the /4 RGB (png), its depth points with their color, the
predicted SQs (exist > 0.5) and the target SQs as meshes (vertices f32, faces u32, SQ index u16 per vertex), per SQ its
instance colour key (target: its instance; prediction: the instance of its matched target, -1 unmatched) and class,
and the instance masks at patch level (predicted: `train.group_instances`; GT: instance share > 0.5).
"""

import json
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "code"))
from model import CX, CY, FX, FY  # noqa: E402
from train import FRAMES, OUT, group_instances, instance_masks  # noqa: E402

RES = 12


def fexp(x, p):
    return np.sign(x) * np.abs(x) ** p


def sq_mesh(scale, shape, rot, trans):
    eta = np.linspace(-np.pi / 2, np.pi / 2, RES)
    omega = np.linspace(-np.pi, np.pi, 2 * RES, endpoint=False)
    e, o = np.meshgrid(eta, omega, indexing="ij")
    p = np.stack([scale[0] * fexp(np.cos(e), shape[0]) * fexp(np.cos(o), shape[1]),
                  scale[1] * fexp(np.cos(e), shape[0]) * fexp(np.sin(o), shape[1]),
                  scale[2] * fexp(np.sin(e), shape[0])], -1).reshape(-1, 3) @ rot.T + trans
    i, j = np.meshgrid(np.arange(RES - 1), np.arange(2 * RES), indexing="ij")
    a, b = i * 2 * RES + j, i * 2 * RES + (j + 1) % (2 * RES)
    c, d = a + 2 * RES, b + 2 * RES
    faces = np.concatenate([np.stack([a, c, b], -1).reshape(-1, 3), np.stack([b, c, d], -1).reshape(-1, 3)])
    return p, faces


def meshes(scale, shape, rot, trans):
    V, F, S, n = [], [], [], 0
    for s in range(len(scale)):
        v, f = sq_mesh(scale[s], shape[s], rot[s], trans[s])
        V.append(v), F.append(f + n), S.append(np.full(len(v), s))
        n += len(v)
    if not V:
        return b"", [0, 0]
    V, F = np.concatenate(V), np.concatenate(F)
    return V.astype("<f4").tobytes() + F.astype("<u4").tobytes() + np.concatenate(S).astype("<u2").tobytes(), [len(V), len(F)]


def main():
    run = sys.argv[1]
    res = json.loads((HERE.parent / "results" / f"{run}.json").read_text())
    classes = json.loads((FRAMES / "classes.json").read_text())
    out = HERE / "data" / "frames" / run
    out.mkdir(parents=True, exist_ok=True)
    index = []
    for row in res["per_frame"]:
        name = row["name"]
        scene, frame = name.rsplit("_", 1)
        v = np.load(FRAMES / scene / f"{frame}_view.npz")
        p = np.load(OUT / run / "predictions" / f"{name}.npz")
        Image.fromarray(v["rgb"]).save(out / f"{name}.png")
        depth = v["depth"].astype(np.float32)
        vv, uu = np.indices(depth.shape)
        ok = depth > 0
        xyz = np.stack([(uu - CX) / FX * depth, (vv - CY) / FY * depth, depth], -1)[ok]
        keep = p["exist"] > 0.5
        inst_of_query = np.full(len(p["exist"]), -1)
        inst_of_query[p["match_q"]] = v["instance"][p["match_t"]]
        pb, pc = meshes(p["scale"][keep], p["shape"][keep], p["rotate"][keep], p["trans"][keep])
        tb, tc = meshes(v["scale"], v["exponents"], v["rotation"], v["translation"])
        groups = group_instances(torch.tensor(p["mask"].astype(np.float32)), torch.tensor(p["exist"]))
        pm = np.full((42, 75), -1, np.int16)
        for g, m in enumerate(groups):
            pm[m.numpy() & (pm < 0)] = g
        lab = torch.tensor(v["label"].astype(np.int64))
        ids = torch.unique(lab[lab >= 0])
        gm = np.full((42, 75), -1, np.int16)
        if len(ids):
            share = instance_masks(lab, ids).numpy()
            best = share.argmax(0)
            gm = np.where(share.max(0) > 0.5, ids.numpy()[best], -1).astype(np.int16)
        (out / f"{name}.bin").write_bytes(xyz.astype("<f4").tobytes() + v["rgb"][ok].astype("u1").tobytes() + pb + tb
                                          + pm.tobytes() + gm.tobytes())
        meta = {k: row.get(k) for k in ("cd", "occ_iou", "mask_cov", "targets", "predicted", "found", "cls_acc")}
        meta |= {"points": int(ok.sum()), "pred_mesh": pc, "target_mesh": tc,
                 "pred_inst": inst_of_query[keep].tolist(), "pred_cls": [classes[c] for c in p["cls"][keep]],
                 "pred_exist": np.round(p["exist"][keep], 2).tolist(),
                 "target_inst": v["instance"].tolist(), "target_cls": [classes[c] for c in v["cls"]]}
        (out / f"{name}.json").write_text(json.dumps(meta))
        index.append({"name": name, "occ_iou": row.get("occ_iou"), "cd": row.get("cd"), "mask_cov": row.get("mask_cov")})
    (out / "index.json").write_text(json.dumps({"run": run, "test": res["test"], "frames": index}))
    runs = sorted(d.name for d in (HERE / "data" / "frames").iterdir() if (d / "index.json").exists())
    (HERE / "data" / "frames" / "runs.json").write_text(json.dumps(runs))
    print(run, len(index), "frames")


if __name__ == "__main__":
    main()
