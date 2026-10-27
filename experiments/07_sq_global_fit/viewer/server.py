"""Viewer for optimized SQ sets (experiment 07).

Run from the repo root:
    uv run python experiments/07_sq_global_fit/viewer/server.py
then open http://localhost:8007
"""

from functools import lru_cache
import json
from pathlib import Path
import sys

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, Response
import numpy as np
import uvicorn
import yaml

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE.parent / "code"))
from global_fit import RESULTS, load_points, load_scene, load_semantics  # noqa: E402

RUNS = RESULTS / "runs"
LARGE_RUNS = RESULTS / "large_files" / "runs"  # >1MB snapshots live here (not versioned, may be missing)
app = FastAPI()


def snapshot_path(name: str) -> Path | None:
    for p in (RUNS / name / "snapshots.npz", LARGE_RUNS / name / "snapshots.npz"):
        if p.exists():
            return p
    return None


def run_dir(name: str) -> Path:
    if snapshot_path(name) is None:
        raise HTTPException(404, f"unknown run {name}")
    return RUNS / name


@app.get("/")
def index():
    return FileResponse(HERE / "index.html")


@app.get("/api/runs")
def runs():
    out = []
    for d in sorted(RUNS.glob("*/metrics.json"), key=lambda p: p.stat().st_mtime, reverse=True):
        m = json.loads(d.read_text())
        if snapshot_path(m["name"]) is None:  # snapshot not available locally
            continue
        out.append({"name": m["name"], "final": m["final"], "seconds": m["seconds"]})
    return out


@app.get("/api/run")
def run(name: str):
    d = run_dir(name)
    z = np.load(snapshot_path(name))
    cfg = yaml.safe_load((d / "config.yaml").read_text())
    sem = load_semantics(cfg["data"]["scene"])
    l2 = sem["id_to_label"]
    ids = z["instance_ids"].tolist()
    names = [sem["classes"].get(l2[i] if i < len(l2) else 0, "unlabeled") for i in ids]
    metrics = json.loads((d / "metrics.json").read_text())
    r4 = lambda a: np.round(a, 4).tolist()  # noqa: E731
    return {
        "config": cfg, "steps": z["step"].tolist(), "instance_ids": ids, "instance_names": names,
        "scale": r4(z["scale"]), "eps": r4(z["eps"]), "rot": r4(z["rot"]), "trans": r4(z["trans"]),
        "existence": r4(z["existence"]), "sq_label": z["sq_label"].tolist(), "purity": r4(z["purity"]),
        "active": z["active"].tolist(), "owned": z["owned"].tolist(),
        "history": metrics["history"], "final_iou": r4(np.array(metrics["final_iou"])),
    }


@lru_cache(maxsize=2)
def packed_points(scene: str, version: int, drop_unlabeled: bool) -> tuple[bytes, dict]:
    cfg = {"data": {"scene": scene, "version": version, "drop_unlabeled": drop_unlabeled,
                    "tokens_dir": "/home/sebastian/Documents/Uni/Thesis/data/replica_v1_sonata/stage4"}}
    sc = load_scene(cfg)
    pos = sc["pos"].cpu().numpy()
    y = sc["y"].cpu().numpy()
    # RGB of the same (filtered) points
    raw = load_points(scene)
    col = raw["col"]
    if drop_unlabeled:
        l2 = np.array(load_semantics(scene)["id_to_label"])
        inst = raw["instance"].astype(np.int64)
        col = col[l2[np.minimum(inst, len(l2) - 1)] > 0]
    lo, hi = pos.min(0), pos.max(0)
    ext = np.maximum(hi - lo, 1e-6)
    blocks = [np.round((pos - lo) / ext * 65535).astype("<u2"), y.astype("<u2"), np.ascontiguousarray(col)]
    headers = {"X-Count": str(len(pos)), "X-Min": ",".join(f"{x:.6f}" for x in lo), "X-Extent": ",".join(f"{x:.6f}" for x in ext)}
    return b"".join(b.tobytes() for b in blocks), headers


@app.get("/api/points")
def points(name: str):
    cfg = yaml.safe_load((run_dir(name) / "config.yaml").read_text())["data"]
    body, headers = packed_points(cfg["scene"], cfg["version"], cfg["drop_unlabeled"])
    return Response(body, media_type="application/octet-stream", headers=headers)


if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8007)
