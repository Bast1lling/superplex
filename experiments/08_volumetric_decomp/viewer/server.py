"""Experiment 06's viewer, pointed at the exported runs of this experiment (code/export_viewer.py).

    uv run python experiments/08_volumetric_decomp/viewer/server.py     -> http://localhost:8008
export_viewer.py remaps sq_label into the instance index space of 06's point loader, so SQ and point colors match.
Single snapshot per run (final SQs); the metrics panel shows this experiment's summary metrics.
index.html is 06's page plus a voxel layer: the labeled volume the run was fitted to (code/volume.py), only voxels
on the surface of their label (a 6-neighbor has another label). First request per scene builds the volume (~15 s,
cached like fit.py). Degraded-input runs show the clean volume.
"""

from functools import lru_cache
from pathlib import Path
import importlib.util
import re
import sys

from fastapi.responses import FileResponse, HTMLResponse, Response
import numpy as np
import uvicorn
import yaml

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("viewer06", HERE.parents[1] / "06_sq_gt_instance_fit" / "viewer" / "server.py")
viewer06 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(viewer06)
viewer06.RUNS = HERE.parent / "results" / "viewer_runs"
viewer06.LARGE_RUNS = viewer06.RUNS
sys.path.insert(0, str(HERE.parent / "code"))
from volume import EXTERIOR, FREE, UNKNOWN, cached_volume, fill_planar, load_scene  # noqa: E402

app = viewer06.app
# replace 06's page and run list (sorted by mtime) with ours
app.router.routes = [r for r in app.router.routes if getattr(r, "path", None) not in ("/", "/api/runs")]

# voxel categories sent to the page
SOLID, PLANAR, CAT_FREE, CAT_UNKNOWN, CAT_EXTERIOR = range(5)


def natural_key(name: str):
    """Sort by scene, then version: room0_v2 < room0_v9_exterior < room0_v10 < room1_v2."""
    return [int(p) if p.isdigit() else p for p in re.split(r"(\d+)", name)]


@app.get("/")
def index():
    return FileResponse(HERE / "index.html")


@app.get("/api/runs")
def runs():
    return sorted(viewer06.runs(), key=lambda r: natural_key(r["name"]))


def surface(L: np.ndarray) -> np.ndarray:
    """Voxels with a 6-neighbor of another label (or at the grid border)."""
    P = np.pad(L, 1, constant_values=np.iinfo(L.dtype).min)
    c = P[1:-1, 1:-1, 1:-1]
    out = np.zeros(L.shape, bool)
    for ax in range(3):
        for s in (-1, 1):
            out |= np.roll(P, s, ax)[1:-1, 1:-1, 1:-1] != c
    return out


@lru_cache(maxsize=4)
def packed_voxels(scene: str, res: float, min_points: int, labels: str, planar: tuple | None, viewer_ids: tuple) -> tuple[bytes, dict]:
    cfg = {"scene": scene, "res": res, "min_points": min_points, "labels": labels}
    sc = load_scene(scene, min_points=min_points, labels=labels)
    vol = cached_volume(cfg, sc, "clean2")
    L, dims = vol["label"].astype(np.int32), vol["dims"]
    keep = surface(L)
    idx = np.argwhere(keep)
    lab = L[keep]
    cat = np.select([lab == FREE, lab == UNKNOWN, lab == EXTERIOR], [CAT_FREE, CAT_UNKNOWN, CAT_EXTERIOR], SOLID)
    inst = np.maximum(lab, 0)
    if planar is not None:  # amodal targets added by fill_planar (as in fit.prepare)
        extra_idx, extra_inst = [], []
        for k in range(len(sc["instance_ids"])):
            vox = np.argwhere(L == k)
            if len(vox) == 0:
                continue
            e = fill_planar(vox, res, *planar)
            e = e[((e >= 0) & (e < dims)).all(1)]
            e = e[L[tuple(e.T)] != k]
            extra_idx.append(e), extra_inst.append(np.full(len(e), k))
        if extra_idx:
            idx = np.concatenate([idx] + extra_idx)
            inst = np.concatenate([inst] + extra_inst)
            cat = np.concatenate([cat, np.full(len(idx) - len(cat), PLANAR)])
    if labels == "instance":  # this experiment's instance index -> the viewer's (06 loader) index, as in export_viewer
        inst = np.searchsorted(np.array(viewer_ids), sc["instance_ids"])[inst]
    blocks = [idx.astype("<u2"), cat.astype("u1"), inst.astype("<u2")]
    counts = np.bincount(cat, minlength=5)
    headers = {"X-Count": str(len(idx)), "X-Lo": ",".join(f"{x:.6f}" for x in vol["lo"]), "X-Res": f"{res}",
               "X-Counts": ",".join(str(c) for c in counts)}
    return b"".join(b.tobytes() for b in blocks), headers


@app.get("/api/voxels")
def voxels(name: str):
    d = viewer06.run_dir(name)
    cfg = yaml.safe_load((d / "config.yaml").read_text())["exp08"]
    z = np.load(viewer06.snapshot_path(name))
    planar = (cfg["planar_thick"], cfg["planar_min"]) if cfg.get("fill_planar", False) else None
    body, headers = packed_voxels(cfg["scene"], float(cfg["res"]), int(cfg["min_points"]), cfg.get("labels", "instance"),
                                  planar, tuple(z["instance_ids"].tolist()))
    # voxels live in the raw point frame; the viewer shows the frame of Sonata token version 0
    headers["X-R"] = ",".join(f"{x:.8f}" for x in z["R"].ravel())
    headers["X-T"] = ",".join(f"{x:.8f}" for x in z["t"])
    return Response(body, media_type="application/octet-stream", headers=headers)


def retarget(runs_dir: Path, title: str):
    """Serve another experiment's exported runs (same format) under its own title; used by 09 / 10's viewer/server.py."""
    viewer06.RUNS = viewer06.LARGE_RUNS = runs_dir
    app.router.routes = [r for r in app.router.routes if getattr(r, "path", None) != "/"]
    html = (HERE / "index.html").read_text()
    html = html.replace("<title>Volumetric SQ Decomposition</title>", f"<title>{title}</title>")
    html = html.replace("<h1>Exp 08: volumetric SQ decomposition</h1>", f"<h1>{title}</h1>")
    app.get("/")(lambda: HTMLResponse(html))


if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8008)
