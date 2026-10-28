"""Experiment 08's viewer (06's page + voxel layer), serving this experiment's exported runs.

    uv run python experiments/08_volumetric_decomp/code/export_viewer.py --exp 09 <run> [...]   # -> results/viewer_runs
    uv run python experiments/09_scene_blob/viewer/server.py     -> http://localhost:8009
SQs are colored by the GT instance they are assigned to (majority of their voxels); the fit never saw the labels.
"""

from pathlib import Path
import importlib.util

import uvicorn

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("viewer08", HERE.parents[1] / "08_volumetric_decomp" / "viewer" / "server.py")
viewer08 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(viewer08)
viewer08.retarget(HERE.parent / "results" / "viewer_runs", "Exp 09: SQs for the whole scene without semantics")
app = viewer08.app

if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8009)
