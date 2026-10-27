"""Experiment 06's viewer, pointed at the exported runs of this experiment (code/export_viewer.py).

    uv run python experiments/08_volumetric_decomp/viewer/server.py     -> http://localhost:8008
Note: 06's viewer colors SQs by sq_label, here the instance index of this experiment's own (>= 20 points) instance
list, so instance colors may not match the point colors of 06's loader exactly.
"""

from pathlib import Path
import importlib.util

import uvicorn

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("viewer06", HERE.parents[1] / "06_sq_gt_instance_fit" / "viewer" / "server.py")
viewer06 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(viewer06)
viewer06.RUNS = HERE.parent / "results" / "viewer_runs"
viewer06.LARGE_RUNS = viewer06.RUNS

if __name__ == "__main__":
    uvicorn.run(viewer06.app, host="127.0.0.1", port=8008)
