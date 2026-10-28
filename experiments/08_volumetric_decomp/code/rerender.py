"""Re-render runs from their sqs.npz: python rerender.py run [run ...]"""

import json
import sys

import numpy as np

from evaluate import render_all
from fit import RUNS
from volume import load_scene

for name in sys.argv[1:]:
    cfg = json.loads((RUNS / name / "config.json").read_text())
    sc = load_scene(cfg["scene"], min_points=cfg.get("min_points", 20), labels=cfg.get("labels", "instance"))
    render_all(sc, dict(np.load(RUNS / name / "sqs.npz")), RUNS / name)
    print("rendered", name)
