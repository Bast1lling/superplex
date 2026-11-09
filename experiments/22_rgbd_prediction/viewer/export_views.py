"""Camera poses of the NICE-SLAM renderings for index.html's view orbs: data/<scene>/views.json (camera-to-world 4 x 4
per frame, row-major, OpenCV axes) and data/<scene>/frames -> the renderings folder (symlink, not versioned).

    uv run python export_views.py
"""

import json
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
RENDERS = Path("/home/sebastian/Documents/Uni/Thesis/data/replica_nice_slam/Replica")

for d in sorted((HERE / "data").glob("*/scene.json")):
    scene = d.parent.name
    src = RENDERS / scene.replace("_", "") / "results"
    c2w = np.loadtxt(src.parent / "traj.txt").reshape(-1, 16)
    (d.parent / "views.json").write_text(json.dumps({"c2w": np.round(c2w, 4).tolist(), "image": "frames/frame{:06d}.jpg"}))
    link = d.parent / "frames"
    if not link.exists():
        link.symlink_to(src)
    print(scene, len(c2w), "frames")
