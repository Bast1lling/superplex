"""Trackio dashboard of this experiment, local only.

    uv run python experiments/21_teacher_regression/code/dashboard.py      -> http://127.0.0.1:7862
"""

import os
from pathlib import Path

EXP = Path(__file__).resolve().parents[1]
os.environ.setdefault("TRACKIO_DIR", str(EXP / "results" / "large_files" / "trackio"))
import trackio  # noqa: E402

if __name__ == "__main__":
    trackio.show(project="teacher_regression", open_browser=False, server_port=7862, block_thread=True)
