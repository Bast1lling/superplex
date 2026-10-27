"""Custom trackio dashboard for this experiment (frontend in ../dashboard, data in ../results/large_files/trackio).

    uv run python experiments/07_sq_global_fit/code/dashboard.py
then open the printed URL (default http://127.0.0.1:7862).
"""

import os
from pathlib import Path

EXP = Path(__file__).resolve().parent.parent
os.environ.setdefault("TRACKIO_DIR", str(EXP / "results/large_files/trackio"))
# Fixed write token so the custom frontend can delete runs. The server only listens on 127.0.0.1.
os.environ.setdefault("TRACKIO_WRITE_TOKEN", "superplex-local")
import trackio  # noqa: E402

if __name__ == "__main__":
    trackio.show(project="sq_global_fit", frontend_dir=EXP / "dashboard", open_browser=False,
                 server_port=int(os.environ.get("PORT", 7862)), block_thread=True)
