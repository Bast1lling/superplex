"""Trackio dashboard of exp 28 (exp 27's patched frontend: groups open / runs grouped by default).

    uv run python experiments/28_slot_generalization/code/dashboard.py      -> http://127.0.0.1:7828
"""

import os
import shutil
import sys
from pathlib import Path

EXP = Path(__file__).resolve().parents[1]
os.environ.setdefault("TRACKIO_DIR", str(EXP / "results" / "large_files" / "trackio"))
S = ["train", "views", "scenes"]
PLOT_ORDER = (["main/train_loss"] + [f"main/{s}_{k}" for k in ("pair_iou_fr", "mask_miou", "part_iou_fr") for s in S]
              + ["loss/*"] + [f"other/{s}_{k}" for k in ("ari", "parts_active", "part_targets", "sdf_own", "matched_active",
                                                          "unmatched_active") for s in S] + ["other/*"])
os.environ["TRACKIO_PLOT_ORDER"] = ",".join(PLOT_ORDER)
sys.path.insert(0, str(EXP.parent / "27_slot_attention" / "code"))
import trackio  # noqa: E402
from dashboard import PATCHES  # noqa: E402  (exp 27's)


def frontend():
    src = Path(trackio.__file__).parent / "frontend" / "dist"
    dst = EXP / "results" / "large_files" / "trackio_frontend"
    shutil.rmtree(dst, ignore_errors=True)
    shutil.copytree(src, dst)
    js = next(f for f in (dst / "assets").glob("index-*.js") if PATCHES[0][0] in f.read_text())
    code = js.read_text()
    for old, new in PATCHES:
        assert code.count(old) == 1, f"trackio frontend changed, patch not found: {old[:60]}"
        code = code.replace(old, new)
    js.write_text(code)
    return dst


if __name__ == "__main__":
    trackio.show(project="slot_generalization", open_browser=False, server_port=7828, block_thread=True, frontend_dir=frontend())
