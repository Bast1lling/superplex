"""Trackio dashboard of this experiment, local only.

    uv run python experiments/22_rgbd_prediction/code/dashboard.py      -> http://127.0.0.1:7863

Metric groups (name prefixes, see train.py's `train_log` / `eval_log`) in the order of PLOT_ORDER; only "main" is
open by default. Runs are grouped by their trackio group (train.py's `group`) by default. Trackio 0.42 has no option
for either, so this serves a copy of its built frontend (results/large_files/trackio_frontend) with two patches.
"""

import os
import shutil
from pathlib import Path

EXP = Path(__file__).resolve().parents[1]
os.environ.setdefault("TRACKIO_DIR", str(EXP / "results" / "large_files" / "trackio"))
PLOT_ORDER = ["main/train_loss", "main/pair_iou", "main/found", "main/predicted", "main/mask_cov", "main/recall@0.5",
              "test/occ_iou@0.3", "test/occ_iou", "test/found@0.3", "test/predicted@0.3", "test/cd", "test/cls_acc",
              "test/*", "loss/cd", "loss/forced", "loss/vol", "loss/vol_forced", "loss/exist", "loss/*",
              "proposals/*", "train_frames/pair_iou", "train_frames/found", "train_frames/occ_iou@0.3",
              "train_frames/*", "optim/*"]
os.environ["TRACKIO_PLOT_ORDER"] = ",".join(PLOT_ORDER)
import trackio  # noqa: E402

PATCHES = [  # (compiled Svelte, replacement)
    # Metrics page: a group's accordion starts open only for "main"
    ('get label(){return`${h(bt)??""} (${h(Ue)??""})`},open:!0',
     'get label(){return`${h(bt)??""} (${h(Ue)??""})`},open:h(bt)==="main"'),
    # Sidebar: group runs by their trackio group by default (and keep it while the configs load)
    ('Q=J(null);Pt(()=>{a(),$(Q,null)}),Pt(()=>{h(ne).some(ge=>ge.value===h(Q))||$(Q,null)})',
     'Q=J("_Group");Pt(()=>{a(),$(Q,"_Group")}),Pt(()=>{h(Q)==="_Group"||h(ne).some(ge=>ge.value===h(Q))||$(Q,null)})'),
]


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
    trackio.show(project="rgbd_prediction", open_browser=False, server_port=7863, block_thread=True, frontend_dir=frontend())
