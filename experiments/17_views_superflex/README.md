# 17 — SuperFlex's objective on the view inputs of 12–15

**Question.** Experiment 16 showed SuperFlex's objective is much better than ours on the 3D GT. How much does it
improve the view-based settings of 12–15 (RGBD / RGB × multi / single view)?

**Setup** (`code/fitviews.py`, see its docstring). Inputs exactly as the final runs of 12–15. Per instance: surface
points = back-projected pixels with its label, query points labelled from the carved grid. Multi-view: unknown =
nearest solid instance, objective unchanged. Single view: unknown query points left out of the soft IoU, bounding-box
term dropped. Init: SuperFlex robust on the partial points (single view RGBD also k-means with the network's count).
Evaluation as in 16; the old 12–15 fits are re-evaluated the same way (`--baseline`).

**Status: running** (`run_all.sh`). First results (room_0): old fits re-evaluated 0.634 (12 RGBD multi, was 0.630),
0.709 (3D upper bound, was 0.708); RGBD single view frame 800: 0.550 (old 0.55), precision 0.90, recall 0.60.

    cd code && uv run python fitviews.py --setting rgbd_multi --scene room_0 [--frame 800 --init kmeans_net]
