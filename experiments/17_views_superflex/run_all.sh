#!/bin/bash
# Experiment 17: old 12-15 fits re-evaluated, then SuperFlex's objective on all four view settings (skips existing runs).
cd "$(dirname "$0")/code"
F="0 400 800 1200 1600"
B=""
for s in room0 room1 office0; do
  B="$B 12:${s}_gt3d 12:${s}_rgbd 13:${s}_depthpro"
  for f in $F; do B="$B 14:${s}_f$f 15:${s}_f$f"; done
done
uv run python fitviews.py --baseline $B
run() { [ -e "../results/runs/$1/metrics.json" ] || uv run python fitviews.py "${@:2}" | grep -v "^  step"; }
for S in room_0 room_1 office_0; do
  s=${S/_/}
  run ${s}_rgbd_multi --setting rgbd_multi --scene $S
  run ${s}_rgb_multi --setting rgb_multi --scene $S
  for f in $F; do
    run ${s}_rgbd_single_f$f --setting rgbd_single --scene $S --frame $f
    run ${s}_rgb_single_f$f --setting rgb_single --scene $S --frame $f
    run ${s}_rgbd_single_f${f}_kmeans_net --setting rgbd_single --scene $S --frame $f --init kmeans_net
  done
done
