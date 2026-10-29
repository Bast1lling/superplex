#!/bin/bash
# Main runs of experiments 12-15 on all scenes (best settings from the room_0 sweeps).
cd "$(dirname "$0")/code"
F="0 400 800 1200 1600"
for s in room_0 room_1 office_0; do
  t=${s/_/}
  uv run python fit.py --exp 12 --name ${t}_gt3d --set scene=$s --set source=gt3d
  uv run python fit.py --exp 12 --name ${t}_rgbd --set scene=$s --set erode=1
  uv run python fit.py --exp 13 --name ${t}_depthpro --set scene=$s --set depth=depthpro --set align=1 --set erode=1
  for f in $F; do
    uv run python fit.py --exp 14 --name ${t}_f$f --set scene=$s --set frames=$f
    uv run python fit.py --exp 15 --name ${t}_f$f --set scene=$s --set frames=$f --set depth=depthpro --set erode=1
    # control: single-view RGB with the oracle per-view depth scale
    uv run python fit.py --exp 15 --name ${t}_f${f}_gtscale --set scene=$s --set frames=$f --set depth=depthpro_gtscale --set erode=1 --no-render
  done
done
