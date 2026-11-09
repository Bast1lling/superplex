#!/bin/bash
# Targets for all 8 NICE-SLAM scenes, one process each, skipping finished ones. ./run_all.sh [filled|solid|winding]
cd "$(dirname "$0")/code"
O=${1:-filled}
D=../results/targets$([ $O != filled ] && echo _$O)
mkdir -p ../results/logs
for S in room_0 room_1 room_2 office_0 office_1 office_2 office_3 office_4; do
  [ -e "$D/$S/metrics.json" ] || uv run python fit_targets.py --scene $S --occ $O 2>&1 | tee ../results/logs/${S}_$O.txt
done
