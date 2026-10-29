#!/bin/bash
# All initialization x objective combinations of experiment 16 for one scene (default room_0).
cd "$(dirname "$0")/code"
S=${1:-room_0}
uv run python sfopt.py --scene $S --init network --objective theirs --steps 0
for i in network kmeans_net kmeans16; do
  for o in theirs ours; do
    [ -e "../results/runs/${S/_/}_${i}_${o}/metrics.json" ] || uv run python sfopt.py --scene $S --init $i --objective $o
  done
done
