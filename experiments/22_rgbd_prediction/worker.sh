#!/bin/bash
# Pulls "<name> <train.py args>" lines from queue.txt (one at a time, locked) and runs them until the queue is empty.
# Several workers can run in parallel; lines can be appended while they run. Finished runs (results/<name>.json) are skipped.
cd "$(dirname "$0")"
R=$(pwd)/results
while true; do
  line=$(flock queue.txt -c 'head -n 1 queue.txt; sed -i 1d queue.txt')
  [ -z "$line" ] && break
  set -- $line
  [ -f $R/$1.json ] && continue
  echo "$(date +%H:%M) start $1" >> $R/queue.log
  (cd code && uv run python train.py --name $1 ${@:2} > $R/$1.log 2>&1)
  echo "$(date +%H:%M) end $1" >> $R/queue.log
done
