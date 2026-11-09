#!/bin/bash
# Runs the lines of queue.txt (train.py arguments, first token = --name value) one after another; start N workers.
# A line is claimed by moving it to queue.done. Logs: results/large_files/logs/<name>.log
cd "$(dirname "$0")"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
mkdir -p results/large_files/logs
while true; do
  line=$(flock queue.txt bash -c 'l=$(head -n1 queue.txt); [ -n "$l" ] && sed -i 1d queue.txt && echo "$l" >> queue.done; echo "$l"')
  [ -z "$line" ] && break
  name=$(echo "$line" | sed -E 's/.*--name ([^ ]+).*/\1/')
  (cd code && ../../../.venv/bin/python train.py $line) > results/large_files/logs/$name.log 2>&1
done
