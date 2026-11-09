#!/bin/bash
# Every split and K in its own process. Test first: a complete sweep exists after about 2 h. Rerun to resume.
set -e
cd "$(dirname "$0")/code"
OUT=/home/sebastian/data/ShapeNet-Chairs-RGBD/sq_fits
for split in test train val; do
    for k in $(seq 1 16); do
        ../../../.venv/bin/python fit_sqs.py --split "$split" --k "$k" --out "$OUT"
    done
done
