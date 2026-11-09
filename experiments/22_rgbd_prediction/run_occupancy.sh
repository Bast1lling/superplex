#!/bin/bash
# Occupancy signals on the 16 frames (6000 steps each), one run after the other. Baselines: f16_dense, f16_topk.
# A: Chamfer + pairwise volumetric loss; B: + soft-IoU of the decoder's union vs the target union; C: vs the depth map.
cd "$(dirname "$0")/code"
R=$(cd ../results && pwd)
run() { [ -f $R/$1.json ] && return; uv run python train.py --name $1 ${@:2} > $R/$1.log 2>&1; }  # skips finished runs
F="--epochs 750 --eval_every 150 --frames 16"
run f16_dense_vol $F --dense --loss both
run f16_topk_vol $F --topk --loss both
run f16_topk_union_target $F --topk --union target
run f16_topk_union_obs $F --topk --union observed
