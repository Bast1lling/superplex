#!/bin/bash
# Two-stage selection: one-frame overfit (3000 steps, as run_overfit.sh), then 16 training frames (6000 steps)
# against plain patch queries on the same frames.
cd "$(dirname "$0")/code"
run() { uv run python train.py --name $1 ${@:2} > ../results/$1.log 2>&1; }
run of_twostage_cd --epochs 375 --eval_every 75 --overfit room_2_000500 --patch_queries --two_stage
F="--epochs 750 --eval_every 150 --frames 16 --patch_queries"
run f16_twostage_cd $F --two_stage & run f16_patch_cd $F & wait
