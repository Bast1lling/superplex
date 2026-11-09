#!/bin/bash
# Single-frame overfit runs (room_2_000500, 3000 steps), two at a time.
cd "$(dirname "$0")/code"
A="--epochs 375 --eval_every 75 --overfit room_2_000500"
run() { uv run python train.py --name $1 $A ${@:2} > ../results/$1.log 2>&1; }
run of_base_cd & run of_base_vol --loss vol & wait
run of_patch_cd --patch_queries & run of_patch_vol --patch_queries --loss vol & wait
run of_patch_both --patch_queries --loss both
run of_patch_sqmask --patch_queries --mask sq & run of_patch_sqamodal --patch_queries --mask sq_amodal & wait
