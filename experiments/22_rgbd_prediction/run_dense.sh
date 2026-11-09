#!/bin/bash
# One-to-many encoder (--dense) and + top-k decoder (--topk): 16 training frames (6000 steps), then the full set
# (150 epochs), one run after the other.
cd "$(dirname "$0")/code"
R=../results
run() { uv run python train.py --name $1 ${@:2} > $R/$1.log 2>&1; }
F="--epochs 750 --eval_every 150 --frames 16"
run f16_dense $F --dense
run f16_topk $F --topk
A="--epochs 150 --eval_every 10"
run full_dense $A --dense
run full_topk $A --topk
