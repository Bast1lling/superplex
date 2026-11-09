#!/bin/bash
# Waits for the train split's SuperFlex fits, then trains on all train chairs (7k steps, batch 8).
cd "$(dirname "$0")"
until [ -f /home/sebastian/data/ShapeNet-Chairs-RGBD/sq_fits/train/superflex_opt_plain.npz ]; do sleep 30; done
sleep 30
(cd code && ../../../.venv/bin/python train.py --name train4745 --train_split train --steps 7000 --eval_every 2000) > results/large_files/logs/train4745.log 2>&1
