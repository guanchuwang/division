#!/bin/bash
## division training

datapath="./data"
outputdir="./checkpoint_densenet161_division/"
resume="${outputdir}checkpoint.pth"

torchrun --nproc_per_node=8 train.py --gpu_devices 0 1 2 3 4 5 6 7 -b 64 --epochs 120 --data-path $datapath --output-dir $outputdir --model densenet161 \
          --lr 0.2 --lr-scheduler cosineannealinglr --wd 1e-4 --lr-warmup-epochs 0 --log_fname log_densenet161_division.txt




