#!/bin/bash
## division training

datapath=<Your Dataset Folder>
outputdir="./checkpoint_resnet50_division/"
resume="${outputdir}checkpoint.pth"

torchrun --nproc_per_node=4 train.py --gpu_devices 0 1 2 3 -b 64 --epochs 120 --data-path $datapath --output-dir $outputdir --model resnet50 \
     --lr 0.1 --lr-scheduler cosineannealinglr --wd 1e-4 --lr-warmup-epochs 0 --log_fname log_resnet50_division.txt

