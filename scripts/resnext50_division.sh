#!/bin/bash
## tamu_datalab3 division training

datapath="/home/grads/z/zhimengj/Guanchu/python_project/division_imagenet/data/"
outputdir="./checkpoint_resnext50_32x4d_division/"
resume="${outputdir}checkpoint.pth"
torchrun --nproc_per_node=4 --master_port 29501 train.py --gpu_devices 0 1 2 3 -b 64 --data-path $datapath --resume --output-dir $outputdir --model resnext50_32x4d --epochs 150 --log_fname log_resnext50_32x4d_division_tamu_datalab3.txt

