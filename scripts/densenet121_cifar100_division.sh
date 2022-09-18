#!/bin/bash
## Division training

python train_cifar100.py --gpu_device 0 1 2 3 --dist-url 'tcp://localhost:10000' --dist-backend nccl --multiprocessing-distributed --world-size 1 --rank 0 --data-path "./data" --epoch 100 -b 256 --lr 0.1 --wd 5e-4 --arch densenet121 --scheduler coslr --lfc_block 8 --hfc_bit_num 2 --workers 16 --seed 0











