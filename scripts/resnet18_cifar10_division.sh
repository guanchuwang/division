#!/bin/bash
## Division training

python train_cifar10.py --gpu_device 0 1 2 3 --dist-url 'tcp://localhost:10009' --dist-backend nccl --multiprocessing-distributed --world-size 1 --rank 0 --epoch 100 -b 256 --lr 0.1 --wd 5e-4 --arch resnet18 --opt SGD --workers 16 --seed 0 2>&1 | tee log/cifar10/resnet18_cifar10_lb_8_hq_2_coslr_e_100_b_256.log

