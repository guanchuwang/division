#!/bin/bash
## Division training

python train_cifar10.py --gpu_device 0 1 2 3 --dist-url 'tcp://localhost:10000' --dist-backend nccl --multiprocessing-distributed --world-size 1 --rank 0 --data-path "./data" --epoch 100 -b 256 --lr 0.1 --wd 5e-4 --arch resnet18 --opt SGD --workers 16 --seed 0

