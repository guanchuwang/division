#!/bin/bash
## Division training

python train_cifar10.py --gpu_device 0 1 2 3 --dist-url 'tcp://localhost:10003' \
                        --dist-backend nccl --multiprocessing-distributed \
                        --world-size 1 --rank 0 \
                        --data-path <Your Dataset Folder> \
                        --epoch 100 -b 256 --lr 0.1 --wd 5e-4 \
                        --arch resnet164 --opt SGD \
                        --workers 16 --seed 7 2>&1 | tee log/cifar10/resnet164_cifar10_division_coslr_e_100_b_256.txt


