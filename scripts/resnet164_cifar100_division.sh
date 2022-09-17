#!/bin/bash
## Vanilla training

python train_cifar100.py --gpu_device 4 5 6 7 --dist-url 'tcp://localhost:10001' \
                        --dist-backend nccl --multiprocessing-distributed \
                        --world-size 1 --rank 0 \
                        --epoch 200 -b 256 --lr 0.15 --wd 5e-4 \
                        --arch resnet164 --scheduler coslr --lfc_block 8 --hfc_bit_num 2 \
                        --workers 16 --seed 0 2>&1 | tee log/cifar100/resnet164_cifar100_division_coslr_e_200_b_256_lr015.log











