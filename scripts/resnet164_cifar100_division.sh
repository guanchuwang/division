#!/bin/bash
## Vanilla training

python train_cifar100.py --gpu_device 0 1 2 3 --dist-url 'tcp://localhost:10001' \
                        --dist-backend nccl --multiprocessing-distributed \
                        --world-size 1 --rank 0 \
                        --data-path "./data" \
                        --epoch 200 -b 256 --lr 0.15 --wd 5e-4 \
                        --arch resnet164 --scheduler coslr --lfc_block 8 --hfc_bit_num 2 \
                        --workers 16 --seed 0











