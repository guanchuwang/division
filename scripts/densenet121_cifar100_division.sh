#!/bin/bash
## Division training

# division lb 8 hq 2 steplr        epoch     100     batchsize 256   lr 0.1
python train_cifar100.py --gpu_device 4 5 6 7 --dist-url 'tcp://localhost:10002' --dist-backend nccl --multiprocessing-distributed --world-size 1 --rank 0 --epoch 100 -b 256 --lr 0.1 --wd 5e-4 --arch densenet121 --scheduler coslr --lfc_block 8 --hfc_bit_num 2 --workers 16 --seed 0 2>&1 | tee log/cifar100/densenet121_cifar100_lb_8_hq_2_coslr_e_120_b_256_lr01.log











