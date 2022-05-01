## nn Parallel 1 GPU

# vanilla       steplr        epoch     200     batchsize 1024
python main_cifar100.py --gpu_device 1 --dist-url 'tcp://localhost:10001' --dist-backend nccl --multiprocessing-distributed --world-size 1 --rank 0 --epoch 200 -b 1024 --lr 0.5 --wd 4e-5 --arch shufflenet_v2_x1_0 --scheduler steplr --gamma 0.1 --vanilla --workers 16 --seed 0 2>&1 | tee shufflenet_v2_x1_0_cifar100_baseline_steplr_e_200_b_1024.log

# vanilla       warmup_coslr  epoch     200     batchsize 1024
python main_cifar100.py --gpu_device 3 --dist-url 'tcp://localhost:10002' --dist-backend nccl --multiprocessing-distributed --world-size 1 --rank 0 --epoch 200 -b 1024 --lr 0.5 --wd 4e-5 --arch shufflenet_v2_x1_0 --scheduler warmup_coslr --vanilla --workers 16 --seed 0 2>&1 | tee shufflenet_v2_x1_0_cifar100_baseline_coslr_e_200_b_1024.log

# vanilla       steplr        epoch     200     batchsize 256
python main_cifar100.py --gpu_device 3 --dist-url 'tcp://localhost:10002' --dist-backend nccl --multiprocessing-distributed --world-size 1 --rank 0 --epoch 200 -b 256 --lr 0.5 --wd 4e-5 --arch shufflenet_v2_x1_0 --scheduler steplr --gamma 0.1 --vanilla --workers 16 --seed 0 2>&1 | tee shufflenet_v2_x1_0_cifar100_baseline_steplr_e_200_b_256.log

# vanilla       warmup_coslr  epoch     200     batchsize 256
python main_cifar100.py --gpu_device 3 --dist-url 'tcp://localhost:10005' --dist-backend nccl --multiprocessing-distributed --world-size 1 --rank 0 --epoch 200 -b 256 --lr 0.5 --wd 4e-5 --arch shufflenet_v2_x1_0 --scheduler warmup_coslr --vanilla --workers 16 --seed 0 2>&1 | tee shufflenet_v2_x1_0_cifar100_baseline_coslr_e_200_b_256.log


# fdq lb 8 hq 2 steplr        epoch     200     batchsize 1024
python main_cifar100.py --gpu_device 0 --dist-url 'tcp://localhost:10008' --dist-backend nccl --multiprocessing-distributed --world-size 1 --rank 0 --epoch 200 -b 1024 --lr 0.5 --wd 4e-5 --arch shufflenet_v2_x1_0 --scheduler steplr --gamma 0.1 --lfc_block 8 --hfc_bit_num 2 --workers 16 --seed 0 2>&1 | tee shufflenet_v2_x1_0_cifar100_lb_8_hq_2_steplr_e_200_b_1024.log

# fdq lb 8 hq 2 warmup_coslr  epoch     200     batchsize 1024
python main_cifar100.py --gpu_device 2 --dist-url 'tcp://localhost:10006' --dist-backend nccl --multiprocessing-distributed --world-size 1 --rank 0 --epoch 200 -b 1024 --lr 0.5 --wd 4e-5 --arch shufflenet_v2_x1_0 --scheduler warmup_coslr --lfc_block 8 --hfc_bit_num 2 --workers 16 --seed 0 2>&1 | tee shufflenet_v2_x1_0_cifar100_lb_8_hq_2_coslr_e_200_b_1024.log

# fdq lb 8 hq 2 steplr        epoch     200     batchsize 256
python main_cifar100.py --gpu_device 3 --dist-url 'tcp://localhost:10002' --dist-backend nccl --multiprocessing-distributed --world-size 1 --rank 0 --epoch 200 -b 256 --lr 0.5 --wd 4e-5 --arch shufflenet_v2_x1_0 --scheduler steplr --gamma 0.1 --lfc_block 8 --hfc_bit_num 2 --workers 16 --seed 0 2>&1 | tee shufflenet_v2_x1_0_cifar100_lb_8_hq_2_steplr_e_200_b_256.log

# fdq lb 8 hq 2 warmup_coslr  epoch     200     batchsize 256
python main_cifar100.py --gpu_device 1 --dist-url 'tcp://localhost:10006' --dist-backend nccl --multiprocessing-distributed --world-size 1 --rank 0 --epoch 200 -b 256 --lr 0.5 --wd 4e-5 --arch shufflenet_v2_x1_0 --scheduler warmup_coslr --lfc_block 8 --hfc_bit_num 2 --workers 16 --seed 0 2>&1 | tee shufflenet_v2_x1_0_cifar100_lb_8_hq_2_coslr_e_200_b_256.log








