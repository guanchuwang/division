## nn Parallel 1 GPU

# vanilla       steplr        epoch     200     batchsize 128
python main_cifar100.py --gpu_device 0 --dist-url 'tcp://localhost:10001' --dist-backend nccl --multiprocessing-distributed --world-size 1 --rank 0 --epoch 200 -b 128 --lr 0.1 --wd 5e-4 --arch densenet121 --scheduler steplr --vanilla --workers 16 --seed 0 2>&1 | tee densenet121_cifar100_baseline_steplr_e_200_b_128.log

# vanilla       warmup_coslr  epoch     200     batchsize 128
python main_cifar100.py --gpu_device 1 --dist-url 'tcp://localhost:10002' --dist-backend nccl --multiprocessing-distributed --world-size 1 --rank 0 --epoch 200 -b 128 --lr 0.1 --wd 5e-4 --arch densenet121 --scheduler warmup_coslr --vanilla --workers 16 --seed 0 2>&1 | tee densenet121_cifar100_baseline_coslr_e_200_b_128.log

# vanilla       steplr        epoch     200     batchsize 256
python main_cifar100.py --gpu_device 2 --dist-url 'tcp://localhost:10003' --dist-backend nccl --multiprocessing-distributed --world-size 1 --rank 0 --epoch 200 -b 256 --lr 0.1 --wd 5e-4 --arch densenet121 --scheduler steplr --vanilla --workers 16 --seed 0 2>&1 | tee densenet121_cifar100_baseline_steplr_e_200_b_256.log

# vanilla       warmup_coslr  epoch     200     batchsize 256
python main_cifar100.py --gpu_device 3 --dist-url 'tcp://localhost:10005' --dist-backend nccl --multiprocessing-distributed --world-size 1 --rank 0 --epoch 200 -b 256 --lr 0.1 --wd 5e-4 --arch densenet121 --scheduler warmup_coslr --vanilla --workers 16 --seed 0 2>&1 | tee densenet121_cifar100_baseline_coslr_e_200_b_256.log


# vanilla       steplr        epoch     100     batchsize 128
python main_cifar100.py --gpu_device 0 --dist-url 'tcp://localhost:10007' --dist-backend nccl --multiprocessing-distributed --world-size 1 --rank 0 --epoch 100 -b 128 --lr 0.1 --wd 5e-4 --arch densenet121 --scheduler steplr --vanilla --workers 16 --seed 0 2>&1 | tee densenet121_cifar100_baseline_steplr_e_100_b_128.log

# vanilla       warmup_coslr  epoch     100     batchsize 128
python main_cifar100.py --gpu_device 1 --dist-url 'tcp://localhost:10007' --dist-backend nccl --multiprocessing-distributed --world-size 1 --rank 0 --epoch 100 -b 128 --lr 0.1 --wd 5e-4 --arch densenet121 --scheduler warmup_coslr --vanilla --workers 16 --seed 0 2>&1 | tee densenet121_cifar100_baseline_coslr_e_100_b_128.log

# vanilla       steplr        epoch     100     batchsize 256
python main_cifar100.py --gpu_device 0 --dist-url 'tcp://localhost:10007' --dist-backend nccl --multiprocessing-distributed --world-size 1 --rank 0 --epoch 100 -b 256 --lr 0.1 --wd 5e-4 --arch densenet121 --scheduler steplr --vanilla --workers 16 --seed 0 2>&1 | tee densenet121_cifar100_baseline_steplr_e_100_b_256.log

# vanilla       warmup_coslr  epoch     100     batchsize 256
python main_cifar100.py --gpu_device 1 --dist-url 'tcp://localhost:10007' --dist-backend nccl --multiprocessing-distributed --world-size 1 --rank 0 --epoch 100 -b 256 --lr 0.1 --wd 5e-4 --arch densenet121 --scheduler warmup_coslr --vanilla --workers 16 --seed 0 2>&1 | tee densenet121_cifar100_baseline_coslr_e_100_b_256.log



# fdq lb 8 hq 2 steplr        epoch     200     batchsize 128
python main_cifar100.py --gpu_device 1 --dist-url 'tcp://localhost:10007' --dist-backend nccl --multiprocessing-distributed --world-size 1 --rank 0 --epoch 200 -b 128 --lr 0.1 --wd 5e-4 --arch densenet121 --scheduler steplr --lfc_block 8 --hfc_bit_num 2 --workers 16 --seed 0 2>&1 | tee densenet121_cifar100_lb_8_hq_2_steplr_e_200_b_128.log

# fdq lb 8 hq 2 warmup_coslr  epoch     200     batchsize 128
python main_cifar100.py --gpu_device 0 --dist-url 'tcp://localhost:10003' --dist-backend nccl --multiprocessing-distributed --world-size 1 --rank 0 --epoch 200 -b 128 --lr 0.1 --wd 5e-4 --arch densenet121 --scheduler warmup_coslr --lfc_block 8 --hfc_bit_num 2 --workers 16 --seed 2>&1 | tee densenet121_cifar100_lb_8_hq_2_coslr_e_200_b_128.log

# fdq lb 8 hq 2 steplr        epoch     200     batchsize 256
python main_cifar100.py --gpu_device 3 --dist-url 'tcp://localhost:10008' --dist-backend nccl --multiprocessing-distributed --world-size 1 --rank 0 --epoch 200 -b 256 --lr 0.1 --wd 5e-4 --arch densenet121 --scheduler steplr --lfc_block 8 --hfc_bit_num 2 --workers 16 --seed 0 2>&1 | tee densenet121_cifar100_lb_8_hq_2_steplr_e_200_b_256.log

# fdq lb 8 hq 2 warmup_coslr  epoch     200     batchsize 256
python main_cifar100.py --gpu_device 1 --dist-url 'tcp://localhost:10006' --dist-backend nccl --multiprocessing-distributed --world-size 1 --rank 0 --epoch 200 -b 256 --lr 0.1 --wd 5e-4 --arch densenet121 --scheduler warmup_coslr --lfc_block 8 --hfc_bit_num 2 --workers 16 --seed 0 2>&1 | tee densenet121_cifar100_lb_8_hq_2_coslr_e_200_b_256.log

# fdq lb 8 hq 2 steplr        epoch     100     batchsize 128
python main_cifar100.py --gpu_device 1 --dist-url 'tcp://localhost:10006' --dist-backend nccl --multiprocessing-distributed --world-size 1 --rank 0 --epoch 100 -b 128 --lr 0.1 --wd 5e-4 --arch densenet121 --scheduler steplr --lfc_block 8 --hfc_bit_num 2 --workers 16 --seed 0 2>&1 | tee densenet121_cifar100_lb_8_hq_2_coslr_e_100_b_128.log

# fdq lb 8 hq 2 warmup_coslr  epoch     100     batchsize 128
python main_cifar100.py --gpu_device 2 --dist-url 'tcp://localhost:10006' --dist-backend nccl --multiprocessing-distributed --world-size 1 --rank 0 --epoch 100 -b 128 --lr 0.1 --wd 5e-4 --arch densenet121 --scheduler warmup_coslr --lfc_block 8 --hfc_bit_num 2 --workers 16 --seed 0 2>&1 | tee densenet121_cifar100_lb_8_hq_2_coslr_e_100_b_128.log

# fdq lb 8 hq 2 steplr        epoch     100     batchsize 256
python main_cifar100.py --gpu_device 1 --dist-url 'tcp://localhost:10006' --dist-backend nccl --multiprocessing-distributed --world-size 1 --rank 0 --epoch 100 -b 256 --lr 0.1 --wd 5e-4 --arch densenet121 --scheduler steplr --lfc_block 8 --hfc_bit_num 2 --workers 16 --seed 0 2>&1 | tee densenet121_cifar100_lb_8_hq_2_coslr_e_100_b_256.log

# fdq lb 8 hq 2 warmup_coslr  epoch     100     batchsize 256
python main_cifar100.py --gpu_device 2 --dist-url 'tcp://localhost:10006' --dist-backend nccl --multiprocessing-distributed --world-size 1 --rank 0 --epoch 100 -b 256 --lr 0.1 --wd 5e-4 --arch densenet121 --scheduler warmup_coslr --lfc_block 8 --hfc_bit_num 2 --workers 16 --seed 0 2>&1 | tee densenet121_cifar100_lb_8_hq_2_coslr_e_100_b_256.log








