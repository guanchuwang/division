## Single-processing 1 GPU

python main_cifar100.py --gpu_device 2 --gpu 0 --epoch 100 --lr 0.1 --arch resnet18 --vanilla --workers 16 -b 256 --seed 0 2>&1 | tee resnet18_cifar100_baseline.log

python main_cifar100.py --gpu_device 1 --gpu 0 --epoch 100 --lr 0.1 --arch resnet18 --lfc_block 8 --hfc_bit_num 2 --workers 16 -b 256 --seed 0 2>&1 | tee resnet18_cifar100_lb_8_hq_2.log


## nn Parallel 1 GPU

python main_cifar100.py --gpu_device 1 --dist-url 'tcp://localhost:10009' --dist-backend nccl --multiprocessing-distributed --world-size 1 --rank 0 --epoch 200 -b 128 --lr 0.1 --wd 5e-4 --arch resnet18 --scheduler steplr --vanilla --workers 16 --seed 0 2>&1 | tee resnet18_cifar100_baseline_steplr_e_200.log

python main_cifar100.py --gpu_device 3 --dist-url 'tcp://localhost:10005' --dist-backend nccl --multiprocessing-distributed --world-size 1 --rank 0 --epoch 200 -b 128 --lr 0.1 --wd 5e-4 --arch resnet18 --scheduler warmup_coslr --vanilla --workers 16 --seed 0 2>&1 | tee resnet18_cifar100_baseline_coslr_e_200.log

python main_cifar100.py --gpu_device 0 --dist-url 'tcp://localhost:10007' --dist-backend nccl --multiprocessing-distributed --world-size 1 --rank 0 --epoch 100 -b 128 --lr 0.1 --wd 5e-4 --arch resnet18 --scheduler steplr --vanilla --workers 16 --seed 0 2>&1 | tee resnet18_cifar100_baseline_steplr_e_100.log

python main_cifar100.py --gpu_device 1 --dist-url 'tcp://localhost:10007' --dist-backend nccl --multiprocessing-distributed --world-size 1 --rank 0 --epoch 100 -b 128 --lr 0.1 --wd 5e-4 --arch resnet18 --scheduler warmup_coslr --vanilla --workers 16 --seed 0 2>&1 | tee resnet18_cifar100_baseline_coslr_e_100.log



python main_cifar100.py --gpu_device 0 --dist-url 'tcp://localhost:10003' --dist-backend nccl --multiprocessing-distributed --world-size 1 --rank 0 --epoch 200 -b 128 --lr 0.1 --wd 5e-4 --arch resnet18 --scheduler steplr --lfc_block 8 --hfc_bit_num 2 --workers 16 --seed 0 2>&1 | tee resnet18_cifar100_lb_8_hq_2_steplr_e_200.log

python main_cifar100.py --gpu_device 3 --dist-url 'tcp://localhost:10006' --dist-backend nccl --multiprocessing-distributed --world-size 1 --rank 0 --epoch 200 -b 128 --lr 0.1 --wd 5e-4 --arch resnet18 --scheduler warmup_coslr --lfc_block 8 --hfc_bit_num 2 --workers 16 --seed 0 2>&1 | tee resnet18_cifar100_lb_8_hq_2_coslr_e_200.log

python main_cifar100.py --gpu_device 2 --dist-url 'tcp://localhost:10006' --dist-backend nccl --multiprocessing-distributed --world-size 1 --rank 0 --epoch 100 -b 128 --lr 0.1 --wd 5e-4 --arch resnet18 --scheduler steplr --lfc_block 8 --hfc_bit_num 2 --workers 16 --seed 0 2>&1 | tee resnet18_cifar100_lb_8_hq_2_coslr_e_100.log

python main_cifar100.py --gpu_device 2 --dist-url 'tcp://localhost:10006' --dist-backend nccl --multiprocessing-distributed --world-size 1 --rank 0 --epoch 100 -b 128 --lr 0.1 --wd 5e-4 --arch resnet18 --scheduler warmup_coslr --lfc_block 8 --hfc_bit_num 2 --workers 16 --seed 0 2>&1 | tee resnet18_cifar100_lb_8_hq_2_coslr_e_100.log


