## Single-processing 1 GPU

python main_cifar100.py --gpu_device 3 --gpu 0 --epoch 100 --lr 0.05 --wd 4e-5 --arch mobilenet_v2 --vanilla --workers 16 -b 256 --seed 0 2>&1 | tee mobilenet_cifar100_baseline.log

python main_cifar100.py --gpu_device 3 --gpu 0 --epoch 100 --lr 0.1 --arch mobilenet_v2 --conv_window_size 0.1 --bn_window_size 0.1 --hfc_bit_num 2 --workers 16 -b 256 --seed 0 2>&1 | tee mobilenet_cifar100_win_0_2_hq_2.log


## Single-processing 1 GPU simulate

python main_cifar100.py --gpu_device 3 --gpu 0 --epoch 100 --lr 0.1 --arch mobilenet_v2 --simulate --conv_window_size 0.1 --bn_window_size 0.1 --hfc_bit_num 2 --workers 16 -b 256 --seed 0 2>&1 | tee mobilenet_cifar100_win_0_2_hq_2.log


## Multi-processing 4 GPU

python main_cifar100.py --gpu_device 0 1 2 3 --dist-url 'tcp://localhost:10002' --dist-backend nccl --multiprocessing-distributed --world-size 1 --rank 0 --epoch 100 --lr 0.1 --arch mobilenet_v2 --conv_window_size 0.1 --bn_window_size 0.1 --hfc_bit_num 2 --workers 16 --seed 0 2>&1 | tee mdct_test.log



