## Single-processing 1 GPU

python main_cifar10.py --gpu_device 2 --gpu 0 --epoch 50 --vanilla --lr 0.1 --arch resnet18 --workers 16 -b 256 --seed 0 2>&1 | tee resnet18_cifar10_baseline.log

python main_cifar10.py --gpu_device 3 --gpu 0 --epoch 50 --conv_window_size 0.2 --bn_window_size 0.2 --lr 0.1 --arch resnet18 --hfc_bit_num 0 --workers 16 -b 256 --seed 0 2>&1 | tee resnet.log

python main_cifar10.py --gpu_device 3 --gpu 0 --epoch 50 --conv_window_size 0.1 --bn_window_size 0.1 --lr 0.1 --hfc_bit_num 2 --arch resnet18 --workers 16 -b 256 --seed 0 2>&1 | tee resnet18_cifar10_win_0_2_hf_2.log

## simulate

python main_cifar10.py --gpu_device 3 --gpu 0 --epoch 50 --lr 0.1 --arch resnet18 --conv_window_size 0.1 --bn_window_size 0.1 --hfc_bit_num 2 --workers 16 -b 256 --seed 0 --simulate 2>&1 | tee resnet18_cifar10_win_0_2_hf_2_3.log

## Single-processing 1 GPU simulate

python main_cifar10.py --gpu_device 3 --gpu 0 --epoch 200 --conv_window_size 0.2 --bn_window_size 0.2 --lr 0.1 --arch resnet18 --hfc_bit_num 0 --workers 16 -b 256 --rm_hfc --simulate --seed 0 2>&1 | tee resnet18_cifar10_win_0_2_hf_0.log


## Multi-processing 1 GPU

python main_cifar10.py --gpu_device 1 --gpu 0 --dist-url 'tcp://localhost:10002' --dist-backend nccl --multiprocessing-distributed --world-size 1 --rank 0 --epoch 200 --conv_window_size 0.1 --bn_window_size 0.1 --lr 0.1 --arch resnet18 --hfc_bit_num 2 --workers 16 --seed 0 2>&1 | tee mdct_test.log

## Multi-processing 4 GPU

python main_cifar10.py --gpu_device 0 1 2 3 --dist-url 'tcp://localhost:10002' --dist-backend nccl --multiprocessing-distributed --world-size 1 --rank 0 --epoch 200 --conv_window_size 0.1 --bn_window_size 0.1 --lr 0.1 --arch resnet18 --hfc_bit_num 2 --workers 16 --seed 0 2>&1 | tee mdct_test.log



