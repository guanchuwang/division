## Single-processing 1 GPU

python main_dctb_cifar100.py --gpu_device 1 --gpu 0 --epoch 200 --conv_window_size 0.1 --bn_window_size 0.1 --lr 0.1 --arch resnet34 --hfc_bit_num 2 --seed 0 2>&1 | tee mdct_test.log

## Multi-processing 1 GPU

python main_dctb_cifar100.py --gpu_device 1 --gpu 0 --dist-url 'tcp://localhost:10002' --dist-backend nccl --multiprocessing-distributed --world-size 1 --rank 0 --epoch 200 --conv_window_size 0.1 --bn_window_size 0.1 --lr 0.1 --arch resnet34 --hfc_bit_num 2 --seed 0 2>&1 | tee mdct_test.log

## Multi-processing 4 GPU

python main_dctb_cifar100.py --gpu_device 0 1 2 4 --dist-url 'tcp://localhost:10002' --dist-backend nccl --multiprocessing-distributed --world-size 1 --rank 0 --epoch 200 --conv_window_size 0.1 --bn_window_size 0.1 --lr 0.1 --arch resnet34 --hfc_bit_num 2 --seed 0 2>&1 | tee mdct_test.log

