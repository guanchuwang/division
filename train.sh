python main_dctb_cifar100.py --gpu_device 0 1 2 3 --dist-url 'tcp://localhost:10002' --dist-backend nccl --multiprocessing-distributed --world-size 1 --rank 0 --epoch 200 --conv_window_size 1 --bn_window_size 1 --lr 0.1 --arch resnet101 2>&1 | tee cifar100_dctb_resnet101_lr_01_convwin_1_bnwin_1.log