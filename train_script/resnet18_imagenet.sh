## Multi-processing 2 GPU

python main_imagenet.py --gpu_device 2 3 --dist-url 'tcp://localhost:10002' --dist-backend nccl --multiprocessing-distributed --world-size 1 --rank 0 --epoch 200 --lr 0.05 --wd 4e-5 --arch resnet18 --conv_window_size 0.1 --bn_window_size 0.1 --hfc_bit_num 2 --lmdb_dataset --workers 16 --seed 0 2>&1 | tee mdct_test.log

python main_imagenet.py --gpu_device 2 3 --dist-url 'tcp://localhost:10002' --dist-backend nccl --multiprocessing-distributed --world-size 1 --rank 0 --epoch 200 --lr 0.05 --wd 4e-5 --arch resnet18 --lfc_block 8 --hfc_bit_num 2 --lmdb_dataset --workers 16 --seed 0 --print-freq 10



