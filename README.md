# A Concise Framework of Memory Efficient Training via Dual Activation Precision

## About This Wrok

### Research Motivation

Existing work of activation compressed training (ACT) relies on searching for optimal bit-width during DNN training to reduce the quantization noise, which makes the procedure complicated and less transparent.

In our project, we have an instructive observation: **DNN backward propagation mainly utilizes the low-frequency component (LFC) of the activation maps, while the majority of memory is for caching the high-frequency component (HFC) during the training.** 
This indicates the HFC of activation maps is highly redundant and compressible during DNN training.
To this end, we propose a concise and transparent framework to reduce the memory cost of DNN training, Dual ActIVation PrecISION (DIVISION).
During the training, DIVISION preserves the high-precision copy of LFC and compresses the HFC into a light-weight copy with low numerical precision.
This can significantly reduce the memory cost without negatively affecting the precision of DNN backward propagation such that it maintains competitive model accuracy.

### DIVISION Framework
The framework of DIVISION is shown in the following figure. After the feed-forward operation of each layer, DIVISION estimates the LFC and compresses the HFC into a low-precision copy such that the total memory cost is significantly decreased after the compression. Before the backward propagation of each layer, the low-precision HFC is decompressed and combined with LFC to reconstruct the activation map. 
<div align=center>
<img width="1000" height="160" src="https://anonymous.4open.science/r/division-5CC0/figure/FDMP_forward_backward.png">
</div>

### Advantages of DIVISION
Compared with the existing frameworks that integrate searching into learning, DIVISION has a more concise compressor and decompressor, speeding up the procedure of ACT.
More importantly, it reveals the compressible (HFC) and non-compressible factors (LFC) during DNN training, improving the transparency of ACT. 


## Dependency
````angular2html
python >= 3.6
torch >= 1.10.2+cu113
torchvision >= 0.11.2+cu113
lmdb >= 1.3.0
pyarrow >= 6.0.1
````
## Run this Repo

### Prepare the ImageNet dataset

First, download the ImageNet dataset from [image-net.org](https://image-net.org/challenges/LSVRC/index.php). Then, generate the LMDB-format ImageNet dataset by running:
````angular2html
cd data
python folder2lmdb.py -f [Your ImageNet folder] -s train
python folder2lmdb.py -f [Your ImageNet folder] -s split
cd ../
````
Transformation to the LMDB-format aims to reduce the communication cost. It will be fine to use the original dataset.

### Generate the CUDA executive (*.so) file 

Generate the "*.so" file by running:
````angular2html
cd cpp_extension
python setup.py build_ext --inplace
cd ../
````
You should find a "backward_func.cpython-36m-x86_64-linux-gnu.so", 
"calc_precision.cpython-36m-x86_64-linux-gnu.so", 
"minimax.cpython-36m-x86_64-linux-gnu.so", 
and "quantization.cpython-36m-x86_64-linux-gnu.so" in the "cpp_extension" folder.


### Train a deep neural network via DIVISION

Train a DNN using DIVISION by running the bash commend:
````angular2html
bash scripts/resnet18_cifar10_division.sh
bash scripts/resnet164_cifar10_division.sh
bash scripts/densenet121_cifar100_division.sh
bash scripts/resnet164_cifar100_division.sh
bash scripts/resnet50_division.sh
bash scripts/densenet161_division.sh
````

Check the model accuracy and training log files.

|  Dataset   |  Architecture | Top-1 Validation Accuracy | Normal Training Accuracy | Log file | 
| :---: | :---: | :---: | :---: | :---: |
| CIFAR-10   | ResNet-18     | 94.7 | 94.9 | [LOG](https://anonymous.4open.science/r/division-5CC0/log/resnet18_cifar10_lb_8_hq_2_coslr_e_100_b_256.txt)          |   
| CIFAR-10   | ResNet-164    | 94.5 | 94.9 | [LOG](https://anonymous.4open.science/r/division-5CC0/log/resnet164_cifar10_division_coslr_e_100_b_256.txt)          | 
| CIFAR-100  | DenseNet-121  | 79.5 | 79.8 | [LOG](https://anonymous.4open.science/r/division-5CC0/log/densenet121_cifar100_lb_8_hq_2_coslr_e_120_b_256_lr01.txt) | 
| CIFAR-100  | ResNet-164    | 76.9 | 77.3 | [LOG](https://anonymous.4open.science/r/division-5CC0/log/resnet164_cifar100_division_coslr_e_200_b_256_lr015.txt)   | 
| ImageNet   | ResNet-50     | 75.9 | [76.2](https://paperswithcode.com/lib/torchvision/resnet) | [LOG](https://anonymous.4open.science/r/division-5CC0/log/log_resnet50_division_B_8_Q_2.txt)                         | 
| ImageNet   | DenseNet-161  | 77.6 | [77.6](https://paperswithcode.com/lib/torchvision/densenet) | [LOG](https://anonymous.4open.science/r/division-5CC0/log/log_densenet161_division_B_8_Q_2.txt)                      | 


### Benchmark the training memory cost by running the bash commend:
````angular2html
bash scripts/mem_benchmark.sh
````

### Benchmark the training throughput of DIVISION

Benchmark the training throughput by running the bash commend:
````angular2html
bash scripts/speed_benchmark.sh
````

## Reproduce our experiment results


### Model Accuracy

<div align=center>
<img width="250" height="200" src="https://anonymous.4open.science/r/division-5CC0/figure/acc_vs_blpa.png">
<img width="350" height="200" src="https://anonymous.4open.science/r/division-5CC0/figure/acc_vs_acgc.png">
<img width="420" height="200" src="https://anonymous.4open.science/r/division-5CC0/figure/acc_vs_actnn.png">
</div>



### Training Memory Cost
<div align=center>
<img width="650" height="250" src="https://anonymous.4open.science/r/division-5CC0/figure/memory_cost_table.png">
</div>

### Training Throughput
<div align=center>
<img width="350" height="250" src="https://anonymous.4open.science/r/division-5CC0/figure/resnet50_throughput_imagenet.png">
<img width="350" height="250" src="https://anonymous.4open.science/r/division-5CC0/figure/wrn50_2_throughput_imagenet.png">
</div>

### Overall Evaluation
<div align=center>
<img width="470" height="250" src="https://anonymous.4open.science/r/division-5CC0/figure/radar.png">
</div>

## Acknowledgment

The LMDB-format data loading is developed based on the opensource repo of [Efficient-PyTorch](https://github.com/Lyken17/Efficient-PyTorch).
The cuda kernel of activation map quantization is developed based on the opensource repo of [ActNN](https://arxiv.org/abs/2104.14129).

Thanks those teams for their contribution to the ML community!

