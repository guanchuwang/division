# A Concise Framework of Memory Efficient Training via Dual Activation Precision

## Research Motivation


## Dependency
````angular2html
python >= 3.6
torch >= 1.10.2+cu113
torchvision >= 0.11.2+cu113
lmdb >= 1.3.0
pyarrow >= 6.0.1
````

## Prepare the ImageNet dataset

First, download the ImageNet dataset from [image-net.org](https://image-net.org/challenges/LSVRC/index.php). Then, generate the LMDB-format ImageNet dataset:
````angular2html
cd data
python folder2lmdb.py -f [Your ImageNet folder] -s train
python folder2lmdb.py -f [Your ImageNet folder] -s split
cd ../
````
Transformation to the LMDB-format aims to reduce the communication cost. It will be fine to use the original dataset.

## Train a deep neural network via DIVISION

Take your dataset folder into <Your Dataset Folder>. Then, run the bash commend:
````angular2html
bash script/resnet18_cifar10_division.sh
bash script/resnet164_cifar10_division.sh
bash script/densenet121_cifar100_division.sh
bash script/resnet164_cifar100_division.sh
bash script/resnet50_division.sh
bash script/densenet161_division.sh
````

## Benchmark the training memory cost of DIVISION

Take your dataset folder into <Your Dataset Folder>. Then, run the bash commend:
````angular2html
bash script/mem_benchmark.sh
````

## Benchmark the training throughput of DIVISION

Take your dataset folder into <Your Dataset Folder>. Then, run the bash commend:
````angular2html
bash script/speed_benchmark.sh
````

## Reproduce our experiment results:


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
