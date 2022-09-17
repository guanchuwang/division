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


* Download the ImageNet dataset from [image-net.org](https://image-net.org/challenges/LSVRC/index.php). 
* Generate the LMDB-format ImageNet dataset:
````angular2html
cd data
python folder2lmdb.py -f [Your ImageNet folder] -s train
python folder2lmdb.py -f [Your ImageNet folder] -s split
cd ../
````
Transformation to the LMDB-format aims to reduce the communication cost. It will be fine to use the original dataset.

## Reproduce our experiment results:


### Model Accuracy


### Training Memory Cost


### Training Throughput

