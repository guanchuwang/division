#!/bin/bash
## division training


script="mem_speed_benchmark.py"
data=<Your Dataset Folder>
model_resnet50="resnet50"
model_wrn50_2="wide_resnet50_2"

python $script --amp --debug_option speed --gpu 0 -a $model_resnet50 --data $data -b 64  --log_fname speed_debug_resnet50_division_b64.json
python $script --amp --debug_option speed --gpu 0 -a $model_resnet50 --data $data -b 128 --log_fname speed_debug_resnet50_division_b128.json
python $script --amp --debug_option speed --gpu 0 -a $model_resnet50 --data $data -b 256 --log_fname speed_debug_resnet50_division_b256.json
python $script --amp --debug_option speed --gpu 0 -a $model_resnet50 --data $data -b 512 --log_fname speed_debug_resnet50_division_b512.json

python $script --amp --debug_option speed --gpu 0 -a $model_wrn50_2 --data $data -b 64  --log_fname speed_debug_wrn50_2_division_b64.json
python $script --amp --debug_option speed --gpu 0 -a $model_wrn50_2 --data $data -b 128 --log_fname speed_debug_wrn50_2_division_b128.json
python $script --amp --debug_option speed --gpu 0 -a $model_wrn50_2 --data $data -b 256 --log_fname speed_debug_wrn50_2_division_b256.json
python $script --amp --debug_option speed --gpu 0 -a $model_wrn50_2 --data $data -b 512 --log_fname speed_debug_wrn50_2_division_b512.json












