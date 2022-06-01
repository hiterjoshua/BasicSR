#!/usr/bin/env bash

# This script is used to test pretrained models. More specific setttings can
# be found and modified in a test.yml file under the experiment dir

# basic settings
degradation=$1
model=$2
gpu_id=5
exp_id=fnet0529three_enhance   #001  fnet1124three


# run
python ./codes/main.py \
  --exp_dir ./experiments_${degradation}/${model}/${exp_id} \
  --mode test \
  --model ${model} \
  --opt test.yml \
  --gpu_id ${gpu_id}
