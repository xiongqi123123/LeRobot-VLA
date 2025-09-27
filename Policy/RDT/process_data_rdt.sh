#!/bin/bash

data_dir=${1}
output_dir=${2}
episode_num=${3}
gpu_id=${4}

export CUDA_VISIBLE_DEVICES=${gpu_id}
python scripts/process_data_lerobot.py --data_dir $data_dir --output_dir $output_dir --episode_num $episode_num --gpu $gpu_id