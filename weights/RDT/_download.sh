#!/bin/bash


export HF_ENDPOINT=https://hf-mirror.com

huggingface-cli download google/t5-v1_1-xxl --local-dir t5-v1_1-xxl
huggingface-cli download google/siglip-so400m-patch14-384 --local-dir siglip-so400m-patch14-384
huggingface-cli download robotics-diffusion-transformer/rdt-1b --local-dir rdt-1b
huggingface-cli download robotics-diffusion-transformer/rdt-170m --local-dir rdt-170m