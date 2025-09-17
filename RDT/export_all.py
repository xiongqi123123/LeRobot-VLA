# Copyright (c) 2025, Cauchy WuChao D-Robotics.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import re
import yaml
import logging
import argparse
from time import time
from collections import OrderedDict

import cv2
import numpy as np
import torch
import h5py
from PIL import Image as PImage

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms

from diffusers.schedulers.scheduling_ddpm import DDPMScheduler
from diffusers.schedulers.scheduling_dpmsolver_multistep import DPMSolverMultistepScheduler

from scripts.agilex_model import create_model
from configs.state_vec import STATE_VEC_IDX_MAPPING
from models.hub_mixin import CompatiblePyTorchModelHubMixin
from models.rdt.blocks import (FinalLayer, RDTBlock, TimestepEmbedder, get_1d_sincos_pos_embed_from_grid, get_multimodal_cond_pos_embed)
from models.multimodal_encoder.siglip_encoder import SiglipVisionTower
from models.multimodal_encoder.t5_encoder import T5Embedder


# taskset -c 0-7 docker run [--gpus all] -it -v <ws>:/open_explorer ai_toolchain_ubuntu_22_s100_gpu:v3.2.0


logging.basicConfig(
    level = logging.DEBUG,
    format = '[%(name)s] [%(asctime)s.%(msecs)03d] [%(levelname)s] %(message)s',
    datefmt='%H:%M:%S')
logger = logging.getLogger("RDK_RDT")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--export_path', type=str, default="rdt_export_ws_test", help="")
    parser.add_argument('--config_path', type=str, default="configs/base.yaml", help="")
    parser.add_argument('--pretrained_vision_encoder', type=str, default="../weights/RDT/siglip-so400m-patch14-384", help="")
    parser.add_argument('--pretrained_model', type=str, default="checkpoints/RDT170M-LeRobot/checkpoint-9000/pytorch_model/mp_rank_00_model_states.pt", help="")
    parser.add_argument('--train_data', type=str, default="training_data/RDT170M-LeRobot", help="")
    # --train_data exampe:
    # .
    # ├── adjust_bottle-demo_clean-300
    # │   ├── episode_0
    # │   │   ├── episode_0.hdf5
    # │   │   └── instructions
    # │   ├── episode_1
    # │   │   ├── episode_1.hdf5
    # │   │   └── instructions
    # ...
    # ├── place_dual_shoes-demo_clean-300
    # │   ├── episode_0
    # │   │   ├── episode_0.hdf5
    # │   │   └── instructions
    # ...
    # └── place_empty_cup-demo_clean-300
    #     ├── episode_0
    #     │   ├── episode_0.hdf5
    #     │   └── instructions
    # ...
    parser.add_argument('--num_samples', type=int, default=50, help="")
    parser.add_argument('--jobs', type=int, default=8, help="")
    parser.add_argument('--optimized_level', type=str, default="O2", help="")
    parser.add_argument('--ctrl_freq', type=int, default=25, help="")
    parser.add_argument('--left_arm_dim', type=int, default=6, help="")
    parser.add_argument('--right_arm_dim', type=int, default=6, help="")
    parser.add_argument('--cal_data_device', type=str, default='cuda:6', help="")
    parser.add_argument('--instructions_per_episode', type=int, default=1, help="")
    opt = parser.parse_args()
    logger.info(opt)

    # Create WorkSpace
    os.makedirs(opt.export_path, exist_ok=True)

    ## BPU_RDT_Policy
    bpu_rdt_name = "BPU_RDT_Policy"
    bpu_rdt_path = os.path.join(opt.export_path, bpu_rdt_name)
    os.makedirs(bpu_rdt_path, exist_ok=True)
    os.system(f"cp {opt.config_path} {bpu_rdt_path}")
    bash_build_all_name = "build_all.sh"

    ## test datas
    test_data_name = "test_data"
    test_data_path = os.path.join(opt.export_path, test_data_name)
    os.makedirs(test_data_path, exist_ok=True)

    ## instruction
    instruction_ws_name = "instructions"
    instruction_ws_path = os.path.join(opt.export_path, instruction_ws_name)
    os.makedirs(instruction_ws_path, exist_ok=True)
    for name in os.listdir(opt.train_data):
        os.makedirs(os.path.join(instruction_ws_path, name), exist_ok=True)

    ## image adaptor
    img_adaptor_ws_name = "img_adaptor_WorkSpace"
    img_adaptor_cal_name = "rdt_image_adaptor_calibration"
    img_adaptor_name = "rdt_image_adaptor.onnx"
    img_adaptor_config_name = "config.yaml"
    img_adaptor_bash_name = "build.sh"
    img_adaptor_path = os.path.join(opt.export_path, img_adaptor_ws_name, img_adaptor_name)
    img_adaptor_ws = os.path.join(opt.export_path, img_adaptor_ws_name)
    os.makedirs(img_adaptor_ws, exist_ok=True)

    global img_adaptor_cal_ws
    img_adaptor_cal_ws = os.path.join(img_adaptor_ws, img_adaptor_cal_name)
    os.makedirs(img_adaptor_cal_ws, exist_ok=True)

    ## action adaptor
    state_adaptor_name1 = "rdt_state_adaptor_1x1x256.onnx"
    state_adaptor_path1 = os.path.join(opt.export_path, bpu_rdt_name, state_adaptor_name1)
    state_adaptor_name2 = "rdt_state_adaptor_1x64x256.onnx"
    state_adaptor_path2 = os.path.join(opt.export_path, bpu_rdt_name, state_adaptor_name2)

    ## lang adaptor 
    lang_adaptor_name = "rdt_lang_adaptor.onnx"
    lang_adaptor_path = os.path.join(opt.export_path, bpu_rdt_name, lang_adaptor_name)

    ## DiT Policy
    dit_ws_name = "DiT_WorkSpace"
    dit_cal_name = "rdt_dit_calibration"
    dit_name = "rdt_dit.onnx"
    dit_config_name = "config.yaml"
    dit_json_name = "quant_config.json"
    dit_bash_name = "build.sh"
    dit_path = os.path.join(opt.export_path, dit_ws_name, dit_name)
    dit_ws = os.path.join(opt.export_path, dit_ws_name)
    os.makedirs(dit_ws, exist_ok=True)
    dit_cal_path = os.path.join(opt.export_path, dit_ws_name, dit_cal_name)
    os.makedirs(dit_cal_path, exist_ok=True)

    global dit_cal_path_x, dit_cal_path_freq, dit_cal_path_t, dit_cal_path_lang_c, dit_cal_path_img_c, dit_cal_path_lang_mask
    dit_cal_path_x = os.path.join(opt.export_path, dit_ws_name, dit_cal_name, "x")
    os.makedirs(dit_cal_path_x, exist_ok=True)
    dit_cal_path_freq = os.path.join(opt.export_path, dit_ws_name, dit_cal_name, "freq")
    os.makedirs(dit_cal_path_freq, exist_ok=True)
    dit_cal_path_t = os.path.join(opt.export_path, dit_ws_name, dit_cal_name, "t")
    os.makedirs(dit_cal_path_t, exist_ok=True)
    dit_cal_path_lang_c = os.path.join(opt.export_path, dit_ws_name, dit_cal_name, "lang_c")
    os.makedirs(dit_cal_path_lang_c, exist_ok=True)
    dit_cal_path_img_c = os.path.join(opt.export_path, dit_ws_name, dit_cal_name, "img_c")
    os.makedirs(dit_cal_path_img_c, exist_ok=True)
    dit_cal_path_lang_mask = os.path.join(opt.export_path, dit_ws_name, dit_cal_name, "lang_mask")
    os.makedirs(dit_cal_path_lang_mask, exist_ok=True)



    # Prepare Calibrate Data
    with open(opt.config_path, "r") as fp:
        config_base_yaml = yaml.safe_load(fp)

    config_base_yaml["arm_dim"] = {"left_arm_dim": opt.left_arm_dim, "right_arm_dim": opt.right_arm_dim}


    dump_model = create_dump_model(
        args=config_base_yaml,
        dtype=torch.float32,
        pretrained=opt.pretrained_model,
        pretrained_vision_encoder_name_or_path=opt.pretrained_vision_encoder,
        control_frequency=opt.ctrl_freq,
        device=opt.cal_data_device
    )

    # Prepare Calbriation Data
    # load training data    
    global dump_cnt, dump_dataset_name
    test_data_cnt = 0
    for dump_dataset_name in os.listdir(opt.train_data):
        dump_dataset_path = os.path.join(opt.train_data, dump_dataset_name)
        training_samples = get_training_samples(dump_dataset_path, num_samples=opt.num_samples, instructions_per_episode=opt.instructions_per_episode)
        for dump_cnt in range(min(opt.num_samples, len(training_samples))):
            sample = training_samples[dump_cnt]
            instruction_emb = {
                "lang_cond": sample['lang_embed'].float().cpu(),
                "lang_str": sample['lang_str']
            }
            ins_str_name = sample['lang_str'].replace(" ", "_")+"__"

            torch.save(instruction_emb, os.path.join(instruction_ws_path, dump_dataset_name, f"{ins_str_name}.pt"))

            # 兼容缺失相机：按键取值，缺失则用 [None, None]
            cam_high_imgs = sample['multi_cam_images'].get('cam_high', [None, None])
            cam_right_wrist_imgs = sample['multi_cam_images'].get('cam_right_wrist', [None, None])
            cam_left_wrist_imgs = sample['multi_cam_images'].get('cam_left_wrist', [None, None])
            image_arrs = [
                cam_high_imgs[0],
                cam_right_wrist_imgs[0],
                cam_left_wrist_imgs[0],
                cam_high_imgs[1],
                cam_right_wrist_imgs[1],
                cam_left_wrist_imgs[1],
            ]
            test_data_cnt += 1
            # 仅对存在的相机保存
            if cam_high_imgs[0] is not None:
                np.save(os.path.join(test_data_path, f"{test_data_cnt}_cam_high_0.npy"), cam_high_imgs[0])
            if cam_right_wrist_imgs[0] is not None:
                np.save(os.path.join(test_data_path, f"{test_data_cnt}_cam_right_wrist_0.npy"), cam_right_wrist_imgs[0])
            if cam_left_wrist_imgs[0] is not None:
                np.save(os.path.join(test_data_path, f"{test_data_cnt}_cam_left_wrist_0.npy"), cam_left_wrist_imgs[0])
            if cam_high_imgs[1] is not None:
                np.save(os.path.join(test_data_path, f"{test_data_cnt}_cam_high_1.npy"), cam_high_imgs[1])
            if cam_right_wrist_imgs[1] is not None:
                np.save(os.path.join(test_data_path, f"{test_data_cnt}_cam_right_wrist_1.npy"), cam_right_wrist_imgs[1])
            if cam_left_wrist_imgs[1] is not None:
                np.save(os.path.join(test_data_path, f"{test_data_cnt}_cam_left_wrist_1.npy"), cam_left_wrist_imgs[1])
            images = [PImage.fromarray(arr) if arr is not None else None for arr in image_arrs]
            proprio = torch.from_numpy(sample['joints']).float().unsqueeze(0).to(opt.cal_data_device) 
            np.save(os.path.join(test_data_path, f"{test_data_cnt}_joints.npy"), sample['joints'])
            lang_embeddings = sample['lang_embed'].float().unsqueeze(0).to(opt.cal_data_device) 
            torch.save(lang_embeddings, os.path.join(test_data_path, f"{test_data_cnt}_lang_embeddings.pt"))
            dump_model.reset()
            begin_time = time()
            actions = dump_model.step(proprio=proprio, images=images, text_embeds=lang_embeddings).squeeze(0).cpu().numpy()
            np.save(os.path.join(test_data_path, f"{test_data_cnt}_actions.npy"), actions)
            logger.debug(f"Dump: Cost {(1000*(time() - begin_time)):.1f} ms, cnt: {dump_cnt}, name: {dump_dataset_name}")
    logger.info("End Generate Calibration Data.")
    del dump_model

    # Load RDT Policy: CPU Model For ONNX Export
    with open(opt.config_path, "r") as fp:
        config_base_yaml = yaml.safe_load(fp)

    config_base_yaml["arm_dim"] = {"left_arm_dim": opt.left_arm_dim, "right_arm_dim": opt.right_arm_dim}

    model = create_model(
        args=config_base_yaml,
        dtype=torch.float32,
        pretrained=opt.pretrained_model,
        pretrained_vision_encoder_name_or_path=opt.pretrained_vision_encoder,
        control_frequency=opt.ctrl_freq,
        device="cpu"
    )

    bash_build_all = ""

    # image adaptor: ONNX Model
    m = model.policy.img_adaptor
    m.eval()

    input_data = torch.randn(1, 4374, 1152)  # 假设批量大小为1
    output = m(input_data)

    torch.onnx.export(
        m,                            # 要转换的模型
        input_data,                   # 模型的输入
        img_adaptor_path,             # 输出文件名
        export_params=True,           # 存储训练后的参数
        opset_version=17,             # ONNX版本（降低以兼容 ReduceMean axes 属性）
        do_constant_folding=True,     # 是否执行常量折叠优化
        input_names=['img_tokens'],   # 输入节点名称
        output_names=['adpated_img'], # 输出节点名称
        dynamic_axes=None,
        verbose=False
    )

    logger.info("Export RDT [img_adaptor] Model Success.")

    yaml_str = f'''
model_parameters:
    onnx_model: '{img_adaptor_name}'
    march: nash-m
    layer_out_dump: False
    working_dir: bpu_output
    output_model_file_prefix: rdt_img_adaptor
    enable_vpu: True
input_parameters:
    input_name: ''
    input_type_rt: 'featuremap;'
    input_layout_rt: 'NCHW;'
    input_type_train: 'featuremap;'
    input_layout_train: 'NCHW;'
    norm_type: 'no_preprocess;'
calibration_parameters:
    cal_data_dir: '{img_adaptor_cal_name}'
    cal_data_type: 'float32'
    calibration_type: 'default'
    # quant_config: {{"op_config": {{"softmax": {{"qtype": "int8"}}}}}}
    quant_config: {{
        "model_config": {{
            "all_node_type": "int16",
            "model_output_type": "int16",
        }}
    }}
compiler_parameters:
    extra_params: {{'input_no_padding': True, 'output_no_padding': True}}
    jobs: {opt.jobs}
    compile_mode: 'latency'
    debug: True
    advice: 1
    optimize_level: 'O2'
    core_num: 2
    '''

    with open(os.path.join(opt.export_path, img_adaptor_ws_name, img_adaptor_config_name), "w", encoding="utf-8") as f:
        f.write(yaml_str)

    bash_str = f'''
    hb_compile --config {img_adaptor_config_name}
    cp bpu_output/*.hbm ../{bpu_rdt_name}/
    '''

    with open(os.path.join(opt.export_path, img_adaptor_ws_name, img_adaptor_bash_name), "w", encoding="utf-8") as f:
        f.write(bash_str)

    bash_build_all += f"cd {img_adaptor_ws_name}" + "\n"
    bash_build_all += f"bash {img_adaptor_bash_name}" + "\n"
    bash_build_all += f"cd .." + "\n"


    # DiT

    m = model.policy.model
    m = m.eval().cpu()
    x = torch.randn(1, 65, 1024)
    freq = torch.tensor([1], dtype=torch.int32)
    t = torch.tensor([10], dtype=torch.int32)
    lang_c = torch.randn(1, 64, 1024)
    img_c = torch.randn(1, 4374, 1024)
    lang_mask = torch.ones(1, 64, dtype=torch.float32)
    dummy_inputs = (x, freq, t, lang_c, img_c, lang_mask)
    outputs = m(x, freq, t, lang_c, img_c, lang_mask)
    torch.onnx.export(
                m,                      # 要导出的模型
                dummy_inputs,               # 模型的输入
                dit_path,            # 保存路径
                # export_params=True,       # 是否导出训练参数
                opset_version=17,           # ONNX 的版本，降低以兼容 ReduceMean axes 属性
                do_constant_folding=True,   # 是否执行常量折叠优化
                input_names=["x", "freq", "t", "lang_c", "img_c", "lang_mask"],    # 输入名称
                output_names=["actions"],  # 输出名称
                verbose=False,
            )

    logger.info("Export RDT [dit] Model Success.")

    yaml_str = f'''
calibration_parameters:
    cal_data_dir: '{dit_cal_name}/x/;{dit_cal_name}/freq/;{dit_cal_name}/t/;{dit_cal_name}/lang_c/;{dit_cal_name}/img_c/;{dit_cal_name}/lang_mask/;'
    quant_config: {dit_json_name}
    run_on_cpu: '/t_embedder/Cos;/t_embedder/Sin;/freq_embedder/Cos;/freq_embedder/Sin'
compiler_parameters:
    compile_mode: latency
    core_num: 1
    debug: true
    jobs: {opt.jobs}
    max_time_per_fc: 0
    optimize_level: O2
    advice: 1
input_parameters:
    input_layout_rt: NCHW;NCHW;NCHW;NCHW;NCHW;NCHW
    input_layout_train: NCHW;NCHW;NCHW;NCHW;NCHW;NCHW
    input_name: x;freq;t;lang_c;img_c;lang_mask;
    input_shape: 1x65x1024;1;1;1x64x1024;1x4374x1024;1x64
    input_space_and_range: ''
    input_type_rt: featuremap;featuremap;featuremap;featuremap;featuremap;featuremap
    input_type_train: featuremap;featuremap;featuremap;featuremap;featuremap;featuremap
    norm_type: no_preprocess;no_preprocess;no_preprocess;no_preprocess;no_preprocess;no_preprocess
model_parameters:
    layer_out_dump: false
    debug_mode: "dump_calibration_data"
    enable_vpu: True
    march: nash-m
    onnx_model: {dit_name}
    output_model_file_prefix: rdt_dit
    working_dir: bpu_output

    '''
    with open(os.path.join(opt.export_path, dit_ws_name, dit_config_name), "w", encoding="utf-8") as f:
        f.write(yaml_str)

    json_str = '''
    {
        "model_config": {
            "all_node_type": "int16",
            "model_output_type": "float32",
            "activation": {
                "calibration_type": ["max"],
                "num_bin": [1024, 2048, 4096],
                "max_num_bin": 16384,
                "max_percentile": 1.0,
                "per_channel": true,
                "asymmetric": [true]
            },
            "weight": {
                "bias_correction": {
                    "metric": "mae"
                }
            },
            "modelwise_search": {
                "metric": "mae"
            }
        },
        "op_config": {
            "ReduceMean": {"qtype": "int16"},
            "Sub": {"qtype": "int16"},
            "Softmax": {"qtype": "int16"}
        },
        "node_config": {
            "/t_embedder/Mul": {"qtype": "float32"},
            "/t_embedder/Cos": {"qtype": "float32"},
            "/t_embedder/Sin": {"qtype": "float32"},
            "/t_embedder/Concat": {"qtype": "float32"},
            "/freq_embedder/Mul": {"qtype": "float32"},
            "/freq_embedder/Cos": {"qtype": "float32"},
            "/freq_embedder/Sin": {"qtype": "float32"},
            "/freq_embedder/Concat": {"qtype": "float32"},
            "/blocks.0/attn/MatMul": {"InputType0": "int16", "InputType1": "int16"},
            "/blocks.0/attn/MatMul_1": {"InputType0": "int16", "InputType1": "int16"},
            "/blocks.0/cross_attn/MatMul": {"InputType0": "int16", "InputType1": "int16"},
            "/blocks.0/cross_attn/MatMul_1": {"InputType0": "int16", "InputType1": "int16"},
            "/blocks.0/cross_attn/k_norm/Mul_1": {"qtype": "int16"},
            "/blocks.0/ffn/fc1/MatMul": {"qtype": "int16"},
            "/blocks.0/ffn/act/Mul": {"qtype": "int16"},
            "/blocks.0/ffn/act/Mul_1": {"qtype": "int16"},
            "/blocks.0/ffn/act/Mul_2": {"qtype": "int16"},
            "/blocks.0/ffn/act/Add": {"qtype": "int16"},
            "/blocks.0/ffn/act/Mul_3": {"qtype": "int16"},
            "/blocks.0/ffn/act/Tanh": {"qtype": "int16"},
            "/blocks.0/norm1/Mul_2": {"qtype": "int16"},
            "/blocks.0/cross_attn/k_norm/Div_1_reciprocal": {"qtype": "int16"},
            "/blocks.0/Add": {"qtype": "int16"},
            "/blocks.1/attn/MatMul": {"InputType0": "int16", "InputType1": "int16"},
            "/blocks.1/attn/MatMul_1": {"InputType0": "int16", "InputType1": "int16"},
            "/blocks.1/cross_attn/MatMul": {"InputType0": "int16", "InputType1": "int16"},
            "/blocks.1/cross_attn/MatMul_1": {"InputType0": "int16", "InputType1": "int16"},
            "/blocks.1/cross_attn/k_norm/Mul_1": {"qtype": "int16"},
            "/blocks.1/ffn/fc1/MatMul": {"qtype": "int16"},
            "/blocks.1/ffn/act/Mul": {"qtype": "int16"},
            "/blocks.1/ffn/act/Mul_1": {"qtype": "int16"},
            "/blocks.1/ffn/act/Mul_2": {"qtype": "int16"},
            "/blocks.1/ffn/act/Add": {"qtype": "int16"},
            "/blocks.1/ffn/act/Mul_3": {"qtype": "int16"},
            "/blocks.1/ffn/act/Tanh": {"qtype": "int16"},
            "/blocks.1/norm1/Mul_2": {"qtype": "int16"},
            "/blocks.1/cross_attn/k_norm/Div_1_reciprocal": {"qtype": "int16"},
            "/blocks.1/Add": {"qtype": "int16"},
            "/blocks.2/attn/MatMul": {"InputType0": "int16", "InputType1": "int16"},
            "/blocks.2/attn/MatMul_1": {"InputType0": "int16", "InputType1": "int16"},
            "/blocks.2/cross_attn/MatMul": {"InputType0": "int16", "InputType1": "int16"},
            "/blocks.2/cross_attn/MatMul_1": {"InputType0": "int16", "InputType1": "int16"},
            "/blocks.2/cross_attn/k_norm/Mul_1": {"qtype": "int16"},
            "/blocks.2/ffn/fc1/MatMul": {"qtype": "int16"},
            "/blocks.2/ffn/act/Mul": {"qtype": "int16"},
            "/blocks.2/ffn/act/Mul_1": {"qtype": "int16"},
            "/blocks.2/ffn/act/Mul_2": {"qtype": "int16"},
            "/blocks.2/ffn/act/Add": {"qtype": "int16"},
            "/blocks.2/ffn/act/Mul_3": {"qtype": "int16"},
            "/blocks.2/ffn/act/Tanh": {"qtype": "int16"},
            "/blocks.2/norm1/Mul_2": {"qtype": "int16"},
            "/blocks.2/cross_attn/k_norm/Div_1_reciprocal": {"qtype": "int16"},
            "/blocks.2/Add": {"qtype": "int16"},
            "/blocks.3/attn/MatMul": {"InputType0": "int16", "InputType1": "int16"},
            "/blocks.3/attn/MatMul_1": {"InputType0": "int16", "InputType1": "int16"},
            "/blocks.3/cross_attn/MatMul": {"InputType0": "int16", "InputType1": "int16"},
            "/blocks.3/cross_attn/MatMul_1": {"InputType0": "int16", "InputType1": "int16"},
            "/blocks.3/cross_attn/k_norm/Mul_1": {"qtype": "int16"},
            "/blocks.3/ffn/fc1/MatMul": {"qtype": "int16"},
            "/blocks.3/ffn/act/Mul": {"qtype": "int16"},
            "/blocks.3/ffn/act/Mul_1": {"qtype": "int16"},
            "/blocks.3/ffn/act/Mul_2": {"qtype": "int16"},
            "/blocks.3/ffn/act/Add": {"qtype": "int16"},
            "/blocks.3/ffn/act/Mul_3": {"qtype": "int16"},
            "/blocks.3/ffn/act/Tanh": {"qtype": "int16"},
            "/blocks.3/norm1/Mul_2": {"qtype": "int16"},
            "/blocks.3/cross_attn/k_norm/Div_1_reciprocal": {"qtype": "int16"},
            "/blocks.3/Add": {"qtype": "int16"},
            "/blocks.4/attn/MatMul": {"InputType0": "int16", "InputType1": "int16"},
            "/blocks.4/attn/MatMul_1": {"InputType0": "int16", "InputType1": "int16"},
            "/blocks.4/cross_attn/MatMul": {"InputType0": "int16", "InputType1": "int16"},
            "/blocks.4/cross_attn/MatMul_1": {"InputType0": "int16", "InputType1": "int16"},
            "/blocks.4/cross_attn/k_norm/Mul_1": {"qtype": "int16"},
            "/blocks.4/ffn/fc1/MatMul": {"qtype": "int16"},
            "/blocks.4/ffn/act/Mul": {"qtype": "int16"},
            "/blocks.4/ffn/act/Mul_1": {"qtype": "int16"},
            "/blocks.4/ffn/act/Mul_2": {"qtype": "int16"},
            "/blocks.4/ffn/act/Add": {"qtype": "int16"},
            "/blocks.4/ffn/act/Mul_3": {"qtype": "int16"},
            "/blocks.4/ffn/act/Tanh": {"qtype": "int16"},
            "/blocks.4/norm1/Mul_2": {"qtype": "int16"},
            "/blocks.4/cross_attn/k_norm/Div_1_reciprocal": {"qtype": "int16"},
            "/blocks.4/Add": {"qtype": "int16"},
            "/blocks.5/attn/MatMul": {"InputType0": "int16", "InputType1": "int16"},
            "/blocks.5/attn/MatMul_1": {"InputType0": "int16", "InputType1": "int16"},
            "/blocks.5/cross_attn/MatMul": {"InputType0": "int16", "InputType1": "int16"},
            "/blocks.5/cross_attn/MatMul_1": {"InputType0": "int16", "InputType1": "int16"},
            "/blocks.5/cross_attn/k_norm/Mul_1": {"qtype": "int16"},
            "/blocks.5/ffn/fc1/MatMul": {"qtype": "int16"},
            "/blocks.5/ffn/act/Mul": {"qtype": "int16"},
            "/blocks.5/ffn/act/Mul_1": {"qtype": "int16"},
            "/blocks.5/ffn/act/Mul_2": {"qtype": "int16"},
            "/blocks.5/ffn/act/Add": {"qtype": "int16"},
            "/blocks.5/ffn/act/Mul_3": {"qtype": "int16"},
            "/blocks.5/ffn/act/Tanh": {"qtype": "int16"},
            "/blocks.5/norm1/Mul_2": {"qtype": "int16"},
            "/blocks.5/cross_attn/k_norm/Div_1_reciprocal": {"qtype": "int16"},
            "/blocks.5/Add": {"qtype": "int16"},
            "/blocks.6/attn/MatMul": {"InputType0": "int16", "InputType1": "int16"},
            "/blocks.6/attn/MatMul_1": {"InputType0": "int16", "InputType1": "int16"},
            "/blocks.6/cross_attn/MatMul": {"InputType0": "int16", "InputType1": "int16"},
            "/blocks.6/cross_attn/MatMul_1": {"InputType0": "int16", "InputType1": "int16"},
            "/blocks.6/cross_attn/k_norm/Mul_1": {"qtype": "int16"},
            "/blocks.6/ffn/fc1/MatMul": {"qtype": "int16"},
            "/blocks.6/ffn/act/Mul": {"qtype": "int16"},
            "/blocks.6/ffn/act/Mul_1": {"qtype": "int16"},
            "/blocks.6/ffn/act/Mul_2": {"qtype": "int16"},
            "/blocks.6/ffn/act/Add": {"qtype": "int16"},
            "/blocks.6/ffn/act/Mul_3": {"qtype": "int16"},
            "/blocks.6/ffn/act/Tanh": {"qtype": "int16"},
            "/blocks.6/norm1/Mul_2": {"qtype": "int16"},
            "/blocks.6/cross_attn/k_norm/Div_1_reciprocal": {"qtype": "int16"},
            "/blocks.6/Add": {"qtype": "int16"},
            "/blocks.7/attn/MatMul": {"InputType0": "int16", "InputType1": "int16"},
            "/blocks.7/attn/MatMul_1": {"InputType0": "int16", "InputType1": "int16"},
            "/blocks.7/cross_attn/MatMul": {"InputType0": "int16", "InputType1": "int16"},
            "/blocks.7/cross_attn/MatMul_1": {"InputType0": "int16", "InputType1": "int16"},
            "/blocks.7/cross_attn/k_norm/Mul_1": {"qtype": "int16"},
            "/blocks.7/ffn/fc1/MatMul": {"qtype": "int16"},
            "/blocks.7/ffn/act/Mul": {"qtype": "int16"},
            "/blocks.7/ffn/act/Mul_1": {"qtype": "int16"},
            "/blocks.7/ffn/act/Mul_2": {"qtype": "int16"},
            "/blocks.7/ffn/act/Add": {"qtype": "int16"},
            "/blocks.7/ffn/act/Mul_3": {"qtype": "int16"},
            "/blocks.7/ffn/act/Tanh": {"qtype": "int16"},
            "/blocks.7/norm1/Mul_2": {"qtype": "int16"},
            "/blocks.7/cross_attn/k_norm/Div_1_reciprocal": {"qtype": "int16"},
            "/blocks.7/Add": {"qtype": "int16"},
            "/blocks.8/attn/MatMul": {"InputType0": "int16", "InputType1": "int16"},
            "/blocks.8/attn/MatMul_1": {"InputType0": "int16", "InputType1": "int16"},
            "/blocks.8/cross_attn/MatMul": {"InputType0": "int16", "InputType1": "int16"},
            "/blocks.8/cross_attn/MatMul_1": {"InputType0": "int16", "InputType1": "int16"},
            "/blocks.8/cross_attn/k_norm/Mul_1": {"qtype": "int16"},
            "/blocks.8/ffn/fc1/MatMul": {"qtype": "int16"},
            "/blocks.8/ffn/act/Mul": {"qtype": "int16"},
            "/blocks.8/ffn/act/Mul_1": {"qtype": "int16"},
            "/blocks.8/ffn/act/Mul_2": {"qtype": "int16"},
            "/blocks.8/ffn/act/Add": {"qtype": "int16"},
            "/blocks.8/ffn/act/Mul_3": {"qtype": "int16"},
            "/blocks.8/ffn/act/Tanh": {"qtype": "int16"},
            "/blocks.8/norm1/Mul_2": {"qtype": "int16"},
            "/blocks.8/cross_attn/k_norm/Div_1_reciprocal": {"qtype": "int16"},
            "/blocks.8/Add": {"qtype": "int16"},
            "/blocks.9/attn/MatMul": {"InputType0": "int16", "InputType1": "int16"},
            "/blocks.9/attn/MatMul_1": {"InputType0": "int16", "InputType1": "int16"},
            "/blocks.9/cross_attn/MatMul": {"InputType0": "int16", "InputType1": "int16"},
            "/blocks.9/cross_attn/MatMul_1": {"InputType0": "int16", "InputType1": "int16"},
            "/blocks.9/cross_attn/k_norm/Mul_1": {"qtype": "int16"},
            "/blocks.9/ffn/fc1/MatMul": {"qtype": "int16"},
            "/blocks.9/ffn/act/Mul": {"qtype": "int16"},
            "/blocks.9/ffn/act/Mul_1": {"qtype": "int16"},
            "/blocks.9/ffn/act/Mul_2": {"qtype": "int16"},
            "/blocks.9/ffn/act/Add": {"qtype": "int16"},
            "/blocks.9/ffn/act/Mul_3": {"qtype": "int16"},
            "/blocks.9/ffn/act/Tanh": {"qtype": "int16"},
            "/blocks.9/norm1/Mul_2": {"qtype": "int16"},
            "/blocks.9/cross_attn/k_norm/Div_1_reciprocal": {"qtype": "int16"},
            "/blocks.9/Add": {"qtype": "int16"},
            "/blocks.10/attn/MatMul": {"InputType0": "int16", "InputType1": "int16"},
            "/blocks.10/attn/MatMul_1": {"InputType0": "int16", "InputType1": "int16"},
            "/blocks.10/cross_attn/MatMul": {"InputType0": "int16", "InputType1": "int16"},
            "/blocks.10/cross_attn/MatMul_1": {"InputType0": "int16", "InputType1": "int16"},
            "/blocks.10/cross_attn/k_norm/Mul_1": {"qtype": "int16"},
            "/blocks.10/ffn/fc1/MatMul": {"qtype": "int16"},
            "/blocks.10/ffn/act/Mul": {"qtype": "int16"},
            "/blocks.10/ffn/act/Mul_1": {"qtype": "int16"},
            "/blocks.10/ffn/act/Mul_2": {"qtype": "int16"},
            "/blocks.10/ffn/act/Add": {"qtype": "int16"},
            "/blocks.10/ffn/act/Mul_3": {"qtype": "int16"},
            "/blocks.10/ffn/act/Tanh": {"qtype": "int16"},
            "/blocks.10/norm1/Mul_2": {"qtype": "int16"},
            "/blocks.10/cross_attn/k_norm/Div_1_reciprocal": {"qtype": "int16"},
            "/blocks.10/Add": {"qtype": "int16"},
            "/blocks.11/attn/MatMul": {"InputType0": "int16", "InputType1": "int16"},
            "/blocks.11/attn/MatMul_1": {"InputType0": "int16", "InputType1": "int16"},
            "/blocks.11/cross_attn/MatMul": {"InputType0": "int16", "InputType1": "int16"},
            "/blocks.11/cross_attn/MatMul_1": {"InputType0": "int16", "InputType1": "int16"},
            "/blocks.11/cross_attn/k_norm/Mul_1": {"qtype": "int16"},
            "/blocks.11/ffn/fc1/MatMul": {"qtype": "int16"},
            "/blocks.11/ffn/act/Mul": {"qtype": "int16"},
            "/blocks.11/ffn/act/Mul_1": {"qtype": "int16"},
            "/blocks.11/ffn/act/Mul_2": {"qtype": "int16"},
            "/blocks.11/ffn/act/Add": {"qtype": "int16"},
            "/blocks.11/ffn/act/Mul_3": {"qtype": "int16"},
            "/blocks.11/ffn/act/Tanh": {"qtype": "int16"},
            "/blocks.11/norm1/Mul_2": {"qtype": "int16"},
            "/blocks.11/cross_attn/k_norm/Div_1_reciprocal": {"qtype": "int16"},
            "/blocks.11/Add": {"qtype": "int16"},
            "/blocks.12/attn/MatMul": {"InputType0": "int16", "InputType1": "int16"},
            "/blocks.12/attn/MatMul_1": {"InputType0": "int16", "InputType1": "int16"},
            "/blocks.12/cross_attn/MatMul": {"InputType0": "int16", "InputType1": "int16"},
            "/blocks.12/cross_attn/MatMul_1": {"InputType0": "int16", "InputType1": "int16"},
            "/blocks.12/cross_attn/k_norm/Mul_1": {"qtype": "int16"},
            "/blocks.12/ffn/fc1/MatMul": {"qtype": "int16"},
            "/blocks.12/ffn/act/Mul": {"qtype": "int16"},
            "/blocks.12/ffn/act/Mul_1": {"qtype": "int16"},
            "/blocks.12/ffn/act/Mul_2": {"qtype": "int16"},
            "/blocks.12/ffn/act/Add": {"qtype": "int16"},
            "/blocks.12/ffn/act/Mul_3": {"qtype": "int16"},
            "/blocks.12/ffn/act/Tanh": {"qtype": "int16"},
            "/blocks.12/norm1/Mul_2": {"qtype": "int16"},
            "/blocks.12/cross_attn/k_norm/Div_1_reciprocal": {"qtype": "int16"},
            "/blocks.12/Add": {"qtype": "int16"},
            "/blocks.13/attn/MatMul": {"InputType0": "int16", "InputType1": "int16"},
            "/blocks.13/attn/MatMul_1": {"InputType0": "int16", "InputType1": "int16"},
            "/blocks.13/cross_attn/MatMul": {"InputType0": "int16", "InputType1": "int16"},
            "/blocks.13/cross_attn/MatMul_1": {"InputType0": "int16", "InputType1": "int16"},
            "/blocks.13/cross_attn/k_norm/Mul_1": {"qtype": "int16"},
            "/blocks.13/ffn/fc1/MatMul": {"qtype": "int16"},
            "/blocks.13/ffn/act/Mul": {"qtype": "int16"},
            "/blocks.13/ffn/act/Mul_1": {"qtype": "int16"},
            "/blocks.13/ffn/act/Mul_2": {"qtype": "int16"},
            "/blocks.13/ffn/act/Add": {"qtype": "int16"},
            "/blocks.13/ffn/act/Mul_3": {"qtype": "int16"},
            "/blocks.13/ffn/act/Tanh": {"qtype": "int16"},
            "/blocks.13/norm1/Mul_2": {"qtype": "int16"},
            "/blocks.13/cross_attn/k_norm/Div_1_reciprocal": {"qtype": "int16"},
            "/blocks.13/Add": {"qtype": "int16"},
            "/blocks.13/norm3/Div_1_reciprocal": {"qtype": "int16"},
            "/final_layer/ffn_final/act/Mul_1": {"qtype": "int16"},
            "/final_layer/ffn_final/act/Mul_2 ": {"qtype": "int16"},
            "/final_layer/norm_final/Div_1_reciprocal": {"qtype": "float32"}
        }
    }
    '''
    with open(os.path.join(opt.export_path, dit_ws_name, dit_json_name), "w", encoding="utf-8") as f:
        f.write(json_str)

    bash_str = f'''
    hb_compile --config {dit_config_name}
    cp bpu_output/*.hbm ../{bpu_rdt_name}/
    '''
    with open(os.path.join(opt.export_path, dit_ws_name, dit_bash_name), "w", encoding="utf-8") as f:
        f.write(bash_str)

    bash_build_all += f"cd {dit_ws_name}" + "\n"
    bash_build_all += f"bash {dit_bash_name}" + "\n"
    bash_build_all += f"cd .." + "\n"

    with open(os.path.join(opt.export_path, bash_build_all_name), "w", encoding="utf-8") as f:
        f.write(bash_build_all)

    # state adaptor

    m = model.policy.state_adaptor
    m.eval()

    input_data = torch.randn(1, 1, 256)  # 假设批量大小为1
    output = m(input_data)

    torch.onnx.export(
        m,                            # 要转换的模型
        input_data,                   # 模型的输入
        state_adaptor_path1,             # 输出文件名
        export_params=True,           # 存储训练后的参数
        opset_version=17,             # ONNX版本（降低以兼容 ReduceMean axes 属性）
        do_constant_folding=True,     # 是否执行常量折叠优化
        input_names=['state_tokens'],   # 输入节点名称
        output_names=['state_traj'], # 输出节点名称
        dynamic_axes=None,
        verbose=False
    )

    logger.info("Export RDT [state 1x1x256] Model Success.")

    input_data = torch.randn(1, 64, 256)  # 假设批量大小为1
    output = m(input_data)

    torch.onnx.export(
        m,                            # 要转换的模型
        input_data,                   # 模型的输入
        state_adaptor_path2,             # 输出文件名
        export_params=True,           # 存储训练后的参数
        opset_version=17,             # ONNX版本（降低以兼容 ReduceMean axes 属性）
        do_constant_folding=True,     # 是否执行常量折叠优化
        input_names=['state_tokens'],   # 输入节点名称
        output_names=['state_traj'], # 输出节点名称
        dynamic_axes=None,
        verbose=False
    )

    logger.info("Export RDT [state 1x64x256] Model Success.")

    # lang adaptor

    m = model.policy.lang_adaptor
    m.eval()

    input_data = torch.randn(1, 14, 4096)  # 假设批量大小为1
    output = m(input_data)

    torch.onnx.export(
        m,                            # 要转换的模型
        input_data,                   # 模型的输入
        lang_adaptor_path,             # 输出文件名
        export_params=True,           # 存储训练后的参数
        opset_version=17,             # ONNX版本（降低以兼容 ReduceMean axes 属性）
        do_constant_folding=True,     # 是否执行常量折叠优化
        input_names=['text_embeds'],   # 输入节点名称
        output_names=['lang_cond'], # 输出节点名称
        dynamic_axes={
            "text_embeds": {1: "N"},
            "lang_cond": {1: "N"}
        },
        verbose=False
    )

    logger.info("Export RDT [lang adaptor] Model Success.")

######## Prepare Calbibration Data


def create_dump_model(args, **kwargs):
    left_arm_dim, right_arm_dim = (args["arm_dim"]["left_arm_dim"], args["arm_dim"]["right_arm_dim"],)
    # 仅右臂6关节映射到 [0, 6) 位置
    AGILEX_STATE_INDICES = [
        STATE_VEC_IDX_MAPPING[f"right_arm_joint_{i}_pos"] for i in range(6)
    ]
    model = RoboticDiffusionTransformerModel_Dump(args, **kwargs)
    pretrained = kwargs.get("pretrained", None)
    if pretrained is not None and os.path.isfile(pretrained):
        model.load_pretrained_weights(pretrained)
    return model

class RDT_Dump(nn.Module):
    def __init__(self,
                 output_dim=128,
                 horizon=32,
                 hidden_size=1152,
                 depth=28,
                 num_heads=16,
                 max_lang_cond_len=1024,
                 img_cond_len=4096,
                 lang_pos_embed_config=None,
                 img_pos_embed_config=None,
                 dtype=torch.bfloat16):
        super().__init__()
        self.horizon = horizon
        self.hidden_size = hidden_size
        self.max_lang_cond_len = max_lang_cond_len
        self.img_cond_len = img_cond_len
        self.dtype = dtype
        self.lang_pos_embed_config = lang_pos_embed_config
        self.img_pos_embed_config = img_pos_embed_config
        self.t_embedder = TimestepEmbedder(hidden_size, dtype=dtype)
        self.freq_embedder = TimestepEmbedder(hidden_size, dtype=dtype)
        # We will use trainable sin-cos embeddings
        # [timestep; state; action]
        self.x_pos_embed = nn.Parameter(torch.zeros(1, horizon + 3, hidden_size))
        # Language conditions
        self.lang_cond_pos_embed = nn.Parameter(torch.zeros(1, max_lang_cond_len, hidden_size))
        # Image conditions
        self.img_cond_pos_embed = nn.Parameter(torch.zeros(1, img_cond_len, hidden_size))
        self.blocks = nn.ModuleList([RDTBlock(hidden_size, num_heads) for _ in range(depth)])
        self.final_layer = FinalLayer(hidden_size, output_dim)
        self.initialize_weights()
    def initialize_weights(self):
        # Initialize transformer layers:
        def _basic_init(module):
            if isinstance(module, nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
        self.apply(_basic_init)
        # Initialize pos_embed by sin-cos embedding
        x_pos_embed = get_multimodal_cond_pos_embed(embed_dim=self.hidden_size,
                                                    mm_cond_lens=OrderedDict([
                                                        ('timestep', 1),
                                                        ('ctrl_freq', 1),
                                                        ('state', 1),
                                                        ('action', self.horizon),
                                                    ]))
        self.x_pos_embed.data.copy_(torch.from_numpy(x_pos_embed).float().unsqueeze(0))
        if self.lang_pos_embed_config is None:
            lang_cond_pos_embed = get_1d_sincos_pos_embed_from_grid(self.hidden_size, torch.arange(self.max_lang_cond_len))
        else:
            lang_cond_pos_embed = get_multimodal_cond_pos_embed(embed_dim=self.hidden_size, mm_cond_lens=OrderedDict(self.lang_pos_embed_config), embed_modality=False)
        self.lang_cond_pos_embed.data.copy_(torch.from_numpy(lang_cond_pos_embed).float().unsqueeze(0))
        if self.img_pos_embed_config is None:
            img_cond_pos_embed = get_1d_sincos_pos_embed_from_grid(self.hidden_size, torch.arange(self.img_cond_len))
        else:
            img_cond_pos_embed = get_multimodal_cond_pos_embed(embed_dim=self.hidden_size, mm_cond_lens=OrderedDict(self.img_pos_embed_config), embed_modality=False)
        self.img_cond_pos_embed.data.copy_(torch.from_numpy(img_cond_pos_embed).float().unsqueeze(0))
        # Initialize timestep and control freq embedding MLP
        nn.init.normal_(self.t_embedder.mlp[0].weight, std=0.02)
        nn.init.normal_(self.t_embedder.mlp[2].weight, std=0.02)
        nn.init.normal_(self.freq_embedder.mlp[0].weight, std=0.02)
        nn.init.normal_(self.freq_embedder.mlp[2].weight, std=0.02)
        # Initialize the final layer: zero-out the final linear layer
        nn.init.constant_(self.final_layer.ffn_final.fc2.weight, 0)
        nn.init.constant_(self.final_layer.ffn_final.fc2.bias, 0)
        # Move all the params to given data type:
        self.to(self.dtype)
    def forward(self, x, freq, t, lang_c, img_c, lang_mask=None, img_mask=None):
        t = self.t_embedder(t).unsqueeze(1)  # (B, 1, D) or (1, 1, D)
        freq = self.freq_embedder(freq).unsqueeze(1)  # (B, 1, D)
        # Append timestep to the input tokens
        if t.shape[0] == 1:
            t = t.expand(x.shape[0], -1, -1)
        x = torch.cat([t, freq, x], dim=1)  # (B, T+1, D)
        # Add multimodal position embeddings
        x = x + self.x_pos_embed
        # Note the lang is of variable length
        lang_c = lang_c + self.lang_cond_pos_embed[:, :lang_c.shape[1]]
        img_c = img_c + self.img_cond_pos_embed
        # Forward pass
        conds = [lang_c, img_c]
        masks = [lang_mask, img_mask]
        for i, block in enumerate(self.blocks):
            c, mask = conds[i % 2], masks[i % 2]
            x = block(x, c, mask)  # (B, T+1, D)
        # Inject the language condition at the final layer
        x = self.final_layer(x)  # (B, T+1, out_channels)
        # Only preserve the action tokens
        x = x[:, -self.horizon:]
        return x

def dump_dit(state_action_traj, ctrl_freqs, t, lang_cond, img_cond, lang_attn_mask):
    t_str = str(t)
    x = state_action_traj.float().contiguous().cpu().detach().numpy()
    freq = ctrl_freqs.float().contiguous().cpu().detach().numpy().astype(np.int32).copy()
    t_ = t.float().contiguous().cpu().detach().numpy()
    t_ = np.expand_dims(t_.astype(np.int32), axis=0).copy()
    lang_c = lang_cond.float().contiguous().cpu().detach().numpy()
    img_c = img_cond.float().contiguous().cpu().detach().numpy()
    lang_mask = lang_attn_mask.float().contiguous().cpu().detach().numpy()
    pad_rows = 64 - lang_mask.shape[1]
    padded = np.pad(lang_mask, ((0,0), (0,pad_rows)), mode="constant")
    mask_float = np.where(padded, 0.0, -512.0).astype(np.float32)
    lang_cond_padded = np.pad(lang_c, pad_width=((0, 0), (0, pad_rows), (0,0)), mode="constant", constant_values=0)
    global dit_cal_path_x, dit_cal_path_freq, dit_cal_path_t, dit_cal_path_lang_c, dit_cal_path_img_c, dit_cal_path_lang_mask
    global dump_cnt, dump_dataset_name
    np.save(os.path.join(dit_cal_path_x, f"x_{t_str}_{dump_dataset_name}_{dump_cnt}.npy"), x)
    np.save(os.path.join(dit_cal_path_freq, f"freq_{t_str}_{dump_dataset_name}_{dump_cnt}.npy"), freq)
    np.save(os.path.join(dit_cal_path_t, f"t_{t_str}_{dump_dataset_name}_{dump_cnt}.npy"), t_)
    np.save(os.path.join(dit_cal_path_lang_c, f"lang_c_{t_str}_{dump_dataset_name}_{dump_cnt}.npy"), lang_cond_padded)
    np.save(os.path.join(dit_cal_path_img_c, f"img_{t_str}_{dump_dataset_name}_{dump_cnt}.npy"), img_c)
    np.save(os.path.join(dit_cal_path_lang_mask, f"lang_mask_{t_str}_{dump_dataset_name}_{dump_cnt}.npy"), mask_float)


def dump_img_adaptor(img_tokens):
    global img_adaptor_cal_ws
    global dump_cnt, dump_dataset_name
    np.save(os.path.join(img_adaptor_cal_ws, f"img_adaptor_{dump_dataset_name}_{dump_cnt}.npy"), img_tokens.float().contiguous().cpu().detach().numpy())


class RDTRunner_Dump(nn.Module,
                CompatiblePyTorchModelHubMixin,
                repo_url="https://huggingface.co/robotics-diffusion-transformer/rdt-1b"):
    def __init__(self,
                 *,
                 action_dim,
                 pred_horizon,
                 config,
                 lang_token_dim,
                 img_token_dim,
                 state_token_dim,
                 max_lang_cond_len,
                 img_cond_len,
                 lang_pos_embed_config=None,
                 img_pos_embed_config=None,
                 dtype=torch.bfloat16):
        super(RDTRunner_Dump, self).__init__()
        # Create diffusion model
        hidden_size = config['rdt']['hidden_size']
        self.model = RDT_Dump(
            output_dim=action_dim,
            horizon=pred_horizon,
            hidden_size=hidden_size,
            depth=config['rdt']['depth'],
            num_heads=config['rdt']['num_heads'],
            max_lang_cond_len=max_lang_cond_len,
            img_cond_len=img_cond_len,
            lang_pos_embed_config=lang_pos_embed_config,
            img_pos_embed_config=img_pos_embed_config,
            dtype=dtype,
        )
        # Create adpators for various conditional inputs
        self.lang_adaptor = self.build_condition_adapter(config['lang_adaptor'], in_features=lang_token_dim, out_features=hidden_size)
        self.img_adaptor = self.build_condition_adapter(config['img_adaptor'], in_features=img_token_dim, out_features=hidden_size)
        # A `state` refers to an action or a proprioception vector
        self.state_adaptor = self.build_condition_adapter(
            config['state_adaptor'],
            in_features=state_token_dim * 2,  # state + state mask (indicator)
            out_features=hidden_size)
        # Create the noise scheduler
        noise_scheduler_config = config['noise_scheduler']
        self.noise_scheduler = DDPMScheduler(
            num_train_timesteps=noise_scheduler_config['num_train_timesteps'],
            beta_schedule=noise_scheduler_config['beta_schedule'],
            prediction_type=noise_scheduler_config['prediction_type'],
            clip_sample=noise_scheduler_config['clip_sample'],
        )
        self.noise_scheduler_sample = DPMSolverMultistepScheduler(
            num_train_timesteps=noise_scheduler_config['num_train_timesteps'],
            beta_schedule=noise_scheduler_config['beta_schedule'],
            prediction_type=noise_scheduler_config['prediction_type'],
        )
        self.num_train_timesteps = noise_scheduler_config['num_train_timesteps']
        self.num_inference_timesteps = noise_scheduler_config['num_inference_timesteps']
        self.prediction_type = noise_scheduler_config['prediction_type']
        self.pred_horizon = pred_horizon
        self.action_dim = action_dim
        print("Diffusion params: %e" %
              sum([p.numel() for p in self.model.parameters()] + [p.numel() for p in self.lang_adaptor.parameters()] +
                  [p.numel()
                   for p in self.img_adaptor.parameters()] + [p.numel() for p in self.state_adaptor.parameters()]))
    def build_condition_adapter(self, projector_type, in_features, out_features):
        projector = None
        if projector_type == 'linear':
            projector = nn.Linear(in_features, out_features)
        else:
            mlp_gelu_match = re.match(r'^mlp(\d+)x_gelu$', projector_type)
            if mlp_gelu_match:
                mlp_depth = int(mlp_gelu_match.group(1))
                modules = [nn.Linear(in_features, out_features)]
                for _ in range(1, mlp_depth):
                    modules.append(nn.GELU(approximate="tanh"))
                    modules.append(nn.Linear(out_features, out_features))
                projector = nn.Sequential(*modules)
        if projector is None:
            raise ValueError(f'Unknown projector type: {projector_type}')
        return projector
    def adapt_conditions(self, lang_tokens, img_tokens, state_tokens):
        adpated_lang = self.lang_adaptor(lang_tokens)
        dump_img_adaptor(img_tokens)
        adpated_img = self.img_adaptor(img_tokens)
        adpated_state = self.state_adaptor(state_tokens)
        return adpated_lang, adpated_img, adpated_state
    def conditional_sample(self, lang_cond, lang_attn_mask, img_cond, state_traj, action_mask, ctrl_freqs):
        device = state_traj.device
        dtype = state_traj.dtype
        noisy_action = torch.randn(size=(state_traj.shape[0], self.pred_horizon, self.action_dim), dtype=dtype, device=device)
        action_mask = action_mask.expand(-1, self.pred_horizon, -1)
        # Set step values
        self.noise_scheduler_sample.set_timesteps(self.num_inference_timesteps)
        for t in self.noise_scheduler_sample.timesteps:
            # Prepare state-action trajectory
            action_traj = torch.cat([noisy_action, action_mask], dim=2)
            action_traj = self.state_adaptor(action_traj)
            state_action_traj = torch.cat([state_traj, action_traj], dim=1)
            # dump
            dump_dit(state_action_traj, ctrl_freqs, t, lang_cond, img_cond, lang_attn_mask)
            # Predict the model output
            model_output = self.model(state_action_traj,
                                      ctrl_freqs,
                                      t.unsqueeze(-1).to(device),
                                      lang_cond,
                                      img_cond,
                                      lang_mask=lang_attn_mask)
            # Compute previous actions: x_t -> x_t-1
            noisy_action = self.noise_scheduler_sample.step(model_output, t, noisy_action).prev_sample
            noisy_action = noisy_action.to(state_traj.dtype)
        # Finally apply the action mask to mask invalid action dimensions
        noisy_action = noisy_action * action_mask
        return noisy_action
    def compute_loss(self, lang_tokens, lang_attn_mask, img_tokens, state_tokens, action_gt, action_mask,
                     ctrl_freqs) -> torch.Tensor:
        '''
        lang_tokens: (batch_size, lang_len, lang_token_dim)
        lang_attn_mask: (batch_size, lang_len), a mask for valid language tokens,
            which should be True-False bool tensor.
        img_tokens: (batch_size, img_len, img_token_dim)
        state_tokens: (batch_size, 1, state_token_dim)
        action_gt: (batch_size, horizon, state_token_dim), ground-truth actions for supervision
        action_mask: (batch_size, 1, state_token_dim), a 0-1 **float** tensor.
        ctrl_freqs: (batch_size,), control frequency for each sample.
        
        return: loss_value, a scalar tensor
        '''
        batch_size = lang_tokens.shape[0]
        device = lang_tokens.device
        # Sample noise that we'll add to the actions
        noise = torch.randn(action_gt.shape, dtype=action_gt.dtype, device=device)
        # Sample random diffusion timesteps
        timesteps = torch.randint(0, self.num_train_timesteps, (batch_size, ), device=device).long()
        # Add noise to the clean actions according to the noise magnitude at each timestep
        # (this is the forward diffusion process)
        noisy_action = self.noise_scheduler.add_noise(action_gt, noise, timesteps)
        # Concatenate the state and action tokens to form the input sequence
        state_action_traj = torch.cat([state_tokens, noisy_action], dim=1)
        # Append the action mask to the input sequence
        action_mask = action_mask.expand(-1, state_action_traj.shape[1], -1)
        state_action_traj = torch.cat([state_action_traj, action_mask], dim=2)
        # Align the dimension with the hidden size
        lang_cond, img_cond, state_action_traj = self.adapt_conditions(lang_tokens, img_tokens, state_action_traj)
        # Predict the denoised result
        pred = self.model(state_action_traj, ctrl_freqs, timesteps, lang_cond, img_cond, lang_mask=lang_attn_mask)
        pred_type = self.prediction_type
        if pred_type == 'epsilon':
            target = noise
        elif pred_type == 'sample':
            target = action_gt
        else:
            raise ValueError(f"Unsupported prediction type {pred_type}")
        loss = F.mse_loss(pred, target)
        return loss
    def predict_action(self, lang_tokens, lang_attn_mask, img_tokens, state_tokens, action_mask, ctrl_freqs):
        '''
        lang_tokens: (batch_size, lang_len, lang_token_dim)
        lang_attn_mask: (batch_size, lang_len), a mask for valid language tokens,
            which should be True-False bool tensor.
        img_tokens: (batch_size, img_len, img_token_dim)
        state_tokens: (batch_size, 1, state_token_dim)
        action_mask: (batch_size, 1, action_dim),
            which should be a 0-1 **float** tensor.
        ctrl_freqs: (batch_size,), control frequency for each sample.
        
        return: (batch_size, horizon, action_dim), predicted action sequence
        '''
        # Prepare the state and conditions
        state_tokens = torch.cat([state_tokens, action_mask], dim=2)
        lang_cond, img_cond, state_traj = self.adapt_conditions(lang_tokens, img_tokens, state_tokens)
        # Run sampling
        action_pred = self.conditional_sample(
            lang_cond,
            lang_attn_mask,
            img_cond,
            state_traj,
            action_mask,
            ctrl_freqs,
        )
        return action_pred
    def forward(self, *args, **kwargs) -> torch.Tensor:
        return self.compute_loss(*args, **kwargs)


class RoboticDiffusionTransformerModel_Dump(object):
    def __init__(
        self,
        args,
        device="cuda",
        dtype=torch.bfloat16,
        image_size=None,
        control_frequency=25,
        pretrained=None,
        pretrained_vision_encoder_name_or_path=None,
    ):
        self.args = args
        self.dtype = dtype
        self.image_size = image_size
        self.device = device
        self.control_frequency = control_frequency
        # We do not use the text encoder due to limited GPU memory
        # self.text_tokenizer, self.text_model = self.get_text_encoder(pretrained_text_encoder_name_or_path)
        self.image_processor, self.vision_model = self.get_vision_encoder(pretrained_vision_encoder_name_or_path)
        self.policy = self.get_policy(pretrained)
        self.left_arm_dim, self.right_arm_dim = (
            args["arm_dim"]["left_arm_dim"],
            args["arm_dim"]["right_arm_dim"],
        )
        self.reset()
    def get_policy(self, pretrained):
        # Initialize model with arguments
        if pretrained is None or os.path.isfile(pretrained):
            img_cond_len = (self.args["common"]["img_history_size"] * self.args["common"]["num_cameras"] * self.vision_model.num_patches)
            _model = RDTRunner_Dump(
                action_dim=self.args["common"]["state_dim"],
                pred_horizon=self.args["common"]["action_chunk_size"],
                config=self.args["model"],
                lang_token_dim=self.args["model"]["lang_token_dim"],
                img_token_dim=self.args["model"]["img_token_dim"],
                state_token_dim=self.args["model"]["state_token_dim"],
                max_lang_cond_len=self.args["dataset"]["tokenizer_max_length"],
                img_cond_len=img_cond_len,
                img_pos_embed_config=[
                    # No initial pos embed in the last grid size
                    # since we've already done in ViT
                    (
                        "image",
                        (
                            self.args["common"]["img_history_size"],
                            self.args["common"]["num_cameras"],
                            -self.vision_model.num_patches,
                        ),
                    ),
                ],
                lang_pos_embed_config=[
                    # Similarly, no initial pos embed for language
                    ("lang", -self.args["dataset"]["tokenizer_max_length"]),
                ],
                dtype=self.dtype,
            )
        else:
            _model = RDTRunner_Dump.from_pretrained(pretrained)
        return _model
    def get_text_encoder(self, pretrained_text_encoder_name_or_path):
        text_embedder = T5Embedder(
            from_pretrained=pretrained_text_encoder_name_or_path,
            model_max_length=self.args["dataset"]["tokenizer_max_length"],
            device=self.device,
        )
        tokenizer, text_encoder = text_embedder.tokenizer, text_embedder.model
        return tokenizer, text_encoder
    def get_vision_encoder(self, pretrained_vision_encoder_name_or_path):
        vision_encoder = SiglipVisionTower(vision_tower=pretrained_vision_encoder_name_or_path, args=None)
        image_processor = vision_encoder.image_processor
        return image_processor, vision_encoder
    def reset(self):
        device = self.device
        weight_dtype = self.dtype
        self.policy.eval()
        # self.text_model.eval()
        self.vision_model.eval()
        self.policy = self.policy.to(device, dtype=weight_dtype)
        # self.text_model = self.text_model.to(device, dtype=weight_dtype)
        self.vision_model = self.vision_model.to(device, dtype=weight_dtype)
    def load_pretrained_weights(self, pretrained=None):
        if pretrained is None:
            return
        print(f"Loading weights from {pretrained}")
        filename = os.path.basename(pretrained)
        if filename.endswith(".pt"):
            checkpoint = torch.load(pretrained)
            self.policy.load_state_dict(checkpoint["module"])
        elif filename.endswith(".safetensors"):
            from safetensors.torch import load_model
            load_model(self.policy, pretrained)
        else:
            raise NotImplementedError(f"Unknown checkpoint format: {pretrained}")
    def encode_instruction(self, instruction, device="cuda"):
        tokens = self.text_tokenizer(instruction, return_tensors="pt", padding="longest", truncation=True)["input_ids"].to(device)
        tokens = tokens.view(1, -1)
        with torch.no_grad():
            pred = self.text_model(tokens).last_hidden_state.detach()
        return pred
    def _format_joint_to_state(self, joints):
        AGILEX_STATE_INDICES = [
                STATE_VEC_IDX_MAPPING[f"right_arm_joint_{i}_pos"] for i in range(6)
            ]
        # Rescale the gripper to the range of [0, 1]
        joints = joints / torch.tensor(
            [[[180, 180, 180, 180, 180, 180]]],
            device=joints.device,
            dtype=joints.dtype,
        )

        B, N, _ = joints.shape
        state = torch.zeros(
            (B, N, self.args["model"]["state_token_dim"]),
            device=joints.device,
            dtype=joints.dtype,
        )
        # Fill into the unified state vector
        state[:, :, AGILEX_STATE_INDICES] = joints
        # Assemble the mask indicating each dimension's availability
        state_elem_mask = torch.zeros(
            (B, self.args["model"]["state_token_dim"]),
            device=joints.device,
            dtype=joints.dtype,
        )
        state_elem_mask[:, AGILEX_STATE_INDICES] = 1
        return state, state_elem_mask

    def _unformat_action_to_joint(self, action):
        AGILEX_STATE_INDICES = [
                STATE_VEC_IDX_MAPPING[f"right_arm_joint_{i}_pos"] for i in range(6)
            ]
        action_indices = AGILEX_STATE_INDICES
        joints = action[:, :, action_indices]
        # Rescale the gripper back to the action range
        # Note that the action range and proprioception range are different
        # for Mobile ALOHA robot
        joints = joints * torch.tensor(
            [[[180, 180, 180, 180, 180, 180]]],
            device=joints.device,
            dtype=joints.dtype,
        )
        return joints
    @torch.no_grad()
    def step(self, proprio, images, text_embeds):
        device = self.device
        dtype = self.dtype
        # The background image used for padding
        background_color = np.array([int(x * 255) for x in self.image_processor.image_mean], dtype=np.uint8).reshape(1, 1, 3)
        background_image = (np.ones(
            (
                self.image_processor.size["height"],
                self.image_processor.size["width"],
                3,
            ),
            dtype=np.uint8,
        ) * background_color)
        # Preprocess the images by order and encode them
        image_tensor_list = []
        for image in images:
            if image is None:
                # Replace it with the background image
                image = PImage.fromarray(background_image)
            if self.image_size is not None:
                image = transforms.Resize(self.data_args.image_size)(image)
            if self.args["dataset"].get("auto_adjust_image_brightness", False):
                pixel_values = list(image.getdata())
                average_brightness = sum(sum(pixel) for pixel in pixel_values) / (len(pixel_values) * 255.0 * 3)
                if average_brightness <= 0.15:
                    image = transforms.ColorJitter(brightness=(1.75, 1.75))(image)
            if self.args["dataset"].get("image_aspect_ratio", "pad") == "pad":
                def expand2square(pil_img, background_color):
                    width, height = pil_img.size
                    if width == height:
                        return pil_img
                    elif width > height:
                        result = PImage.new(pil_img.mode, (width, width), background_color)
                        result.paste(pil_img, (0, (width - height) // 2))
                        return result
                    else:
                        result = PImage.new(pil_img.mode, (height, height), background_color)
                        result.paste(pil_img, ((height - width) // 2, 0))
                        return result
                image = expand2square(image, tuple(int(x * 255) for x in self.image_processor.image_mean))
            image = self.image_processor.preprocess(image, return_tensors="pt")["pixel_values"][0]
            image_tensor_list.append(image)
        image_tensor = torch.stack(image_tensor_list, dim=0).to(device, dtype=dtype)
        image_embeds = self.vision_model(image_tensor).detach()
        image_embeds = image_embeds.reshape(-1, self.vision_model.hidden_size).unsqueeze(0)
        # Prepare the proprioception states and the control frequency
        joints = proprio.to(device).unsqueeze(0)  # (1, 1, 14)
        states, state_elem_mask = self._format_joint_to_state(joints)  # (1, 1, 128), (1, 128)
        states, state_elem_mask = states.to(device, dtype=dtype), state_elem_mask.to(device, dtype=dtype)
        states = states[:, -1:, :]  # (1, 1, 128)
        ctrl_freqs = torch.tensor([self.control_frequency]).to(device)
        text_embeds = text_embeds.to(device, dtype=dtype)
        # Predict the next action chunk given the inputs
        trajectory = self.policy.predict_action(
            lang_tokens=text_embeds,
            lang_attn_mask=torch.ones(text_embeds.shape[:2], dtype=torch.bool, device=text_embeds.device),
            img_tokens=image_embeds,
            state_tokens=states,
            action_mask=state_elem_mask.unsqueeze(1),
            ctrl_freqs=ctrl_freqs,
        )
        trajectory = self._unformat_action_to_joint(trajectory).to(torch.float32)
        return trajectory

######## Prepare Training Data
def fill_in_state(values, left_arm_dim, right_arm_dim):
    # RDT的Action及状态需要按照要求映射导一个128维的统一向量
    uni_vec = np.zeros(values.shape[:-1] + (128,))
    # 仅右臂6关节映射到 [0, 6) 位置
    for i in range(min(6, right_arm_dim[0])):
        uni_vec[..., i] = values[..., i]
    return uni_vec

def get_training_samples(data_dir, num_samples=100, instructions_per_episode=1):
    training_samples = []
    logger.info(f"Get Training Data From: {data_dir}.")
    for root, dirs, files in os.walk(data_dir):
        for file in files:
            if file.endswith('.hdf5') and len(training_samples) < num_samples:
                file_path = os.path.join(root, file)
                try:
                    with h5py.File(file_path, 'r') as f:
                        observations = f['observations']
                        actions = f['action'][:]
                        images = observations['images']
                        # left_arm_dim = observations['left_arm_dim'][:]
                        # right_arm_dim = observations['right_arm_dim'][:]
                        qpos = observations['qpos'][:]
                        episode_dir = os.path.dirname(file_path)
                        instructions_dir = os.path.join(episode_dir, 'instructions')
                        num_steps = len(qpos)
                        if num_steps > 1:
                            # 收集该 episode 可用的 instruction 索引
                            step_indices = []
                            if os.path.isdir(instructions_dir):
                                for name in os.listdir(instructions_dir):
                                    if name.startswith('lang_embed_') and name.endswith('.pt'):
                                        try:
                                            idx = int(name[len('lang_embed_'):-3])
                                            if 0 <= idx < num_steps:
                                                step_indices.append(idx)
                                        except Exception:
                                            continue
                            step_indices = sorted(set(step_indices))
                            # 按每个 episode 上限采样/截断
                            if len(step_indices) == 0:
                                # 回退策略：如果没有任何配套 instruction，就尝试用一个随机 step，但很可能会在后续被过滤
                                step_indices = []
                            for step_idx in step_indices[:max(1, instructions_per_episode)]:
                                if len(training_samples) >= num_samples:
                                    break
                                # 获取多摄像头多历史帧图像
                                multi_cam_images = {}
                                for cam_name in ['cam_high', 'cam_left_wrist', 'cam_right_wrist']:
                                    if cam_name in images:
                                        cam_images = []
                                        # 取 step_idx 的前一帧与当前帧，共两帧
                                        for i in range(max(step_idx - 1, 0), step_idx + 1):
                                            img_bits = images[cam_name][i]
                                            img = cv2.imdecode(np.frombuffer(img_bits, np.uint8), cv2.IMREAD_COLOR)
                                            if img is not None:
                                                cam_images.append(img)
                                        if len(cam_images) == 1:
                                            cam_images = [cam_images[0], cam_images[0]]
                                        if len(cam_images) >= 2:
                                            multi_cam_images[cam_name] = cam_images[:2]
                                if len(multi_cam_images) == 0:
                                    logger.warning(f"Skip sample (missing images): {file_path}, step {step_idx}")
                                    continue
                                # 语言嵌入与文本
                                lang_embed = None
                                lang_str = ""
                                lang_embed_path = os.path.join(instructions_dir, f'lang_embed_{step_idx}.pt')
                                if os.path.exists(lang_embed_path):
                                    try:
                                        lang_embed = torch.load(lang_embed_path)
                                    except Exception as e:
                                        logger.error(f"Error reading text file {lang_embed_path}: {e}")
                                        lang_embed = None
                                lang_str_path = os.path.join(instructions_dir, f'txt_lang_embed_{step_idx}.txt')
                                if os.path.exists(lang_str_path):
                                    try:
                                        with open(lang_str_path, 'r', encoding='utf-8') as f:
                                            lang_str = f.read().strip()
                                    except Exception as e:
                                        logger.error(f"Error reading text file {lang_str_path}: {e}")
                                        lang_str = ""
                                if lang_embed is None:
                                    logger.warning(f"Skip sample (missing lang): {file_path}, step {step_idx}")
                                    continue
                                training_samples.append({
                                    'multi_cam_images': multi_cam_images,
                                    'joints': actions[step_idx],
                                    'lang_embed': lang_embed,
                                    'lang_str': lang_str,
                                    'source': file_path,
                                    'step': step_idx
                                })
                                logger.debug(f"TimeStep: {step_idx}, Sample: {file_path}")
                except Exception as e:
                    logger.error(f"Faild: {file_path} : {e}")
                    continue
    logger.info(f"Total Num: {len(training_samples)}.")
    return training_samples


if __name__ == "__main__":
    main()