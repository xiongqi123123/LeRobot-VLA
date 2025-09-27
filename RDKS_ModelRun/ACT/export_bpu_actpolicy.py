#!/user/bin/env python

# Copyright (c) 2025，WuChao D-Robotics.
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

# 注意: 此程序在开发机的训练环境中运行
# Attention: This program runs on developer machine training environment.

import logging
import os
import shutil
import cv2
import numpy as np
import torch
import torch.nn as nn
import argparse
import logging
import onnx
from copy import deepcopy
from termcolor import colored
from onnxsim import simplify
from termcolor import colored
from pprint import pformat

from lerobot.policies.act.modeling_act import ACTPolicy
from lerobot.datasets.factory import make_dataset
from lerobot.utils.utils import get_safe_torch_device, init_logging
from lerobot.configs import parser
from lerobot.configs.train import TrainPipelineConfig

BPU_VisionEncoder = "BPU_ACTPolicy_VisionEncoder"
BPU_TransformerLayers = "BPU_ACTPolicy_TransformerLayers"

REPOSITORY = "REPOSITORY"
TAG = "TAG"

# REPOSITORY = "openexplorer/ai_toolchain_ubuntu_20_x5_gpu"
# TAG = "v1.2.8-py310"

@parser.wrap()
def main(cfg: TrainPipelineConfig):
    # LeRobot的参数列表
    cfg.validate()
    logging.info(pformat(cfg.to_dict()))
    # 这里只是为了美观, 不支持从外传参, 需要在文件内修改
    parser = argparse.ArgumentParser()
    parser.add_argument('--act-path', type=str, default='outputs/train/act_so100_test/checkpoints/001000/pretrained_model', help='Path to LeRobot ACT Policy model.')
    """ 
    # example: --act-path pretrained_model
    ./pretrained_model/
    ├── config.json
    ├── model.safetensors
    └── train_config.json
    """
    parser.add_argument('--export-path', type=str, default='marcelo_test1', help='Path to save LeRobot ACT Policy model.') 
    parser.add_argument('--cal-num', type=int, default=400, help='Num of images to generate')
    parser.add_argument('--onnx-sim', type=bool, default=True, help='Simplify onnx or not.') 
    parser.add_argument('--type', type=str, default="nash-e", help='Optional: nash-e, nash-m, nash-p, bayes-e, bayes') 
    parser.add_argument('--combine-jobs', type=int, default=6, help='combie jobs for OpenExplore.')

    opt = parser.parse_args([])
    logging.info(f"opt: {opt}")
    # 所有的导出会基于opt.export_path这个文件夹
    ## 如果存在这个文件夹则删除
    if os.path.exists(opt.export_path): 
        shutil.rmtree(opt.export_path)
    visionEncoder_ws = os.path.join(opt.export_path, BPU_VisionEncoder)
    transformersLayers_ws = os.path.join(opt.export_path, BPU_TransformerLayers)
    ## 导出的ONNX文件路径
    onnx_name_BPU_ACTPolicy_VisionEncoder = BPU_VisionEncoder + ".onnx"
    onnx_path_BPU_ACTPolicy_VisionEncoder = os.path.join(visionEncoder_ws, onnx_name_BPU_ACTPolicy_VisionEncoder)
    onnx_name_BPU_ACTPolicy_TransformerLayers = BPU_TransformerLayers + ".onnx"
    onnx_path_BPU_ACTPolicy_TransformerLayers = os.path.join(transformersLayers_ws, onnx_name_BPU_ACTPolicy_TransformerLayers)
    ## 导出校准文件路径
    calbrate_data_name_BPU_ACTPolicy_VisionEncoder = "calibration_data_" + BPU_VisionEncoder
    calbrate_data_path_BPU_ACTPolicy_VisionEncoder = os.path.join(visionEncoder_ws, calbrate_data_name_BPU_ACTPolicy_VisionEncoder)
    calbrate_data_name_BPU_ACTPolicy_TransformerLayers = "calibration_data_" + BPU_TransformerLayers
    calbrate_data_path_BPU_ACTPolicy_TransformerLayers = os.path.join(transformersLayers_ws, calbrate_data_name_BPU_ACTPolicy_TransformerLayers)
    state_calbrate_data_path_BPU_ACTPolicy_TransformerLayers = os.path.join(calbrate_data_path_BPU_ACTPolicy_TransformerLayers, "state")
    ## 导出yaml配置文件路径
    config_yaml_name_BPU_ACTPolicy_VisionEncoder = "config_" + BPU_VisionEncoder + ".yaml"
    config_yaml_path_BPU_ACTPolicy_VisionEncoder = os.path.join(visionEncoder_ws, config_yaml_name_BPU_ACTPolicy_VisionEncoder)
    config_yaml_name_BPU_ACTPolicy_TransformerLayers = "config_" + BPU_TransformerLayers + ".yaml"
    config_yaml_path_BPU_ACTPolicy_TransformerLayers = os.path.join(transformersLayers_ws, config_yaml_name_BPU_ACTPolicy_TransformerLayers)
    ## 导出bash编译脚本路径
    bash_name_BPU_ACTPolicy_VisionEncoder = "build_" + BPU_VisionEncoder + ".sh"
    bash_path_BPU_ACTPolicy_VisionEncoder = os.path.join(visionEncoder_ws, bash_name_BPU_ACTPolicy_VisionEncoder)
    bash_name_BPU_ACTPolicy_TransformerLayers = "build_" + BPU_TransformerLayers + ".sh"
    bash_path_BPU_ACTPolicy_TransformerLayers = os.path.join(transformersLayers_ws, bash_name_BPU_ACTPolicy_TransformerLayers)
    ## 发布文件夹的脚本路径
    bpu_output_name = "bpu_output"
    bpu_output_path = os.path.join(opt.export_path, bpu_output_name)
    bash_build_all_path = os.path.join(opt.export_path, "build_all.sh") 
    ## 前后处理参数文件路径
    action_std_path = os.path.join(bpu_output_path, "action_std.npy")
    action_mean_path = os.path.join(bpu_output_path, "action_mean.npy")
    action_std_unnormalize_path = os.path.join(bpu_output_path, "action_std_unnormalize.npy")
    action_mean_unnormalize_path = os.path.join(bpu_output_path, "action_mean_unnormalize.npy")
    ## 新建工作目录
    os.makedirs(visionEncoder_ws, exist_ok=True)
    logging.info(colored(f"mkdir: {visionEncoder_ws} Success.", 'green'))
    os.makedirs(transformersLayers_ws, exist_ok=True)
    logging.info(colored(f"mkdir: {transformersLayers_ws} Success.", 'green'))
    os.makedirs(calbrate_data_path_BPU_ACTPolicy_VisionEncoder, exist_ok=True)
    logging.info(colored(f"mkdir: {calbrate_data_path_BPU_ACTPolicy_VisionEncoder} Success.", 'green'))
    os.makedirs(calbrate_data_path_BPU_ACTPolicy_TransformerLayers, exist_ok=True)
    logging.info(colored(f"mkdir: {calbrate_data_path_BPU_ACTPolicy_TransformerLayers} Success.", 'green'))
    os.makedirs(state_calbrate_data_path_BPU_ACTPolicy_TransformerLayers, exist_ok=True)
    logging.info(colored(f"mkdir: {state_calbrate_data_path_BPU_ACTPolicy_TransformerLayers} Success.", 'green'))
    os.makedirs(bpu_output_path, exist_ok=True)
    logging.info(colored(f"mkdir: {bpu_output_path} Success.", 'green'))

    # 加载 ACT Policy 模型
    policy = ACTPolicy.from_pretrained(opt.act_path).cpu().eval()
    logging.info(colored(f"Load ACT Policy Model: {opt.act_path} Success.", 'light_red'))
    
    # CUDA Configs
    device = get_safe_torch_device(cfg.policy.device, log=True)
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True

    # 构造数据集，读取数据集，并准备校准数据
    dataset = make_dataset(cfg)
    shuffle = True
    sampler = None
    dataloader = torch.utils.data.DataLoader(
        dataset,
        num_workers=0,
        batch_size=1,
        shuffle=True,
        sampler=None,
        pin_memory=device.type != "cpu",
        drop_last=False,
    )
    logging.info(colored(f"Load ACT Policy Dataset: \n{dataset} Success.", 'light_red'))

    # Export
    ## 动态获取相机名称和数据
    batch = next(iter(dataloader))
    image_keys = [key for key in batch.keys() if key.startswith('observation.images.')]
    camera_names = [key.split('.')[-1] for key in image_keys]  # 提取相机名称
    kvs = image_keys + ['observation.state']
    batch = dict(filter(lambda item: item[0] in kvs, batch.items()))
    
    logging.info(f"Detected cameras: {camera_names}")
    logging.info(f"Using keys: {kvs}")
    
    ## dirty run
    outputs = policy.select_action(deepcopy(batch))

    ## 动态获取前后处理参数
    # 为每个相机保存归一化参数
    for camera_name in camera_names:
        buffer_name = f"buffer_observation_images_{camera_name}"
        if hasattr(policy.normalize_inputs, buffer_name):
            buffer = getattr(policy.normalize_inputs, buffer_name)
            camera_std = buffer.std.data.detach().cpu().numpy()
            camera_mean = buffer.mean.data.detach().cpu().numpy()
            
            camera_std_path = os.path.join(bpu_output_path, f"{camera_name}_std.npy")
            camera_mean_path = os.path.join(bpu_output_path, f"{camera_name}_mean.npy")
            
            np.save(camera_std_path, camera_std)
            np.save(camera_mean_path, camera_mean)
            logging.info(f"Saved {camera_name} normalization parameters")

    # 保存状态和动作归一化参数
    action_std = policy.normalize_inputs.buffer_observation_state.std.data.detach().cpu().numpy()
    action_mean = policy.normalize_inputs.buffer_observation_state.mean.data.detach().cpu().numpy()
    action_std_unnormalize = policy.unnormalize_outputs.buffer_action.std.data.detach().cpu().numpy()
    action_mean_unnormalize = policy.unnormalize_outputs.buffer_action.mean.data.detach().cpu().numpy()

    np.save(action_std_path, action_std)
    np.save(action_mean_path, action_mean)
    np.save(action_std_unnormalize_path, action_std_unnormalize)   
    np.save(action_mean_unnormalize_path, action_mean_unnormalize)

    ## Vision Encoder
    batch = policy.normalize_inputs(batch)
    m_VisionEncoder = BPU_ACTPolicy_VisionEncoder(policy)
    m_VisionEncoder.eval()

    # # 基于huggingFace的前处理
    # mean = policy.normalize_inputs.buffer_observation_images_phone.mean
    # std = policy.normalize_inputs.buffer_observation_images_phone.std

    # input_tensor = batch['observation.images.phone']
    # # np.save(f"new_observation.images.phone.npy", input_tensor.detach().cpu().numpy())
    # input_tensor = (input_tensor - mean) / std
    # # np.save(f"new_observation.images.phone_meanstd.npy", input_tensor.detach().cpu().numpy())
    # # input_tensor = policy.normalize_inputs({"observation.images": batch['observation.images.phone']})["observation.images"]
    # vision_feature1 = m(input_tensor)
    # # np.save(f"new_can0feature.npy", vision_feature1.detach().cpu().numpy())

    # input_tensor = batch['observation.images.laptop']
    # # np.save(f"new_observation.images.laptop.npy", input_tensor.detach().cpu().numpy())
    # input_tensor = (input_tensor - mean) / std
    # # np.save(f"new_observation.images.laptop_meanstd.npy", input_tensor.detach().cpu().numpy())
    # # input_tensor = policy.normalize_inputs({"observation.images": batch['observation.images.laptop']})["observation.images"]
    # vision_feature2 = m(input_tensor)
    # # np.save(f"new_can1feature.npy", vision_feature2.detach().cpu().numpy())


    # ### 基于板端前处理对齐
    # yaml_mean = (255.0 * policy.normalize_inputs.buffer_observation_images_phone.mean[:,0,0]).cpu().numpy()[np.newaxis,:,np.newaxis,np.newaxis]
    # yaml_scale = (1.0 / (255.0 * policy.normalize_inputs.buffer_observation_images_phone.std[:,0,0])).cpu().numpy()[np.newaxis,:,np.newaxis,np.newaxis]

    # img = lerobotTensor2cvmat(batch['observation.images.laptop'])
    # # cv2.imwrite("new_observation.images.laptop.jpg", img)
    # input_tensor = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)     # BGR2RGB
    # input_tensor = np.transpose(input_tensor, (2, 0, 1))    # HWC2CHW
    # input_tensor = np.expand_dims(input_tensor, axis=0).astype(np.float32)  # CHW -> NCHW
    # input_tensor = (input_tensor - yaml_mean) * yaml_scale
    # # np.save(f"new_cv2_observation.images.laptop_meanstd.npy", input_tensor)
    # input_tensor = torch.from_numpy(input_tensor).to(device)
    # vision_feature1 = m(input_tensor)
    # # np.save(f"new_cv2_can1feature.npy", vision_feature1.detach().cpu().numpy())

    # img = lerobotTensor2cvmat(batch['observation.images.phone'])
    # # cv2.imwrite("new_observation.images.phone.jpg", img)
    # input_tensor = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)     # BGR2RGB
    # input_tensor = np.transpose(input_tensor, (2, 0, 1))    # HWC2CHW
    # input_tensor = np.expand_dims(input_tensor, axis=0).astype(np.float32)  # CHW -> NCHW
    # input_tensor = (input_tensor - yaml_mean) * yaml_scale
    # # np.save(f"new_cv2_observation.images.phone_meanstd.npy", input_tensor)
    # input_tensor = torch.from_numpy(input_tensor).to(device)
    # vision_feature2 = m(input_tensor)
    # # np.save(f"new_cv2_can0feature.npy", vision_feature2.detach().cpu().numpy())

    # 动态获取相机视觉特征
    vision_features = []
    for camera_name in camera_names:
        input_tensor = batch[f'observation.images.{camera_name}']
        vision_feature = m_VisionEncoder(input_tensor)
        vision_features.append(vision_feature)
        logging.info(f"Generated vision features for {camera_name}: {vision_feature.shape}")

    # 确定ONNX版本
    opset_version = 11 if "bayes" in opt.type else 19
    logging.info(f"Using ONNX opset version: {opset_version} for type: {opt.type}")

    onnx_path = onnx_path_BPU_ACTPolicy_VisionEncoder
    torch.onnx.export(
        m_VisionEncoder,  # 要转换的模型
        input_tensor,  # 模型的输入
        onnx_path,  # 输出文件名
        export_params=True,  # 存储训练后的参数
        opset_version=opset_version,  # 动态ONNX版本
        do_constant_folding=True,  # 是否执行常量折叠优化
        input_names=['images'],  # 输入节点名称
        output_names=['Vision_Features'],  # 输出节点名称
        dynamic_axes=None
    )
    onnx_sim(onnx_path, opt.onnx_sim)
    logging.info(colored(f"Export {onnx_path} Success.", 'green'))

    m_TransformerLayers = BPU_ACTPolicy_TransformerLayers(policy, camera_names)
    m_TransformerLayers.eval()

    state = batch["observation.state"]
    actions = m_TransformerLayers(state, *vision_features)
    np.save(f"new_actions.npy", actions.detach().cpu().numpy())

    # 动态构建输入名称
    input_names = ['states'] + [f'{camera_name}_features' for camera_name in camera_names]
    logging.info(f"Transformer input names: {input_names}")

    onnx_path = onnx_path_BPU_ACTPolicy_TransformerLayers
    torch.onnx.export(
        m_TransformerLayers,  # 要转换的模型
        (state, *vision_features),  # 模型的输入
        onnx_path,  # 输出文件名
        export_params=True,  # 存储训练后的参数
        opset_version=opset_version,  # 动态ONNX版本
        do_constant_folding=True,  # 是否执行常量折叠优化
        input_names=input_names,  # 动态输入节点名称
        output_names=['Actions'],  # 输出节点名称
        dynamic_axes=None
    )
    onnx_sim(onnx_path, opt.onnx_sim)
    logging.info(colored(f"Export {onnx_path} Success.", 'green'))

    # 准备编译的校准数据, yaml文件和脚本
    if "nash" in opt.type:
        ## config yaml
        ### VisionEncoder
        yaml = f'''
model_parameters:
  onnx_model: '{onnx_name_BPU_ACTPolicy_VisionEncoder}'
  march: "{opt.type}"
  layer_out_dump: False
  working_dir: 'bpu_model_output'
  output_model_file_prefix: '{BPU_VisionEncoder}'
input_parameters:
  input_name: ""
  input_type_rt: 'featuremap'
  input_layout_rt: 'NCHW'
  input_type_train: 'featuremap'
  input_layout_train: 'NCHW'
  norm_type: 'no_preprocess'
calibration_parameters:
  cal_data_dir: '{calbrate_data_name_BPU_ACTPolicy_VisionEncoder}'
  cal_data_type: 'float32'
  calibration_type: 'default'
  optimization: set_all_nodes_int16
compiler_parameters:
  extra_params: {{'input_no_padding': True, 'output_no_padding': True}}
  jobs: {opt.combine_jobs}
  compile_mode: 'latency'
  debug: true
  optimize_level: 'O2'
'''
        with open(config_yaml_path_BPU_ACTPolicy_VisionEncoder, "w", encoding="utf-8") as file:
            file.write(yaml)
        logging.info(colored(f"Export config yaml: {config_yaml_path_BPU_ACTPolicy_VisionEncoder} success", 'green'))

        ### TransformerLayers - 动态生成相机配置
        # 构建输入名称字符串
        input_name_list = ['states'] + [f'{camera_name}_features' for camera_name in camera_names]
        input_name_str = ';'.join(input_name_list) + ';'
        
        # 构建输入类型字符串
        input_type_list = ['featuremap'] * len(input_name_list)
        input_type_str = ';'.join(input_type_list) + ';'
        
        # 构建校准数据路径字符串
        cal_data_dirs = [os.path.join(calbrate_data_name_BPU_ACTPolicy_TransformerLayers, "state").replace("\", "/")]
        cal_data_dirs.extend([os.path.join(calbrate_data_name_BPU_ACTPolicy_TransformerLayers, camera_name).replace("\", "/") for camera_name in camera_names])
        cal_data_dir_str = ';'.join(cal_data_dirs) + ';'
        
        # 构建数据类型字符串
        cal_data_type_str = ';'.join(['float32'] * len(input_name_list)) + ';'
        nchw_str = 'NCHW' * len(input_name_list))
        norm_type_str = 'no_preprocess;' * len(input_name_list)
        
        yaml = f'''
model_parameters:
  onnx_model: '{onnx_name_BPU_ACTPolicy_TransformerLayers}'
  march: "{opt.type}"
  layer_out_dump: False
  working_dir: 'bpu_model_output'
  output_model_file_prefix: '{BPU_TransformerLayers}'
input_parameters:
  input_name: "{input_name_str}"
  input_type_rt: '{input_type_str}'
  input_layout_rt: '{nchw_str}'
  input_type_train: '{input_type_str}'
  input_layout_train: '{nchw_str}'
  norm_type: '{norm_type_str}'
calibration_parameters:
  cal_data_dir: '{cal_data_dir_str}'
  cal_data_type: '{cal_data_type_str}'
  calibration_type: 'default'
  optimization: set_all_nodes_int16
compiler_parameters:
  extra_params: {{'input_no_padding': True, 'output_no_padding': True}}
  jobs: {opt.combine_jobs}
  compile_mode: 'latency'
  debug: False
  optimize_level: 'O2'
'''
        with open(config_yaml_path_BPU_ACTPolicy_TransformerLayers, "w", encoding="utf-8") as file:
            file.write(yaml)
        logging.info(colored(f"Export config yaml: {config_yaml_path_BPU_ACTPolicy_TransformerLayers} success", 'green'))
        ## bash scripts
        ### VisionEncoder
        bash = f'''
#!/bin/bash
set -e -v
cd $(dirname $0) || exit
hb_compile --config {config_yaml_name_BPU_ACTPolicy_VisionEncoder}
chmod 777 ./*
cp bpu_model_output/{BPU_VisionEncoder}.hbm ../{bpu_output_name}
'''
        with open(bash_path_BPU_ACTPolicy_VisionEncoder, "w", encoding="utf-8") as file:
            file.write(bash)
        logging.info(colored(f"Export bash scripts: {config_yaml_path_BPU_ACTPolicy_VisionEncoder} success", 'green'))

        ### TransformerLayers
        bash = f'''
#!/bin/bash
set -e -v
cd $(dirname $0) || exit
hb_compile --config {config_yaml_name_BPU_ACTPolicy_TransformerLayers}
chmod 777 ./*
cp bpu_model_output/{BPU_TransformerLayers}.hbm ../{bpu_output_name}
'''
        with open(bash_path_BPU_ACTPolicy_TransformerLayers, "w", encoding="utf-8") as file:
            file.write(bash)
        logging.info(colored(f"Export bash scripts: {bash_path_BPU_ACTPolicy_TransformerLayers} success", 'green'))

        ## all in one bash
        bash = f'''
#!/bin/bash
cd {BPU_VisionEncoder} && bash {bash_name_BPU_ACTPolicy_VisionEncoder} && cd ..
cd {BPU_TransformerLayers} && bash {bash_name_BPU_ACTPolicy_TransformerLayers} && cd ..
echo "End of build all."
'''
        with open(bash_build_all_path, "w", encoding="utf-8") as file:
            file.write(bash)
        logging.info(colored(f"Export bash scripts: {bash_build_all_path} success", 'green'))

        ## calibrate data - 动态生成相机校准数据目录
        input_names_TransformerLayers = camera_names + ["state"]
        input_cal_path = []
        for input_name in input_names_TransformerLayers:
            p = os.path.join(calbrate_data_path_BPU_ACTPolicy_TransformerLayers, input_name)
            input_cal_path.append(p)
            os.makedirs(p, exist_ok=True)
            logging.info(colored(f"mkdir: {p} Success.", 'green'))

        for i, batch in enumerate(dataloader):
            name = "%.10d.npy"%i
            batch = policy.normalize_inputs(batch)
            
            # 动态处理所有相机输入
            camera_inputs = {}
            for camera_name in camera_names:
                camera_inputs[camera_name] = batch[f'observation.images.{camera_name}']
            
            state_input = batch["observation.state"]
            
            ## VisionEncoder - 动态保存所有相机的校准数据
            if i%4 == 0:
                for camera_name in camera_names:
                    p = os.path.join(calbrate_data_path_BPU_ACTPolicy_VisionEncoder, f"{camera_name}_" + name)
                    np.save(p, camera_inputs[camera_name].detach().cpu().numpy())
                    logging.info(colored(f"save to: {p}", 'light_blue'))
            
            ## TransformerLayers - 动态处理所有相机的视觉特征
            for camera_name in camera_names:
                vision_feature = m_VisionEncoder(camera_inputs[camera_name])
                camera_cal_path = os.path.join(calbrate_data_path_BPU_ACTPolicy_TransformerLayers, camera_name)
                p = os.path.join(camera_cal_path, name)
                np.save(p, vision_feature.detach().cpu().numpy())
                logging.info(colored(f"save to: {p}", 'light_magenta'))

            p = os.path.join(state_calbrate_data_path_BPU_ACTPolicy_TransformerLayers, name)
            np.save(p, state_input.detach().cpu().numpy())
            logging.info(colored(f"save to: {p}", 'light_magenta'))

            if i >= opt.cal_num:
                break


    if "bayes" in opt.type:
        ## config yaml
        ### VisionEncoder
        yaml = f'''
model_parameters:
  onnx_model: '{onnx_name_BPU_ACTPolicy_VisionEncoder}'
  march: "{opt.type}"
  layer_out_dump: False
  working_dir: 'bpu_model_output'
  output_model_file_prefix: '{BPU_VisionEncoder}'
input_parameters:
  input_name: ""
  input_type_rt: 'featuremap'
  input_layout_rt: 'NCHW'
  input_type_train: 'featuremap'
  input_layout_train: 'NCHW'
  norm_type: 'no_preprocess'
calibration_parameters:
  cal_data_dir: '{calbrate_data_name_BPU_ACTPolicy_VisionEncoder}'
  cal_data_type: 'float32'
  calibration_type: 'default'
  optimization: set_all_nodes_int16;set_Softmax_input_int16;set_Softmax_output_int16;
compiler_parameters:
  jobs: {opt.combine_jobs}
  compile_mode: 'latency'
  debug: true
  optimize_level: 'O3'
'''
        with open(config_yaml_path_BPU_ACTPolicy_VisionEncoder, "w", encoding="utf-8") as file:
            file.write(yaml)
        logging.info(colored(f"Export config yaml: {config_yaml_path_BPU_ACTPolicy_VisionEncoder} success", 'green'))

        ### TransformerLayers - 动态生成相机配置 (Bayes版本)
        # 构建输入名称字符串
        input_name_list = ['states'] + [f'{camera_name}_features' for camera_name in camera_names]
        input_name_str = ';'.join(input_name_list) + ';'
        
        # 构建输入类型字符串
        input_type_list = ['featuremap'] * len(input_name_list)
        input_type_str = ';'.join(input_type_list) + ';'
        
        # 构建校准数据路径字符串
        cal_data_dirs = [os.path.join(calbrate_data_name_BPU_ACTPolicy_TransformerLayers, "state")]
        cal_data_dirs.extend([os.path.join(calbrate_data_name_BPU_ACTPolicy_TransformerLayers, camera_name) for camera_name in camera_names])
        cal_data_dir_str = ';'.join(cal_data_dirs) + ';'
        
        # 构建数据类型字符串
        cal_data_type_str = ';'.join(['float32'] * len(input_name_list)) + ';'
        nchw_str = ';'.join(['NCHW'] * len(input_name_list)) + ';'
        norm_type_str = ';'.join(['no_preprocess'] * len(input_name_list)) + ';'
        
        yaml = f'''
model_parameters:
  onnx_model: '{onnx_name_BPU_ACTPolicy_TransformerLayers}'
  march: "{opt.type}"
  layer_out_dump: False
  working_dir: 'bpu_model_output'
  output_model_file_prefix: '{BPU_TransformerLayers}'
input_parameters:
  input_name: "{input_name_str}"
  input_type_rt: '{input_type_str}'
  input_layout_rt: '{nchw_str}'
  input_type_train: '{input_type_str}'
  input_layout_train: '{nchw_str}'
  norm_type: '{norm_type_str}'
calibration_parameters:
  cal_data_dir: '{cal_data_dir_str}'
  cal_data_type: '{cal_data_type_str}'
  calibration_type: 'default'
  optimization: set_all_nodes_int16;set_Softmax_input_int16;set_Softmax_output_int16;
compiler_parameters:
  jobs: {opt.combine_jobs}
  compile_mode: 'latency'
  debug: False
  optimize_level: 'O3'
'''
        with open(config_yaml_path_BPU_ACTPolicy_TransformerLayers, "w", encoding="utf-8") as file:
            file.write(yaml)
        logging.info(colored(f"Export config yaml: {config_yaml_path_BPU_ACTPolicy_TransformerLayers} success", 'green'))
        ## bash scripts
        ### VisionEncoder
        bash = f'''
#!/bin/bash
set -e -v
cd $(dirname $0) || exit
hb_mapper makertbin --model-type onnx --config {config_yaml_name_BPU_ACTPolicy_VisionEncoder}
chmod 777 ./*
cp bpu_model_output/{BPU_VisionEncoder}.bin ../{bpu_output_name}
'''
        with open(bash_path_BPU_ACTPolicy_VisionEncoder, "w", encoding="utf-8") as file:
            file.write(bash)
        logging.info(colored(f"Export bash scripts: {config_yaml_path_BPU_ACTPolicy_VisionEncoder} success", 'green'))

        ### TransformerLayers
        bash = f'''
#!/bin/bash
set -e -v
cd $(dirname $0) || exit
hb_mapper makertbin --model-type onnx --config {config_yaml_name_BPU_ACTPolicy_TransformerLayers}
chmod 777 ./*
cp bpu_model_output/{BPU_TransformerLayers}.bin ../{bpu_output_name}
'''
        with open(bash_path_BPU_ACTPolicy_TransformerLayers, "w", encoding="utf-8") as file:
            file.write(bash)
        logging.info(colored(f"Export bash scripts: {bash_path_BPU_ACTPolicy_TransformerLayers} success", 'green'))

        ## all in one bash
        bash = f'''
#!/bin/bash
cd {BPU_VisionEncoder} && bash {bash_name_BPU_ACTPolicy_VisionEncoder} && cd ..
cd {BPU_TransformerLayers} && bash {bash_name_BPU_ACTPolicy_TransformerLayers} && cd ..
echo "End of build all."
'''
        with open(bash_build_all_path, "w", encoding="utf-8") as file:
            file.write(bash)
        logging.info(colored(f"Export bash scripts: {bash_build_all_path} success", 'green'))

        ## calibrate data - 动态生成相机校准数据目录
        input_names_TransformerLayers = camera_names + ["state"]
        input_cal_path = []
        for input_name in input_names_TransformerLayers:
            p = os.path.join(calbrate_data_path_BPU_ACTPolicy_TransformerLayers, input_name)
            input_cal_path.append(p)
            os.makedirs(p, exist_ok=True)
            logging.info(colored(f"mkdir: {p} Success.", 'green'))

        for i, batch in enumerate(dataloader):
            name = "%.10d.nchw"%i
            batch = policy.normalize_inputs(batch)
            
            # 动态处理所有相机输入
            camera_inputs = {}
            for camera_name in camera_names:
                camera_inputs[camera_name] = batch[f'observation.images.{camera_name}']
            
            state_input = batch["observation.state"]
            
            ## VisionEncoder - 动态保存所有相机的校准数据 (Bayes格式)
            if i%4 == 0:
                for camera_name in camera_names:
                    p = os.path.join(calbrate_data_path_BPU_ACTPolicy_VisionEncoder, f"{camera_name}_" + name)
                    camera_inputs[camera_name].detach().cpu().numpy().tofile(p)
                    logging.info(colored(f"save to: {p}", 'light_blue'))
            
            ## TransformerLayers - 动态处理所有相机的视觉特征 (Bayes格式)
            for camera_name in camera_names:
                vision_feature = m_VisionEncoder(camera_inputs[camera_name])
                camera_cal_path = os.path.join(calbrate_data_path_BPU_ACTPolicy_TransformerLayers, camera_name)
                p = os.path.join(camera_cal_path, name)
                vision_feature.detach().cpu().numpy().tofile(p)
                logging.info(colored(f"save to: {p}", 'light_magenta'))

            p = os.path.join(state_calbrate_data_path_BPU_ACTPolicy_TransformerLayers, name)
            state_input.detach().cpu().numpy().tofile(p)
            logging.info(colored(f"save to: {p}", 'light_magenta'))

            if i >= opt.cal_num:
                break

    if "bernoulli2" in opt.type:
        print("I have no time to do this. But Bernoulli2 is very similiar with Bayes.")
        exit()

    # 提示词
    print()
    print(colored("="*80, 'light_green'))
    print(colored(f"Export Success.", 'light_red'))
    os.system(f"tree {opt.export_path} -L 2 -h")
    print()

    print(colored("="*80, 'light_green'))
    print(colored("Reference Command: ", 'light_red'))
    print(f"[Docker] Run Command: [sudo] docker run [--gpus all] -it -v {os.path.join(os.getcwd(), opt.export_path)}:/open_explorer {REPOSITORY}:{TAG}")
    print(f"[BPU] Run Command: bash build_all.sh")
    print()

    print(colored("="*80, 'light_green'), "\n")





def onnx_sim(opt):
    if opt.onnx_sim:
        model_onnx = onnx.load(opt.onnx_path)  # load onnx model
        onnx.checker.check_model(model_onnx)  # check onnx model
        model_onnx, check = simplify(
            model_onnx,
            dynamic_input_shape=False,
            input_shapes=None)
        assert check, 'assert check failed'
        onnx.save(model_onnx, opt.onnx_path)    

class BPU_ACTPolicy_VisionEncoder(nn.Module):
    def __init__(self, act_policy):
        super().__init__()
        self.backbone = deepcopy(act_policy.model.backbone)
        self.encoder_img_feat_input_proj = deepcopy(act_policy.model.encoder_img_feat_input_proj)
    def forward(self, images):
        cam_features = self.backbone(images)["feature_map"]
        cam_features = self.encoder_img_feat_input_proj(cam_features)
        cam_features = cam_features
        return cam_features

class BPU_ACTPolicy_TransformerLayers(nn.Module):
    def __init__(self, act_policy, camera_names):
        super().__init__()
        self.model = deepcopy(act_policy.model)
        self.camera_names = camera_names

    def forward(self, states, *vision_features):
        latent_sample = torch.zeros([1, self.model.config.latent_dim], dtype=torch.float32)

        encoder_in_tokens = [self.model.encoder_latent_input_proj(latent_sample)]
        encoder_in_pos_embed = self.model.encoder_1d_feature_pos_embed.weight.unsqueeze(1).unbind(dim=0)
        encoder_in_tokens.append(self.model.encoder_robot_state_input_proj(states))

        all_cam_features = []
        all_cam_pos_embeds = []

        # 动态处理所有相机的视觉特征
        for vision_feature in vision_features:
            cam_pos_embed = self.model.encoder_cam_feat_pos_embed(vision_feature)
            all_cam_features.append(vision_feature)
            all_cam_pos_embeds.append(cam_pos_embed)


        tokens = []
        for token in encoder_in_tokens:
            tokens.append(token.view(1,1,self.model.config.dim_model))
        all_cam_features = torch.cat(all_cam_features, axis=-1).permute(2, 3, 0, 1).view(-1,1,self.model.config.dim_model)
        tokens.append(all_cam_features)
        encoder_in_tokens = torch.cat(tokens, axis=0)

        pos_embeds = []
        for pos_embed in encoder_in_pos_embed:
            pos_embeds.append(pos_embed.view(1,1,self.model.config.dim_model))
        all_cam_pos_embeds = torch.cat(all_cam_pos_embeds, axis=-1).permute(2, 3, 0, 1).view(-1,1,self.model.config.dim_model)
        pos_embeds.append(all_cam_pos_embeds)
        encoder_in_pos_embed = torch.cat(pos_embeds, axis=0)


        # all_cam_features = torch.cat(all_cam_features, axis=-1)
        # encoder_in_tokens.extend(einops.rearrange(all_cam_features, "b c h w -> (h w) b c"))
        # all_cam_pos_embeds = torch.cat(all_cam_pos_embeds, axis=-1)
        # encoder_in_pos_embed.extend(einops.rearrange(all_cam_pos_embeds, "b c h w -> (h w) b c"))

        # encoder_in_tokens = torch.stack(encoder_in_tokens, axis=0)
        # encoder_in_pos_embed = torch.stack(encoder_in_pos_embed, axis=0)

        encoder_out = self.model.encoder(encoder_in_tokens, pos_embed=encoder_in_pos_embed)

        decoder_in = torch.zeros(
            (self.model.config.chunk_size, 1, self.model.config.dim_model),
            dtype=encoder_in_pos_embed.dtype,
            device=encoder_in_pos_embed.device,
        )
        decoder_out = self.model.decoder(
            decoder_in,
            encoder_out,
            encoder_pos_embed=encoder_in_pos_embed,
            decoder_pos_embed=self.model.decoder_pos_embed.weight.unsqueeze(1),
        )

        decoder_out = decoder_out.transpose(0, 1)

        actions = self.model.action_head(decoder_out)

        return actions

def lerobotTensor2cvmat(tensor):
    img = (tensor*255).permute(0, 2, 3, 1).cpu().numpy().astype(np.uint8)[0,:,:,:]
    # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    return img

def onnx_sim(onnx_path, onnx_sim):   
    if onnx_sim:
        model_onnx = onnx.load(onnx_path)  # load onnx model
        onnx.checker.check_model(model_onnx)  # check onnx model
        model_onnx, check = simplify(
            model_onnx,
            dynamic_input_shape=False,
            input_shapes=None)
        assert check, 'assert check failed'
        onnx.save(model_onnx, onnx_path)    

if __name__ == "__main__":
    init_logging()
    main()

