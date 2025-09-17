# Copyright (c) 2025, Cauchy WuChao, D-Robotics.
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

import numpy as np
import pickle
import os
import yaml
import argparse
import logging
from time import time, sleep

import requests
import onnxruntime as ort
import cv2
import numpy as np
import torch

from diffusers.schedulers.scheduling_ddpm import DDPMScheduler
from diffusers.schedulers.scheduling_dpmsolver_multistep import DPMSolverMultistepScheduler

from server_client import Client
from libpyCauchyKesai import CauchyKesai


logging.basicConfig(
    level = logging.DEBUG,
    format = '[%(name)s] [%(asctime)s.%(msecs)03d] [%(levelname)s] %(message)s',
    datefmt='%H:%M:%S')
logger = logging.getLogger("RDK_RDT")



def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--bpu_rdt_path', type=str, default="./BPU_RDT_Policy/", help="")
    # example: $ tree BPU_RDT_Policy
    # .
    # |-- base.yaml
    # |-- bpu_siglip_so400m_patch14_nashm_384x384_featuremaps.hbm
    # |-- rdt_dit.hbm
    # |-- rdt_img_adaptor.hbm
    # |-- rdt_lang_adaptor.onnx
    # |-- rdt_state_adaptor_1x1x256.onnx
    # `-- rdt_state_adaptor_1x64x256.onnx
    parser.add_argument('--test_data_path', type=str, default="./test_data/", help="")
    # example: $ tree test_data
    # .
    # |-- 9_actions.npy
    # |-- 9_cam_high_0.npy
    # |-- 9_cam_high_1.npy
    # |-- 9_cam_left_wrist_0.npy
    # |-- 9_cam_left_wrist_1.npy
    # |-- 9_cam_right_wrist_0.npy
    # |-- 9_cam_right_wrist_1.npy
    # |-- 9_joints.npy
    # `-- 9_lang_embeddings.pt
    parser.add_argument('--ctrl_freq', type=int, default=25, help="")
    parser.add_argument('--left_arm_dim', type=int, default=0, help="")
    parser.add_argument('--right_arm_dim', type=int, default=6, help="")

    opt = parser.parse_args()
    logger.info(opt)

    with open(os.path.join(opt.bpu_rdt_path, "base.yaml"), "r") as fp:
        config_base_yaml = yaml.safe_load(fp)
    config_base_yaml["arm_dim"] = {"left_arm_dim": opt.left_arm_dim, "right_arm_dim": opt.right_arm_dim}
    config_base_yaml['ctrl_freq'] = opt.ctrl_freq
    m = BPU_RDT_Policy(opt.bpu_rdt_path, config_base_yaml)
    # m = BPU_RDT_Policy(opt.bpu_rdt_path, config_base_yaml, SERVER_URL = 'http://10.64.60.208:5000/process')

    for i in range(100):
        dump_cnt = i + 1
        logger.info(f"=== Compare Actions: {dump_cnt}_actions.npy ===")
        imgs_list = [
        np.load(os.path.join(opt.test_data_path, f"{dump_cnt}_cam_high_0.npy")),
        np.load(os.path.join(opt.test_data_path, f"{dump_cnt}_cam_right_wrist_0.npy")),
        # np.load(os.path.join(opt.test_data_path, f"{dump_cnt}_cam_left_wrist_0.npy")), 
        np.load(os.path.join(opt.test_data_path, f"{dump_cnt}_cam_high_1.npy")), 
        np.load(os.path.join(opt.test_data_path, f"{dump_cnt}_cam_right_wrist_1.npy")), 
        # np.load(os.path.join(opt.test_data_path, f"{dump_cnt}_cam_left_wrist_1.npy")), 
        ]

        proprio = np.load(os.path.join(opt.test_data_path, f"{dump_cnt}_joints.npy"))
        logger.debug(f"{proprio.shape = }")
                                
        lang_embeddings = torch.load(os.path.join(opt.test_data_path, f"{dump_cnt}_lang_embeddings.pt"), map_location=torch.device('cpu')).numpy()
        logger.debug(f"{lang_embeddings.shape = }")

        m.set_lang_condition(lang_embeddings)

        actions_fp32 = np.load(os.path.join(opt.test_data_path, f"{dump_cnt}_actions.npy"))

        actions = m.step(imgs_list, proprio)
        
        cosine_similarity(actions, actions_fp32)
 

def cosine_similarity(A, B):
    A_flat = A.flatten()
    B_flat = B.flatten()
    dot_product = np.dot(A_flat, B_flat)
    norm_A = np.linalg.norm(A_flat)
    norm_B = np.linalg.norm(B_flat)
    if norm_A == 0 or norm_B == 0:
        return 0
    cos = dot_product / (norm_A * norm_B)
    error = A - B
    error_sum = np.sum(abs(error))
    abs_error_A = error_sum / np.sum(abs(A))
    abs_error_B = error_sum / np.sum(abs(B))
    print(f"COS: \033[1;31m{cos:.5f}\033[0m, A: {A.min():.5f} ~ {A.max():.5f}, B: {B.min():.5f} ~ {B.max():.5f}, Error: {error.min():.5f} ~ {error.max():.5f}")


STATE_VEC_IDX_MAPPING = {
    # [0, 10): right arm joint positions
    **{
        'arm_joint_{}_pos'.format(i): i for i in range(10)
    },
    **{
        'right_arm_joint_{}_pos'.format(i): i for i in range(10)
    },
    # [10, 15): right gripper joint positions
    **{
        'gripper_joint_{}_pos'.format(i): i + 10 for i in range(5)
    },
    **{
        'right_gripper_joint_{}_pos'.format(i): i + 10 for i in range(5)
    },
    'gripper_open': 10, # alias of right_gripper_joint_0_pos
    'right_gripper_open': 10,
    # [15, 25): right arm joint velocities
    **{
        'arm_joint_{}_vel'.format(i): i + 15 for i in range(10)
    },
    **{
        'right_arm_joint_{}_vel'.format(i): i + 15 for i in range(10)
    },
    # [25, 30): right gripper joint velocities
    **{
        'gripper_joint_{}_vel'.format(i): i + 25 for i in range(5)
    },
    **{
        'right_gripper_joint_{}_vel'.format(i): i + 25 for i in range(5)
    },
    'gripper_open_vel': 25, # alias of right_gripper_joint_0_vel
    'right_gripper_open_vel': 25,
    # [30, 33): right end effector positions
    'eef_pos_x': 30,
    'right_eef_pos_x': 30,
    'eef_pos_y': 31,
    'right_eef_pos_y': 31,
    'eef_pos_z': 32,
    'right_eef_pos_z': 32,
    # [33, 39): right end effector 6D pose
    'eef_angle_0': 33,
    'right_eef_angle_0': 33,
    'eef_angle_1': 34,
    'right_eef_angle_1': 34,
    'eef_angle_2': 35,
    'right_eef_angle_2': 35,
    'eef_angle_3': 36,
    'right_eef_angle_3': 36,
    'eef_angle_4': 37,
    'right_eef_angle_4': 37,
    'eef_angle_5': 38,
    'right_eef_angle_5': 38,
    # [39, 42): right end effector velocities
    'eef_vel_x': 39,
    'right_eef_vel_x': 39,
    'eef_vel_y': 40,
    'right_eef_vel_y': 40,
    'eef_vel_z': 41,
    'right_eef_vel_z': 41,
    # [42, 45): right end effector angular velocities
    'eef_angular_vel_roll': 42,
    'right_eef_angular_vel_roll': 42,
    'eef_angular_vel_pitch': 43,
    'right_eef_angular_vel_pitch': 43,
    'eef_angular_vel_yaw': 44,
    'right_eef_angular_vel_yaw': 44,
    # [45, 50): reserved 
    # [50, 60): left arm joint positions
    **{
        'left_arm_joint_{}_pos'.format(i): i + 50 for i in range(10)
    },
    # [60, 65): left gripper joint positions
    **{
        'left_gripper_joint_{}_pos'.format(i): i + 60 for i in range(5)
    },
    'left_gripper_open': 60, # alias of left_gripper_joint_0_pos
    # [65, 75): left arm joint velocities
    **{
        'left_arm_joint_{}_vel'.format(i): i + 65 for i in range(10)
    },
    # [75, 80): left gripper joint velocities
    **{
        'left_gripper_joint_{}_vel'.format(i): i + 75 for i in range(5)
    },
    'left_gripper_open_vel': 75, # alias of left_gripper_joint_0_vel
    # [80, 83): left end effector positions
    'left_eef_pos_x': 80,
    'left_eef_pos_y': 81,
    'left_eef_pos_z': 82,
    # [83, 89): left end effector 6D pose
    'left_eef_angle_0': 83,
    'left_eef_angle_1': 84,
    'left_eef_angle_2': 85,
    'left_eef_angle_3': 86,
    'left_eef_angle_4': 87,
    'left_eef_angle_5': 88,
    # [89, 92): left end effector velocities
    'left_eef_vel_x': 89,
    'left_eef_vel_y': 90,
    'left_eef_vel_z': 91,
    # [92, 95): left end effector angular velocities
    'left_eef_angular_vel_roll': 92,
    'left_eef_angular_vel_pitch': 93,
    'left_eef_angular_vel_yaw': 94,
    # [95, 100): reserved
    # [100, 102): base linear velocities
    'base_vel_x': 100,
    'base_vel_y': 101,
    # [102, 103): base angular velocities
    'base_angular_vel': 102,
    # [103, 128): reserved
}
STATE_VEC_LEN = 128

# 仅右臂 6 关节
AGILEX_STATE_INDICES  = [STATE_VEC_IDX_MAPPING[f"right_arm_joint_{i}_pos"] for i in range(6)]


class BPU_RDT_Policy():
    def __init__(self, bpu_rdt_path, config, SERVER_URL=None):

        if SERVER_URL is not None:
            logger.info("Using Double RDK S100(P) mode.")
            self.SERVER_URL = SERVER_URL
            self.vision_emb = self.vision_emb_double
        else:
            logger.info("Using Single RDK S100(P) mode.")
            self.vision_emb = self.vision_emb_single
            
        self.control_frequency = config['ctrl_freq']
        self.pred_horizon = config['common']['action_chunk_size']     # 64
        self.action_dim = config['common']['state_dim']               # 128
        self.num_inference_timesteps = config['model']['noise_scheduler']['num_inference_timesteps']  # 5

        # Create the noise scheduler
        noise_scheduler_config = config['model']['noise_scheduler']
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

        # Load ONNX Models (CPU) and hbm Models (BPU)
        session_options = ort.SessionOptions()
        session_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        session_options.log_severity_level = 4
        session_options.intra_op_num_threads = 6  # 使用全部 6 个核心进行单个操作并行
        session_options.execution_mode = ort.ExecutionMode.ORT_PARALLEL  # 启用并行执行模式
        session_options.inter_op_num_threads = 0  # 让 runtime 自动决定（通常为 1，除非有多个并行子图）

        logger.info("Loading dit ...")
        self.dit = CauchyKesai(os.path.join(bpu_rdt_path, "rdt_dit.hbm"))
        self.dit.s()

        logger.info("Loading img adaptor ...")
        self.img_adaptor = CauchyKesai(os.path.join(bpu_rdt_path, "rdt_img_adaptor.hbm"))
        self.img_adaptor.s()

        logger.info("loading lang adaptor ...")
        self.lang_adaptor = ort.InferenceSession(os.path.join(bpu_rdt_path, "rdt_lang_adaptor.onnx"), sess_options=session_options, providers=['CPUExecutionProvider'])
        self.show_onnx_info(self.lang_adaptor)

        logger.info("Loading state sdaptor 1x1x256 ...")
        self.state_adaptor1 = ort.InferenceSession(os.path.join(bpu_rdt_path, "rdt_state_adaptor_1x1x256.onnx"), sess_options=session_options, providers=['CPUExecutionProvider'])
        self.show_onnx_info(self.state_adaptor1)

        logger.info("Loading state sdaptor 1x64x256 ...")
        self.state_adaptor2 = ort.InferenceSession(os.path.join(bpu_rdt_path, "rdt_state_adaptor_1x64x256.onnx"), sess_options=session_options, providers=['CPUExecutionProvider'])
        self.show_onnx_info(self.state_adaptor2)

        if not os.path.exists(os.path.join(bpu_rdt_path, "bpu_siglip_so400m_patch14_nashm_384x384_featuremaps.hbm")):
            logger.error("Please Download bpu_siglip_so400m_patch14_nashm_384x384_featuremaps.hbm")
            logger.info("command: wget https://archive.d-robotics.cc/downloads/rdk_model_zoo/rdk_s100/RoboticsDiffusionTransformers/bpu_siglip_so400m_patch14_nashm_384x384_featuremaps.hbm")
            exit()
        logger.info("Loading bpu_siglip_so400m_patch14_nashm_384x384_featuremaps.hbm ... (Please wait for 20 seconds.)")
        # 两路相机 × 两帧 = 4 张图
        self.siglip = CauchyKesai(os.path.join(bpu_rdt_path, "bpu_siglip_so400m_patch14_nashm_384x384_featuremaps.hbm"), n_task=4)
        self.siglip.s()

    def show_onnx_info(self, m):
        for input_ in m.get_inputs():
            logger.info(input_)
        for output_ in m.get_outputs():
            logger.info(output_)
    
    def vision_emb_single(self, imgs_list):
        begin_time = time()
        last_hidden_states = []
        # 4 张图：[ext_{t-1}, right_{t-1}, ext_{t}, right_{t}]
        for i in range(4):
            # Pre Process
            img_resized = cv2.resize(imgs_list[i], (384, 288))
            img_padded = cv2.copyMakeBorder( img_resized, 48, 48, 0, 0, cv2.BORDER_CONSTANT, value=(127, 127, 127))
            input_tensor = np.expand_dims(img_padded.transpose((2, 0, 1)), axis=0).astype(np.float32) / 127.5 - 1.0
            # Forward
            begin_time = time()
            self.siglip.start([input_tensor.copy()], task_id=i)
        for i in range(4):
            last_hidden_states.append(torch.from_numpy(self.siglip.wait(task_id=i)[0]))
        logger.debug("\033[92m" + f"SigLIP BPU Forward time = {1000*(time() - begin_time):.2f} ms" + "\033[0m")
        return torch.cat(last_hidden_states, dim=0)

    def vision_emb_double(self, imgs_list):
        begin_time = time()
        last_hidden_states = []
        for i in [0,1,2]:
            # Pre Process
            img_resized = cv2.resize(imgs_list[i], (384, 288))
            img_padded = cv2.copyMakeBorder( img_resized, 48, 48, 0, 0, cv2.BORDER_CONSTANT, value=(127, 127, 127))
            input_tensor = np.expand_dims(img_padded.transpose((2, 0, 1)), axis=0).astype(np.float32) / 127.5 - 1.0
            # Forward
            begin_time = time()
            self.siglip.start([input_tensor.copy()], task_id=i)

        response = requests.post(
            self.SERVER_URL,
            data=pickle.dumps(np.stack([imgs_list[3],imgs_list[4],imgs_list[5]], axis=0), protocol=pickle.HIGHEST_PROTOCOL),
            headers={'Content-Type': 'application/octet-stream'},
            timeout=30  # 设置超时
        )    
        
        if response.status_code != 200:
            raise Exception(f"Server returned status code {response.status_code}")

        datas = torch.from_numpy(pickle.loads(response.content))
        print(f"{datas.shape = }")

        for i in [0,1,2]:
            last_hidden_states.append(torch.from_numpy(self.siglip.wait(task_id=i)[0]))
        logger.debug("\033[92m" + f"SigLIP BPU Forward time = {1000*(time() - begin_time):.2f} ms" + "\033[0m")
        return torch.cat(last_hidden_states + [datas], dim=0)


    def step(self, imgs, joints):
        # imgs: list[np.array]，长度为 4：[ext_{t-1}, right_{t-1}, ext_{t}, right_{t}]，(480, 640, 3), np.uint8, RGB
        # joints: np.array, (6,), np.float32（右臂 6 关节）

        rdt_begin_time = time()

        # img embs
        image_embeds = self.vision_emb(imgs)
        image_embeds = image_embeds.reshape(-1, 1152).unsqueeze(0)

        expected_tokens = 4374
        if image_embeds.shape[1] < expected_tokens:
            pad_tokens = expected_tokens - image_embeds.shape[1]
            pad = torch.zeros((image_embeds.shape[0], pad_tokens, image_embeds.shape[2]), dtype=image_embeds.dtype)
            image_embeds = torch.cat([image_embeds, pad], dim=1)
    
        # joints
        joints = torch.from_numpy(joints[np.newaxis, np.newaxis, :])
        states, state_elem_mask = self._format_joint_to_state(joints)    # (1, 1, 128), (1, 128)
        state_tokens = states[:, -1:, :]  # (1, 1, 128)
        action_mask=state_elem_mask.unsqueeze(1)
        state_tokens = torch.cat([state_tokens, action_mask], dim=2)

        # adaptor
        begin_time = time()
        img_cond = torch.from_numpy(self.img_adaptor([image_embeds.float().contiguous().cpu().detach().numpy()])[0])
        logger.debug("\033[92m" + f"image adaptor time = {1000*(time() - begin_time):.2f} ms" + "\033[0m")

        begin_time = time()
        state_traj = torch.from_numpy(self.state_adaptor1.run([self.state_adaptor1.get_outputs()[0].name], {self.state_adaptor1.get_inputs()[0].name: state_tokens.numpy()})[0])
        logger.debug("\033[92m" + f"state adaptor time = {1000*(time() - begin_time):.2f} ms" + "\033[0m")

        # conditional_sample
        begin_time = time()
        ctrl_freqs = torch.tensor([self.control_frequency])
        lang_attn_mask=torch.ones(self.lang_c.shape[:2], dtype=torch.bool)
        noisy_action = torch.randn(size=(state_traj.shape[0], self.pred_horizon, self.action_dim),dtype=torch.float32)
        action_mask = action_mask.expand(-1, self.pred_horizon, -1)
        # Set step values
        self.noise_scheduler_sample.set_timesteps(self.num_inference_timesteps)
        logger.debug("\033[92m" + f"DiT conditional_sample time = {1000*(time() - begin_time):.2f} ms" + "\033[0m")

        for t in self.noise_scheduler_sample.timesteps:
            begin_time = time()
            # Prepare state-action trajectory
            action_traj = torch.cat([noisy_action, action_mask], dim=2)
            action_traj = torch.from_numpy(self.state_adaptor2.run([self.state_adaptor2.get_outputs()[0].name], {self.state_adaptor2.get_inputs()[0].name: action_traj.numpy()})[0])
            state_action_traj = torch.cat([state_traj, action_traj], dim=1)
            # Predict the model output
            x = state_action_traj.float().contiguous().cpu().detach().numpy()
            freq = ctrl_freqs.float().contiguous().cpu().detach().numpy()
            t_ = t.float().contiguous().cpu().detach().numpy()
            lang_c = self.lang_c.float().contiguous().cpu().detach().numpy()
            img_c = img_cond.float().contiguous().cpu().detach().numpy()
            lang_mask = lang_attn_mask.float().contiguous().cpu().detach().numpy()
            pad_rows = 64 - lang_mask.shape[1]
            padded = np.pad(lang_mask, ((0,0), (0,pad_rows)), mode="constant")
            mask_float = np.where(padded, 0.0, -512.0).astype(np.float32)
            lang_cond_padded = np.pad(lang_c, pad_width=((0, 0), (0, pad_rows), (0,0)), mode="constant", constant_values=0)
            logger.debug("\033[92m" + f"DiT PreProcess time = {1000*(time() - begin_time):.2f} ms" + "\033[0m")
            # BPU
            begin_time = time()
            model_output = torch.from_numpy(self.dit([x.copy(), freq.astype(np.int32).copy(), np.expand_dims(t_.astype(np.int32), axis=0).copy(), lang_cond_padded.copy(), img_c.copy(), mask_float.copy()])[0])
            logger.debug("\033[92m" + f"DiT BPU Forward time = {1000*(time() - begin_time):.2f} ms" + "\033[0m")

            # Compute previous actions: x_t -> x_t-1
            begin_time = time()
            noisy_action = self.noise_scheduler_sample.step(model_output, t, noisy_action).prev_sample
            noisy_action = noisy_action.to(state_traj.dtype)
            logger.debug("\033[92m" + f"DiT BPU PostProcess time = {1000*(time() - begin_time):.2f} ms" + "\033[0m")

        # Finally apply the action mask to mask invalid action dimensions
        noisy_action = noisy_action * action_mask
        actions = self._unformat_action_to_joint(noisy_action).to(torch.float32).squeeze(0).cpu().numpy()

        logger.info("\033[1;31m" + f"BPU RDT time = {1000*(time() - rdt_begin_time):.2f} ms" + "\033[0m")
        return actions

    def set_lang_condition(self, lang_emb):
        # lang_emb: np.array, (1, length, 4096)
        begin_time = time()
        self.lang_c = torch.from_numpy(self.lang_adaptor.run([self.lang_adaptor.get_outputs()[0].name], {self.lang_adaptor.get_inputs()[0].name: lang_emb})[0])
        logger.debug("\033[1;31m" + "set_lang_condition time = %.2f ms"%(1000*(time() - begin_time)) + "\033[0m")
        logger.debug(f"Language Condition Shape: {self.lang_c.shape}")

    def _format_joint_to_state(self, joints):
        joints = joints / torch.tensor(
            [[[180, 180, 180, 180, 180, 180]]],
            device=joints.device,
            dtype=joints.dtype,
        )
        B, N, _ = joints.shape
        state = torch.zeros(
            (B, N, self.action_dim), 
            device=joints.device, dtype=joints.dtype
        )
        state[:, :, AGILEX_STATE_INDICES] = joints
        state_elem_mask = torch.zeros(
            (B, self.action_dim),
            device=joints.device, dtype=joints.dtype
        )
        state_elem_mask[:, AGILEX_STATE_INDICES] = 1
        return state, state_elem_mask

    def _unformat_action_to_joint(self, action):
        action_indices = AGILEX_STATE_INDICES
        joints = action[:, :, action_indices]
        joints = joints * torch.tensor(
            [[[180, 180, 180, 180, 180, 180]]],
            device=joints.device,
            dtype=joints.dtype,
        )
        return joints
    

if __name__ == "__main__":
    main()