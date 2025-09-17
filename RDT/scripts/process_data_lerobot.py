#!/usr/bin/env python3
"""
LeRobot到RDT数据转换脚本

LeRobot机器人结构：
- 5个关节 (shoulder_pan, shoulder_lift, elbow_flex, wrist_flex, wrist_roll)
- 1个夹爪 (gripper)
- 总计：6个自由度 (6DOF)

维度映射（匹配RDT训练代码）：
- left_arm_dim = 0 (单臂机器人，左臂不存在)
- right_arm_dim = 6 (5关节 + 1夹爪，映射到RDT的right_arm部分)
- 状态向量：6维 [joint1, joint2, joint3, joint4, joint5, gripper]
- RDT索引映射：right_arm_joint_0_pos到right_arm_joint_5_pos (索引0-5)
"""

import sys
import os
import h5py
import numpy as np
import cv2
import argparse
import yaml
import json
import subprocess
from pathlib import Path
import pandas as pd
import torch

current_dir = os.path.dirname(__file__)
sys.path.append(os.path.join(current_dir, ".."))
from models.multimodal_encoder.t5_encoder import T5Embedder

def extract_frames_from_video(video_path, output_dir, episode_idx):
    if not os.path.exists(video_path):
        print(f"  No video file: {video_path}")
        return []
    
    temp_dir = os.path.join(output_dir, f"temp_frames_{episode_idx}")
    if not os.path.exists(temp_dir):
        os.makedirs(temp_dir)
    
    output_pattern = os.path.join(temp_dir, "frame_%04d.jpg")
    
    try:
        cmd = [
            'ffmpeg', '-i', video_path,
            '-vf', 'fps=30',
            '-q:v', '2',
            output_pattern,
            '-y'
        ]
        
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        if result.returncode != 0:
            print(f"  Failed to extract frames with ffmpeg: {result.stderr}")
            return []
        
        frames = []
        frame_files = sorted([f for f in os.listdir(temp_dir) if f.endswith('.jpg')])
        
        for frame_file in frame_files:
            frame_path = os.path.join(temp_dir, frame_file)
            frame = cv2.imread(frame_path)
            if frame is not None:
                frame_resized = cv2.resize(frame, (640, 480))
                frames.append(frame_resized)
        
        print(f"  Successfully extracted {len(frames)} frames")
        
        for frame_file in frame_files:
            os.remove(os.path.join(temp_dir, frame_file))
        os.rmdir(temp_dir)
        
        return frames
        
    except Exception as e:
        print(f"  Error extracting frames: {e}")
        return []

def load_lerobot_episode(data_dir, episode_idx, output_dir):
    """加载LeRobot的单个episode数据
    
    LeRobot数据结构：
    - action: 6维 [shoulder_pan, shoulder_lift, elbow_flex, wrist_flex, wrist_roll, gripper]
    - observation.state: 6维 [shoulder_pan, shoulder_lift, elbow_flex, wrist_flex, wrist_roll, gripper]
    - 图像: 高位相机 + 手臂相机
    """
    parquet_path = os.path.join(data_dir, "data/chunk-000", f"episode_{episode_idx:06d}.parquet")
    if not os.path.exists(parquet_path):
        print(f"Episode {episode_idx} parquet file does not exist: {parquet_path}")
        return None
    
    df = pd.read_parquet(parquet_path)
    
    actions = []
    qpos = []
    
    for i in range(len(df)):
        action = df['action'].iloc[i]
        state = df['observation.state'].iloc[i]
        
        if isinstance(action, np.ndarray):
            actions.append(action.astype(np.float32))
        else:
            actions.append(np.array(action, dtype=np.float32))
            
        if isinstance(state, np.ndarray):
            qpos.append(state.astype(np.float32))
        else:
            qpos.append(np.array(state, dtype=np.float32))
    
    high_cam_path = os.path.join(data_dir, "videos/chunk-000/observation.images.front", f"episode_{episode_idx:06d}.mp4")
    arm_cam_path = os.path.join(data_dir, "videos/chunk-000/observation.images.arm", f"episode_{episode_idx:06d}.mp4")
    
    print(f"  Extracting high camera frames...")
    high_images = extract_frames_from_video(high_cam_path, output_dir, episode_idx)
    
    print(f"  Extracting arm camera frames...")
    arm_images = extract_frames_from_video(arm_cam_path, output_dir, episode_idx)
    
    target_frames = len(df)
    if len(high_images) > target_frames:
        high_images = high_images[:target_frames]
    if len(arm_images) > target_frames:
        arm_images = arm_images[:target_frames]
    
    while len(high_images) < target_frames and high_images:
        high_images.append(high_images[-1])
    while len(arm_images) < target_frames and arm_images:
        arm_images.append(arm_images[-1])
    
    return {
        'actions': np.array(actions),
        'qpos': np.array(qpos),
        'high_images': high_images,
        'arm_images': arm_images,
        'episode_length': len(df)
    }

def images_encoding(imgs):
    if not imgs:
        return [], 0
        
    encode_data = []
    padded_data = []
    max_len = 0
    
    for i in range(len(imgs)):
        success, encoded_image = cv2.imencode(".jpg", imgs[i])
        if success:
            jpeg_data = encoded_image.tobytes()
            encode_data.append(jpeg_data)
            max_len = max(max_len, len(jpeg_data))
        else:
            print(f"  Image encoding failed: {i}")
            empty_data = b""
            encode_data.append(empty_data)
    
    for i in range(len(imgs)):
        padded_data.append(encode_data[i].ljust(max_len, b"\0"))
    
    return encode_data, max_len

def load_task_instructions(data_dir):
    tasks_file = os.path.join(data_dir, "meta/tasks.jsonl")
    if not os.path.exists(tasks_file):
        print(f"Warning: tasks file not found: {tasks_file}")
        return None
    
    instructions = []
    with open(tasks_file, 'r') as f:
        for line in f:
            if line.strip():
                task_data = json.loads(line.strip())
                instructions.append(task_data["task"])
    
    print(f"  加载了 {len(instructions)} 个任务指令")
    return instructions

def encode_language_instruction(instruction_text, t5_embedder, device):
    try:
        text_embeds, attn_mask = t5_embedder.get_text_embeddings([instruction_text])
        
        valid_embeds = text_embeds[0][attn_mask[0]].float()
        return valid_embeds.cpu().numpy()
        
    except Exception as e:
        print(f"  Language encoding failed: {e}")
        return np.zeros((1, 4096))

def convert_lerobot_to_rdt(data_dir, output_dir, episode_num, gpu=0, no_language=False):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    print(f"Start converting LeRobot data to RDT format...")
    print(f"Data source: {data_dir}")
    print(f"Output directory: {output_dir}")
    print(f"Processing episode number: {episode_num}")
    print(f"GPU device: {gpu}")
    
    scene_name = os.path.basename(data_dir)
    
    instructions = None
    if not no_language:
        instructions = load_task_instructions(data_dir)
    
    t5_embedder = None
    if not no_language and instructions:
        try:
            print(f"  Initializing T5 encoder...")
            t5_model_path = "/home/qi.xiong/Data_Qi/t5-v1_1-xxl"
            if not os.path.exists(t5_model_path):
                print(f"  Warning: T5 model path does not exist: {t5_model_path}")
                print(f"  Will skip language processing")
                no_language = True
            else:
                t5_embedder = T5Embedder(
                    from_pretrained=t5_model_path,
                    device=f"cuda:{gpu}" if torch.cuda.is_available() else "cpu",
                    model_max_length=120,
                    use_offload_folder=None,
                )
                print(f"  T5 encoder initialized successfully")
        except Exception as e:
            print(f"  T5 encoder initialization failed: {e}")
            print(f"  Will skip language processing")
            no_language = True
    
    for i in range(episode_num):
        print(f"Processing episode {i}...")
        
        episode_data = load_lerobot_episode(data_dir, i, output_dir)
        if episode_data is None:
            print(f"Skipping episode {i}")
            continue
        
        episode_output_dir = os.path.join(output_dir, f"episode_{i}")
        if not os.path.exists(episode_output_dir):
            os.makedirs(episode_output_dir)
        
        hdf5_path = os.path.join(episode_output_dir, f"episode_{i}.hdf5")
        
        with h5py.File(hdf5_path, "w") as f:
            f.create_dataset("action", data=episode_data['actions'])
            
            obs = f.create_group("observations")
            obs.create_dataset("qpos", data=episode_data['qpos'])
            
            image = obs.create_group("images")
            
            if episode_data['high_images']:
                print(f"  Encoding high camera images...")
                high_enc, len_high = images_encoding(episode_data['high_images'])
                if high_enc and len_high > 0:
                    image.create_dataset("cam_high", data=high_enc, dtype=f"S{len_high}")
                    print(f"  Saved high camera images: {len(episode_data['high_images'])} frames")
                else:
                    print(f"  Warning: High camera images encoding failed")
            
            if episode_data['arm_images']:
                print(f"  Encoding arm camera images...")
                arm_enc, len_arm = images_encoding(episode_data['arm_images'])
                if arm_enc and len_arm > 0:
                    image.create_dataset("cam_right_wrist", data=arm_enc, dtype=f"S{len_arm}")
                    print(f"  Saved arm camera images: {len(episode_data['arm_images'])} frames")
                else:
                    print(f"  Warning: Arm camera images encoding failed")
            
            # 添加机器人维度信息（LeRobot: 5个关节 + 1个夹爪）
            # 根据process_data.py的逻辑，每个时间步都需要记录维度信息
            # LeRobot是单臂机器人，只有右臂：5个关节 + 1个夹爪 = 6维
            # 左臂：0维（单臂机器人）
            
            # 为每个时间步记录维度信息
            left_arm_dim = [0] * len(episode_data['actions'])  # 左臂0维（单臂机器人）
            right_arm_dim = [6] * len(episode_data['actions'])  # 右臂6维（5关节+1夹爪）
            
            obs.create_dataset("left_arm_dim", data=np.array(left_arm_dim))
            obs.create_dataset("right_arm_dim", data=np.array(right_arm_dim))
        
        print(f"  Episode {i} converted successfully: {hdf5_path}")
        print(f"  Data length: {episode_data['episode_length']}")
        print(f"  Action shape: {episode_data['actions'].shape}")
        print(f"  Qpos shape: {episode_data['qpos'].shape}")
        print(f"  High camera frames: {len(episode_data['high_images'])}")
        print(f"  Arm camera frames: {len(episode_data['arm_images'])}")
        
        if not no_language and t5_embedder and instructions:
            print(f"  Processing language instructions...")
            try:
                instruction = instructions[0]
                
                language_features = encode_language_instruction(instruction, t5_embedder, f"cuda:{gpu}")
                
                instructions_dir = os.path.join(episode_output_dir, "instructions")
                if not os.path.exists(instructions_dir):
                    os.makedirs(instructions_dir)
                
                lang_embed_path = os.path.join(instructions_dir, "lang_embed_0.pt")
                torch.save(torch.from_numpy(language_features), lang_embed_path)
                
                print(f"  Language instruction encoded successfully: {instruction}")
                print(f"  Language features saved to: {lang_embed_path}")
                print(f"  Language features shape: {language_features.shape}, data type: {language_features.dtype}")
                
            except Exception as e:
                print(f"  Language instruction processing failed: {e}")
    
    print(f"\nConversion completed! Processed {episode_num} episodes")
    print(f"Output directory: {output_dir}")

def main():
    parser = argparse.ArgumentParser(description="Convert LeRobot data to RDT format")
    parser.add_argument("--data_dir", type=str, required=True, 
                       help="LeRobot data directory path")
    parser.add_argument("--output_dir", type=str, required=True,
                       help="Output directory path")
    parser.add_argument("--episode_num", type=int, default=5,
                       help="Number of episodes to process")
    parser.add_argument("--gpu", type=int, default=0,
                       help="GPU device ID")
    parser.add_argument("--no_language", action="store_true",
                       help="Skip language processing")
    
    args = parser.parse_args()
    
    if not os.path.exists(args.data_dir):
        print(f"Error: Data directory does not exist: {args.data_dir}")
        return
    
    meta_file = os.path.join(args.data_dir, "meta/info.json")
    if not os.path.exists(meta_file):
        print(f"Error: Meta information file not found: {meta_file}")
        return
    
    try:
        subprocess.run(['ffmpeg', '-version'], capture_output=True, check=True)
        print("ffmpeg is available, will use ffmpeg to extract video frames")
    except (subprocess.CalledProcessError, FileNotFoundError):
        print("Warning: ffmpeg is not available, image data may not be extracted correctly")
        print("Please install ffmpeg: conda install -c conda-forge ffmpeg=6.1")
        return
    
    with open(meta_file, 'r') as f:
        meta_info = yaml.safe_load(f)
    
    total_episodes = meta_info.get('total_episodes', 10)
    if args.episode_num > total_episodes:
        print(f"Warning: Requested episode number ({args.episode_num}) exceeds available number ({total_episodes})")
        args.episode_num = total_episodes
    
    convert_lerobot_to_rdt(
        args.data_dir, 
        args.output_dir, 
        args.episode_num,
        args.gpu,
        args.no_language
    )

if __name__ == "__main__":
    main()
