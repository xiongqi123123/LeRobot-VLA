import logging
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from pprint import pformat
from collections import deque
import torch
from PIL import Image
import numpy as np
from lerobot.cameras import (  # noqa: F401
    CameraConfig,  # noqa: F401
)
from lerobot.cameras.opencv.configuration_opencv import OpenCVCameraConfig  # noqa: F401
from lerobot.configs import parser
from lerobot.configs.policies import PreTrainedConfig
from lerobot.datasets.image_writer import safe_stop_image_writer
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.datasets.utils import build_dataset_frame, hw_to_dataset_features
from lerobot.datasets.video_utils import VideoEncodingManager
from lerobot.policies.factory import make_policy
from lerobot.policies.pretrained import PreTrainedPolicy
from lerobot.robots import (  # noqa: F401
    Robot,
    RobotConfig,
    bi_so100_follower,
    hope_jr,
    koch_follower,
    make_robot_from_config,
    so100_follower,
    so101_follower,
)
from lerobot.robots.so101_follower.so101_follower import SO101Follower
from lerobot.robots.so101_follower.config_so101_follower import SO101FollowerConfig
from lerobot.teleoperators import (  # noqa: F401
    Teleoperator,
    TeleoperatorConfig,
    bi_so100_leader,
    homunculus,
    koch_leader,
    make_teleoperator_from_config,
    so100_leader,
    so101_leader,
)
from lerobot.teleoperators.keyboard.teleop_keyboard import KeyboardTeleop
from lerobot.utils.control_utils import (
    init_keyboard_listener,
    is_headless,
    predict_action,
    sanity_check_dataset_name,
    sanity_check_dataset_robot_compatibility,
)
from lerobot.utils.robot_utils import busy_wait
from lerobot.utils.visualization_utils import _init_rerun, log_rerun_data
from server_client import *

debug_save_img = True 

_action_queue = deque([], maxlen=64)
_message_id = 0 
_last_cam_high = None  
_last_cam_right_wrist = None  

@safe_stop_image_writer
def record_loop(
    robot: Robot,
    client: Client,
):
    global _action_queue, _message_id, _last_cam_high, _last_cam_right_wrist

    observation = robot.get_observation()

    cam_high = observation['high']
    cam_right_wrist = observation['arm']
    image_arrs = [
        _last_cam_high,
        _last_cam_right_wrist,
        None,
        cam_high,
        cam_right_wrist,
        None
    ]

    images = [arr if arr is not None else None
            for arr in image_arrs]
    joint_positions = [observation[key] for key in observation.keys() if key.endswith('.pos')]
    proprio = torch.tensor(joint_positions, dtype=torch.float32).unsqueeze(0)

###################Debug图像########################
    if debug_save_img:
        imgs_to_show = [cam_high, cam_right_wrist, _last_cam_high, _last_cam_right_wrist]
        if all(img is not None for img in imgs_to_show):
            pil_imgs = []
            for img in imgs_to_show:
                if img.dtype != np.uint8:
                    img = np.clip(img, 0, 1)
                    img = (img * 255).astype(np.uint8)
                if img.ndim == 2:
                    img = np.stack([img]*3, axis=-1)  
                elif img.shape[-1] == 1:
                    img = np.repeat(img, 3, axis=-1)
                pil_imgs.append(Image.fromarray(img))
            w, h = pil_imgs[0].size
            for i in range(4):
                if pil_imgs[i].size != (w, h):
                    pil_imgs[i] = pil_imgs[i].resize((w, h))
            new_img = Image.new('RGB', (w*2, h*2))
            new_img.paste(pil_imgs[0], (0, 0))       # 左上：新high
            new_img.paste(pil_imgs[1], (w, 0))       # 右上：新wrist
            new_img.paste(pil_imgs[2], (0, h))       # 左下：老high
            new_img.paste(pil_imgs[3], (w, h))       # 右下：老wrist

            debug_save_path = "debug_2x2.png"
            new_img.save(debug_save_path)
            print(f"Have been saved at: {debug_save_path}")
            # new_img.show()
###################Debug图像########################
    rdt_data = {
        'message_id': _message_id,
        'proprio': proprio,
        'images': images,
        'text_embeds': ""
    }
    client.send(rdt_data)
    _message_id += 1 
    print(f"send new rdt data done, message_id: {_message_id-1}")
    action_data = client.receive()
    if action_data is None:
        print("ERROR: Server returned None. Is the RDT server running?")
        print("Please start the RDT server first!")
        raise ConnectionError("Failed to receive response from RDT server")
    actions = action_data['actions']
    action_message_id = action_data["message_id"]
    print(f"receive actions done, message_id: {action_message_id}")
    # print(f"receive actions contents: {actions}")
    actions_array = np.array(actions)
    if len(actions_array.shape) == 3: 
        action_sequence = actions_array[0, :, :]  # 取第一个batch的所有时间步
    else:
        print(f"action shape should be 3 dim, but get {actions_array.shape} ")
    
    joint_names = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper"]
    for step_idx in range(0, len(action_sequence), 4):  # 64个动作隔4个执行一次动作
        action_values = action_sequence[step_idx]
        action_dict = {f"{joint}.pos": float(action_values[i]) for i, joint in enumerate(joint_names)}
        sent_action = robot.send_action(action_dict)
        time.sleep(0.1)  

    _last_cam_high = cam_high
    _last_cam_right_wrist = cam_right_wrist



def main():
    robot = SO101Follower(SO101FollowerConfig(
        port="/dev/tty.usbmodem5AB90671801",
        id="my_awesome_follower_arm",
        cameras={
            "arm": OpenCVCameraConfig(index_or_path=0, width=1920, height=1080, fps=30),
            "high": OpenCVCameraConfig(index_or_path=2, width=1920, height=1080, fps=30)
        }
    ))
    
    robot.connect()
    client = Client(host="localhost", port=5002)

    try:
        while True:
            record_loop(
                robot,
                client
            )
            time.sleep(0.1)
    except KeyboardInterrupt:
        pass

    robot.disconnect()


if __name__ == "__main__":
    main()
