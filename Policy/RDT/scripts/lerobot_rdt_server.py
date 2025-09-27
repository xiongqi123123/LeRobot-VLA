import os, sys

import numpy as np
import torch
from PIL import Image
from torchvision import transforms
import yaml

from pathlib import Path

# get current workspace
current_file = Path(__file__)
sys.path.append(os.path.join(current_file.parent.parent, "models"))
sys.path.append(os.path.join(current_file.parent.parent, "models"))
sys.path.append(os.path.join(current_file.parent.parent))  
from configs.state_vec import STATE_VEC_IDX_MAPPING
from multimodal_encoder.siglip_encoder import SiglipVisionTower
from multimodal_encoder.t5_encoder import T5Embedder
from rdt_runner import RDTRunner
from server_client import *

# The indices that the raw vector should be mapped to in the unified action vector
AGILEX_STATE_INDICES = [
    STATE_VEC_IDX_MAPPING[f"right_arm_joint_{i}_pos"] for i in range(6)
]

# Create the RDT model
def create_model(args, **kwargs):
    model = RoboticDiffusionTransformerModel(args, **kwargs)
    pretrained = kwargs.get("pretrained", None)
    if pretrained is not None and os.path.isfile(pretrained):
        model.load_pretrained_weights(pretrained)

    return model

class RoboticDiffusionTransformerModel(object):
    """A wrapper for the RDT model, which handles
    1. Model initialization
    2. Encodings of instructions
    3. Model inference
    """

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

        self.reset()

    def get_policy(self, pretrained):
        """Initialize the model."""
        # Initialize model with arguments
        if pretrained is None or os.path.isfile(pretrained):
            img_cond_len = (self.args["common"]["img_history_size"] * self.args["common"]["num_cameras"] *
                            self.vision_model.num_patches)

            _model = RDTRunner(
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
            _model = RDTRunner.from_pretrained(pretrained)

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
        """Set model to evaluation mode."""
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
        """Encode string instruction to latent embeddings.

        Args:
            instruction: a string of instruction
            device: a string of device

        Returns:
            pred: a tensor of latent embeddings of shape (text_max_length, 512)
        """
        tokens = self.text_tokenizer(instruction, return_tensors="pt", padding="longest",
                                     truncation=True)["input_ids"].to(device)

        tokens = tokens.view(1, -1)
        with torch.no_grad():
            pred = self.text_model(tokens).last_hidden_state.detach()

        return pred

    def _format_joint_to_state(self, joints):
        """
        Format the joint proprioception into the unified action vector.

        Args:
            joints (torch.Tensor): The joint proprioception to be formatted.
                qpos ([B, N, 14]).

        Returns:
            state (torch.Tensor): The formatted vector for RDT ([B, N, 128]).
        """
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
        """
        Unformat the unified action vector into the joint action to be executed.

        Args:
            action (torch.Tensor): The unified action vector to be unformatted.
                ([B, N, 128])

        Returns:
            joints (torch.Tensor): The unformatted robot joint action.
                qpos ([B, N, 14]).
        """
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
        """
        Predict the next action chunk given the
        proprioceptive states, images, and instruction embeddings.

        Args:
            proprio: proprioceptive states
            images: RGB images, the order should be
                [ext_{t-1}, right_wrist_{t-1}, left_wrist_{t-1},
                ext_{t}, right_wrist_{t}, left_wrist_{t}]
            text_embeds: instruction embeddings

        Returns:
            action: predicted action
        """
        device = self.device
        dtype = self.dtype

        # The background image used for padding
        background_color = np.array([int(x * 255) for x in self.image_processor.image_mean],
                                    dtype=np.uint8).reshape(1, 1, 3)
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
                image = Image.fromarray(background_image)
            else:
                # Convert numpy array to PIL Image if needed
                if isinstance(image, np.ndarray):
                    image = Image.fromarray(image)

            if self.image_size is not None:
                image = transforms.Resize(self.image_size)(image)

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
                        result = Image.new(pil_img.mode, (width, width), background_color)
                        result.paste(pil_img, (0, (width - height) // 2))
                        return result
                    else:
                        result = Image.new(pil_img.mode, (height, height), background_color)
                        result.paste(pil_img, ((height - width) // 2, 0))
                        return result

                image = expand2square(image, tuple(int(x * 255) for x in self.image_processor.image_mean))
            image = self.image_processor.preprocess(image, return_tensors="pt")["pixel_values"][0]
            image_tensor_list.append(image)

        image_tensor = torch.stack(image_tensor_list, dim=0).to(device, dtype=dtype)

        image_embeds = self.vision_model(image_tensor).detach()
        image_embeds = image_embeds.reshape(-1, self.vision_model.hidden_size).unsqueeze(0)

        # Prepare the proprioception states and the control frequency
        # Convert numpy array to tensor if needed
        if isinstance(proprio, np.ndarray):
            # Copy the array to make it writable
            proprio = torch.from_numpy(proprio.copy())
        
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

class LERobotRDTServer:
    def __init__(self, pretrained_vision_encoder_name_or_path, pretrained, args, lang_model):
        self.policy = create_model(
            args=args,
            dtype=torch.bfloat16,
            pretrained=pretrained,
            pretrained_vision_encoder_name_or_path=pretrained_vision_encoder_name_or_path,
            control_frequency=30,
        )
        self.server = Server(host="0.0.0.0", port=5002)
        
        # Load and debug language embeddings
        self.lang_embeddings = torch.load(lang_model)
        print(f"Loaded language embeddings shape: {self.lang_embeddings.shape}")
        print(f"Model expects tokenizer_max_length: {self.policy.args['dataset']['tokenizer_max_length']}")
        print(f"Model lang_token_dim: {self.policy.args['model']['lang_token_dim']}")
        
        # Check if dimensions match
        expected_seq_len = self.policy.args["dataset"]["tokenizer_max_length"]
        expected_hidden_dim = self.policy.args["model"]["lang_token_dim"]
        
        # Handle different embedding formats
        if len(self.lang_embeddings.shape) == 2:
            # Format: [seq_len, hidden_dim]
            actual_seq_len, actual_hidden_dim = self.lang_embeddings.shape
            if actual_seq_len != expected_seq_len:
                print(f"WARNING: Sequence length mismatch! Expected {expected_seq_len}, got {actual_seq_len}")
            if actual_hidden_dim != expected_hidden_dim:
                print(f"WARNING: Hidden dimension mismatch! Expected {expected_hidden_dim}, got {actual_hidden_dim}")
        elif len(self.lang_embeddings.shape) == 3:
            # Format: [batch_size, seq_len, hidden_dim]
            actual_batch, actual_seq_len, actual_hidden_dim = self.lang_embeddings.shape
            if actual_seq_len != expected_seq_len:
                print(f"WARNING: Sequence length mismatch! Expected {expected_seq_len}, got {actual_seq_len}")
            if actual_hidden_dim != expected_hidden_dim:
                print(f"WARNING: Hidden dimension mismatch! Expected {expected_hidden_dim}, got {actual_hidden_dim}")
        else:
            print(f"WARNING: Unexpected embedding shape: {self.lang_embeddings.shape}")
        
    def run(self):
        print("LERobot RDT Server started, waiting for messages...")
        try:
            while True:
                print("Waiting for RDT data...")
                rdt_data = self.server.receive()
                print(f"Received RDT data, message_id: {rdt_data['message_id']}")
                
                # Perform inference
                # Ensure language embeddings have correct shape
                if len(self.lang_embeddings.shape) == 2:
                    # [seq_len, hidden_dim] -> [1, seq_len, hidden_dim]
                    text_embeds = self.lang_embeddings.unsqueeze(0)
                else:
                    # Already [batch_size, seq_len, hidden_dim]
                    text_embeds = self.lang_embeddings
                
                action = self.policy.step(
                    proprio=rdt_data["proprio"],
                    images=rdt_data["images"],
                    text_embeds=text_embeds,
                )
                
                # Convert tensor to numpy for serialization
                if isinstance(action, torch.Tensor):
                    action_np = action.cpu().numpy()
                else:
                    action_np = action
                
                # Prepare response - use 'actions' key to match client expectation
                message_id = rdt_data["message_id"]
                action_data = {
                    "message_id": message_id,
                    "actions": action_np,  # Now numpy array, can be serialized
                }
                
                # Send response
                print(f"send action data, action_data: {action_data}")
                self.server.send(action_data)
                print(f"Sent action data for message_id: {message_id}")
                
        except KeyboardInterrupt:
            print("\nServer stopped by user")
            self.server.close()
        except Exception as e:
            print(f"Error in server loop: {e}")
            self.server.close()
            raise

if __name__ == "__main__":
    path_to_rdt_model_wights = "/home/qi.xiong/DualArm/RoboTwin/policy/RDT-LeRobot/checkpoints/RDT170M-LeRobot/checkpoint-10000/pytorch_model/mp_rank_00_model_states.pt"
    path_to_vision_encoder_model = "/home/qi.xiong/DualArm/RoboTwin/policy/weights/RDT/siglip-so400m-patch14-384"
    lang_model = "/home/qi.xiong/Data_Qi/LeRobot/RDT170M-LeRobot/blackmarker_scence1/episode_2/instructions/lang_embed_0.pt"
    with open("/home/qi.xiong/DualArm/RoboTwin/policy/RDT-LeRobot/configs/base.yaml", "r") as fp:
        config = yaml.safe_load(fp)
    rdt_server = LERobotRDTServer(path_to_vision_encoder_model, path_to_rdt_model_wights, config, lang_model)
    rdt_server.run()
