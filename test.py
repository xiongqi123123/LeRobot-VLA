import numpy as np
import torch
from PIL import Image
from torchvision import transforms
from transformers import SiglipImageProcessor

# 检查 SigLIP 处理器的默认尺寸
processor = SiglipImageProcessor.from_pretrained('/home/qi.xiong/DualArm/LeRobot-VLA/weights/RDT/siglip-so400m-patch14-384')
print('SigLIP processor size:', processor.size)

# 检查实际图像尺寸
test_img = np.random.randint(0, 255, (1080, 1920, 3), dtype=np.uint8)
print('Input image shape:', test_img.shape)

# GPU端处理
img_pil = Image.fromarray(test_img)
processed = processor.preprocess(img_pil, return_tensors='pt')
print('GPU processed shape:', processed['pixel_values'].shape)

# 板端处理 (模拟)
img_resized = transforms.Resize((384, 288))(img_pil)
img_padded = transforms.Pad((0, 48, 0, 48), fill=127)(img_resized)
print('BPU resized shape:', img_resized.size)
print('BPU padded shape:', img_padded.size)