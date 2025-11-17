import subprocess
import sys

cmd = [
    'lerobot-train',
    '--dataset.repo_id=/opt/ml/input/data/dataset/', 
    '--dataset.video_backend=pyav',
    '--policy.type=act',
    '--output_dir=outputs/train/act_so100_test',
    '--job_name=act_so100_test',
    '--wandb.enable=false',
    '--policy.push_to_hub=false',
    '--policy.device=cuda',
    '--batch_size=2',
    '--steps=100000'
]

print(f"Running: {' '.join(cmd)}")
subprocess.run(cmd, check=True)