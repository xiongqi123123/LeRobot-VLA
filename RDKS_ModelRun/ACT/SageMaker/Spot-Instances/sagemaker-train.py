import time
import os
import re
from sagemaker import Session
from sagemaker.pytorch import PyTorch
from sagemaker.inputs import TrainingInput
import boto3

# 配置区域与资源
ROLE = "arn:aws:iam::your-account-id:role/robotics"
REGION = "ap-northeast-2"
TRAINING_SCRIPT = "train.py"
SOURCE_DIR = "/home/ssm-user/LeRobot-VLA"

# 所有数据集目录
DATASET_URIS = [
    "s3://skyxz-test/lerobot-data/blackmarker_scence1/",
    "s3://skyxz-test/lerobot-data/blackmarker_scence2/",
    "s3://skyxz-test/lerobot-data/blackmarker_scence3/",
    "s3://skyxz-test/lerobot-data/blackmarker_scence4/",
    "s3://skyxz-test/lerobot-data/blackmarker_scence5/",
    "s3://skyxz-test/lerobot-data/blackmarker_scence6/",
    "s3://skyxz-test/lerobot-data/redmarker_scence3/",
    "s3://skyxz-test/lerobot-data/redmarker_scence4/",
    "s3://skyxz-test/lerobot-data/redmarker_scence5/",
    "s3://skyxz-test/lerobot-data/redmarker_scence6/"
]

CHECKPOINT_URI = "s3://skyxz-test/checkpoints/"
OUTPUT_URI = "s3://skyxz-test/output/"

def create_session():
    boto3_session = boto3.Session(region_name=REGION)
    return Session(boto_session=boto3_session)

def create_valid_job_name(instance_type):
    clean_name = instance_type.replace('ml.', '').replace('.', '-')
    timestamp = int(time.time())
    job_name = f"train-{clean_name}-{timestamp}"
    job_name = re.sub(r'[^a-zA-Z0-9-]', '', job_name)
    return job_name[:63]

def create_dataset_inputs():
    """为所有数据集创建输入通道"""
    inputs = {}

    print("Creating input channels for datasets:")
    for i, data_uri in enumerate(DATASET_URIS):
        channel_name = f"dataset_{i+1:02d}"  # 使用两位数字，如 dataset_01
        inputs[channel_name] = TrainingInput(
            data_uri,
            distribution="FullyReplicated"
        )
        print(f"  {channel_name}: {data_uri}")

    print(f"Total datasets: {len(inputs)}")
    return inputs

def main():
    session = create_session()

    # 创建所有数据集的输入通道
    dataset_inputs = create_dataset_inputs()

    # G5 实例类型优先级
    G5_TYPES = [
        "ml.g5.2xlarge",
        "ml.g5.xlarge",
        "ml.g5.4xlarge",
        "ml.g5.8xlarge",
    ]

    best_job = None

    for instance_type in G5_TYPES:
        job_name = create_valid_job_name(instance_type)

        try:
            print(f"\nSubmitting training job: {job_name} with {instance_type}")
            print(f"Total datasets: {len(DATASET_URIS)}")

            estimator = PyTorch(
                entry_point=TRAINING_SCRIPT,
                source_dir=SOURCE_DIR,
                role=ROLE,
                instance_count=1,
                instance_type=instance_type,
                framework_version="2.1",
                py_version="py310",
                sagemaker_session=session,
                use_spot_instances=True,
                max_wait=4 * 60 * 60,
                max_run=3 * 60 * 60,
                checkpoint_s3_uri=CHECKPOINT_URI,
                output_path=OUTPUT_URI,
                # 添加环境变量传递数据集信息
                environment={
                    'DATASET_PATHS': ','.join(DATASET_URIS),
                    'TOTAL_DATASETS': str(len(DATASET_URIS)),
                    'TRAINING_JOB_NAME': job_name
                }
            )

            # 使用所有数据集输入
            estimator.fit(dataset_inputs, job_name=job_name, wait=True, logs=True)
            best_job = job_name
            print(f"Training job {job_name} started successfully.")
            break
        except Exception as e:
            msg = str(e)
            print(f"Failed to start {instance_type} due to: {msg}")

            if "InsufficientInstanceCapacity" in msg or "Capacity" in msg:
                print("容量不足，尝试下一个实例类型。")
                time.sleep(5)
                continue
            else:
                print(f"其他错误，继续尝试下一个实例类型。")
                continue

    if best_job:
        print(f"Training job launched: {best_job}")
    else:
        print("所有 G5 实例尝试失败")

if __name__ == "__main__":
    main()