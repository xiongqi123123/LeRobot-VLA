import subprocess
import sys
import os
import logging
import time
import random

# 设置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def get_unique_output_dir():
    """生成唯一的输出目录路径"""
    # 使用毫秒级时间戳 + 随机数确保唯一性
    unique_id = f"{int(time.time() * 1000)}_{random.randint(1000, 9999)}"
    return f'/opt/ml/model/lerobot_output_{unique_id}'

def setup_environment():
    """设置训练环境"""
    try:
        directories = [
            '/home/ssm-user/LeRobot-VLA/Policy/RDT/checkpoints'
        ]

        for directory in directories:
            os.makedirs(directory, exist_ok=True)
            logger.info(f"Created directory: {directory}")

        logger.info("Environment setup completed")

    except Exception as e:
        logger.error(f"Environment setup failed: {e}")
        raise

def find_dataset():
    """查找可用的数据集目录"""
    base_path = '/opt/ml/input/data'

    # 优先检查默认目录
    default_path = os.path.join(base_path, 'dataset')
    if os.path.exists(default_path):
        logger.info(f"Using default dataset path: {default_path}")
        return default_path

    # 查找其他数据集目录
    for item in os.listdir(base_path):
        item_path = os.path.join(base_path, item)
        if os.path.isdir(item_path):
            # 检查是否包含数据集文件
            metadata_file = os.path.join(item_path, 'metadata.json')
            episode_dir = os.path.join(item_path, 'episode_data')

            if os.path.exists(metadata_file) or os.path.exists(episode_dir):
                logger.info(f"Found dataset at: {item_path}")
                return item_path

    # 如果没有找到有效数据集，返回第一个目录
    for item in os.listdir(base_path):
        item_path = os.path.join(base_path, item)
        if os.path.isdir(item_path):
            logger.info(f"Using first available directory: {item_path}")
            return item_path

    return default_path  # 即使不存在也返回，让后续验证处理

def validate_dataset(dataset_path):
    """验证数据集"""
    if not os.path.exists(dataset_path):
        logger.error(f"Dataset directory does not exist: {dataset_path}")
        return False

    logger.info(f"Dataset directory exists: {dataset_path}")

    contents = os.listdir(dataset_path)
    logger.info(f"Found {len(contents)} items in dataset directory")

    return True

def install_lerobot():
    """安装 LeRobot"""
    try:
        logger.info("Installing LeRobot from GitHub...")

        # 直接从 GitHub 安装 LeRobot
        install_cmd = [
            sys.executable, '-m', 'pip', 'install',
            'git+https://github.com/huggingface/lerobot.git@v0.3.2'
        ]

        result = subprocess.run(install_cmd, capture_output=True, text=True)

        if result.returncode == 0:
            logger.info("LeRobot installed successfully from GitHub")
            return True
        else:
            logger.error(f"Failed to install LeRobot: {result.stderr}")
            return False

    except Exception as e:
        logger.error(f"Error installing LeRobot: {e}")
        return False

def check_lerobot():
    """检查 LeRobot 是否可用"""
    try:
        result = subprocess.run(
            ['lerobot-train', '--help'],
            capture_output=True,
            text=True,
            timeout=30
        )
        if result.returncode == 0:
            logger.info("lerobot-train command is available")
            return True
        else:
            return False
    except Exception:
        return False

def main():
    try:
        logger.info("Starting training script")

        # 设置环境
        setup_environment()

        # 查找数据集路径
        dataset_path = find_dataset()

        # 验证数据集
        if not validate_dataset(dataset_path):
            sys.exit(1)

        # 检查并安装 LeRobot
        if not check_lerobot():
            logger.info("LeRobot not found, installing...")
            if not install_lerobot():
                logger.error("Failed to install LeRobot")
                sys.exit(1)

        # 生成唯一的输出目录和任务名称
        output_dir = get_unique_output_dir()
        job_name = f'act_so100_test_{int(time.time())}'

        logger.info(f"Output directory will be: {output_dir}")
        logger.info(f"Job name: {job_name}")

        # 训练命令
        cmd = [
            'lerobot-train',
            f'--dataset.repo_id={dataset_path}',
            '--dataset.video_backend=pyav',
            '--policy.type=act',
            f'--output_dir={output_dir}',
            f'--job_name={job_name}',
            '--wandb.enable=false',
            '--policy.push_to_hub=false',
            '--policy.device=cuda',
            '--batch_size=2',
            '--steps=1000'
            # 不设置 --resume 参数
        ]

        logger.info(f"Starting training: {' '.join(cmd)}")
        logger.info("Note: Output directory will be created by lerobot-train command")

        # 执行训练
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)

        logger.info("Training completed successfully")
        if result.stdout:
            logger.info(f"Output: {result.stdout}")

    except subprocess.CalledProcessError as e:
        logger.error(f"Training failed: {e.stderr}")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()