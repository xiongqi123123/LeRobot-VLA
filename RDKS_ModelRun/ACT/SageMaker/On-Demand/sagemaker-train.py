from sagemaker.pytorch import PyTorch
from sagemaker.inputs import TrainingInput
import boto3, sagemaker

# === 数据集路径 ===
dataset_input = TrainingInput(
    "s3://your-bucket-name/",
    distribution="FullyReplicated"
)

# === 创建 boto3 会话 ===
boto_sess = boto3.Session(
    aws_access_key_id="AccessKeyId",
    aws_secret_access_key="AccessKeySecret",
    region_name="Region"
)

# === SageMaker 会话 ===
sess = sagemaker.Session(boto_session=boto_sess)

# === PyTorch 训练任务 ===
estimator = PyTorch(
    entry_point="lerobot/train.py", 
    source_dir="/your-source-dir/", 
    dependencies=["lerobot/requirements.txt"],
    role="your-role-arn",
    instance_count=1,
    instance_type="ml.m5.large",    
    framework_version="2.1",
    py_version="py310",
    sagemaker_session=sess,
)

# === 发起训练 ===
estimator.fit({"dataset": dataset_input})