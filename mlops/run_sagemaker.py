#!/usr/bin/env python3

import os
import argparse
import boto3
import sagemaker
from sagemaker.pytorch import PyTorch
from sagemaker.inputs import FileSystemInput


def launch_sagemaker_job(args):
    try:
        sagemaker_session = sagemaker.Session()
        role = sagemaker.get_execution_role()
    except ValueError:
        role = args.role_arn if args.role_arn else "arn:aws:iam::123456789012:role/service-role/AmazonSageMaker-ExecutionRole-MLFF"
        sagemaker_session = sagemaker.Session(default_bucket=args.bucket)

    print(f"SageMaker Role ARN: {role}")
    print(f"Default S3 Bucket: {sagemaker_session.default_bucket()}")

    hyperparameters = {
        "data_dir": "/opt/ml/input/data/training",
        "model_path": "/opt/ml/model/model_checkpoint.pt",
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "lr": args.lr,
        "hidden_dim": args.hidden_dim,
        "energy_weight": args.energy_weight,
        "force_weight": args.force_weight,
        "mlflow_uri": args.mlflow_uri
    }

    distribution = {
        "pytorchddp": {
            "enabled": True
        }
    }

    estimator = PyTorch(
        entry_point="train_pipeline.py",
        source_dir="mlops",
        role=role,
        framework_version="2.0",
        py_version="py310",
        instance_count=args.instance_count,
        instance_type=args.instance_type,
        hyperparameters=hyperparameters,
        distribution=distribution,
        sagemaker_session=sagemaker_session,
        volume_size=200,
        max_run=86400
    )

    if args.use_fsx:
        data_channel = FileSystemInput(
            file_system_id=args.fsx_id,
            file_system_type="FSxLustre",
            directory_path=args.fsx_path,
            file_system_access_mode="ro"
        )
    else:
        s3_data_uri = f"s3://{sagemaker_session.default_bucket()}/mlff/data"
        data_channel = s3_data_uri

    print("Launching job on AWS SageMaker...")
    estimator.fit(
        inputs={"training": data_channel},
        wait=args.wait_for_completion
    )
    print("SageMaker job submitted successfully.")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--instance_type", type=str, default="ml.p4d.24xlarge")
    parser.add_argument("--instance_count", type=int, default=2)
    parser.add_argument("--role_arn", type=str)
    parser.add_argument("--bucket", type=str)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--hidden_dim", type=int, default=64)
    parser.add_argument("--energy_weight", type=float, default=1.0)
    parser.add_argument("--force_weight", type=float, default=20.0)
    parser.add_argument("--mlflow_uri", type=str, default="http://mlflow-server.mycompany.internal:5000")
    parser.add_argument("--use_fsx", action="store_true")
    parser.add_argument("--fsx_id", type=str, default="fs-0123456789abcdef0")
    parser.add_argument("--fsx_path", type=str, default="/fsx/mlff-dataset")
    parser.add_argument("--wait_for_completion", action="store_true")
    
    args = parser.parse_args()
    launch_sagemaker_job(args)


if __name__ == "__main__":
    main()
