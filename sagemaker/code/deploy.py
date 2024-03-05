import os
import boto3
import argparse
import sagemaker
from datetime import datetime
from sagemaker.compute_resource_requirements.resource_requirements import ResourceRequirements
from sagemaker.predictor import Predictor
from sagemaker.enums import EndpointType
from sagemaker.model import Model
from sagemaker.session import Session
from sagemaker.pytorch import PyTorchModel
from sagemaker.pipeline import PipelineModel


def main():
    parser = argparse.ArgumentParser(description='Example script to demonstrate argument parsing.')
    parser.add_argument('--model-data', type=str, required=True,
                    help='s3 path to trained pytorch model')
    args = parser.parse_args()

    model_name = "pytorch"
    endpoint_name = "pytorch"
    endpoint_config_name = "pytorch"
    role = "arn:aws:iam::767397884728:role/SageMakerSnowFlakeExampleIAMRole-"
    # boto3.setup_default_session(profile_name='ml-staging-admin')
    region = os.environ["AWS_REGION"]
    os.environ["AWS_DEFAULT_REGION"] = region
    sagemaker_client = boto3.client('sagemaker', region_name=region)

    # Create or update the PyTorch model
    try:
        # Try to delete the existing model if it exists
        sagemaker_client.delete_model(ModelName=model_name)
    except:
        # If the model does not exist, ignore the error
        pass

    sagemaker_session = sagemaker.Session(sagemaker_client=sagemaker_client)
    try:
        # delete old pytorch model if exists
        sagemaker_session.delete_model(model_name=model_name)
    except:
        pass

    pytorch_model = PyTorchModel(
        model_data=args.model_data,
        role=role,
        framework_version='1.11.0',
        source_dir="/opt/program/code",
        entry_point='inference.py',
        sagemaker_session=sagemaker_session,
        name=model_name,
        py_version='py38',
    )

    # construct a pipeline model
    # only pipeline models support endpoint update in sagemaker
    pipeline_model_name = "pipeline"
    try:
        sagemaker_session.delete_model(model_name=pipeline_model_name)
    except:
        pass
    pipeline_model = PipelineModel(models=[pytorch_model],role=role,sagemaker_session=sagemaker_session,
                                name=pipeline_model_name)

    endpoint_name = "pytorch"
    
    try:
        # Pipeline model uses 2 endpoint configurations at once: for pytorch model and for pipeline wrapper
        # during update, pytorch configuration can stay in place, while pipeline configuration has to be deleted
        sagemaker_session.delete_endpoint_config(endpoint_config_name=pipeline_model_name)
        print(f"Deleted existing endpoint configuration: {pipeline_model_name}")
    except:
        # If the endpoint configuration does not exist, do nothing
        print(f"Endpoint configuration {pipeline_model_name} does not exist. Proceeding with deployment.")

    try:
        pipeline_model.deploy(instance_type='ml.m4.xlarge',
        initial_instance_count=1,
        endpoint_name=endpoint_name,
        update_endpoint=True)
    except:
        pipeline_model.deploy(instance_type='ml.m4.xlarge',
        initial_instance_count=1,
        endpoint_name=endpoint_name)
        

if __name__ == "__main__":
    main()