import boto3
import sagemaker
from sagemaker.processing import ScriptProcessor, ProcessingInput, ProcessingOutput
from sagemaker.workflow.steps import ProcessingStep, TrainingStep
from sagemaker.workflow.model_step import ModelStep
from sagemaker.workflow.pipeline import Pipeline
from sagemaker.workflow.triggers import PipelineSchedule
from sagemaker.pytorch.estimator import PyTorch
from sagemaker.workflow.pipeline_context import PipelineSession
from sagemaker.pytorch.model import PyTorchModel
from sagemaker.inputs import CreateModelInput
from sagemaker.workflow.step_collections import RegisterModel

sf_account_id = "lnb99345.us-east-1"
sf_secret_id = "snowflake_credentials"

def main():
    sf_account_id = "lnb99345.us-east-1"
    sf_secret_id = "snowflake_credentials"

    warehouse = "XSMALL"
    database = "PRODUCTION"
    schema = "SIGNALS"
    session = boto3.session.Session(profile_name="ml-staging-admin")
    #sagemaker_session = sagemaker.Session(session)
    pipeline_session = PipelineSession(boto_session=session)
    region = session.region_name
    print(f"region={region}")

    env = {"SECRET_ID": sf_secret_id, 
       "SF_ACCOUNT": sf_account_id,
       "SF_WAREHOUSE": warehouse,
       "SF_DATABASE": database,
       "SF_SCHEMA": schema,
       "AWS_REGION": region,
       "PYTHONPATH": "/opt/program/code"}
    
    snowflake_image = "767397884728.dkr.ecr.us-east-1.amazonaws.com/snowflake:latest"
    role = "arn:aws:iam::767397884728:role/SageMakerSnowFlakeExampleIAMRole-"

    # collect dataset on Snowflake
    # script_processor = ScriptProcessor(command=['python3'],
    #         image_uri=snowflake_image,
    #         role=role,
    #         instance_count=1,
    #         instance_type='ml.m5.xlarge',
    #         env=env,
    #         sagemaker_session=pipeline_session)
    
    
    # snowflake_step = ProcessingStep(
    #     name="SnowflakeSQLQueries",
    #     processor=script_processor,
    #     outputs=[ProcessingOutput(output_name="records_count", source='/opt/ml/processing/output')],
    #     code='code/collect_data.py',
    #     job_arguments=["--input", "COMMUNITYFEEDMEDIASIGNAL", "--output", "TRAIN_DATASET_V4"]
    # )

    # train pytorch
    pytorch_estimator = PyTorch('train.py',
        instance_type='ml.p3.2xlarge',
        instance_count=1,
        framework_version='1.11.0',
        py_version='py38',
        hyperparameters = {
            "input": "TRAIN_DATASET_V4",
            'num-epochs': 100, 
            'lr': 0.001},
        source_dir = "code",
        role=role,
        sagemaker_session=pipeline_session,
        environment=env,
    )
    
    training_step = TrainingStep(
        name='PyTorchModelTraining',
        estimator=pytorch_estimator,
        # inputs={
        #     'records_count': sagemaker.inputs.TrainingInput(s3_data=snowflake_step.properties.ProcessingOutputConfig.Outputs[
        #         "records_count"
        #     ].S3Output.S3Uri,
        #     content_type='text/csv')
        # },
    )

    # inference
    pytorch_model = PyTorchModel(model_data=training_step.properties.ModelArtifacts.S3ModelArtifacts,
        role=role,
        entry_point='inference.py', # Your script for processing inference requests
        framework_version='1.11.0',
        py_version='py38',
        source_dir="code",
        sagemaker_session=pipeline_session,
    )

    model_step = ModelStep(
        name="CreateModel",
        step_args=pytorch_model.create(instance_type="ml.m5.xlarge"),
    )

    # register
    # step_register = RegisterModel(
    #     name="PytorchRegisterModel",
    #     model=pytorch_model,
    #     content_types=["application/json"],
    #     response_types=["application/json"],
    #     inference_instances=["ml.t2.medium"],
    #     model_package_group_name='sipgroup',
    # )
    # register_model_step_args = pytorch_model.register(
    #     content_types=["application/json"],
    #     response_types=["application/json"],
    #     inference_instances=["ml.t2.medium"],
    #     model_package_group_name='sipgroup',
    # )

    # step_register = ModelStep(
    #     name="PytorchRegisterModel",
    #     step_args=register_model_step_args,
    # )

    # deploy for inference
    # predictor = pytorch_model.deploy(instance_type='ml.m5.large', initial_instance_count=1)

    # assemble pipeline
    pipeline_name = f"SnowflakePipeline"
    pipeline = Pipeline(
        name=pipeline_name,
        #steps=[snowflake_step, training_step],
        #steps=[training_step, model_step, step_register],
        steps=[training_step, model_step],
        sagemaker_session=pipeline_session
    )

    pipeline.upsert(role_arn=role)
    execution = pipeline.start()
    print(execution.describe())

    my_datetime_schedule = PipelineSchedule(
        name="snowflake-pipeline-schedule", 
        cron="0 6 ? * * *"
    )
    pipeline.put_triggers(triggers=[my_datetime_schedule], role_arn=role)
    print(pipeline.describe_trigger("snowflake-pipeline-schedule"))
    



if __name__ == "__main__":
    main()