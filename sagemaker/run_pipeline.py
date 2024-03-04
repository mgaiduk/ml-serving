import boto3
import sagemaker
from sagemaker.processing import ScriptProcessor, ProcessingInput, ProcessingOutput
from sagemaker.workflow.steps import ProcessingStep, TrainingStep
from sagemaker.workflow.pipeline import Pipeline
from sagemaker.workflow.triggers import PipelineSchedule
from sagemaker.pytorch.estimator import PyTorch
from sagemaker.workflow.pipeline_context import PipelineSession

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
    script_processor = ScriptProcessor(command=['python3'],
            image_uri=snowflake_image,
            role=role,
            instance_count=1,
            instance_type='ml.m5.xlarge',
            env=env,
            sagemaker_session=pipeline_session)
    
    
    snowflake_step = ProcessingStep(
        name="SnowflakeSQLQueries",
        processor=script_processor,
        outputs=[ProcessingOutput(output_name="records_count", source='/opt/ml/processing/output')],
        code='code/collect_data.py',
        job_arguments=["--input", "COMMUNITYFEEDMEDIASIGNAL", "--output", "TRAIN_DATASET_V4"]
    )

    # train pytorch
    # role = sagemaker.get_execution_role(sagemaker_session=sagemaker_session)
    pytorch_estimator = PyTorch('train.py',
        instance_type='ml.p3.2xlarge',
        instance_count=1,
        framework_version='1.8.0',
        py_version='py3',
        hyperparameters = {
            "input": "TRAIN_DATASET_V4",
            'epochs': 100, 
            'learning-rate': 0.001},
        source_dir = "code",
        role=role,
        sagemaker_session=pipeline_session,
    )
    
    training_step = TrainingStep(
        name='PyTorchModelTraining',
        estimator=pytorch_estimator,
        inputs={
            'records_count': sagemaker.inputs.TrainingInput(s3_data=snowflake_step.properties.ProcessingOutputConfig.Outputs[
                "records_count"
            ].S3Output.S3Uri,
            content_type='text/csv')
        },
    )


    # assemble pipeline
    pipeline_name = f"SnowflakePipeline"
    pipeline = Pipeline(
        name=pipeline_name,
        steps=[snowflake_step, training_step],
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