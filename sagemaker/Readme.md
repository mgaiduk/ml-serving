`run_pipeline.py` assembles, runs and attaches to a trigger (currently, launching it every day at 6:00 am) the entire pipeline: snowflake dataset collection, pytorch and xgboost training and deployment to sagemaker inference.   

To run it, login with `aws sso login --profile <my_profile>` and pass in a name of the profile:
```bash
python run_pipeline.py --profile-name ml-staging-admin
```


To set it up for a new project, it is recommended to go step-by-step, debugging and launching a step locally first, and then through the sagemaker pipeline.
## Setting up
For the pipeline and some of the scripts to work, you will need to setup:
- Sagemaker Domain, described here (https://aws.amazon.com/tutorials/machine-learning-tutorial-build-model-locally/). Only first chapter is relevant
- Sagemaker role, described here https://docs.aws.amazon.com/sagemaker/latest/dg/define-pipeline.html (or just copy https://us-east-1.console.aws.amazon.com/iam/home?region=us-east-1#/roles/details/SageMakerSnowFlakeExampleIAMRole-?section=permissions)
- Build the docker image and upload to the registry
```bash
sudo docker build -t snowflake . --platform=linux/amd64
sudo docker tag snowflake:latest 767397884728.dkr.ecr.us-east-1.amazonaws.com/snowflake:latest
aws ecr get-login-password --profile ml-staging-admin --region us-east-1| sudo docker login --username AWS --password-stdin 767397884728.dkr.ecr.us-east-1.amazonaws.com/cdk-hnb659fds-container-assets-767397884728-us-east-1
sudo docker push 767397884728.dkr.ecr.us-east-1.amazonaws.com/snowflake:latest
```
Replace relevant account name and docker registry path to the one used for docker registry.

## Snowflake dataset collection
### Locally
Setup an aws secret with snowflake credentials. Described here: https://aws.amazon.com/blogs/machine-learning/use-snowflake-as-a-data-source-to-train-ml-models-with-amazon-sagemaker/, in the "Store Snowflake credentials in Secrets Manager" part.  


To run locally:
```bash
SM_MODEL_DIR=model_dir SF_ACCOUNT=lnb99345.us-east-1 SECRET_ID=snowflake_credentials SF_WAREHOUSE=XSMALL SF_DATABASE=PRODUCTION SF_SCHEMA=SIGNALS AWS_REGION=us-east-1 python code/collect_data2.py --input COMMUNITYFEEDMEDIASIGNAL --output TRAIN_DATASET_V2 --profile-name <my_profile_name>
```
Replace snowflake credentials id and account respectively.
This should run fresh dataset collection on Snowflake. In the end, if all went well, it will print out the most recent event time that made it into the dataset:
```
2024-03-26 13:46:20.736000
```
Verify that it indeed corresponds to a time within the last 15 minutes.