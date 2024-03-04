"""
Read the HOUSING table (this is the california housing dataset used by this example)
"""
import argparse
import os
import pandas as pd
import snowflake.connector
import json
import boto3

def get_credentials(secret_id: str, region_name: str) -> str:
    
    client = boto3.client('secretsmanager', region_name=region_name)
    response = client.get_secret_value(SecretId=secret_id)
    secrets_value = json.loads(response['SecretString'])    
    
    return secrets_value

def connect(secret_id: str, account: str, warehouse: str, database: str, schema: str, protocol: str, region: str) -> snowflake.connector.SnowflakeConnection:
    
    secret_value = get_credentials(secret_id, region)
    sf_user = secret_value['username']
    sf_password = secret_value['password']
    sf_account = account
    sf_warehouse = warehouse
    sf_database = database
    sf_schema = schema
    sf_protocol = protocol
    
    print(f"sf_user={sf_user}, sf_password=****, sf_account={sf_account}, sf_warehouse={sf_warehouse}, "
          f"sf_database={sf_database}, sf_schema={sf_schema}, sf_protocol={sf_protocol}")    
    
    # Read to connect to snowflake
    ctx = snowflake.connector.connect(user=sf_user,
                                      password=sf_password,
                                      account=sf_account,
                                      warehouse=sf_warehouse,
                                      database=sf_database,
                                      schema=sf_schema,
                                      protocol=sf_protocol)
    
    return ctx

def collect_dataset(ctx: snowflake.connector.SnowflakeConnection, input: str, output: str, schema: str, database: str) -> pd.DataFrame:
    # Collect dataset
    cursor = ctx.cursor()
    #print(f"USE DATABASE {database}")
    #cursor.execute(f"USE DATABASE {database}")
    #print(f"USE {schema}")
    #cursor.execute(f"USE {schema}")
    cursor.execute(f"""
    create or replace table {output} as (
        with events as (SELECT 
            SRC:__signalType::string AS signalType,
            SRC:__userId::string AS userId,
            TO_TIMESTAMP(SRC:__timestamp::STRING) AS timestamp,
            SRC:mediaId::string AS mediaId,
            SRC:mediaTakenById::string AS mediaTakenById,
            SRC:requestId::string AS requestId,
            coalesce(JSON_EXTRACT_PATH_TEXT(SRC:signalData::variant, 'timeSpent'), 0.0) AS timespent
        FROM {input})


        select 
            userId, mediaId, mediaTakenById, requestId, 
            count_if(signalType = 'CFMediaViewedSignal') as nViews,
            sum(timespent) as timespent,
            min(timestamp) as min_timestamp,
            count_if(signalType = 'CFMediaAddedReactionSignal') as nReactions
        from events
        where timestamp > '2024-02-20'
        group by userId, mediaId, mediaTakenById, requestId
        )
    """)
    cursor.execute(f"""
        select max(min_timestamp) as last_ts from {output}               
        """)
    one_row = cursor.fetchone()
    max_timestamp = one_row[0]
    with open("/opt/ml/processing/output/output.txt", "w") as file:
        print(max_timestamp, file=file)

    return max_timestamp

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Example script to demonstrate argument parsing.')
    parser.add_argument('--input', type=str, required=True,
                    help='Input table with signals')
    parser.add_argument('--output', type=str, required=True,
                help='Output dataset table')
    args = parser.parse_args()
    
    # snowflake related params are read from environment variables
    secret_id = os.environ["SECRET_ID"]
    account = os.environ["SF_ACCOUNT"]
    warehouse = os.environ["SF_WAREHOUSE"]
    database = os.environ["SF_DATABASE"].upper()
    schema = os.environ["SF_SCHEMA"].upper()
    region = os.environ["AWS_REGION"]
    
    protocol = "https"
    ctx = connect(secret_id, account, warehouse, database, schema, protocol, region)
    num_records = collect_dataset(ctx, input=args.input, output=args.output, schema=schema, database=database)