"""
Read the HOUSING table (this is the california housing dataset used by this example)
"""
import argparse
import os
import pandas as pd
import snowflake.connector

from snowflake_utils import connect




def collect_dataset(ctx: snowflake.connector.SnowflakeConnection, input: str, output: str) -> pd.DataFrame:
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
    num_records = collect_dataset(ctx, input=args.input, output=args.output)