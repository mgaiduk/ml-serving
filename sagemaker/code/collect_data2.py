"""
Read the HOUSING table (this is the california housing dataset used by this example)
"""
import argparse
from jinja2 import Template
import os
import pandas as pd
import snowflake.connector


from snowflake_utils import connect




def collect_dataset(ctx: snowflake.connector.SnowflakeConnection, input: str, output: str) -> pd.DataFrame:
    # Collect dataset
    cursor = ctx.cursor()
    media_features = "PRODUCTION.TABLES.COMMUNITYFEEDMEDIAFEATURES"
    takenby_features = "PRODUCTION.TABLES.COMMUNITYFEEDTAKENBYFEATURES"
    user_features = "PRODUCTION.TABLES.COMMUNITYFEEDUSERFEATURES"
    cross_features = "PRODUCTION.TABLES.COMMUNITYFEEDUSERCROSSTAKENBYFEATURES"

    # List all your columns
    all_signals = ['CFMediaAddedCommentSignal', 'CFMediaAddedReactionSignal', 'CFMediaDwellTimeSignal', 'CFMediaTappedTakenBySignal', 'CFMediaViewedSignal']
    all_intervals = ["1m", "5m", "15m", "1h", "6h", "1d", "7d", "30d"]

    # Define your Jinja template for the SQL query
    sql_template = """
    create or replace table {{output}} as (


        with events as (
            SELECT 
            SRC:__signalType::string AS signalType,
            SRC:__userId::string AS userId,
            TO_TIMESTAMP(SRC:__timestamp::STRING) AS timestamp,
            SRC:mediaId::string AS mediaId,
            SRC:mediaTakenById::string AS mediaTakenById,
            SRC:requestId::string AS requestId,
            coalesce(JSON_EXTRACT_PATH_TEXT(SRC:signalData::variant, 'timeSpent'), 0.0) AS timespent
            FROM {{input}}
        ),


        takenByEvents as
            (SELECT
            SRC:data:eventTime::BIGINT AS eventTime,
            f.value:featureName::STRING AS featureName,
            f.value:takenBy::STRING AS takenBy,
            SRC:data:requestId::STRING AS requestId,
            {% for signal in all_signals %}
            {% for interval in all_intervals %}
                CASE WHEN f.value:featureName = '{{ signal }}' THEN f.value:counts:"{{interval}}"::FLOAT else null end AS {{signal}}_{{interval}},
            {% endfor %}
            {% endfor %}
            FROM
            {{takenby_features}},
            LATERAL FLATTEN(input => SRC:data:features) f
        ), 
        

        collapsedTakenByEvents as (
            select max(eventTime) as eventTime, takenBy, requestId,
            {% for signal in all_signals %}
            {% for interval in all_intervals %}
                max({{signal}}_{{interval}}) as TAKENBY_{{signal}}_{{interval}},
            {% endfor %}
            {% endfor %}
            from takenByEvents
            group by takenBy, requestId
        ),

        
        mediaEvents as (
            SELECT
            f.value:featureName::STRING AS featureName,
            f.value:mediaId::STRING AS mediaId,
            SRC:data:requestId::STRING AS requestId,
            {% for signal in all_signals %}
            {% for interval in all_intervals %}
                CASE WHEN f.value:featureName = '{{signal}}' THEN f.value:counts:"1m"::FLOAT else null end AS {{signal}}_{{interval}},
            {% endfor %}
            {% endfor %}
            FROM
            {{media_features}},
            LATERAL FLATTEN(input => SRC:data:features) f
        ), 
        
        
        collapsedMediaEvents as (
            select mediaId, requestId,
            {% for signal in all_signals %}
            {% for interval in all_intervals %}
                max({{signal}}_{{interval}}) as MEDIA_{{signal}}_{{interval}},
            {% endfor %}
            {% endfor %}
            from mediaEvents
            group by mediaId, requestId
        ),


        userEvents as (
            SELECT
            f.value:featureName::STRING AS featureName,
            f.value:userId::STRING AS userId,
            SRC:data:requestId::STRING AS requestId,
            {% for signal in all_signals %}
            {% for interval in all_intervals %}
                CASE WHEN f.value:featureName = '{{signal}}' THEN f.value:counts:"1m"::FLOAT else null end AS {{signal}}_{{interval}},
            {% endfor %}
            {% endfor %} 
            FROM
            {{user_features}},
            LATERAL FLATTEN(input => SRC:data:features) f
        ), 
        
        
        collapsedUserEvents as (
            select userId, requestId,
            {% for signal in all_signals %}
            {% for interval in all_intervals %}
                max({{signal}}_{{interval}}) as USER_{{signal}}_{{interval}},
            {% endfor %}
            {% endfor %}
            from userEvents
            group by userId, requestId
        ),


        crossEvents as (
            SELECT
            f.value:featureName::STRING AS featureName,
            f.value:userId::STRING AS userId,
            f.value:takenBy::STRING as takenBy,
            SRC:data:requestId::STRING AS requestId,
            {% for signal in all_signals %}
            {% for interval in all_intervals %}
                CASE WHEN f.value:featureName = '{{signal}}' THEN f.value:counts:"1m"::FLOAT else null end AS {{signal}}_{{interval}},
            {% endfor %}
            {% endfor %}
            FROM
            {{cross_features}},
            LATERAL FLATTEN(input => SRC:data:features) f
        ), 
        
        
        collapsedCrossEvents as (
            select userId, takenBy, requestId,
            {% for signal in all_signals %}
            {% for interval in all_intervals %}
                max({{signal}}_{{interval}}) as CROSS_{{signal}}_{{interval}},
            {% endfor %}
            {% endfor %}
            from crossEvents
            group by takenBy, requestId, userId
        ),


        aggregatedEvents as (
            select 
            userId, mediaId, mediaTakenById, requestId, 
            count_if(signalType = 'CFMediaViewedSignal') as nViews,
            sum(timespent) as timespent,
            min(timestamp) as min_timestamp,
            count_if(signalType = 'CFMediaAddedReactionSignal') as nReactions
            from events as a
            where timestamp >= '2024-03-13'
            group by userId, mediaId, mediaTakenById, requestId
        ),


        joinedEvents as (
            select a.*,
            {% for signal in all_signals %}
            {% for interval in all_intervals %}
                b.TAKENBY_{{signal}}_{{interval}},
            {% endfor %}
            {% endfor %}
            from aggregatedevents as a
            left join collapsedTakenByEvents as b
            on a.requestId = b.requestId
            and a.mediaTakenById = b.takenBy
        ),


        joinedEvents2 as (
            select 
            a.*,
            {% for signal in all_signals %}
            {% for interval in all_intervals %}
                b.MEDIA_{{signal}}_{{interval}},
            {% endfor %}
            {% endfor %}
            from joinedEvents as a
            left join collapsedMediaEvents as b
            on a.requestId = b.requestId
            and a.mediaId = b.mediaId
        ),


        joinedEvents3 as (
            select a.*,
            {% for signal in all_signals %}
            {% for interval in all_intervals %}
                b.USER_{{signal}}_{{interval}},
            {% endfor %}
            {% endfor %}
            from joinedEvents2 as a
            left join collapsedUserEvents as b
            on a.requestId = b.requestId
            and a.userId = b.userId
        ),


        joinedEvents4 as (
            select a.*,
            {% for signal in all_signals %}
            {% for interval in all_intervals %}
                b.CROSS_{{signal}}_{{interval}},
            {% endfor %}
            {% endfor %}
            from joinedEvents3 as a
            left join collapsedCrossEvents as b
            on a.requestId = b.requestId
            and a.userId = b.userId
            and a.mediaTakenById = b.takenBy
        )


        select * from joinedEvents4
    )
    """

    # Create a template object
    template = Template(sql_template)

    # Render the template with your variables
    rendered_sql = template.render(input=input, 
                                   all_signals=all_signals, all_intervals=all_intervals, output=output,
                                    media_features=media_features, takenby_features=takenby_features, user_features=user_features, cross_features=cross_features)

    print(rendered_sql)
    cursor.execute(rendered_sql)
    cursor.execute(f"""
        select max(min_timestamp) as last_ts from {output}               
        """)
    one_row = cursor.fetchone()
    max_timestamp = one_row[0]
    print(max_timestamp)
    #with open("/opt/ml/processing/output/output.txt", "w") as file:
    #    print(max_timestamp, file=file)

    return max_timestamp

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Example script to demonstrate argument parsing.')
    parser.add_argument('--input', type=str, required=True,
                    help='Input table with signals')
    parser.add_argument('--output', type=str, required=True,
                help='Output dataset table')
    parser.add_argument('--profile-name', type=str, required=False,
                help='Profile for aws connect. Needed in local launched only')
    args = parser.parse_args()
    
    # snowflake related params are read from environment variables
    secret_id = os.environ["SECRET_ID"]
    account = os.environ["SF_ACCOUNT"]
    warehouse = os.environ["SF_WAREHOUSE"]
    database = os.environ["SF_DATABASE"].upper()
    schema = os.environ["SF_SCHEMA"].upper()
    region = os.environ["AWS_REGION"]
    
    protocol = "https"
    ctx = connect(secret_id, account, warehouse, database, schema, protocol, region, profile_name=args.profile_name)
    num_records = collect_dataset(ctx, input=args.input, output=args.output)