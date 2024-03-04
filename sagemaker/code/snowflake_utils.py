import boto3
import json
import snowflake.connector


def get_credentials(secret_id: str, region_name: str, profile_name: str = None) -> str:
    if profile_name:
        session = boto3.session.Session(profile_name=profile_name)
        client = session.client('secretsmanager', region_name=region_name)
    else:
        client = boto3.client('secretsmanager', region_name=region_name)
    response = client.get_secret_value(SecretId=secret_id)
    secrets_value = json.loads(response['SecretString'])    
    
    return secrets_value

def connect(secret_id: str, account: str, warehouse: str, database: str, schema: str, protocol: str, region: str,  profile_name: str = None) -> snowflake.connector.SnowflakeConnection:
    
    secret_value = get_credentials(secret_id, region, profile_name=profile_name)
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