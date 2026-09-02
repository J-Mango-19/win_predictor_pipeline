import os
import time
import json
import boto3
import botocore
from datetime import datetime, timezone
from botocore.exceptions import ClientError
from prefect.context import refresh_global_settings_context
from prefect.blocks.system import Secret
from prefect.variables import Variable
from prefect.settings import get_current_settings
from functools import lru_cache

def ensure_utc(dt: datetime) -> datetime:
    """Return `dt` as a timezone-aware UTC datetime.

    Naive datetimes are assumed to already be in UTC (every datetime this
    codebase produces is UTC); aware datetimes are converted to UTC. The result
    is always safe to compare against other `ensure_utc` outputs without
    triggering "can't compare offset-naive and offset-aware datetimes".
    """
    if dt.tzinfo is None:
        return dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


def parse_api_timestamp(s: str) -> datetime:
    """Parse a Clash Royale API timestamp (e.g. "20260611T030424.000Z") into a
    timezone-aware UTC datetime."""
    return ensure_utc(datetime.fromisoformat(s))


def login_to_prefect() -> None:
    """ Obtains Prefect API key and workspace URL via AWS Secrets Manager, then saves them as environment variables. """

    secret_name = "prefect_login_info"
    region_name = "us-east-2"

    session = boto3.session.Session()
    client = session.client(
        service_name="secretsmanager",
        region_name=region_name,
    )

    try:
        get_secret_value_response = client.get_secret_value(
            SecretId=secret_name
        )
    except ClientError as e:
        raise e

    secret = json.loads(get_secret_value_response["SecretString"])

    os.environ["PREFECT_API_KEY"] = secret["PREFECT_API_KEY"]
    os.environ["PREFECT_API_URL"] = secret["PREFECT_API_URL"]
    refresh_global_settings_context()


def get_api_credentials() -> list[str]:
    """Retrieves encrypted secrets directly from Prefect Cloud Blocks."""
    keys = []
    for key_name in ["clash-api-key-1", "clash-api-key-2", "clash-api-key-3"]:
        try:
            # Prefect 3.0 Secret Block load pattern
            token = Secret.load(key_name).get()
            keys.append(token)
        except Exception:
            # Fallback for local debugging if block isn't in Cloud
            raise ValueError("No clash royale API keys found!")
    return keys

@lru_cache(maxsize=1)
def load_database_credentials() -> dict:
    """Retrieves encrypted database credentials directly from Prefect Cloud Blocks."""

    password = Secret.load("db-password").get()
    db_name = Variable.get("db-name")
    db_user = Variable.get("db-user")
    db_host = Variable.get("db-host")
    db_port = Variable.get("db-port")
    return {
        "dbname": db_name,
        "user": db_user,
        "host": db_host,
        "port": db_port,
        "password": password
    }

def get_s3_bucket_name() -> str:
    """ return the name of the S3 bucket to store the clean dataset """
    return Variable.get("s3-bucket-name")

def get_database_dump_prefix() -> str:
    """ return the prefix to the latest database dump file """
    return Variable.get("database-dump-prefix")

def get_parquet_dataset_prefix() -> str:
    """ return the prefix to the cleaned dataset file. """
    return Variable.get("parquet-dataset-prefix")

def get_aws_region() -> str:
    """ return the AWS region for S3 bucket """
    return Variable.get("aws-region")

def get_training_dataset_prefix() -> str:
    """ return the S3 prefix ("folder") of parquet file(s) the model trains on """
    return Variable.get("training-dataset-prefix")

def get_model_weights_prefix() -> str:
    """ return the S3 prefix ("folder") that trained model weights are uploaded to """
    return Variable.get("model-weights-prefix")

def get_wandb_api_key() -> str:
    """ return the wandb api key corresponding to the acct to log training progress on """
    return Secret.load("wandb_api_key").get()

def get_wandb_project_name() -> str:
    """ return the wandb project name where the run metrics are to be recorded """
    return Variable.get("wandb-project-name")

def get_wandb_entity() -> str:
    """ return the preferred entity that's writing run metrics to wandb """
    return Variable.get("entity")

def run_remote_command(instance_id: str, commands: list[str], status_check_interval: int=300) -> str:
    """ Runs a shell command on the remote AWS instance with instance_id. """
    ssm = boto3.client("ssm", region_name=get_aws_region())
    response = ssm.send_command(
        InstanceIds=[instance_id],
        DocumentName="AWS-RunShellScript",
        Parameters={
            "commands": commands
        },
    )

    command_id = response["Command"]["CommandId"]
    print(f"{command_id=}")

    waiter = ssm.get_waiter("command_executed")
    try:
        waiter.wait(CommandId=command_id, InstanceId=instance_id)
    except botocore.exceptions.WaiterError:
        pass 

    while True:
        response = ssm.get_command_invocation(
            CommandId=command_id,
            InstanceId=instance_id,
        )

        status = response["Status"]

        if status in {
            "Success",
            "Failed",
            "Cancelled",
            "TimedOut",
            "Undeliverable",
        }:
            break

        time.sleep(status_check_interval)

    if status != "Success":
        raise RuntimeError(
            f"Command failed with status {status}:\n"
            f"{response['StandardErrorContent']}"
        )

    return response["StandardOutputContent"]