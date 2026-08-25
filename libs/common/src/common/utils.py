import time
import boto3
import botocore
from prefect.blocks.system import Secret
from prefect.variables import Variable

def get_api_credentials() -> list[str]:
    """Retrieves encrypted secrets directly from Prefect Cloud Blocks."""
    keys = []
    for key_name in ["clash-api-key-primary", "clash-api-key-secondary"]:
        try:
            # Prefect 3.0 Secret Block load pattern
            token = Secret.load(key_name).get()
            keys.append(token)
        except Exception:
            # Fallback for local debugging if block isn't in Cloud
            raise ValueError("No clash royale API keys found!")
    return keys

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

def get_database_dump_path() -> str:
    """ return the path to the latest database dump file """
    return Variable.get("s3-dump-path")

def get_aws_region() -> str:
    """ return the AWS region for S3 bucket """
    return Variable.get("aws-region")

def get_aws_profile() -> str:   
    """ return the AWS profile for S3 bucket """
    return Variable.get("aws-profile")

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