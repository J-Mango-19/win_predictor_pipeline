import boto3
from prefect import flow, task, get_run_logger
from datetime import datetime, timezone
from pathlib import Path
from python_terraform import Terraform, TerraformCommandError
from orchestration.utils import construct_prompt, get_response, extract_json_from_tail, wait_for_setup_script, wait_for_ec2_status_checks, wait_for_ssm_registration
from common.constants import PROJECT_ROOT
from common.utils import run_remote_command
from orchestration.prefect_secrets import set_secrets
from orchestration.prefect_vars import set_vars


@task
def check_for_game_changing_events():
    """Check for game-changing events in Clash Royale since the oldest data used ot train the latest model."""
    logger = get_run_logger()
    client = boto3.client('bedrock-agentcore', region_name='us-east-2')

    # TODO: I will need to have a folder in S3 that holds the most recent dataset used to train the model so that I can check for updates since then
    prompt_path = PROJECT_ROOT / "infra/orchestration/src/orchestration/prompt.txt"
    prompt = construct_prompt(prompt_path, "s3://cr-games-bucket/clean-2026-07-20--2026-07-28/")

    response_text = get_response(client, prompt)
    response_list = extract_json_from_tail(response_text)
    logger.info(f"Game-changing events found: {response_list}")
    return len(response_list) > 0

@task(name="provision-ingestion-infrastructure", retries=0)
def provision_ingestion_infrastructure(tf_base_dir: str | Path) -> tuple[str, str, str]:
    """
    Manages Terraform lifecycle:
    1. Apply persistent infrastructure.
    2. Teardown any leftover temporary infrastructure.
    3. Spin up fresh temporary infrastructure, passing in outputs from persistent.
    """
    logger = get_run_logger()
    tf_base_path = Path(tf_base_dir).resolve()

    persistent_dir = tf_base_path / "persistent"
    temporary_dir = tf_base_path / "temporary"

    tf_persistent = Terraform(working_dir=str(persistent_dir))
    tf_temporary = Terraform(working_dir=str(temporary_dir))

    # Step 1: Ensure persistent infrastructure exists
    logger.info(f"Applying persistent infrastructure in {persistent_dir}...")
    return_code, stdout, stderr = tf_persistent.apply(
        skip_plan=True,
        capture_output=True
    )
    if return_code != 0:
        raise TerraformCommandError(return_code, "terraform apply (persistent)", stdout, stderr)

    # Pull the outputs we need for the temporary infra
    persistent_outputs = tf_persistent.output()
    elastic_ip_allocation_id = persistent_outputs["elastic_ip_allocation_id"]['value']
    instance_profile_name = persistent_outputs["instance_profile_name"]['value']
    logger.info(
        f"Collected persistent outputs: "
        f"elastic_ip_allocation_id={elastic_ip_allocation_id}, "
        f"instance_profile_name={instance_profile_name}"
    )

    # Step 2: Tear down lingering temporary infrastructure
    logger.info(f"Destroying stale temporary infrastructure in {temporary_dir}...")
    return_code, stdout, stderr = tf_temporary.destroy(
        force=None,
        auto_approve=True,
        capture_output=True,
        var={
            "elastic_ip_allocation_id": elastic_ip_allocation_id,
            "instance_profile_name": instance_profile_name,
        },
    )
    if return_code != 0:
        raise TerraformCommandError(return_code, "terraform destroy (temporary)", stdout, stderr)

    # Step 3: Rebuild temporary infrastructure, passing in persistent outputs
    logger.info(f"Applying fresh temporary infrastructure in {temporary_dir}...")
    return_code, stdout, stderr = tf_temporary.apply(
        skip_plan=True,
        capture_output=True,
        var={
            "elastic_ip_allocation_id": elastic_ip_allocation_id,
            "instance_profile_name": instance_profile_name,
        },
    )
    if return_code != 0:
        raise TerraformCommandError(return_code, "terraform apply (temporary)", stdout, stderr)

    logger.info("Infrastructure lifecycle initialized successfully.")

    # Step 4: Collect the instance id from the temporary infra
    tmp_outputs = tf_temporary.output()
    tmp_instance_id = tmp_outputs["instance_id"]["value"]
    return elastic_ip_allocation_id, instance_profile_name, tmp_instance_id

@task(name="wait-for-instance-ready", retries=0)
def wait_for_instance_ready(instance_id: str, ssm_timeout: int = 120, setup_script_timeout: int = 300, poll_interval: int = 10) -> None:
    """
    Block until an EC2 instance is fully ready to receive commands:
      1. Passes EC2 status checks (instance + system reachability)
      2. Is registered and pinging with SSM (required for run_remote_command)
      3. `setup.sh` has completed (indicated with sentinel file)
    """
    ec2 = boto3.client('ec2', region_name='us-east-2')
    ssm = boto3.client('ssm', region_name='us-east-2')

    wait_for_ec2_status_checks(ec2, instance_id)
    wait_for_ssm_registration(ssm, instance_id, ssm_timeout, poll_interval)
    wait_for_setup_script(ssm, instance_id, setup_script_timeout, poll_interval)


@task(name="call_ingestion", retries=0)
def call_ingestion(oldest_time_allowed: datetime, instance_id: str) -> None:
    """ call the data ingestion pipeline on an existing EC2 instance """
    commands = [
                "cd /opt/classification-pipeline/services/ingestion",
                f"uv run python -m ingestion.pipeline {str(oldest_time_allowed)}",
            ]

    response = run_remote_command(instance_id, commands)
    if 'failed' in response:
        raise RuntimeError(response)
    
    print(response)


@task(name="destroy-infrastructure", retries=0)
def destroy_ingestion_infrastructure(tf_base_dir: str | Path, elastic_ip_allocation_id: str, instance_profile_name: str) -> None:
    logger = get_run_logger()
    base_path = Path(tf_base_dir).resolve()

    temporary_dir = base_path / "temporary"
    tf_temporary = Terraform(working_dir=str(temporary_dir))

    logger.info(f"Destroying temporary infrastructure in {temporary_dir}...")
    return_code, stdout, stderr = tf_temporary.destroy(
        force=None,
        auto_approve=True,
        capture_output=True,
        var={
            "elastic_ip_allocation_id": elastic_ip_allocation_id,
            "instance_profile_name": instance_profile_name,
        },
    )
    if return_code != 0:
        raise TerraformCommandError(return_code, "terraform destroy (temporary)", stdout, stderr)

    logger.info("Temporary infrastructure destroyed successfully.")


@flow
def main():
    logger = get_run_logger()

    # 1: Check for whether Clash Royale has been updated since the last run
    #game_changed = check_for_game_changing_events()
    game_changed = True; print("HARDCODED game_changed=True for testing")  # FOR TESTING ONLY
    if not game_changed:
        return

    set_secrets()
    set_vars()
    logger.info("Initiating Data Collection & Training")


    # 2: Create fresh infra for ingestion (clearing any existing ephemeral infra)
    tf_base_dir = PROJECT_ROOT / "infra/terraform"


    logger.info(f"Using Terraform base directory: {tf_base_dir}")
    oldest_time_allowed = datetime.now(timezone.utc) 
    from datetime import timedelta
    oldest_time_allowed = oldest_time_allowed - timedelta(weeks=2); print("HARDCODED oldest_time_allowed to be more forgiving for development")
    elastic_ip_allocation_id, instance_profile_name, tmp_instance_id = provision_ingestion_infrastructure(tf_base_dir=tf_base_dir)
    wait_for_instance_ready(tmp_instance_id)

    # 3: Invoke data ingestion
    try:
        call_ingestion(oldest_time_allowed, tmp_instance_id) # TODO: modify ingestion code to build the urls dict/json with every troop encountered and send those to S3
    except:
        logger.error("Ingestion failed!!")
        raise # infra destruction in `finally` will still run before exiting
    
    finally:
        # tear down data ingestion infra
        logger.info("destroying ephemeral infra")
        destroy_ingestion_infrastructure(tf_base_dir, elastic_ip_allocation_id, instance_profile_name)
    
    # 4: Deploy training infra
    # train model
    # quantize
    # save weights to S3
    # save metrics summary to S3
    # tear down training infra

    # 5: Alert Github of the change
    # send a requests.post() trigger to github actions
    # which will download and serve the new weights
    # and the new troop image URLs


def cli_run():
    main.serve(name="cr-pipeline")
