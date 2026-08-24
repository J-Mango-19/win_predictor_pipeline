import boto3
from prefect import flow, task, get_run_logger
from pathlib import Path
from python_terraform import Terraform, TerraformCommandError
from orchestration.utils import construct_prompt, get_response, extract_json_from_tail
from common.constants import PROJECT_ROOT


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

@task(name="provision-infrastructure", retries=0)
def provision_infrastructure(tf_base_dir: str | Path) -> tuple[str, str]:
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
    return elastic_ip_allocation_id, instance_profile_name


@task(name="destroy-infrastructure", retries=0)
def destroy_infrastructure(tf_base_dir: str | Path, elastic_ip_allocation_id: str, instance_profile_name: str) -> None:
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
    game_changed = True  # FOR TESTING ONLY
    if not game_changed:
        return

    # formally, this would all be logged...
    logger.info("Initiating Data Collection & Training")
    # send email or slack or something indicating that a new cycle is beginning & what the changes were


    # 2: Destroy any existing infrastructure for active data collection / training runs, then create fresh infra
    # note that we DO NOT destroy persistent infrastructure
    tf_base_dir = PROJECT_ROOT / "infra/terraform"
    logger.info(f"Using Terraform base directory: {tf_base_dir}")
    elastic_ip_allocation_id, instance_profile_name = provision_infrastructure(tf_base_dir=tf_base_dir)


    # 3: Deploy infrastructure for a fresh data collection run
    # Deploy EC2 instances to collect dataset via terraform
    # use now() as the earliest valid time or perhaps wait a day
    # don't forget to build the urls dict/json with every troop encountered and send those to S3
    # tear down data collection infra

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

    import time
    logger.info("Waiting 5 mins before tearing down infra, starting now.")
    time.sleep(5 * 60)

    destroy_infrastructure(tf_base_dir, elastic_ip_allocation_id, instance_profile_name)



if __name__ == '__main__':
    main.serve(name="cr-pipeline")
