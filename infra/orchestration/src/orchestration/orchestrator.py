import subprocess
import time
import boto3
import requests
from botocore.exceptions import ClientError
from prefect import flow, task, get_run_logger
from datetime import datetime, timezone
from pathlib import Path
from python_terraform import Terraform, TerraformCommandError
from orchestration.utils import construct_prompt, get_response, extract_json_from_tail, wait_for_setup_script, wait_for_ec2_status_checks, wait_for_ssm_registration
from common.constants import PROJECT_ROOT
from common.utils import run_remote_command, ensure_utc, get_github_dispatch_token
from orchestration.prefect_secrets import set_secrets
from orchestration.prefect_vars import set_vars


# The AWS-RunShellScript maximum. Training runs for hours; SSM's own default of
# one hour would kill the command mid-run and bill us for the GPU regardless.
TRAINING_EXECUTION_TIMEOUT = 172800
# The training box's setup.sh clones the repo and `uv sync`s several GB of CUDA
# wheels, which comfortably outlasts wait_for_instance_ready's 300s default.
TRAINING_SETUP_TIMEOUT = 1800
TRAINING_SSM_TIMEOUT = 300
# main.tf caps the instance's own delete at 10 minutes; this is the budget for
# the whole destroy, after which the terminate fallback takes over.
TRAINING_DESTROY_TIMEOUT = 900
AWS_REGION = "us-east-2"
# The frontend is a GitHub Pages site built by .github/workflows/deploy-frontend.yml,
# which pulls the new weights from S3. repository_dispatch is how this flow reaches it.
GITHUB_REPO = "J-Mango-19/win_predictor_pipeline"
GITHUB_DISPATCH_EVENT = "model-updated"


def _run_terraform(
    working_dir: Path,
    subcommand: str,
    logger,
    tf_vars: dict[str, str] | None = None,
    timeout: int | None = None,
) -> int:
    """Run a terraform subcommand, streaming its output into the Prefect logs.

    python_terraform buffers the whole run behind `capture_output=True` and
    offers no timeout, so a destroy waiting on an unresponsive instance printed
    nothing for twenty minutes and was indistinguishable from a hung flow. This
    streams each line as terraform emits it (including its periodic "Still
    destroying..." ticks) and gives up once `timeout` seconds have passed.
    """
    cmd = [
        "terraform",
        f"-chdir={working_dir}",
        subcommand,
        "-no-color",
        "-input=false",
        "-lock-timeout=60s",
    ]
    if subcommand in {"apply", "destroy"}:
        cmd.append("-auto-approve")
    for key, value in (tf_vars or {}).items():
        cmd.extend(["-var", f"{key}={value}"])

    logger.info(f"$ {' '.join(cmd)}")
    process = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )

    deadline = None if timeout is None else time.monotonic() + timeout
    try:
        for line in process.stdout:
            logger.info(line.rstrip())
            if deadline is not None and time.monotonic() > deadline:
                logger.error(
                    f"terraform {subcommand} exceeded its {timeout}s budget; killing it."
                )
                process.kill()
                break
        return process.wait()
    finally:
        if process.poll() is None:
            process.kill()
            process.wait()


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
def provision_ingestion_infrastructure(tf_ingestion_dir: str | Path, tf_persistent_dir : str | Path) -> tuple[str, str, str]:
    """
    Manages Terraform lifecycle:
    1. Apply persistent infrastructure.
    2. Teardown any leftover temporary infrastructure.
    3. Spin up fresh temporary infrastructure, passing in outputs from persistent.
    """
    logger = get_run_logger()
    persistent_dir = Path(tf_persistent_dir).resolve()
    temporary_dir = Path(tf_ingestion_dir).resolve()

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
                f"uv run python -m ingestion.pipeline {ensure_utc(oldest_time_allowed).isoformat()}",
            ]

    response = run_remote_command(instance_id, commands)
    if 'failed' in response:
        raise RuntimeError(response)
    
    print(response)


@task(name="destroy-infrastructure", retries=0)
def destroy_ingestion_infrastructure(tf_ingestion_dir: str | Path, elastic_ip_allocation_id: str, instance_profile_name: str) -> None:
    logger = get_run_logger()
    temporary_dir = Path(tf_ingestion_dir).resolve()

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

@task(name="provision-training-infrastructure", retries=0)
def provision_training_infrastructure(tf_training_dir: str | Path, tf_persistent_dir: str | Path) -> tuple[str, str]:
    """
    Manages Terraform lifecycle for the GPU training box:
    1. Apply persistent infrastructure.
    2. Teardown any leftover temporary infrastructure.
    3. Spin up fresh temporary infrastructure, passing in outputs from persistent.

    Unlike the ingestion stack this one needs no Elastic IP -- training only
    makes outbound calls -- so only the instance profile is threaded through.
    """
    logger = get_run_logger()
    persistent_dir = Path(tf_persistent_dir).resolve()
    temporary_dir = Path(tf_training_dir).resolve()

    tf_persistent = Terraform(working_dir=str(persistent_dir))
    tf_temporary = Terraform(working_dir=str(temporary_dir))

    # Step 1: Ensure persistent infrastructure exists
    logger.info(f"Applying persistent infrastructure in {persistent_dir}...")
    return_code = _run_terraform(persistent_dir, "apply", logger)
    if return_code != 0:
        raise TerraformCommandError(return_code, "terraform apply (persistent)", None, None)

    # Pull the outputs we need for the temporary infra
    persistent_outputs = tf_persistent.output()
    instance_profile_name = persistent_outputs["instance_profile_name"]['value']
    logger.info(f"Collected persistent outputs: instance_profile_name={instance_profile_name}")

    tf_vars = {"instance_profile_name": instance_profile_name}

    # Step 2: Tear down lingering temporary infrastructure
    logger.info(f"Destroying stale temporary infrastructure in {temporary_dir}...")
    return_code = _run_terraform(
        temporary_dir, "destroy", logger, tf_vars, timeout=TRAINING_DESTROY_TIMEOUT
    )
    if return_code != 0:
        raise TerraformCommandError(return_code, "terraform destroy (temporary)", None, None)

    # Step 3: Rebuild temporary infrastructure, passing in persistent outputs
    logger.info(f"Applying fresh temporary infrastructure in {temporary_dir}...")
    return_code = _run_terraform(temporary_dir, "apply", logger, tf_vars)
    if return_code != 0:
        raise TerraformCommandError(return_code, "terraform apply (temporary)", None, None)

    logger.info("Infrastructure lifecycle initialized successfully.")

    # Step 4: Collect the instance id from the temporary infra
    tmp_outputs = tf_temporary.output()
    tmp_instance_id = tmp_outputs["instance_id"]["value"]
    return instance_profile_name, tmp_instance_id


@task(name="call_training", retries=0)
def call_training(instance_id: str) -> None:
    """ call the training pipeline on an existing GPU EC2 instance """
    logger = get_run_logger()

    # The `cd` is load-bearing: config.yaml puts the train/val splits and the
    # ONNX checkpoints at paths relative to the CWD, and SSM runs commands from
    # /usr/bin. training.pipeline takes no arguments -- it reads everything from
    # config.yaml and Prefect Variables.
    commands = [
                "cd /opt/classification-pipeline/services/training",
                "uv run python -m training.pipeline",
            ]

    response = run_remote_command(instance_id, commands, execution_timeout=TRAINING_EXECUTION_TIMEOUT)
    logger.info(response)


@task(name="destroy-training-infrastructure", retries=0)
def destroy_training_infrastructure(
    tf_training_dir: str | Path,
    instance_profile_name: str,
    instance_id: str | None = None,
) -> None:
    """Tear down the GPU stack, terminating the instance directly if terraform can't.

    A destroy that fails or runs past its budget used to leave the box running
    and billing -- the task has no retries, and the flow was already unwinding.
    So `instance_id` gets terminated through the EC2 API as a backstop, and
    terraform is then re-run to reconcile the state file.
    """
    logger = get_run_logger()
    temporary_dir = Path(tf_training_dir).resolve()
    tf_vars = {"instance_profile_name": instance_profile_name}

    logger.info(f"Destroying temporary infrastructure in {temporary_dir}...")
    return_code = _run_terraform(
        temporary_dir, "destroy", logger, tf_vars, timeout=TRAINING_DESTROY_TIMEOUT
    )
    if return_code == 0:
        logger.info("Temporary infrastructure destroyed successfully.")
        return

    logger.error(f"terraform destroy exited {return_code}; falling back to the EC2 API.")

    if instance_id:
        ec2 = boto3.client("ec2", region_name=AWS_REGION)
        try:
            logger.info(f"Terminating {instance_id} directly...")
            ec2.terminate_instances(InstanceIds=[instance_id])
            ec2.get_waiter("instance_terminated").wait(
                InstanceIds=[instance_id],
                WaiterConfig={"Delay": 15, "MaxAttempts": 40},  # ~10 min ceiling
            )
            logger.info(f"Instance {instance_id} terminated.")
        except ClientError as e:
            # An id that no longer exists is the happy case -- nothing is billing.
            logger.warning(f"Could not terminate {instance_id} directly: {e}")
    else:
        logger.warning("No instance id available; cannot terminate directly.")

    # Second pass, now that nothing should be holding the resources open.
    return_code = _run_terraform(
        temporary_dir, "destroy", logger, tf_vars, timeout=TRAINING_DESTROY_TIMEOUT
    )
    if return_code != 0:
        raise TerraformCommandError(return_code, "terraform destroy (temporary)", None, None)

    logger.info("Temporary infrastructure destroyed successfully after direct termination.")


@task(name="trigger-frontend-deploy", retries=3, retry_delay_seconds=30)
def trigger_frontend_deploy() -> None:
    """Ask GitHub Actions to rebuild the Pages site against the new weights.

    The workflow pulls everything it needs from S3 itself, so client_payload is
    purely informational -- that keeps the deploy decoupled from whatever this
    flow happened to compute.
    """
    logger = get_run_logger()

    response = requests.post(
        f"https://api.github.com/repos/{GITHUB_REPO}/dispatches",
        headers={
            "Accept": "application/vnd.github+json",
            "Authorization": f"Bearer {get_github_dispatch_token()}",
            "X-GitHub-Api-Version": "2022-11-28",
        },
        json={
            "event_type": GITHUB_DISPATCH_EVENT,
            "client_payload": {
                "dispatched_at": datetime.now(timezone.utc).isoformat(),
                "source": "prefect",
            },
        },
        timeout=30,
    )

    # A successful dispatch is 204 No Content. A 404 here almost always means the
    # token lacks Contents: write rather than that the repo is missing.
    if response.status_code != 204:
        raise RuntimeError(
            f"repository_dispatch failed: {response.status_code} {response.text}"
        )

    logger.info(f"Dispatched '{GITHUB_DISPATCH_EVENT}' to {GITHUB_REPO}")


@flow
def main():
    logger = get_run_logger()

    tf_persistent_dir = Path(PROJECT_ROOT) / "infra/terraform/persistent"
    tf_ingestion_dir = Path(PROJECT_ROOT) / "infra/terraform/temporary/ingestion"
    tf_training_dir = Path(PROJECT_ROOT) / "infra/terraform/temporary/training"

    # These run unconditionally: `run_remote_command` resolves its region via
    # the Prefect Variable "aws-region", and training reads its wandb key and S3
    # prefixes the same way. With the ingestion block below commented out for
    # testing, the training-only path would otherwise depend on whatever a
    # previous run happened to leave behind on the Prefect server.
    set_secrets()
    set_vars()

    # # 1: Check for whether Clash Royale has been updated since the last run
    # #game_changed = check_for_game_changing_events()
    # game_changed = True; print("HARDCODED game_changed=True for testing")  # FOR TESTING ONLY
    # if not game_changed:
    #     return

    # logger.info("Initiating Data Collection & Training")


    # # 2: Create fresh infra for ingestion (clearing any existing ephemeral infra)


    # logger.info(f"Using Terraform base directory: {tf_ingestion_dir}")
    # oldest_time_allowed = datetime.now(timezone.utc) 
    # from datetime import timedelta
    # oldest_time_allowed = oldest_time_allowed - timedelta(weeks=2); print("HARDCODED oldest_time_allowed to be more forgiving for development")
    # elastic_ip_allocation_id, instance_profile_name, tmp_instance_id = provision_ingestion_infrastructure(tf_ingestion_dir=tf_ingestion_dir, tf_persistent_dir=tf_persistent_dir)

    # # 3: Invoke data ingestion
    # try:
    #     # Inside the try so that an instance which never becomes ready still
    #     # gets torn down by the `finally` rather than left running.
    #     wait_for_instance_ready(tmp_instance_id)
    #     call_ingestion(oldest_time_allowed, tmp_instance_id) # TODO: modify ingestion code to build the urls dict/json with every troop encountered and send those to S3
    # except:
    #     logger.error("Ingestion failed!!")
    #     raise # infra destruction in `finally` will still run before exiting
    
    # finally:
    #     # tear down data ingestion infra
    #     logger.info("destroying ephemeral infra")
    #     destroy_ingestion_infrastructure(tf_ingestion_dir, elastic_ip_allocation_id, instance_profile_name)
    
    # 4: Train, quantize and publish the model on a GPU box
    # training.pipeline handles train -> ONNX export -> INT8 quantize -> upload
    # in one process, so this is a single remote command.
    # TODO: save metrics summary to S3 (currently they only reach wandb)
    logger.info(f"Using Terraform training directory: {tf_training_dir}")
    training_profile_name, training_instance_id = provision_training_infrastructure(
        tf_training_dir=tf_training_dir,
        tf_persistent_dir=tf_persistent_dir,
    )

    try:
        wait_for_instance_ready(
            training_instance_id,
            ssm_timeout=TRAINING_SSM_TIMEOUT,
            setup_script_timeout=TRAINING_SETUP_TIMEOUT,
        )
        call_training(training_instance_id)
    except:
        logger.error("Training failed!!")
        raise # infra destruction in `finally` will still run before exiting

    finally:
        # tear down training infra -- a GPU box left running is expensive
        logger.info("destroying ephemeral training infra")
        destroy_training_infrastructure(
            tf_training_dir, training_profile_name, training_instance_id
        )

    # 5: Tell GitHub Actions to rebuild the site against the new weights.
    # Deliberately allowed to fail the flow: both ephemeral stacks are already
    # torn down by the `finally` above, so nothing is billing, and a failed
    # dispatch means the site is serving stale weights -- worth a red flow run
    # that can be re-run from the UI rather than a silent no-op.
    trigger_frontend_deploy()


def cli_run():
    main.serve(name="cr-pipeline")
