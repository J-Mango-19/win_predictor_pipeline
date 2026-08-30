import uuid
import time
from botocore.exceptions import WaiterError
import polars as pl
from prefect import get_run_logger

def construct_prompt(prompt_file_path: str, games_dataset_path: str) -> str:
    """ Instructs the Agent to look for updates *after* the earliest data used in the last training run """
    print("Constructing prompt for Agent...")

    def get_cutoff_date(games_dataset_path: str) -> str:
        """ returns the date of the oldest battle in the S3 dataset used to train the current model """
        print("Getting cutoff date from dataset...")
        date = pl.scan_parquet(games_dataset_path).select(pl.col("time").min()).collect().item()
        return date

    date = get_cutoff_date(games_dataset_path)

    with open(prompt_file_path, 'r') as fp:
        prompt = fp.read().replace("__DATE__", str(date))

    return prompt


def extract_json_from_tail(full_response: str) -> str:
    """ returns the first JSON object delimited by [] working up from the tail of the input str """
    print("Extracting JSON from Agent's response...")

    # find the closing ]
    end_idx = len(full_response) - 1
    while full_response[end_idx] != ']':
        end_idx -= 1


    # find the opening [
    start_idx = end_idx - 1
    while full_response[start_idx] != '[':
        start_idx -= 1


    # clear out any unwanted characters
    invalid_chars = ['\n', '\r']
    template = full_response[start_idx : end_idx + 1]
    out = ""

    for i in range(len(template)):
        if template[i] not in invalid_chars:
            out += template[i]

    return out


def get_response(client, prompt: str) -> str:
    """ return agent's output to the prompt """

    logger = get_run_logger()
    response = client.invoke_harness(
        harnessArn='arn:aws:bedrock-agentcore:us-east-2:867138159308:harness/harness_y2jm7-32nOx7W3n9',
        runtimeSessionId=str(uuid.uuid4()),
        messages=[
            {
                'role': 'user',
                'content': [{'text': prompt}]
            }
        ]
    )

    # Process the streaming response
    full_response = ""
    for event in response['stream']:
        if 'contentBlockDelta' in event:
            delta = event['contentBlockDelta'].get('delta', {})
            if 'text' in delta:
                full_response += '\n'
                full_response += str(delta['text'])

    logger.info(f"Agent's response: {full_response}")
    return full_response


def wait_for_setup_script(ssm, instance_id: str, timeout: int = 600, poll_interval: int = 10):
    """Wait until setup.sh has created the completion sentinel."""
    logger = get_run_logger()

    deadline = time.monotonic() + timeout

    while time.monotonic() < deadline:
        response = ssm.send_command(
            InstanceIds=[instance_id],
            DocumentName="AWS-RunShellScript",
            Parameters={
                "commands": [
                    "test -f /opt/classification-pipeline/.setup-complete"
                ]
            },
        )

        command_id = response["Command"]["CommandId"]

        # Wait for SSM to actually execute the command.
        while True:
            try:
                invocation = ssm.get_command_invocation(
                    CommandId=command_id,
                    InstanceId=instance_id,
                )
            except ssm.exceptions.InvocationDoesNotExist:
                # SSM has accepted the command but hasn't exposed the
                # invocation through GetCommandInvocation yet.
                time.sleep(1)
                continue

            status = invocation["Status"]

            if status in {"Pending", "InProgress", "Delayed"}:
                time.sleep(1)
                continue

            break

        if status == "Success":
            print("Instance setup complete.")
            return

        # Sentinel doesn't exist yet, so setup is presumably still running.
        time.sleep(poll_interval)

    raise TimeoutError(
        f"Instance {instance_id} did not finish setup within {timeout} seconds"
    )

def wait_for_ec2_status_checks(ec2, instance_id:str ) -> None:
    logger = get_run_logger()
    logger.info(f"Waiting for instance {instance_id} to pass EC2 status checks...")
    waiter = ec2.get_waiter('instance_status_ok')
    try:
        waiter.wait(
            InstanceIds=[instance_id],
            WaiterConfig={'Delay': 15, 'MaxAttempts': 40},  # ~10 min ceiling
        )
    except WaiterError as e:
        raise RuntimeError(f"Instance {instance_id} never passed EC2 status checks: {e}")
    logger.info(f"Instance {instance_id} passed EC2 status checks.")

def wait_for_ssm_registration(ssm, instance_id: str, ssm_timeout: int, ssm_poll_interval: int):

    logger = get_run_logger()
    logger.info(f"Waiting for SSM agent on {instance_id} to come online...")
    elapsed = 0
    ssm_hit_timeout=True
    while elapsed < ssm_timeout:
        resp = ssm.describe_instance_information(
            Filters=[{'Key': 'InstanceIds', 'Values': [instance_id]}]
        )
        infos = resp.get('InstanceInformationList', [])
        if infos and infos[0].get('PingStatus') == 'Online':
            logger.info(f"SSM agent on {instance_id} is online.")
            ssm_hit_timeout=False
            break
        time.sleep(ssm_poll_interval)
        elapsed += ssm_poll_interval

    if ssm_hit_timeout:
        raise RuntimeError(
            f"SSM agent on {instance_id} did not come online within {ssm_timeout}s. "
            f"Check that the instance profile has AmazonSSMManagedInstanceCore and that "
            f"the SSM agent is installed/running on your AMI."
        )
