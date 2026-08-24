import polars as pl
from prefect import get_run_logger
import uuid

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