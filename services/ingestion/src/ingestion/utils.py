import os
import json
import boto3
import psycopg
import requests
import logging
from pathlib import Path
from urllib3.util import Retry
from requests.adapters import HTTPAdapter
from common.utils import load_database_credentials
from common.constants import GAME_MODES

def setup_process_file_logger(log_dir: Path) -> logging.Logger:
    """Configures a logger that writes exclusively to a PID-named file."""
    pid = os.getpid()
    logger_name = f"worker_process_{pid}"
    logger = logging.getLogger(logger_name)

    # Prevent adding duplicate handlers if the same worker process handles multiple items
    if not logger.handlers:
        log_dir.mkdir(parents=True, exist_ok=True)
        log_file_path = log_dir / f"worker_pid_{pid}.log"
        
        file_handler = logging.FileHandler(log_file_path, mode="a")
        formatter = logging.Formatter(
            "%(asctime)s | [PID %(process)d] | %(levelname)s | %(message)s"
        )
        file_handler.setFormatter(formatter)
        
        logger.addHandler(file_handler)
        logger.setLevel(logging.INFO)
        # Prevent logs from bubbling up to Prefect's stdout interception
        logger.propagate = False

    return logger

def make_robust_session(retries=6, backoff_factor=2, status_forcelist=(504,)) -> requests.sessions.Session:
    """Creates a requests session with retry logic for handling transient errors.

    Args:
        retries: Total number of retries for failed requests.
        backoff_factor: A backoff factor to apply between attempts after the second try.
        status_forcelist: A set of HTTP status codes that we should force a retry on.
    
    Returns:
        A requests.Session object configured with retry logic.
    """
    session = requests.Session()
    retries = Retry(
        total=retries,
        backoff_factor=backoff_factor,
        status_forcelist=status_forcelist
    )
    session.mount('https://', HTTPAdapter(max_retries=retries))
    return session

def construct_db_URI() -> str:
    """ returns the database URI using the parameters from the given JSON file """
    db_params = load_database_credentials()
    return f"postgresql://{db_params['user']}:{db_params['password']}@{db_params['host']}:{db_params['port']}/{db_params['dbname']}".format(**db_params)

def make_card_to_idx_mapping() -> dict:
    """ Reads the card_ids table of the database and returns a mapping from card name to card ID. """
    with psycopg.connect(construct_db_URI()) as conn:
        with conn.cursor() as cur:
            cur.execute("SELECT id, name FROM card_ids;")
            rows = cur.fetchall()
            card_to_idx = {name: idx for idx, name in rows}
    return card_to_idx


def create_db_tables() -> None:
    """ 
    Creates tables (if they don't already exist):
        - clans (id:str, members_scraped:bool, claimed:bool)
        - active_players (id:str, claimed: bool)
        - games (winner_id, loser_id, time, game_mode, winner cards, loser cards, winner card lvls, loser card lvls)
        - card IDs (id:int, name:str)
    
    Assumes the database itself already exists.
    """
    with psycopg.connect(construct_db_URI()) as conn:
        with conn.cursor() as cur:
            # Create clans table
            cur.execute("""
                CREATE TABLE IF NOT EXISTS clans (
                    clan_id VARCHAR(20) PRIMARY KEY,
                    members_collected BOOLEAN,
                    claimed BOOLEAN 
                );
            """)
        
            # Create active_players table
            cur.execute("""
                CREATE TABLE IF NOT EXISTS active_players (
                    player_id VARCHAR(20) PRIMARY KEY,
                    claimed BOOLEAN
                );
            """)

            # Create games table
            cur.execute("""
                CREATE TABLE IF NOT EXISTS games (
                    winner_id VARCHAR(12) NOT NULL,
                    loser_id VARCHAR(12) NOT NULL,
                    time TIMESTAMPTZ NOT NULL,
                    game_mode VARCHAR(32) NOT NULL,

                    winner_card_0 INT, winner_card_1 INT, winner_card_2 INT, winner_card_3 INT,
                    winner_card_4 INT, winner_card_5 INT, winner_card_6 INT, winner_card_7 INT,

                    loser_card_0 INT, loser_card_1 INT, loser_card_2 INT, loser_card_3 INT,
                    loser_card_4 INT, loser_card_5 INT, loser_card_6 INT, loser_card_7 INT,

                    winner_card_0_level INT, winner_card_1_level INT, winner_card_2_level INT,
                    winner_card_3_level INT, winner_card_4_level INT, winner_card_5_level INT,
                    winner_card_6_level INT, winner_card_7_level INT,

                    loser_card_0_level INT, loser_card_1_level INT, loser_card_2_level INT,
                    loser_card_3_level INT, loser_card_4_level INT, loser_card_5_level INT,
                    loser_card_6_level INT, loser_card_7_level INT
                );
                
                CREATE UNIQUE INDEX IF NOT EXISTS games_unique_game
                    ON games (winner_id, loser_id, time);
            """)

            # create card IDs table
            # 1. Read existing mapping if available
            cardToIdx = {}
            if os.path.exists('info/cardToIdx.json'):
                with open('info/cardToIdx.json', 'r') as f:
                    cardToIdx = json.load(f)

            # 2. Create table with card ids generated automatically
            cur.execute("""
                CREATE TABLE IF NOT EXISTS card_ids (
                    id INT GENERATED BY DEFAULT AS IDENTITY PRIMARY KEY,
                    name TEXT NOT NULL UNIQUE
                );
            """)

            # 3. Populate the card_ids table with the existing mapping (if available)
            if cardToIdx:
                # Prepare tuple list, eg: [(0, 'bats'), (1, 'lumberjack'), (2, 'wizard')]
                values = [(idx, name) for name, idx in cardToIdx.items()]
                
                cur.executemany(
                    """
                    INSERT INTO card_ids (id, name) 
                    VALUES (%s, %s) 
                    ON CONFLICT (id) DO NOTHING;
                    """,
                    values
                )
                
                # Sync the auto-increment sequence so PostgreSQL knows to start after max(id)
                cur.execute("""
                    SELECT setval(
                        pg_get_serial_sequence('card_ids', 'id'),
                        COALESCE((SELECT MAX(id) FROM card_ids), 0)
                    );
                """)

def translate_gameModes_to_apiNames(selected_gameModes: list) -> list:
    """ Translates human-readable gameMode names to their corresponding API identifiers. """
    return [GAME_MODES.get(gm) for gm in selected_gameModes]

def list_s3_objects(bucket: str, prefix: str) -> list[str]:
    """List all object keys in an S3 bucket under a given prefix.

    Args:
        bucket: Name of the S3 bucket.
        prefix: Prefix to search within the bucket.

    Returns:
        A list of S3 object keys matching the prefix.
    """
    s3 = boto3.client("s3")

    keys = []
    paginator = s3.get_paginator("list_objects_v2")

    for page in paginator.paginate(Bucket=bucket, Prefix=prefix):
        keys.extend(obj["Key"] for obj in page.get("Contents", []))

    return keys
