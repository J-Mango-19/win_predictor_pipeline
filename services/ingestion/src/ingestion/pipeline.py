import gc
import sys
import tempfile
import subprocess
import logging
import boto3
import psycopg
import polars as pl
from pathlib import Path
from datetime import datetime
from botocore.exceptions import ClientError
from multiprocessing import Pool
from ingestion.utils import construct_db_URI, make_card_to_idx_mapping
from ingestion.config import load_pipeline_config,  StoreActivePlayerIDsConfig, StoreGamesConfig, StoreClanIDsConfig
from ingestion.extractors.clans import get_region_IDs, fetch_and_store_clans
from ingestion.extractors.players import fetch_and_store_players
from ingestion.extractors.battles import fetch_and_store_games
from common.utils import get_api_credentials, get_s3_bucket_name, get_database_dump_prefix, get_aws_region, login_to_prefect, get_parquet_dataset_prefix, ensure_utc
from common.constants import PROJECT_ROOT, WINNER_LVL_COLS, LOSER_LVL_COLS, WINNER_CARD_COLS, LOSER_CARD_COLS, INVALID_TOKEN

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def task_store_clan_ids(cfg: StoreClanIDsConfig, cr_api_keys: list[str], worker_log_dir: Path, procs_per_api_key: int):
    """ Sample clan IDs and store them in the DB """
    logger.info("starting process pool for gathering & storing clan IDs")
    valid_reg_IDs = list(get_region_IDs(cr_api_key=cr_api_keys[0]))

    pool_args = []
    api_key_idx = 0
    for i in range(cfg.num_workers):
        if i % procs_per_api_key == 0 and i > 0:
            api_key_idx = api_key_idx + 1
            if api_key_idx >= len(cr_api_keys):
                logger.warning(f"Ran out of API keys to use for workers; using only {i} workers instead of {cfg.num_workers}")
                break
                
        pool_args.append((cfg.clans_per_worker, cr_api_keys[api_key_idx], valid_reg_IDs, worker_log_dir))

    with Pool(processes=len(pool_args)) as pool:
        pool.starmap(fetch_and_store_clans, pool_args)
    logger.info(f"clan ID gathering process pool finished. Worker logs saved to {worker_log_dir}")

def task_store_active_player_ids(cfg: StoreActivePlayerIDsConfig, cr_api_keys: list[str], worker_log_dir: Path, procs_per_api_key: int):
    """ Store active players scraped from collected clan IDs into the DB """
    logger.info("starting process pool for storing clan members")

    pool_args = []
    api_key_idx = 0
    for i in range(cfg.num_workers):
        if i % procs_per_api_key == 0 and i > 0:
            api_key_idx = api_key_idx + 1
            if api_key_idx >= len(cr_api_keys):
                logger.warning(f"Ran out of API keys to use for workers; using only {i} workers instead of {cfg.num_workers}")
                break
                
        pool_args.append((cfg.source_clans_per_worker, cr_api_keys[api_key_idx], worker_log_dir))

    with Pool(processes=len(pool_args)) as pool:
        pool.starmap(fetch_and_store_players, pool_args)
    logger.info(f"Active player ID gathering process pool finished. Worker logs saved to {worker_log_dir}")

def task_store_games(cfg: StoreGamesConfig, cr_api_keys: list[str], worker_log_dir: Path, procs_per_api_key: int, oldest_time_allowed: datetime): 
    """ Store games scraped from collected player IDs into the DB """
    logger.info("starting process pool for storing games")
    card_to_idx = make_card_to_idx_mapping()


    # pre-req: ensure that no player_id's in active_players are claimed
    release_claims_sql = """
    ALTER TABLE active_players DROP COLUMN claimed;
    ALTER TABLE active_players ADD COLUMN claimed BOOLEAN NOT NULL DEFAULT FALSE;
    """
    with psycopg.connect(construct_db_URI()) as conn:
        with conn.cursor() as cur:
            cur.execute(release_claims_sql)

    pool_args = []
    api_key_idx = 0
    for i in range(cfg.num_workers):
        if i % procs_per_api_key == 0 and i > 0:
            api_key_idx = api_key_idx + 1
            if api_key_idx >= len(cr_api_keys):
                logger.warning(f"Ran out of API keys to use for workers; using only {i} workers instead of {cfg.num_workers}")
                break
                
        pool_args.append((cr_api_keys[api_key_idx], oldest_time_allowed, cfg.game_modes_allowed, cfg.inactivity_threshold_weeks, card_to_idx, worker_log_dir, cfg.soft_games_limit))

    with Pool(processes=len(pool_args)) as pool:
        pool.starmap(fetch_and_store_games, pool_args)
    logger.info(f"Active player ID gathering process pool finished. Worker logs saved to {worker_log_dir}")


def task_export_clean_dataset(
    bucket: str,
    prefix: str,
    dataset_filename: str,
    max_games_chunk: int,
    max_lvl_gap: float,
    aws_region: str
) -> bool:
    """
    Read the `games` table, clean it, and upload the result to S3 as a
    single Parquet file at s3://{bucket}/{prefix}{dataset_filename}.parquet.
 
    Returns
    -------
    bool
        True if the cleaned dataset was successfully uploaded to S3,
        False if any step failed.
    """
    dst_s3_path = f"s3://{bucket}/{prefix}{dataset_filename}"
    storage_options = {"region": aws_region}
 
    with tempfile.TemporaryDirectory() as tmpdir:
        chunk_paths: list[Path] = []
 
        # ---- Step 1: stream the table out of Postgres in bounded chunks ----
        try:
            with psycopg.connect(construct_db_URI()) as conn:
                with conn.transaction():
                    with conn.cursor(name="games_chunk_cursor") as cur:
                        cur.execute(
                            "SELECT * FROM games ORDER BY time, winner_id, loser_id;"
                        )
                        colnames = [desc.name for desc in cur.description]
 
                        idx = 0
                        while True:
                            rows = cur.fetchmany(max_games_chunk)
                            if not rows:
                                break
 
                            df = pl.DataFrame(rows, schema=colnames, orient="row")
                            df = df.with_columns(pl.col(pl.Int64).cast(pl.UInt8))
 
                            chunk_path = Path(tmpdir) / f"chunk_{idx}.parquet"
                            df.write_parquet(chunk_path)
                            chunk_paths.append(chunk_path)
 
                            logger.info(
                                f"Buffered chunk {idx} ({len(df)} rows) to local disk."
                            )
 
                            del df
                            gc.collect()
                            idx += 1
        except Exception:
            raise
 
        if not chunk_paths:
            raise ValueError("`games` returned no rows; nothing to upload.")
        
 
        # ---- Step 2: stream-filter the local chunks, sink straight to S3 ----
        # We keep games where: loser_avg <= winner_avg + gap
        try:
            (
                pl.scan_parquet(chunk_paths)
                .filter(
                    (pl.sum_horizontal(LOSER_LVL_COLS) / 8)
                    <= (pl.sum_horizontal(WINNER_LVL_COLS) / 8) + max_lvl_gap
                )
                .filter(
                    ~pl.any_horizontal(
                        pl.col(WINNER_CARD_COLS + LOSER_CARD_COLS) == INVALID_TOKEN
                    )
                )
                .sink_parquet(dst_s3_path, storage_options=storage_options)
            )
        except Exception:
            raise
 
    logger.info(f"Successfully uploaded cleaned games dataset to {dst_s3_path}.")
    return True


def task_save_database_dump(
    bucket: str,
    prefix: str,
    filename: str,
    aws_region: str,
) -> None:
    """
    Delete the games table, save the database as a .dump file,
    rename any existing S3 dump to backup.dump, and upload the new dump.
    """

    db_uri = construct_db_URI()

    # Clear the games table and commit changes
    with psycopg.connect(db_uri) as conn:
        with conn.cursor() as cur:
            cur.execute("TRUNCATE TABLE games CASCADE;")
        conn.commit()

    # 2. Dump the database to a .dump file using pg_dump
    db_dump_dst = PROJECT_ROOT / f"services/ingestion/src/ingestion/{filename}"
    with db_dump_dst.open("wb") as dump_file:
        subprocess.run(
            [
                "docker",
                "exec",
                "postgres",
                "pg_dump",
                "-Fc",
                db_uri,
            ],
            stdout=dump_file,
            check=True,
        )
    # 3. AWS S3 Client setup
    session = boto3.Session(region_name=aws_region)
    s3_client = session.client("s3")

    prefix_clean = prefix.rstrip('/')
    s3_key = f"{prefix_clean}/{filename}"
    backup_key = f"{prefix_clean}/backup.dump"

    # 4. Check if filename exists in S3; if so, rename (copy + delete) to backup.dump
    try:
        s3_client.head_object(Bucket=bucket, Key=s3_key)

        # Copy existing object to backup.dump (automatically overwrites existing backup.dump)
        s3_client.copy_object(
            Bucket=bucket,
            CopySource={"Bucket": bucket, "Key": s3_key},
            Key=backup_key,
        )

        # Delete the original file
        s3_client.delete_object(Bucket=bucket, Key=s3_key)

    except ClientError as e:
        error_code = e.response.get("Error", {}).get("Code")
        if error_code not in ("404", "NoSuchKey"):
            raise

    # 5. Upload the new dump file to S3
    s3_client.upload_file(db_dump_dst, bucket, s3_key)


def main():
    if len(sys.argv) < 2:
        print("Error: Please provide at least one argument.", file=sys.stderr)
        print("Usage: uv run myscript.py <your_argument>")
        sys.exit(1)

    login_to_prefect()

    # A naive timestamp on the CLI is interpreted as UTC so it stays comparable
    # with the timezone-aware battle times pulled from the API.
    oldest_time_allowed = ensure_utc(datetime.fromisoformat(sys.argv[1]))
    config_path = PROJECT_ROOT / "services/ingestion/src/ingestion/config.yaml"
    pipeline_cfg = load_pipeline_config(config_path)

    cr_api_keys = get_api_credentials()

    # check if we have enough clans stored in the clans table
    with psycopg.connect(construct_db_URI()) as conn:
        with conn.cursor() as cur:
            cur.execute("SELECT COUNT(*) FROM clans;")
            num_clans = cur.fetchone()[0]
    
    store_clans = num_clans < pipeline_cfg.clan_ids.min_ids_needed or pipeline_cfg.clan_ids.force_store_clans
    if store_clans:
        task_store_clan_ids(
            cfg=pipeline_cfg.clan_ids, 
            cr_api_keys=cr_api_keys,
            worker_log_dir=Path(pipeline_cfg.global_opts.worker_log_dir),
            procs_per_api_key=pipeline_cfg.global_opts.procs_per_api_key
        )


    # check if we have enough players stored in the active_players table
    with psycopg.connect(construct_db_URI()) as conn:
        with conn.cursor() as cur:
            cur.execute("SELECT COUNT(*) FROM active_players;")
            num_players = cur.fetchone()[0] 
    store_players = num_players < pipeline_cfg.active_players.min_ids_needed  or pipeline_cfg.active_players.force_store_players
    if store_players:
        task_store_active_player_ids(
            cfg=pipeline_cfg.active_players, 
            cr_api_keys=cr_api_keys,
            worker_log_dir=Path(pipeline_cfg.global_opts.worker_log_dir),
            procs_per_api_key=pipeline_cfg.global_opts.procs_per_api_key
        )

    # we populate the games table at each invocation
    task_store_games(
        cfg=pipeline_cfg.games, 
        cr_api_keys=cr_api_keys,
        worker_log_dir=Path(pipeline_cfg.global_opts.worker_log_dir),
        oldest_time_allowed=oldest_time_allowed,
        procs_per_api_key=pipeline_cfg.global_opts.procs_per_api_key
    )
    
    # we create a clean dataset at each invocation
    export_success = task_export_clean_dataset(
        bucket=get_s3_bucket_name(),
        prefix=get_parquet_dataset_prefix(),
        dataset_filename=pipeline_cfg.export_clean_dataset.dataset_filename,
        max_games_chunk=pipeline_cfg.export_clean_dataset.max_games_chunk,
        max_lvl_gap=pipeline_cfg.export_clean_dataset.max_loser_lvl_advantage,
        aws_region=get_aws_region()
    )
    if not export_success:
        raise

    # save the rest of the database to a .dump file in S3 (w/o the games table)
    task_save_database_dump(
        bucket=get_s3_bucket_name(),
        prefix=get_database_dump_prefix(),
        filename=pipeline_cfg.save_db.database_dump_filename,
        aws_region=get_aws_region()
    )

if __name__ == '__main__':
    main()