import yaml
from pydantic import BaseModel, Field

# --- Task-Specific Configuration Models ---

class StoreClanIDsConfig(BaseModel):
    num_workers: int = Field(default=4, description="Parallel worker processes for collecting random clans.")
    clans_per_worker: int = Field(default=10_000, description="Number of clan IDs each worker records")
    min_ids_needed: int = Field(default=100_000, description="Minimum number of clan IDs to store in the DB before skipping this step.")
    force_store_clans: bool = Field(default=False, description="Force storing clans even if we already have enough in the DB.")

class StoreActivePlayerIDsConfig(BaseModel):
    num_workers: int = Field(default=4, description="Parallel worker processes for storing active clan members.")
    source_clans_per_worker: int = Field(default=10_000, description="Number of clans each worker scrapes to find players.")
    min_ids_needed: int = Field(default=100_000, description="Minimum number of player IDs to store in the DB before skipping this step.")
    force_store_players: bool = Field(default=False, description="Force storing players even if we already have enough in the DB.")

class StoreGamesConfig(BaseModel):
    num_workers: int = Field(default=4, description="Parallel worker processes for fetching game logs.")
    inactivity_threshold_weeks: int = Field(default=2, description="Remove players whose latest game is older than X weeks.")
    game_modes_allowed: list[str] = Field(default=["draft", "triple draft", "mega draft", "ladder", "ranked ladder"], description="Game modes to include in the DB.")
    soft_games_limit: int = Field(default=20_000_000, description="Soft limit on the number of games to store in the games table")

class ExportCleanDatasetConfig(BaseModel):
    max_loser_lvl_advantage: int = Field(default=5, description="Max allowed avg card level gap for losing games.")
    max_games_chunk: int = Field(default=1_000_000, description="Max number of games to include in each parquet file chunk.")
    dataset_filename: str = Field(default="clean_games_dataset", description="Name of the parquet file to save the clean dataset to.")

class GlobalPipelineConfig(BaseModel):
    clear_logs: bool = Field(default=True, description="Whether to clear log files prior to execution.")
    setup_db: bool = Field(default=False, description="First time setup flag to build database tables.")
    worker_log_dir: str = Field(default="logs", description="Directory to store worker log files. Main process logs go to prefect cloud.")
    procs_per_api_key: int = Field(default=5, description="Max number of parallel processes to run per CR API key.")

class SaveDatabaseConfig(BaseModel):
    database_dump_filename: str = Field(default="latest_cr_db.dump", description="filename for persistent database dump in S3")

# --- Master Root Model ---

class PipelineConfig(BaseModel):
    global_opts: GlobalPipelineConfig = Field(default_factory=GlobalPipelineConfig)
    clan_ids: StoreClanIDsConfig = Field(default_factory=StoreClanIDsConfig)
    active_players: StoreActivePlayerIDsConfig = Field(default_factory=StoreActivePlayerIDsConfig)
    games: StoreGamesConfig = Field(default_factory=StoreGamesConfig)
    export_clean_dataset: ExportCleanDatasetConfig = Field(default_factory=ExportCleanDatasetConfig)
    save_db: SaveDatabaseConfig = Field(default_factory=SaveDatabaseConfig)


def load_pipeline_config(config_path: str = "config.yaml") -> PipelineConfig:
    """Always returns a single, strongly-typed PipelineConfig instance."""
    with open(config_path, "r") as f:
        data = yaml.safe_load(f) or {}
    return PipelineConfig(**data)