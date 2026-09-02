import logging
from pathlib import Path

import numpy as np
import polars as pl
import torch
from torch.utils.data import DataLoader, TensorDataset

from common.constants import (
    WINNER_CARD_COLS,
    LOSER_CARD_COLS,
    WINNER_LVL_COLS,
    LOSER_LVL_COLS,
)

logger = logging.getLogger(__name__)

# Deck A = winner, Deck B = loser. The model is antisymmetric, so the label is
# always "deck A wins" and never needs to be stored.
CARD_COLS = WINNER_CARD_COLS + LOSER_CARD_COLS      # 16 columns
LVL_COLS = WINNER_LVL_COLS + LOSER_LVL_COLS         # 16 columns
FEATURE_COLS = CARD_COLS + LVL_COLS                 # 32 columns
MAX_CARD_LEVEL = 16


def load_s3_folder(s3_uri: str, storage_options: dict | None = None) -> pl.DataFrame:
    """Download every parquet file under an S3 prefix into a single DataFrame.

    Args:
        s3_uri: a glob such as ``s3://bucket/prefix/*.parquet``.
        storage_options: forwarded to polars (e.g. ``{"region": "us-east-2"}``).
    """
    logger.info("scanning parquet files at %s", s3_uri)
    df = (
        pl.scan_parquet(s3_uri, storage_options=storage_options)
        .select(FEATURE_COLS)
        .collect()
    )
    logger.info(
        "loaded %s games (%.2f GB in memory)",
        f"{df.height:,}",
        df.estimated_size("gb"),
    )
    return df


def card_vocab_size(df: pl.DataFrame) -> int:
    """Smallest embedding-table size that fits every card id in the dataset."""
    return int(df.select(CARD_COLS).to_numpy().max()) + 1


def make_splits(
    df: pl.DataFrame,
    train_fraction: float,
    shuffle: bool,
    seed: int,
    local_dir: str,
) -> tuple[Path, Path]:
    """Split ``df`` into train/val sets and write each to a local parquet file."""
    if shuffle:
        df = df.sample(fraction=1.0, shuffle=True, seed=seed)

    n_train = int(df.height * train_fraction)
    train_df = df.head(n_train)
    val_df = df.tail(df.height - n_train)

    out_dir = Path(local_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    train_path = out_dir / "train.parquet"
    val_path = out_dir / "val.parquet"
    train_df.write_parquet(train_path)
    val_df.write_parquet(val_path)

    logger.info(
        "wrote %s train / %s val games to %s (shuffle=%s)",
        f"{train_df.height:,}",
        f"{val_df.height:,}",
        out_dir,
        shuffle,
    )
    return train_path, val_path


def _to_tensors(df: pl.DataFrame) -> TensorDataset:
    cards = torch.from_numpy(df.select(CARD_COLS).to_numpy().astype(np.int64))
    lvls = torch.from_numpy(df.select(LVL_COLS).to_numpy().astype(np.float32)) / MAX_CARD_LEVEL
    return TensorDataset(cards, lvls)


def build_dataloader(
    parquet_path: Path,
    batch_size: int,
    shuffle: bool,
    workers: int,
) -> DataLoader:
    """Load a split parquet file into an in-memory DataLoader of (cards, levels)."""
    dataset = _to_tensors(pl.read_parquet(parquet_path))
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=workers,
        drop_last=shuffle and len(dataset) > batch_size,
        pin_memory=torch.cuda.is_available(),
    )
