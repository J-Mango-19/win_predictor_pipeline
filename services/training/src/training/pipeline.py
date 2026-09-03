import logging
from pathlib import Path

import torch

from common.constants import PROJECT_ROOT
from common.utils import (
    get_aws_region,
    get_model_weights_prefix,
    get_s3_bucket_name,
    get_training_dataset_prefix,
    login_to_prefect,
)
from training.config import load_training_config
from training.data import build_dataloader, card_vocab_size, load_s3_folder, make_splits
from training.models.transformer import TransformerBinaryClassifier
from training.quantize import (
    export_onnx,
    quantize_onnx_int8,
    upload_model,
    write_metadata_sidecar,
)
from training.train import get_device, train

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

CONFIG_PATH = PROJECT_ROOT / "services/training/src/training/config.yaml"


def main() -> None:
    login_to_prefect()
    cfg = load_training_config(str(CONFIG_PATH))

    bucket = get_s3_bucket_name()
    dataset_prefix = get_training_dataset_prefix().strip("/")
    weights_prefix = get_model_weights_prefix().strip("/")
    storage_options = {"region": get_aws_region()}

    # 1. Download every parquet file in the S3 "folder" into one DataFrame.
    s3_uri = f"s3://{bucket}/{dataset_prefix}/*.parquet"
    full_df = load_s3_folder(s3_uri, storage_options=storage_options)
    vocab_size = card_vocab_size(full_df)
    logger.info("inferred vocab_size=%d", vocab_size)

    # 2. Split into train/val and persist each split locally as parquet.
    train_path, val_path = make_splits(
        full_df,
        cfg.data.train_fraction,
        cfg.data.shuffle,
        cfg.data.seed,
        cfg.data.local_dir,
    )
    del full_df

    train_loader = build_dataloader(
        train_path, cfg.training.batch_size, shuffle=True, workers=cfg.training.dataloader_workers
    )
    val_loader = build_dataloader(
        val_path, cfg.training.batch_size, shuffle=False, workers=cfg.training.dataloader_workers
    )

    # 3. Train.
    device = get_device()
    model = TransformerBinaryClassifier(
        d_model=cfg.model.d_model,
        n_heads=cfg.model.n_heads,
        n_sxa_blocks=cfg.model.n_sxa_blocks,
        n_masked_xa_blocks=cfg.model.n_masked_xa_blocks,
        mlp_hidden=cfg.model.mlp_hidden,
        vocab_size=vocab_size,
        dropout=cfg.model.dropout,
        set_size=cfg.model.set_size,
    )
    if cfg.training.compile and device.type in ("cuda", "cpu"):
        model = torch.compile(model)

    model = train(model, train_loader, val_loader, cfg.training, device, cfg.model)
    raw_model = getattr(model, "_orig_mod", model)  # unwrap torch.compile

    # 4. Export to ONNX, quantize weights to INT8, and upload to the configured S3 "folder".
    out_dir = Path(cfg.output.checkpoint_dir)
    fp32_path = out_dir / "model_fp32.onnx"
    int8_path = out_dir / cfg.output.model_filename
    export_onnx(raw_model, cfg.model, vocab_size, fp32_path)
    quantize_onnx_int8(fp32_path, int8_path)
    upload_model(int8_path, bucket, f"{weights_prefix}/{cfg.output.model_filename}")

    # The frontend build reads this instead of the graph's metadata_props, which
    # onnxruntime-web cannot see. Uploaded after the model so the sidecar never
    # describes weights that failed to land.
    sidecar = write_metadata_sidecar(int8_path, cfg.model, vocab_size)
    upload_model(sidecar, bucket, f"{weights_prefix}/{sidecar.name}")


if __name__ == "__main__":
    main()
