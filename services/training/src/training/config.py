import yaml
from pydantic import BaseModel, Field

# --- Task-Specific Configuration Models ---

class DataConfig(BaseModel):
    train_fraction: float = Field(default=0.8, gt=0.0, lt=1.0, description="Fraction of games used for training; the remainder is validation.")
    shuffle: bool = Field(default=True, description="Shuffle rows before splitting. Set False to keep a chronological (time-ordered) split.")
    seed: int = Field(default=42, description="RNG seed for the shuffle/split so runs are reproducible.")
    local_dir: str = Field(default="./data", description="Local directory where train.parquet / val.parquet are written.")

class ModelConfig(BaseModel):
    d_model: int = Field(default=256, description="Internal feature dimension.")
    n_heads: int = Field(default=4, description="Number of attention heads (must divide d_model).")
    n_sxa_blocks: int = Field(default=3, description="Number of self-attention + cross-attention blocks.")
    n_masked_xa_blocks: int = Field(default=1, description="Number of level-masked cross-attention blocks.")
    mlp_hidden: int = Field(default=256, description="Hidden size of the classification MLP.")
    set_size: int = Field(default=8, description="Number of cards per deck.")
    dropout: float = Field(default=0.0, ge=0.0, lt=1.0, description="Dropout probability.")

class TrainingConfig(BaseModel):
    n_epochs: int = Field(default=3, gt=0, description="Number of passes over the training set.")
    batch_size: int = Field(default=8192, gt=0, description="Number of games per batch.")
    lr: float = Field(default=2e-4, gt=0.0, description="Peak learning rate.")
    weight_decay: float = Field(default=1e-2, ge=0.0, description="AdamW weight decay (not applied to norm/bias params).")
    warmup_ratio: float = Field(default=0.05, ge=0.0, le=1.0, description="Fraction of total steps spent in linear LR warmup.")
    min_lr_ratio: float = Field(default=0.1, ge=0.0, le=1.0, description="Cosine-decay LR floor as a fraction of peak LR.")
    max_grad_norm: float = Field(default=1.0, gt=0.0, description="Gradient-norm clipping threshold.")
    grad_accumulation_steps: int = Field(default=1, ge=1, description="Micro-batches accumulated before an optimizer step.")
    compile: bool = Field(default=False, description="torch.compile the model (CUDA/CPU only).")
    dataloader_workers: int = Field(default=0, ge=0, description="Subprocesses used by the DataLoader.")

class OutputConfig(BaseModel):
    checkpoint_dir: str = Field(default="./checkpoints", description="Local directory for saved model files.")
    model_filename: str = Field(default="model_int8.onnx", description="Basename for the INT8 ONNX model, used locally and as the S3 object name.")

# --- Master Root Model ---

class TrainingPipelineConfig(BaseModel):
    data: DataConfig = Field(default_factory=DataConfig)
    model: ModelConfig = Field(default_factory=ModelConfig)
    training: TrainingConfig = Field(default_factory=TrainingConfig)
    output: OutputConfig = Field(default_factory=OutputConfig)


def load_training_config(config_path: str = "config.yaml") -> TrainingPipelineConfig:
    """Always returns a single, strongly-typed TrainingPipelineConfig instance."""
    with open(config_path, "r") as f:
        data = yaml.safe_load(f) or {}
    return TrainingPipelineConfig(**data)
