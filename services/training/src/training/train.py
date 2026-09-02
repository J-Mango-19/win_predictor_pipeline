import contextlib
import logging
import math

import torch
import torch.nn as nn
import wandb
from torch.utils.data import DataLoader

from training.config import ModelConfig, TrainingConfig
from common.utils import (
    get_wandb_api_key,
    get_wandb_project_name,
    get_wandb_entity,
)

logger = logging.getLogger(__name__)

LOG_EVERY_N_STEPS = 50


def login_to_wandb() -> None:
    """Authenticate the wandb client using the configured API key."""
    wandb.login(key=get_wandb_api_key())


def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    logger.warning("no GPU available; training on CPU")
    return torch.device("cpu")


def _autocast(device: torch.device):
    """bfloat16 autocast on CUDA, no-op everywhere else."""
    if device.type == "cuda":
        return torch.autocast(device_type="cuda", dtype=torch.bfloat16)
    return contextlib.nullcontext()


def _warmup_cosine_scheduler(optimizer, warmup_steps: int, total_steps: int, min_lr_ratio: float):
    """Linear warmup then cosine decay down to ``min_lr_ratio`` of the peak LR."""
    def lr_lambda(step: int) -> float:
        if step < warmup_steps:
            return (step + 1) / max(1, warmup_steps)
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        cosine = 0.5 * (1.0 + math.cos(math.pi * min(1.0, progress)))
        return min_lr_ratio + (1.0 - min_lr_ratio) * cosine

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def _build_optimizer(model: nn.Module, lr: float, weight_decay: float, device: torch.device):
    """AdamW with weight decay applied only to >=2D params (skips norms and biases)."""
    decay, no_decay = [], []
    for param in model.parameters():
        if not param.requires_grad:
            continue
        (decay if param.ndim >= 2 else no_decay).append(param)

    return torch.optim.AdamW(
        [
            {"params": decay, "weight_decay": weight_decay},
            {"params": no_decay, "weight_decay": 0.0},
        ],
        lr=lr,
        fused=(device.type == "cuda"),
    )


def _split_batch(cards: torch.Tensor, lvls: torch.Tensor, device: torch.device):
    cards = cards.to(device, non_blocking=True)
    lvls = lvls.to(device, non_blocking=True)
    half = cards.shape[1] // 2
    return (
        cards[:, :half].int(),      # deck A (winner) card ids
        cards[:, half:].int(),      # deck B (loser) card ids
        lvls[:, :half].float(),     # deck A card levels (already scaled to [0, 1])
        lvls[:, half:].float(),     # deck B card levels
    )


@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader, loss_fn, device: torch.device) -> tuple[float, float]:
    """Mean loss and accuracy over ``loader``. Deck A always wins, so the target is 1."""
    was_training = model.training
    model.eval()
    total_loss, total_correct, total = 0.0, 0, 0
    for cards, lvls in loader:
        deck_a, deck_b, lvls_a, lvls_b = _split_batch(cards, lvls, device)
        with _autocast(device):
            logits = model(deck_a, deck_b, lvls_a, lvls_b).squeeze(-1)
        logits = logits.float()
        total_loss += loss_fn(logits, torch.ones_like(logits)).item() * logits.numel()
        total_correct += (logits > 0).sum().item()
        total += logits.numel()
    model.train(was_training)
    return total_loss / max(1, total), total_correct / max(1, total)


def train(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    cfg: TrainingConfig,
    device: torch.device,
    model_cfg: ModelConfig | None = None,
) -> nn.Module:
    """Train ``model`` in place for ``cfg.n_epochs`` epochs and return it."""
    model = model.to(device)
    model.train()

    login_to_wandb()
    run = wandb.init(
        project=get_wandb_project_name(),
        entity=get_wandb_entity(),
        config={
            **(model_cfg.model_dump() if model_cfg is not None else {}),
            **cfg.model_dump(),
        },
    )

    n_params = sum(p.numel() for p in model.parameters())
    n_trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    run.summary["n_params"] = n_params
    run.summary["n_trainable_params"] = n_trainable_params
    logger.info("model parameter count: %d (%d trainable)", n_params, n_trainable_params)

    accum = cfg.grad_accumulation_steps
    optim_steps_per_epoch = math.ceil(len(train_loader) / accum)
    total_steps = optim_steps_per_epoch * cfg.n_epochs
    warmup_steps = int(cfg.warmup_ratio * total_steps)

    optimizer = _build_optimizer(model, cfg.lr, cfg.weight_decay, device)
    scheduler = _warmup_cosine_scheduler(optimizer, warmup_steps, total_steps, cfg.min_lr_ratio)
    loss_fn = nn.BCEWithLogitsLoss()

    logger.info(
        "training on %s | %d epochs | %d optimizer steps (%d warmup) | effective batch %d",
        device,
        cfg.n_epochs,
        total_steps,
        warmup_steps,
        cfg.batch_size * accum,
    )

    global_step = 0
    for epoch in range(1, cfg.n_epochs + 1):
        running_loss, seen = 0.0, 0
        optimizer.zero_grad(set_to_none=True)

        for i, (cards, lvls) in enumerate(train_loader):
            deck_a, deck_b, lvls_a, lvls_b = _split_batch(cards, lvls, device)
            with _autocast(device):
                logits = model(deck_a, deck_b, lvls_a, lvls_b).squeeze(-1)
            logits = logits.float()
            loss = loss_fn(logits, torch.ones_like(logits)) / accum
            loss.backward()

            running_loss += loss.item() * accum
            seen += 1

            is_step_boundary = (i + 1) % accum == 0 or (i + 1) == len(train_loader)
            if not is_step_boundary:
                continue

            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.max_grad_norm)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad(set_to_none=True)
            global_step += 1

            if global_step % LOG_EVERY_N_STEPS == 0:
                run.log(
                    {
                        "epoch": epoch,
                        "train/loss": running_loss / seen,
                        "train/lr": scheduler.get_last_lr()[0],
                    },
                    step=global_step,
                )

        val_loss, val_acc = evaluate(model, val_loader, loss_fn, device)
        run.log(
            {
                "epoch": epoch,
                "train/epoch_loss": running_loss / max(1, seen),
                "val/loss": val_loss,
                "val/accuracy": val_acc,
            },
            step=global_step,
        )

    run.finish()
    return model
