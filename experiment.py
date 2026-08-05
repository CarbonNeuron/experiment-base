"""Build and run one experiment from structured configuration."""

from __future__ import annotations

import torch

from config import (
    ChainedHydraConfig,
    CompoundQConfig,
    ExperimentConfig,
    HydraConfig,
    TournamentHydraConfig,
)
from data import build_dataloaders
from model import (
    ChainedHydraTransformer,
    CompoundQTransformer,
    GenericTransformer,
    HydraTransformer,
    TournamentHydraTransformer,
)
from trainer import Trainer, resolve_device


def run_experiment(config: ExperimentConfig) -> None:
    """Compose model, data, and trainer without embedding CLI concerns."""
    torch.manual_seed(config.training.seed)
    device = resolve_device(config.runtime.device)
    if isinstance(config.model, TournamentHydraConfig):
        model = TournamentHydraTransformer(
            config.model, config.embed_path
        ).to(device)
    elif isinstance(config.model, ChainedHydraConfig):
        model = ChainedHydraTransformer(
            config.model, config.embed_path
        ).to(device)
    elif isinstance(config.model, HydraConfig):
        model = HydraTransformer(config.model, config.embed_path).to(device)
    elif isinstance(config.model, CompoundQConfig):
        model = CompoundQTransformer(config.model, config.embed_path).to(device)
    else:
        model = GenericTransformer(config.model, config.embed_path).to(device)
    trainable = model.num_parameters(trainable_only=True)
    print(f"device={device} dtype={config.runtime.dtype}")
    print(
        f"vocab={model.vocab_size:,} parameters={model.num_parameters():,} "
        f"total, {trainable:,} trainable"
    )

    train_loader, val_loader = build_dataloaders(
        config.data,
        seq_len=config.model.max_seq_len,
        device=device,
    )
    trainer = Trainer(
        model,
        train_loader,
        val_loader,
        config.model,
        config.training,
        config.runtime,
        device,
    )
    trainer.fit()
