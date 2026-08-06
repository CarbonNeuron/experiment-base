"""Build and run one experiment from structured configuration."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

from config import FFN_TYPES, ExperimentConfig
from data import build_dataloaders
from experiments import (
    DEFAULT_REGISTRY,
    ModelRegistry,
    build_model,
    objective_for,
)
from training import PrettyLogger, Trainer, resolve_device


def _parse_overrides() -> argparse.Namespace:
    """Parse optional CLI overrides without interfering with scripts."""
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--device", type=str, default=None,
                        help="Override device (e.g. cuda:0, cuda:1, cpu)")
    parser.add_argument("--dtype", type=str, default=None,
                        help="Override precision (fp32, bf16, fp16)")
    parser.add_argument("--compile", action="store_true", default=None,
                        help="Enable torch.compile")
    parser.add_argument("--no-compile", dest="compile", action="store_false",
                        help="Disable torch.compile")
    parser.add_argument(
        "--ffn-type",
        choices=FFN_TYPES,
        default=None,
        help="Override FFN type for architectures that support selection",
    )
    parser.add_argument(
        "--n-quats",
        type=int,
        default=None,
        help="Override QuatSpin channels for supported architectures",
    )
    args, _ = parser.parse_known_args()
    return args


def run_experiment(
    config: ExperimentConfig,
    *,
    registry: ModelRegistry = DEFAULT_REGISTRY,
) -> Path:
    """Compose and run a registered model without embedding CLI concerns.

    Supports runtime and FFN CLI overrides without per-script argparse
    boilerplate. FFN overrides require matching fields on the model config.
    """
    overrides = _parse_overrides()
    if overrides.device is not None:
        config.runtime.device = overrides.device
    if overrides.dtype is not None:
        config.runtime.dtype = overrides.dtype
    if overrides.compile is not None:
        config.runtime.compile = overrides.compile
    if overrides.ffn_type is not None:
        if not hasattr(config.model, "ffn_type"):
            raise ValueError(
                f"{type(config.model).__name__} does not support FFN selection"
            )
        config.model.ffn_type = overrides.ffn_type
    if overrides.n_quats is not None:
        if overrides.n_quats <= 0:
            raise ValueError("n_quats must be positive")
        if not hasattr(config.model, "n_quats"):
            raise ValueError(
                f"{type(config.model).__name__} does not support n_quats"
            )
        config.model.n_quats = overrides.n_quats

    logger = PrettyLogger()
    torch.manual_seed(config.training.seed)
    device = resolve_device(config.runtime.device)
    model = build_model(
        config.model, config.embed_path, registry=registry
    ).to(device)
    trainable = model.num_parameters(trainable_only=True)
    logger.experiment_header(
        device,
        config.runtime.dtype,
        model.vocab_size,
        model.num_parameters(),
        trainable,
        config.model,
    )

    train_loader, val_loader = build_dataloaders(
        config.data,
        seq_len=config.model.max_seq_len,
        device=device,
        logger=logger,
    )
    trainer = Trainer(
        model,
        train_loader,
        val_loader,
        config.model,
        config.training,
        config.runtime,
        device,
        objective=objective_for(config.model, registry=registry),
        logger=logger,
    )
    return trainer.fit()
