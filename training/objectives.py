"""Adapters between model forward APIs and the shared training loop."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol

from torch import Tensor, nn
from torch.nn import functional as F

from config import TrainingConfig


class TrainingObjective(Protocol):
    """Compute next-token loss without coupling the trainer to a model class."""

    supports_hard_negatives: bool

    def loss(
        self,
        model: nn.Module,
        batch: Tensor,
        config: TrainingConfig,
        *,
        evaluating: bool,
        hard_negative_trainer: Any | None = None,
        hard_loss_weight: float = 0.0,
    ) -> Tensor: ...


@dataclass(frozen=True)
class ModelProvidedLoss:
    """Use the optimized loss API exposed by the SVD transformer models."""

    supports_hard_negatives: bool = True

    def loss(
        self,
        model: nn.Module,
        batch: Tensor,
        config: TrainingConfig,
        *,
        evaluating: bool,
        hard_negative_trainer: Any | None = None,
        hard_loss_weight: float = 0.0,
    ) -> Tensor:
        kwargs: dict[str, Any] = {
            "loss_chunk_size": config.ce_chunk_size,
            "loss_backend": "tiled" if evaluating else config.ce_backend,
        }
        if not evaluating:
            kwargs["loss_negative_samples"] = config.ce_negative_samples
            if hard_negative_trainer is not None:
                kwargs["hard_negative_trainer"] = hard_negative_trainer
                kwargs["hard_loss_weight"] = hard_loss_weight
        _, loss = model(batch[:, :-1], batch[:, 1:], **kwargs)
        if loss is None:
            raise RuntimeError("model did not return a training loss")
        return loss


@dataclass(frozen=True)
class LogitsCrossEntropy:
    """Apply external cross entropy to models whose forward returns logits."""

    supports_hard_negatives: bool = False

    def loss(
        self,
        model: nn.Module,
        batch: Tensor,
        config: TrainingConfig,
        *,
        evaluating: bool,
        hard_negative_trainer: Any | None = None,
        hard_loss_weight: float = 0.0,
    ) -> Tensor:
        del config, evaluating, hard_negative_trainer, hard_loss_weight
        logits = model(batch[:, :-1])
        return F.cross_entropy(
            logits.reshape(-1, logits.size(-1)),
            batch[:, 1:].reshape(-1),
            ignore_index=-1,
        )
