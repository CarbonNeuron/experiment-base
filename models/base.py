"""Shared tied-embedding language-model behavior."""

from __future__ import annotations

import importlib.util
import warnings
from pathlib import Path
from typing import TYPE_CHECKING, Any

import torch
from torch import Tensor, nn
from torch.nn import functional as F
from svd_embeds import OpenAIEmbedding

from config import COMPILE_BACKENDS

if TYPE_CHECKING:
    from output_retrieval.hard_negative_loss import HardNegativeTrainer


def resolve_compile_backend(requested: str, device_type: str) -> str:
    """Choose a usable compiler backend before the first training batch."""
    if requested not in COMPILE_BACKENDS:
        choices = ", ".join(COMPILE_BACKENDS)
        raise ValueError(f"unknown compile backend {requested!r}; choose: {choices}")
    if requested != "auto":
        return requested
    if device_type == "cuda":
        try:
            has_triton = importlib.util.find_spec("triton") is not None
        except (ImportError, ValueError):
            has_triton = False
        if not has_triton:
            return "aot_eager"
    if device_type not in {"cpu", "cuda"}:
        return "aot_eager"
    return "inductor"


class SVDLanguageModel(nn.Module):
    """Common runtime and loss contract for SVD-embedding architectures."""

    config: Any
    embeddings: OpenAIEmbedding

    def _finish_initialization(
        self,
        width: int,
        max_seq_len: int,
        embed_path: str | Path | None,
    ) -> None:
        self.apply(self._init_weights)
        embedding_kwargs = (
            {"embedding_path": embed_path} if embed_path is not None else {}
        )
        self.embeddings = OpenAIEmbedding(
            width,
            max_seq_len=max_seq_len,
            **embedding_kwargs,
        )
        self._compiled_encoder = None
        self._compile_backend: str | None = None
        self.last_hard_negative_metrics: dict[str, Tensor] | None = None

    @staticmethod
    def _init_weights(module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, nn.LayerNorm):
            nn.init.ones_(module.weight)
            nn.init.zeros_(module.bias)

    def effective_embeddings(self) -> Tensor:
        return self.embeddings.weight

    @property
    def vocab_size(self) -> int:
        return self.embeddings.num_embeddings

    def _encode_hidden(self, input_ids: Tensor) -> Tensor:
        raise NotImplementedError

    def encode(self, input_ids: Tensor) -> Tensor:
        if input_ids.ndim != 2:
            raise ValueError("input_ids must have shape [batch, sequence]")
        seq_len = input_ids.size(1)
        if seq_len > self.config.max_seq_len:
            raise ValueError(
                f"sequence length {seq_len} exceeds max_seq_len="
                f"{self.config.max_seq_len}"
            )
        if self._compiled_encoder is None:
            return self._encode_hidden(input_ids)
        try:
            return self._compiled_encoder(input_ids)
        except Exception as error:
            summary = str(error).splitlines()[0] or type(error).__name__
            warnings.warn(
                "compiled encoder failed; falling back to eager mode: "
                f"{type(error).__name__}: {summary}",
                RuntimeWarning,
                stacklevel=2,
            )
            self.disable_compile()
            return self._encode_hidden(input_ids)

    def compile_encoder(
        self,
        *,
        mode: str = "default",
        dynamic: bool = False,
        backend: str = "auto",
    ) -> str:
        selected_backend = resolve_compile_backend(
            backend, self.embeddings.directions.device.type
        )
        kwargs: dict[str, Any] = {
            "backend": selected_backend,
            "dynamic": dynamic,
        }
        if selected_backend == "inductor":
            kwargs["mode"] = mode
        self._compiled_encoder = torch.compile(self._encode_hidden, **kwargs)
        self._compile_backend = selected_backend
        return selected_backend

    def disable_compile(self) -> None:
        self._compiled_encoder = None
        self._compile_backend = None

    def logits(self, hidden: Tensor) -> Tensor:
        return self.embeddings.project(hidden)

    def transform_hidden_to_fixed_space(self, hidden: Tensor) -> Tensor:
        return hidden @ self.embeddings.rotation.matrix

    def _combine_hard_negative_loss(
        self,
        hidden: Tensor,
        targets: Tensor,
        sampled_loss: Tensor,
        trainer: "HardNegativeTrainer | None",
        hard_loss_weight: float,
    ) -> Tensor:
        if trainer is None:
            self.last_hard_negative_metrics = None
            return sampled_loss
        hard_loss, metrics = trainer.compute(
            hidden,
            targets,
            self.embeddings.rotation.matrix,
            self.embeddings.magnitude,
        )
        total = sampled_loss + hard_loss_weight * hard_loss
        metrics["sampled_loss"] = sampled_loss.detach()
        metrics["weighted_hard_loss"] = (hard_loss_weight * hard_loss).detach()
        metrics["hard_loss_weight"] = torch.tensor(
            hard_loss_weight, device=sampled_loss.device
        )
        metrics["total_loss"] = total.detach()
        self.last_hard_negative_metrics = metrics
        return total

    def forward(
        self,
        input_ids: Tensor,
        targets: Tensor | None = None,
        *,
        loss_chunk_size: int = 0,
        loss_backend: str = "tiled",
        loss_negative_samples: int = 4096,
        hard_negative_trainer: "HardNegativeTrainer | None" = None,
        hard_loss_weight: float = 0.0,
    ) -> tuple[Tensor | None, Tensor | None]:
        hidden = self.encode(input_ids)
        if targets is not None:
            if targets.shape != input_ids.shape:
                raise ValueError("targets must have the same shape as input_ids")
            if loss_chunk_size > 0:
                sampled_loss = self.embeddings.cross_entropy(
                    hidden,
                    targets,
                    chunk_size=loss_chunk_size,
                    backend=loss_backend,
                    num_negative_samples=loss_negative_samples,
                )
                loss = self._combine_hard_negative_loss(
                    hidden,
                    targets,
                    sampled_loss,
                    hard_negative_trainer,
                    hard_loss_weight,
                )
                return None, loss

        logits = self.embeddings.project(hidden)
        loss = None
        if targets is not None:
            loss = F.cross_entropy(
                logits.reshape(-1, logits.size(-1)),
                targets.reshape(-1),
                ignore_index=-1,
            )
        return logits, loss

    def num_parameters(self, trainable_only: bool = False) -> int:
        parameters = sum(
            parameter.numel()
            for parameter in self.parameters()
            if not trainable_only or parameter.requires_grad
        )
        if not trainable_only:
            parameters += self.embeddings.directions.numel()
        return parameters
