"""Optional hard-negative retrieval lifecycle and diagnostics."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
import torch.distributed as dist
from torch import nn

from config import RuntimeConfig, TrainingConfig
from output_retrieval import HardNegativeTrainer, build_or_load_index


class HardNegativeRuntime:
    """Keep retrieval-specific state out of the generic optimizer loop."""

    def __init__(
        self,
        model: nn.Module,
        training: TrainingConfig,
        runtime: RuntimeConfig,
        resume_fingerprint: str | None = None,
    ) -> None:
        self.model = model
        self.training = training
        self.runtime = runtime
        self.resume_fingerprint = resume_fingerprint
        self.trainer: HardNegativeTrainer | None = None
        self.index_fingerprint: str | None = None
        self.index_path: Path | None = None

    def enable(self) -> None:
        config = self.training.hard_negative_retrieval
        if self.training.ce_backend != "sampled" or self.training.ce_chunk_size <= 0:
            raise ValueError(
                "hard-negative retrieval requires the bounded sampled training "
                "loss (--ce-backend sampled and --ce-chunk-size > 0)"
            )
        if config.index.path is None:
            config.index.path = self.runtime.checkpoint_dir / "output_indexes"

        distributed = dist.is_available() and dist.is_initialized()
        rank = dist.get_rank() if distributed else 0
        if distributed and rank != 0:
            dist.barrier()
            rebuild = config.index.rebuild
            config.index.rebuild = False
            try:
                index, fingerprint, path = build_or_load_index(
                    self.model.embeddings.directions, config
                )
            finally:
                config.index.rebuild = rebuild
        else:
            index, fingerprint, path = build_or_load_index(
                self.model.embeddings.directions, config
            )
            if distributed:
                dist.barrier()

        self.trainer = HardNegativeTrainer(
            index, self.model.embeddings.directions, config
        )
        if (
            self.resume_fingerprint is not None
            and self.resume_fingerprint != fingerprint
        ):
            raise ValueError(
                "checkpoint hard-negative index fingerprint does not match "
                "the fixed output directions"
            )
        config.index.rebuild = False
        self.index_fingerprint = fingerprint
        self.index_path = path
        location = str(path) if path is not None else "memory only"
        print(
            f"hard-negative retrieval enabled: backend={config.backend} "
            f"k={config.hard_k} index={location} fingerprint={fingerprint[:12]}"
        )

    def loss_weight(self, step: int) -> float:
        config = self.training.hard_negative_retrieval
        if self.trainer is None:
            return 0.0
        if config.warmup_steps == 0:
            return config.loss_weight
        return config.loss_weight * min(1.0, step / config.warmup_steps)

    def metrics(self) -> dict[str, torch.Tensor] | None:
        return getattr(self.model, "last_hard_negative_metrics", None)

    def maybe_log(
        self,
        progress: Any,
        step: int,
        step_seconds: float,
        device: torch.device,
    ) -> None:
        config = self.training.hard_negative_retrieval
        metrics = self.metrics()
        if (
            self.trainer is not None
            and config.diagnostics.exact_recall_interval > 0
            and step % config.diagnostics.exact_recall_interval == 0
        ):
            recall = self.trainer.exact_recall()
            if recall is not None:
                progress.write(
                    f"step {step}: hard-negative exact "
                    f"recall@{config.hard_k}={recall.item():.4f}"
                )
        if (
            metrics is None
            or config.diagnostics.log_interval <= 0
            or step % config.diagnostics.log_interval != 0
        ):
            return

        rotation = self.model.embeddings.rotation.matrix.detach().float()
        identity = torch.eye(rotation.size(0), device=rotation.device)
        orthogonality_error = torch.linalg.vector_norm(
            rotation.T @ rotation - identity
        ).item()
        peak_memory = (
            torch.cuda.max_memory_allocated(device) if device.type == "cuda" else 0
        )
        progress.write(
            f"step {step}: "
            f"sampled={metrics['sampled_loss'].item():.4f} "
            f"hard={metrics['hard_loss'].item():.4f} "
            f"weighted_hard={metrics['weighted_hard_loss'].item():.4f} "
            f"total={metrics['total_loss'].item():.4f} "
            f"weight={metrics['hard_loss_weight'].item():.4f} "
            f"positive_logit={metrics['mean_positive_logit'].item():.3f} "
            f"max_hard_logit={metrics['mean_max_hard_logit'].item():.3f} "
            f"margin={metrics['mean_hard_margin'].item():.3f} "
            f"hard_error={metrics['hard_error_rate'].item():.3f} "
            f"valid_hard={metrics['mean_valid_hard_negatives'].item():.1f} "
            f"retrieval_ms={1000 * metrics['retrieval_seconds'].item():.2f} "
            f"score_ms={1000 * metrics['candidate_score_seconds'].item():.2f} "
            f"hard_loss_ms={1000 * metrics['hard_loss_seconds'].item():.2f} "
            f"step_ms={1000 * step_seconds:.2f} "
            f"peak_bytes={peak_memory} "
            f"scale={self.model.embeddings.magnitude.item():.4f} "
            f"rotation_orth_error={orthogonality_error:.2e}"
        )
