"""Optimization, evaluation, progress reporting, and checkpointing."""

from __future__ import annotations

import math
import time
from dataclasses import asdict
from pathlib import Path

import torch
import torch.distributed as dist
from torch import Tensor, nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from config import (
    GrowingWidthConfig,
    HardNegativeRetrievalConfig,
    RuntimeConfig,
    TrainingConfig,
    TransformerConfig,
)
from model import GenericTransformer, GrowingWidthTransformer
from output_retrieval import HardNegativeTrainer, build_or_load_index


PROGRESS_REFRESH_SECONDS = 0.2


def resolve_device(requested: str) -> torch.device:
    """Resolve ``auto`` without leaking device policy into other modules."""
    if requested == "auto":
        requested = "cuda" if torch.cuda.is_available() else "cpu"
    return torch.device(requested)


def resolve_precision(
    dtype_name: str, device: torch.device
) -> tuple[torch.dtype, bool]:
    dtype = {
        "fp32": torch.float32,
        "bf16": torch.bfloat16,
        "fp16": torch.float16,
    }[dtype_name]
    amp_enabled = dtype_name != "fp32" and device.type in {"cuda", "cpu"}
    if device.type == "cpu" and dtype == torch.float16:
        amp_enabled = False
    return dtype, amp_enabled


def make_scheduler(
    optimizer: AdamW, warmup_steps: int, total_steps: int
) -> LambdaLR:
    """Linear warmup followed by cosine decay."""

    def multiplier(step: int) -> float:
        if step < warmup_steps:
            return (step + 1) / max(1, warmup_steps)
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        progress = min(max(progress, 0.0), 1.0)
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    return LambdaLR(optimizer, multiplier)


class Trainer:
    """Own the mutable training process for an already-built model and data."""

    def __init__(
        self,
        model: GenericTransformer | GrowingWidthTransformer,
        train_loader: DataLoader[Tensor],
        val_loader: DataLoader[Tensor],
        model_config: TransformerConfig | GrowingWidthConfig,
        training_config: TrainingConfig,
        runtime_config: RuntimeConfig,
        device: torch.device,
    ) -> None:
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.model_config = model_config
        self.config = training_config
        self.runtime = runtime_config
        self.device = device
        self.dtype, self.amp_enabled = resolve_precision(
            runtime_config.dtype, device
        )
        runtime_config.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        self.optimizer = AdamW(
            model.parameters(),
            lr=training_config.lr,
            weight_decay=training_config.weight_decay,
            betas=(0.9, 0.95),
        )
        updates_per_epoch = math.ceil(
            len(train_loader) / training_config.grad_accum_steps
        )
        self.total_steps = (
            training_config.max_steps
            or updates_per_epoch * training_config.epochs
        )
        self.scheduler = make_scheduler(
            self.optimizer,
            training_config.warmup_steps,
            self.total_steps,
        )
        self.step = 0
        self.start_epoch = 0
        self.hard_negative_trainer: HardNegativeTrainer | None = None
        self.hard_negative_index_fingerprint: str | None = None
        self.hard_negative_index_path: Path | None = None
        self._resume_index_fingerprint: str | None = None

        if runtime_config.resume is not None:
            self._resume(runtime_config.resume)
        if training_config.hard_negative_retrieval.enabled:
            self._enable_hard_negatives()
        if runtime_config.compile:
            self._enable_compile()

    def _enable_hard_negatives(self) -> None:
        config = self.config.hard_negative_retrieval
        if self.config.ce_backend != "sampled" or self.config.ce_chunk_size <= 0:
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
        self.hard_negative_trainer = HardNegativeTrainer(
            index, self.model.embeddings.directions, config
        )
        if (
            self._resume_index_fingerprint is not None
            and self._resume_index_fingerprint != fingerprint
        ):
            raise ValueError(
                "checkpoint hard-negative index fingerprint does not match "
                "the fixed output directions"
            )
        config.index.rebuild = False
        self.hard_negative_index_fingerprint = fingerprint
        self.hard_negative_index_path = path
        location = str(path) if path is not None else "memory only"
        print(
            f"hard-negative retrieval enabled: backend={config.backend} "
            f"k={config.hard_k} index={location} fingerprint={fingerprint[:12]}"
        )

    def _hard_loss_weight(self) -> float:
        config = self.config.hard_negative_retrieval
        if self.hard_negative_trainer is None:
            return 0.0
        if config.warmup_steps == 0:
            return config.loss_weight
        return config.loss_weight * min(1.0, self.step / config.warmup_steps)

    def _autocast(self):
        return torch.autocast(
            device_type=self.device.type,
            dtype=self.dtype,
            enabled=self.amp_enabled,
        )

    def _resume(self, path: Path) -> None:
        checkpoint = torch.load(path, map_location=self.device, weights_only=False)
        self.model.load_state_dict(checkpoint["model"])
        self.optimizer.load_state_dict(checkpoint["optimizer"])
        self.scheduler.load_state_dict(checkpoint["scheduler"])
        self.step = int(checkpoint["step"])
        self.start_epoch = int(checkpoint["epoch"])
        saved_training = checkpoint.get("training_config", {})
        saved_hard_config = saved_training.get("hard_negative_retrieval")
        if saved_hard_config is not None:
            self.config.hard_negative_retrieval = HardNegativeRetrievalConfig(
                **saved_hard_config
            )
        self._resume_index_fingerprint = checkpoint.get(
            "hard_negative_index_fingerprint"
        )
        print(f"resumed {path} at step {self.step}")

    def _enable_compile(self) -> None:
        try:
            backend = self.model.compile_encoder(
                mode=self.runtime.compile_mode,
                backend=self.runtime.compile_backend,
            )
            print(
                "torch.compile enabled for encoder "
                f"(backend={backend}, mode={self.runtime.compile_mode})"
            )
            if self.runtime.compile_backend == "auto" and backend == "aot_eager":
                print(
                    "Triton is unavailable for this accelerator; selected "
                    "aot_eager instead of Inductor."
                )
        except Exception as error:
            print(f"torch.compile unavailable; using eager encoder: {error}")

    @torch.no_grad()
    def evaluate(self) -> float:
        """Evaluate without owning dataset construction or experiment policy."""
        self.model.eval()
        total_loss = 0.0
        batches = 0
        max_batches = self.config.eval_batches
        progress_total = (
            min(len(self.val_loader), max_batches)
            if max_batches > 0
            else len(self.val_loader)
        )
        for chunk in tqdm(
            self.val_loader,
            total=progress_total,
            desc="Validating",
            unit="batch",
            leave=False,
        ):
            chunk = chunk.to(self.device, non_blocking=True)
            with self._autocast():
                _, loss = self.model(
                    chunk[:, :-1],
                    chunk[:, 1:],
                    loss_chunk_size=self.config.ce_chunk_size,
                    loss_backend="tiled",
                )
            assert loss is not None
            total_loss += loss.float().item()
            batches += 1
            if max_batches > 0 and batches >= max_batches:
                break
        self.model.train()
        return total_loss / max(1, batches)

    def save_checkpoint(self, filename: str, epoch: int) -> Path:
        """Serialize training state without compiler wrappers or SVD directions."""
        path = self.runtime.checkpoint_dir / filename
        torch.save(
            {
                "model": self.model.state_dict(),
                "optimizer": self.optimizer.state_dict(),
                "scheduler": self.scheduler.state_dict(),
                "config": asdict(self.model_config),
                "training_config": asdict(self.config),
                "hard_negative_index_fingerprint": (
                    self.hard_negative_index_fingerprint
                ),
                "hard_negative_index_path": (
                    str(self.hard_negative_index_path)
                    if self.hard_negative_index_path is not None
                    else None
                ),
                "step": self.step,
                "epoch": epoch,
            },
            path,
        )
        return path

    def fit(self) -> Path:
        """Run training and return the final checkpoint path."""
        self.optimizer.zero_grad(set_to_none=True)
        self.model.train()
        stop = False
        last_epoch = self.start_epoch
        last_metrics_at = 0.0

        with tqdm(
            total=self.total_steps,
            initial=min(self.step, self.total_steps),
            desc="Training",
            unit="step",
            dynamic_ncols=True,
            mininterval=PROGRESS_REFRESH_SECONDS,
        ) as progress:
            for epoch in range(self.start_epoch, self.config.epochs):
                last_epoch = epoch
                batches_in_epoch = len(self.train_loader)
                for batch_index, chunk in enumerate(self.train_loader):
                    batch_start = time.perf_counter()
                    group_start = (
                        batch_index // self.config.grad_accum_steps
                    ) * self.config.grad_accum_steps
                    micro_batches = min(
                        self.config.grad_accum_steps,
                        batches_in_epoch - group_start,
                    )
                    chunk = chunk.to(self.device, non_blocking=True)
                    with self._autocast():
                        model_kwargs = {
                            "loss_chunk_size": self.config.ce_chunk_size,
                            "loss_backend": self.config.ce_backend,
                            "loss_negative_samples": self.config.ce_negative_samples,
                        }
                        if self.hard_negative_trainer is not None:
                            model_kwargs.update(
                                hard_negative_trainer=self.hard_negative_trainer,
                                hard_loss_weight=self._hard_loss_weight(),
                            )
                        _, loss = self.model(
                            chunk[:, :-1], chunk[:, 1:], **model_kwargs
                        )
                        assert loss is not None
                        scaled_loss = loss / micro_batches
                    scaled_loss.backward()

                    group_finished = (
                        (batch_index + 1) % self.config.grad_accum_steps == 0
                        or batch_index + 1 == batches_in_epoch
                    )
                    if not group_finished:
                        continue

                    nn.utils.clip_grad_norm_(
                        self.model.parameters(), self.config.max_grad_norm
                    )
                    self.optimizer.step()
                    self.scheduler.step()
                    self.optimizer.zero_grad(set_to_none=True)
                    self.step += 1
                    step_seconds = time.perf_counter() - batch_start
                    now = time.monotonic()
                    if (
                        now - last_metrics_at >= PROGRESS_REFRESH_SECONDS
                        or self.step >= self.total_steps
                    ):
                        postfix = dict(
                            epoch=f"{epoch + 1}/{self.config.epochs}",
                            loss=f"{loss.item():.4f}",
                            lr=f"{self.scheduler.get_last_lr()[0]:.2e}",
                        )
                        hard_metrics = self.model.last_hard_negative_metrics
                        if hard_metrics is not None:
                            postfix["hard"] = (
                                f"{hard_metrics['hard_loss'].item():.4f}"
                            )
                            postfix["margin"] = (
                                f"{hard_metrics['mean_hard_margin'].item():.3f}"
                            )
                        progress.set_postfix(**postfix, refresh=False)
                        last_metrics_at = now
                    progress.update(1)

                    hard_config = self.config.hard_negative_retrieval
                    hard_metrics = self.model.last_hard_negative_metrics
                    if (
                        self.hard_negative_trainer is not None
                        and hard_config.diagnostics.exact_recall_interval > 0
                        and self.step
                        % hard_config.diagnostics.exact_recall_interval
                        == 0
                    ):
                        recall = self.hard_negative_trainer.exact_recall()
                        if recall is not None:
                            progress.write(
                                f"step {self.step}: hard-negative exact "
                                f"recall@{hard_config.hard_k}={recall.item():.4f}"
                            )
                    if (
                        hard_metrics is not None
                        and hard_config.diagnostics.log_interval > 0
                        and self.step % hard_config.diagnostics.log_interval == 0
                    ):
                        rotation = (
                            self.model.embeddings.rotation.matrix.detach().float()
                        )
                        identity = torch.eye(
                            rotation.size(0), device=rotation.device
                        )
                        orthogonality_error = (
                            rotation.T @ rotation - identity
                        ).norm().item()
                        peak_memory = (
                            torch.cuda.max_memory_allocated(self.device)
                            if self.device.type == "cuda"
                            else 0
                        )
                        retrieval_ms = 1000 * hard_metrics[
                            "retrieval_seconds"
                        ].item()
                        candidate_score_ms = 1000 * hard_metrics[
                            "candidate_score_seconds"
                        ].item()
                        hard_loss_ms = 1000 * hard_metrics[
                            "hard_loss_seconds"
                        ].item()
                        progress.write(
                            f"step {self.step}: "
                            f"sampled={hard_metrics['sampled_loss'].item():.4f} "
                            f"hard={hard_metrics['hard_loss'].item():.4f} "
                            f"weighted_hard="
                            f"{hard_metrics['weighted_hard_loss'].item():.4f} "
                            f"total={hard_metrics['total_loss'].item():.4f} "
                            f"weight={hard_metrics['hard_loss_weight'].item():.4f} "
                            f"positive_logit="
                            f"{hard_metrics['mean_positive_logit'].item():.3f} "
                            f"max_hard_logit="
                            f"{hard_metrics['mean_max_hard_logit'].item():.3f} "
                            f"margin={hard_metrics['mean_hard_margin'].item():.3f} "
                            f"hard_error="
                            f"{hard_metrics['hard_error_rate'].item():.3f} "
                            f"valid_hard="
                            f"{hard_metrics['mean_valid_hard_negatives'].item():.1f} "
                            f"retrieval_ms={retrieval_ms:.2f} "
                            f"score_ms={candidate_score_ms:.2f} "
                            f"hard_loss_ms={hard_loss_ms:.2f} "
                            f"step_ms={1000 * step_seconds:.2f} "
                            f"peak_bytes={peak_memory} "
                            f"scale={self.model.embeddings.magnitude.item():.4f} "
                            f"rotation_orth_error={orthogonality_error:.2e}"
                        )

                    if (
                        self.config.eval_every > 0
                        and self.step % self.config.eval_every == 0
                    ):
                        val_loss = self.evaluate()
                        progress.write(
                            f"step {self.step}: validation loss={val_loss:.4f} "
                            f"ppl={math.exp(min(val_loss, 20)):.2f}"
                        )

                    if (
                        self.config.save_every > 0
                        and self.step % self.config.save_every == 0
                    ):
                        self.save_checkpoint(
                            f"step_{self.step:08d}.pt", epoch
                        )

                    if (
                        self.config.max_steps > 0
                        and self.step >= self.config.max_steps
                    ):
                        stop = True
                        break
                if stop:
                    break

        path = self.save_checkpoint("latest.pt", last_epoch)
        print(f"finished at step {self.step}; wrote {path}")
        return path
