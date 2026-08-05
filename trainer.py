"""Optimization, evaluation, progress reporting, and checkpointing."""

from __future__ import annotations

import math
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any

import torch
from torch import Tensor, nn
from torch.optim import AdamW
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from config import (
    HardNegativeRetrievalConfig,
    RuntimeConfig,
    TrainingConfig,
)
from training.hard_negatives import HardNegativeRuntime
from training.objectives import ModelProvidedLoss, TrainingObjective
from training.runtime import make_scheduler, resolve_device, resolve_precision


PROGRESS_REFRESH_SECONDS = 0.2


class Trainer:
    """Own the mutable training process for an already-built model and data."""

    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader[Tensor],
        val_loader: DataLoader[Tensor],
        model_config: Any,
        training_config: TrainingConfig,
        runtime_config: RuntimeConfig,
        device: torch.device,
        objective: TrainingObjective | None = None,
    ) -> None:
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.model_config = model_config
        self.config = training_config
        self.runtime = runtime_config
        self.device = device
        self.objective = objective or ModelProvidedLoss()
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
        self._resume_index_fingerprint: str | None = None

        if runtime_config.resume is not None:
            self._resume(runtime_config.resume)
        self.hard_negatives = HardNegativeRuntime(
            model,
            training_config,
            runtime_config,
            self._resume_index_fingerprint,
        )
        if (
            training_config.hard_negative_retrieval.enabled
            and not self.objective.supports_hard_negatives
        ):
            raise ValueError(
                "the selected training objective does not support hard-negative "
                "retrieval"
            )
        if training_config.hard_negative_retrieval.enabled:
            self.hard_negatives.enable()
        if runtime_config.compile:
            self._enable_compile()

    @property
    def hard_negative_trainer(self) -> Any | None:
        """Backward-compatible access to the optional retrieval trainer."""
        return self.hard_negatives.trainer

    @property
    def hard_negative_index_fingerprint(self) -> str | None:
        return self.hard_negatives.index_fingerprint

    @property
    def hard_negative_index_path(self) -> Path | None:
        return self.hard_negatives.index_path

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
        compile_encoder = getattr(self.model, "compile_encoder", None)
        if compile_encoder is None:
            print("torch.compile skipped; model has no compile_encoder hook")
            return
        try:
            backend = compile_encoder(
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
                loss = self.objective.loss(
                    self.model,
                    chunk,
                    self.config,
                    evaluating=True,
                )
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
                    self.hard_negatives.index_fingerprint
                ),
                "hard_negative_index_path": (
                    str(self.hard_negatives.index_path)
                    if self.hard_negatives.index_path is not None
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
                        loss = self.objective.loss(
                            self.model,
                            chunk,
                            self.config,
                            evaluating=False,
                            hard_negative_trainer=self.hard_negatives.trainer,
                            hard_loss_weight=self.hard_negatives.loss_weight(
                                self.step
                            ),
                        )
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
                        hard_metrics = self.hard_negatives.metrics()
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

                    self.hard_negatives.maybe_log(
                        progress,
                        self.step,
                        step_seconds,
                        self.device,
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
