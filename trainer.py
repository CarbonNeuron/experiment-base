"""Optimization, evaluation, progress reporting, and checkpointing."""

from __future__ import annotations

import math
import time
from dataclasses import asdict
from pathlib import Path

import torch
from torch import Tensor, nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from config import (
    GrowingWidthConfig,
    RuntimeConfig,
    TrainingConfig,
    TransformerConfig,
)
from model import GenericTransformer, GrowingWidthTransformer


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

        if runtime_config.resume is not None:
            self._resume(runtime_config.resume)
        if runtime_config.compile:
            self._enable_compile()
        runtime_config.checkpoint_dir.mkdir(parents=True, exist_ok=True)

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
                    group_start = (
                        batch_index // self.config.grad_accum_steps
                    ) * self.config.grad_accum_steps
                    micro_batches = min(
                        self.config.grad_accum_steps,
                        batches_in_epoch - group_start,
                    )
                    chunk = chunk.to(self.device, non_blocking=True)
                    with self._autocast():
                        _, loss = self.model(
                            chunk[:, :-1],
                            chunk[:, 1:],
                            loss_chunk_size=self.config.ce_chunk_size,
                            loss_backend=self.config.ce_backend,
                            loss_negative_samples=(
                                self.config.ce_negative_samples
                            ),
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
                    now = time.monotonic()
                    if (
                        now - last_metrics_at >= PROGRESS_REFRESH_SECONDS
                        or self.step >= self.total_steps
                    ):
                        progress.set_postfix(
                            epoch=f"{epoch + 1}/{self.config.epochs}",
                            loss=f"{loss.item():.4f}",
                            lr=f"{self.scheduler.get_last_lr()[0]:.2e}",
                            refresh=False,
                        )
                        last_metrics_at = now
                    progress.update(1)

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
