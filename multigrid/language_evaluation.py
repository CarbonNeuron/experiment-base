"""Matched-mechanism WikiText evaluation with context-length diagnostics."""

from __future__ import annotations

import csv
import json
import math
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import torch
from torch import Tensor
from torch.optim import AdamW

from config import COMPILE_BACKENDS, COMPILE_MODES, DTYPES, LOSS_BACKENDS
from data import load_wikitext
from training.logger import PrettyLogger
from training.runtime import make_scheduler, resolve_device, resolve_precision

from .mechanisms import (
    MECHANISM_NAMES,
    MechanismLanguageModel,
    SymbolicModelConfig,
    matched_model_configs,
)


PROGRESS_REFRESH_SECONDS = 0.2
POSITION_BUCKETS = ("0-25%", "25-50%", "50-75%", "75-100%")


@dataclass
class LanguageEvaluationConfig:
    """Reproducible matched-mechanism language-model protocol."""

    mechanisms: tuple[str, ...] = MECHANISM_NAMES
    train_sequence_length: int = 512
    eval_sequence_lengths: tuple[int, ...] = (128, 256, 512, 1024, 2048)
    eval_tail_tokens: int = 128
    train_steps: int = 1_000
    batch_size: int = 128
    eval_batch_size: int = 2
    eval_batches: int = 16
    learning_rate: float = 3e-4
    weight_decay: float = 0.1
    warmup_steps: int = 100
    max_grad_norm: float = 1.0
    ce_chunk_size: int = 1024
    ce_backend: str = "sampled"
    ce_negative_samples: int = 4096
    log_every: int = 10
    save_every: int = 250
    seed: int = 42
    device: str = "auto"
    dtype: str = "bf16"
    compile: bool = True
    compile_mode: str = "default"
    compile_backend: str = "auto"
    resume: bool = True
    cache_dir: Path = Path("data_cache")
    output_dir: Path = Path("checkpoints/multigrid-nlp-evaluation-v2")
    embed_path: Path | None = None
    model: SymbolicModelConfig = field(
        default_factory=lambda: SymbolicModelConfig(
            vocab_size=100_277,
            d_model=128,
            n_layers=4,
            d_ff=256,
            n_heads=8,
            max_seq_len=2048,
            dropout=0.0,
            n_memory_slots=64,
            d_memory=64,
            d_key=64,
        )
    )

    def __post_init__(self) -> None:
        if isinstance(self.model, dict):
            self.model = SymbolicModelConfig(**self.model)
        self.mechanisms = tuple(self.mechanisms)
        self.eval_sequence_lengths = tuple(self.eval_sequence_lengths)
        self.cache_dir = Path(self.cache_dir)
        self.output_dir = Path(self.output_dir)
        self.embed_path = (
            Path(self.embed_path) if self.embed_path is not None else None
        )

        unknown = set(self.mechanisms) - set(MECHANISM_NAMES)
        if unknown:
            raise ValueError(f"unknown mechanisms: {sorted(unknown)}")
        if not self.mechanisms:
            raise ValueError("mechanisms cannot be empty")
        if not self.eval_sequence_lengths:
            raise ValueError("eval_sequence_lengths cannot be empty")
        dimensions = (
            self.train_sequence_length,
            self.train_steps,
            self.batch_size,
            self.eval_batch_size,
            self.eval_batches,
            self.eval_tail_tokens,
            self.learning_rate,
            self.max_grad_norm,
            self.ce_chunk_size,
            self.ce_negative_samples,
            self.log_every,
        )
        if any(value <= 0 for value in dimensions):
            raise ValueError("evaluation dimensions and rates must be positive")
        if any(length < 4 for length in self.eval_sequence_lengths):
            raise ValueError(
                "evaluation sequence lengths must be at least four tokens"
            )
        if tuple(sorted(set(self.eval_sequence_lengths))) != (
            self.eval_sequence_lengths
        ):
            raise ValueError(
                "eval_sequence_lengths must be sorted and unique"
            )
        if self.eval_tail_tokens > min(self.eval_sequence_lengths):
            raise ValueError(
                "eval_tail_tokens cannot exceed the shortest evaluation context"
            )
        maximum_length = max(
            self.train_sequence_length,
            *self.eval_sequence_lengths,
        )
        if maximum_length > self.model.max_seq_len:
            raise ValueError(
                f"maximum evaluation length {maximum_length} exceeds "
                f"model.max_seq_len={self.model.max_seq_len}"
            )
        if self.warmup_steps < 0 or self.save_every < 0:
            raise ValueError("warmup_steps and save_every must be non-negative")
        if self.weight_decay < 0:
            raise ValueError("weight_decay must be non-negative")
        if self.dtype not in DTYPES:
            raise ValueError(f"dtype must be one of {DTYPES}")
        if self.ce_backend not in LOSS_BACKENDS:
            raise ValueError(f"ce_backend must be one of {LOSS_BACKENDS}")
        if self.compile_mode not in COMPILE_MODES:
            raise ValueError(f"compile_mode must be one of {COMPILE_MODES}")
        if self.compile_backend not in COMPILE_BACKENDS:
            raise ValueError(
                f"compile_backend must be one of {COMPILE_BACKENDS}"
            )


def _sample_training_batch(
    tokens: Tensor,
    *,
    sequence_length: int,
    batch_size: int,
    seed: int,
    step: int,
) -> Tensor:
    """Return the same random token windows for every mechanism and resume."""
    maximum_start = tokens.numel() - sequence_length - 1
    if maximum_start < 0:
        raise ValueError("training token stream is shorter than one example")
    generator = torch.Generator().manual_seed(seed + step)
    starts = torch.randint(
        maximum_start + 1,
        (batch_size,),
        generator=generator,
    )
    offsets = torch.arange(sequence_length + 1)
    return tokens[starts[:, None] + offsets[None, :]]


def _validation_starts(
    token_count: int,
    *,
    sequence_length: int,
    example_count: int,
    maximum_sequence_length: int | None = None,
) -> Tensor:
    """Spread windows while optionally aligning targets across context lengths."""
    maximum_sequence_length = maximum_sequence_length or sequence_length
    if maximum_sequence_length < sequence_length:
        raise ValueError(
            "maximum_sequence_length cannot be shorter than sequence_length"
        )
    maximum_end = token_count - 1
    if maximum_end < maximum_sequence_length:
        raise ValueError("validation token stream is shorter than one example")
    if example_count == 1:
        ends = torch.tensor([maximum_sequence_length], dtype=torch.long)
    else:
        ends = torch.linspace(
            maximum_sequence_length,
            maximum_end,
            example_count,
            dtype=torch.float64,
        ).round().to(torch.long)
    return ends - sequence_length


def _batch_from_starts(
    tokens: Tensor,
    starts: Tensor,
    sequence_length: int,
) -> Tensor:
    offsets = torch.arange(sequence_length + 1)
    return tokens[starts[:, None] + offsets[None, :]]


def _perplexity(loss: float) -> float:
    return math.exp(min(loss, 80.0))


class LanguageEvaluator:
    """Train matched mechanisms on identical WikiText windows and compare."""

    def __init__(
        self,
        config: LanguageEvaluationConfig,
        logger: PrettyLogger | None = None,
    ) -> None:
        self.config = config
        self.logger = logger or PrettyLogger()
        self.device = resolve_device(config.device)
        if self.device.type == "cuda":
            torch.cuda.set_device(self.device)
        self.dtype, self.amp_enabled = resolve_precision(
            config.dtype,
            self.device,
        )
        self.config.output_dir.mkdir(parents=True, exist_ok=True)
        self.model_configs = matched_model_configs(
            config.model,
            config.mechanisms,
        )

    def _autocast(self):
        return torch.autocast(
            device_type=self.device.type,
            dtype=self.dtype,
            enabled=self.amp_enabled,
        )

    def _checkpoint_path(self, mechanism: str) -> Path:
        return self.config.output_dir / mechanism / "latest.pt"

    def _new_model(self, mechanism: str) -> MechanismLanguageModel:
        torch.manual_seed(self.config.seed)
        if self.device.type == "cuda":
            torch.cuda.manual_seed_all(self.config.seed)
        return MechanismLanguageModel(
            self.model_configs[mechanism],
            self.config.embed_path,
        ).to(self.device)

    def _enable_compile(
        self,
        model: MechanismLanguageModel,
        mechanism: str,
    ) -> None:
        if not self.config.compile:
            self.logger.compile_status(False)
            return
        if mechanism == "gru":
            self.logger.compile_status(
                False,
                warning="GRU uses the fused eager MIOpen/cuDNN implementation",
            )
            return
        if mechanism == "softmax" and torch.version.hip is not None:
            self.logger.compile_status(
                False,
                warning="softmax uses eager ROCm Flash Attention",
            )
            return
        try:
            backend = model.compile_encoder(
                mode=self.config.compile_mode,
                backend=self.config.compile_backend,
            )
            self.logger.compile_status(
                True,
                backend=backend,
                mode=self.config.compile_mode,
            )
        except Exception as error:
            self.logger.compile_status(
                False,
                warning=f"torch.compile unavailable: {error}",
            )

    @staticmethod
    def _synchronize(device: torch.device) -> None:
        if device.type == "cuda":
            torch.cuda.synchronize(device)

    def _save_checkpoint(
        self,
        path: Path,
        *,
        model: MechanismLanguageModel,
        optimizer: AdamW,
        scheduler: Any,
        step: int,
        elapsed_seconds: float,
        peak_memory_bytes: int,
        learning_curve: list[dict[str, Any]],
    ) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        checkpoint: dict[str, Any] = {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "step": step,
            "elapsed_seconds": elapsed_seconds,
            "peak_memory_bytes": peak_memory_bytes,
            "learning_curve": learning_curve,
            "model_config": model.config.to_dict(),
            "protocol": {
                "train_sequence_length": self.config.train_sequence_length,
                "batch_size": self.config.batch_size,
                "train_steps": self.config.train_steps,
                "seed": self.config.seed,
            },
            "cpu_rng_state": torch.get_rng_state(),
        }
        if self.device.type == "cuda":
            checkpoint["cuda_rng_state"] = torch.cuda.get_rng_state(
                self.device
            )
        torch.save(checkpoint, path)

    def _load_checkpoint(
        self,
        path: Path,
        *,
        model: MechanismLanguageModel,
        optimizer: AdamW,
        scheduler: Any,
    ) -> tuple[int, float, int, list[dict[str, Any]]]:
        checkpoint = torch.load(
            path,
            map_location=self.device,
            weights_only=False,
        )
        if checkpoint.get("model_config") != model.config.to_dict():
            raise ValueError(
                f"checkpoint model configuration does not match {path}"
            )
        expected_protocol = {
            "train_sequence_length": self.config.train_sequence_length,
            "batch_size": self.config.batch_size,
            "train_steps": self.config.train_steps,
            "seed": self.config.seed,
        }
        saved_protocol = checkpoint.get("protocol", {})
        mismatches = {
            name: (saved_protocol.get(name), expected)
            for name, expected in expected_protocol.items()
            if saved_protocol.get(name) != expected
        }
        if mismatches:
            raise ValueError(
                f"checkpoint training protocol does not match {path}: "
                f"{mismatches}"
            )
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        scheduler.load_state_dict(checkpoint["scheduler"])
        if "cpu_rng_state" in checkpoint:
            torch.set_rng_state(checkpoint["cpu_rng_state"].cpu())
        if self.device.type == "cuda" and "cuda_rng_state" in checkpoint:
            torch.cuda.set_rng_state(
                checkpoint["cuda_rng_state"],
                self.device,
            )
        step = int(checkpoint["step"])
        self.logger.resumed(path, step)
        return (
            step,
            float(checkpoint.get("elapsed_seconds", 0.0)),
            int(checkpoint.get("peak_memory_bytes", 0)),
            list(checkpoint.get("learning_curve", [])),
        )

    def _train(
        self,
        mechanism: str,
        model: MechanismLanguageModel,
        tokens: Tensor,
    ) -> tuple[dict[str, Any], list[dict[str, Any]]]:
        trainable = [
            parameter for parameter in model.parameters()
            if parameter.requires_grad
        ]
        optimizer = AdamW(
            trainable,
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay,
            betas=(0.9, 0.95),
            fused=self.device.type == "cuda",
        )
        scheduler = make_scheduler(
            optimizer,
            self.config.warmup_steps,
            self.config.train_steps,
        )
        checkpoint_path = self._checkpoint_path(mechanism)
        step = 0
        elapsed_before = 0.0
        saved_peak_memory = 0
        learning_curve: list[dict[str, Any]] = []
        if self.config.resume and checkpoint_path.exists():
            (
                step,
                elapsed_before,
                saved_peak_memory,
                learning_curve,
            ) = self._load_checkpoint(
                checkpoint_path,
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
            )
            if step > self.config.train_steps:
                raise ValueError(
                    f"checkpoint step {step} exceeds configured train_steps="
                    f"{self.config.train_steps}"
                )

        if self.device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(self.device)
        model.train()
        optimizer.zero_grad(set_to_none=True)
        if step < self.config.train_steps:
            self._enable_compile(model, mechanism)

        self._synchronize(self.device)
        started = time.perf_counter()
        last_metrics_at = 0.0
        last_loss = (
            float(learning_curve[-1]["training_loss"])
            if learning_curve
            else math.nan
        )
        with self.logger.training_progress(
            self.config.train_steps,
            initial=step,
        ) as progress:
            task = progress.add_task(
                f"NLP {mechanism}",
                total=self.config.train_steps,
                completed=step,
            )
            while step < self.config.train_steps:
                dropout_seed = 2_000_000 + self.config.seed + step
                torch.manual_seed(dropout_seed)
                if self.device.type == "cuda":
                    torch.cuda.manual_seed_all(dropout_seed)
                batch = _sample_training_batch(
                    tokens,
                    sequence_length=self.config.train_sequence_length,
                    batch_size=self.config.batch_size,
                    seed=self.config.seed,
                    step=step,
                ).to(self.device, non_blocking=True)
                negative_generator = torch.Generator(device=self.device)
                negative_generator.manual_seed(1_000_000 + self.config.seed + step)
                with self._autocast():
                    hidden = model.encode(batch[:, :-1])
                    loss = model.embeddings.cross_entropy(
                        hidden,
                        batch[:, 1:],
                        chunk_size=self.config.ce_chunk_size,
                        backend=self.config.ce_backend,
                        num_negative_samples=self.config.ce_negative_samples,
                        generator=negative_generator,
                    )
                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    trainable,
                    self.config.max_grad_norm,
                    foreach=self.device.type == "cuda",
                )
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
                step += 1
                last_loss = float(loss.detach())
                now = time.monotonic()
                should_log = (
                    step == 1
                    or step % self.config.log_every == 0
                    or step == self.config.train_steps
                )
                if should_log:
                    learning_curve.append(
                        {
                            "mechanism": mechanism,
                            "step": step,
                            "tokens_seen": (
                                step
                                * self.config.batch_size
                                * self.config.train_sequence_length
                            ),
                            "training_loss": last_loss,
                            "learning_rate": scheduler.get_last_lr()[0],
                        }
                    )
                if (
                    now - last_metrics_at >= PROGRESS_REFRESH_SECONDS
                    or step == self.config.train_steps
                ):
                    progress.update(
                        task,
                        loss=f"{last_loss:.4f}",
                        lr=f"{scheduler.get_last_lr()[0]:.2e}",
                    )
                    last_metrics_at = now
                progress.advance(task)

                if (
                    self.config.save_every > 0
                    and step % self.config.save_every == 0
                ):
                    self._synchronize(self.device)
                    current_elapsed = elapsed_before + (
                        time.perf_counter() - started
                    )
                    current_peak = saved_peak_memory
                    if self.device.type == "cuda":
                        current_peak = max(
                            current_peak,
                            torch.cuda.max_memory_allocated(self.device),
                        )
                    self._save_checkpoint(
                        checkpoint_path,
                        model=model,
                        optimizer=optimizer,
                        scheduler=scheduler,
                        step=step,
                        elapsed_seconds=current_elapsed,
                        peak_memory_bytes=current_peak,
                        learning_curve=learning_curve,
                    )

        self._synchronize(self.device)
        elapsed_seconds = elapsed_before + (time.perf_counter() - started)
        peak_memory_bytes = saved_peak_memory
        if self.device.type == "cuda":
            peak_memory_bytes = max(
                peak_memory_bytes,
                torch.cuda.max_memory_allocated(self.device),
            )
        self._save_checkpoint(
            checkpoint_path,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            step=step,
            elapsed_seconds=elapsed_seconds,
            peak_memory_bytes=peak_memory_bytes,
            learning_curve=learning_curve,
        )
        tokens_seen = (
            step * self.config.batch_size * self.config.train_sequence_length
        )
        return (
            {
                "steps": step,
                "tokens_seen": tokens_seen,
                "last_training_loss": last_loss,
                "elapsed_seconds": elapsed_seconds,
                "tokens_per_second": (
                    tokens_seen / elapsed_seconds if elapsed_seconds > 0 else 0.0
                ),
                "peak_memory_bytes": peak_memory_bytes,
                "checkpoint": str(checkpoint_path),
            },
            learning_curve,
        )

    @torch.no_grad()
    def _evaluate_length(
        self,
        model: MechanismLanguageModel,
        tokens: Tensor,
        sequence_length: int,
    ) -> dict[str, Any]:
        model.eval()
        starts = _validation_starts(
            tokens.numel(),
            sequence_length=sequence_length,
            example_count=self.config.eval_batch_size * self.config.eval_batches,
            maximum_sequence_length=max(self.config.eval_sequence_lengths),
        )
        bucket_loss_sums = [0.0 for _ in POSITION_BUCKETS]
        bucket_token_counts = [0 for _ in POSITION_BUCKETS]
        tail_loss_sum = 0.0
        tail_token_count = 0
        with self.logger.validation_progress(
            self.config.eval_batches
        ) as progress:
            task = progress.add_task(
                f"Validating {sequence_length:,} tokens",
                total=self.config.eval_batches,
            )
            for batch_index in range(self.config.eval_batches):
                offset = batch_index * self.config.eval_batch_size
                batch_starts = starts[
                    offset : offset + self.config.eval_batch_size
                ]
                batch = _batch_from_starts(
                    tokens,
                    batch_starts,
                    sequence_length,
                ).to(self.device, non_blocking=True)
                with self._autocast():
                    hidden = model.encode(batch[:, :-1])
                    targets = batch[:, 1:]
                    for bucket_index in range(len(POSITION_BUCKETS)):
                        start = bucket_index * sequence_length // 4
                        end = (bucket_index + 1) * sequence_length // 4
                        loss = model.embeddings.cross_entropy(
                            hidden[:, start:end],
                            targets[:, start:end],
                            chunk_size=self.config.ce_chunk_size,
                            backend="tiled",
                        )
                        count = targets[:, start:end].numel()
                        bucket_loss_sums[bucket_index] += float(loss) * count
                        bucket_token_counts[bucket_index] += count
                    tail_start = sequence_length - self.config.eval_tail_tokens
                    tail_loss = model.embeddings.cross_entropy(
                        hidden[:, tail_start:],
                        targets[:, tail_start:],
                        chunk_size=self.config.ce_chunk_size,
                        backend="tiled",
                    )
                    tail_count = targets[:, tail_start:].numel()
                    tail_loss_sum += float(tail_loss) * tail_count
                    tail_token_count += tail_count
                progress.advance(task)

        total_tokens = sum(bucket_token_counts)
        total_loss = sum(bucket_loss_sums) / max(1, total_tokens)
        tail_loss = tail_loss_sum / max(1, tail_token_count)
        bucket_losses = {
            name: loss_sum / max(1, token_count)
            for name, loss_sum, token_count in zip(
                POSITION_BUCKETS,
                bucket_loss_sums,
                bucket_token_counts,
                strict=True,
            )
        }
        return {
            "sequence_length": sequence_length,
            "regime": (
                "trained"
                if sequence_length == self.config.train_sequence_length
                else (
                    "shorter"
                    if sequence_length < self.config.train_sequence_length
                    else "extrapolation"
                )
            ),
            "loss": total_loss,
            "perplexity": _perplexity(total_loss),
            "token_count": total_tokens,
            "tail_tokens": self.config.eval_tail_tokens,
            "tail_token_count": tail_token_count,
            "tail_loss": tail_loss,
            "tail_perplexity": _perplexity(tail_loss),
            "position_loss": bucket_losses,
            "position_perplexity": {
                name: _perplexity(loss)
                for name, loss in bucket_losses.items()
            },
        }

    def _run_mechanism(
        self,
        mechanism: str,
        train_tokens: Tensor,
        validation_tokens: Tensor,
        index: int,
    ) -> tuple[dict[str, Any], list[dict[str, Any]]]:
        model = self._new_model(mechanism)
        total_parameters = model.num_parameters()
        trainable_parameters = model.num_parameters(trainable_only=True)
        effective_n_quats = (
            model.config.n_quats or model.config.d_model
            if model.config.ffn_type == "quatspin"
            else None
        )
        self.logger.language_evaluation_case(
            index=index,
            total=len(self.config.mechanisms),
            mechanism=mechanism,
            parameters=total_parameters,
            trainable_parameters=trainable_parameters,
            d_ff=model.config.d_ff,
            ffn_type=model.config.ffn_type,
            n_quats=effective_n_quats,
        )
        training, learning_curve = self._train(
            mechanism,
            model,
            train_tokens,
        )
        # Fixed-shape compilation accelerates training. Eager evaluation avoids
        # compiling one graph for every context length in the diagnostic sweep.
        model.disable_compile()
        evaluations = [
            self._evaluate_length(model, validation_tokens, length)
            for length in self.config.eval_sequence_lengths
        ]
        result = {
            "mechanism": mechanism,
            "parameters": total_parameters,
            "trainable_parameters": trainable_parameters,
            "d_ff": model.config.d_ff,
            "ffn_type": model.config.ffn_type,
            "n_quats": effective_n_quats,
            "training": training,
            "evaluations": evaluations,
        }
        del model
        if self.device.type == "cuda":
            torch.cuda.empty_cache()
        return result, learning_curve

    def run(
        self,
        *,
        train_tokens: Tensor | None = None,
        validation_tokens: Tensor | None = None,
    ) -> Path:
        """Run all mechanisms and return the generated Markdown report."""
        if train_tokens is None:
            train_tokens = load_wikitext(
                self.config.cache_dir,
                self.config.train_sequence_length,
                "train",
                self.logger,
            ).tokens
        if validation_tokens is None:
            validation_tokens = load_wikitext(
                self.config.cache_dir,
                self.config.train_sequence_length,
                "validation",
                self.logger,
            ).tokens
        if train_tokens.ndim != 1 or validation_tokens.ndim != 1:
            raise ValueError("WikiText token streams must be one-dimensional")

        self.logger.language_evaluation_header(
            device=self.device,
            dtype=self.config.dtype,
            output_dir=self.config.output_dir,
            train_steps=self.config.train_steps,
            train_sequence_length=self.config.train_sequence_length,
            eval_sequence_lengths=self.config.eval_sequence_lengths,
            eval_tail_tokens=self.config.eval_tail_tokens,
            batch_size=self.config.batch_size,
            eval_batch_size=self.config.eval_batch_size,
            eval_batches=self.config.eval_batches,
            mechanisms=self.config.mechanisms,
        )
        results: list[dict[str, Any]] = []
        learning_curves: list[dict[str, Any]] = []
        for index, mechanism in enumerate(self.config.mechanisms, start=1):
            result, curve = self._run_mechanism(
                mechanism,
                train_tokens,
                validation_tokens,
                index,
            )
            results.append(result)
            learning_curves.extend(curve)
        return self._write_report(
            results,
            learning_curves,
            train_token_count=train_tokens.numel(),
            validation_token_count=validation_tokens.numel(),
        )

    def _write_report(
        self,
        results: list[dict[str, Any]],
        learning_curves: list[dict[str, Any]],
        *,
        train_token_count: int,
        validation_token_count: int,
    ) -> Path:
        reference_parameters = next(
            result["parameters"]
            for result in results
            if result["mechanism"] == "multigrid"
        ) if any(
            result["mechanism"] == "multigrid" for result in results
        ) else results[0]["parameters"]
        report = {
            "evaluation": "WikiText-103 matched-mechanism language modeling",
            "config": asdict(self.config),
            "dataset": {
                "name": "Salesforce/wikitext",
                "subset": "wikitext-103-raw-v1",
                "tokenizer": "cl100k_base",
                "train_tokens_available": train_token_count,
                "validation_tokens_available": validation_token_count,
            },
            "results": results,
            "notes": {
                "fairness": (
                    "Every mechanism sees identical deterministic training "
                    "windows and uses the same tied SVD token frontend."
                ),
                "positions": (
                    "Fixed sinusoidal positions are used so contexts longer "
                    "than training do not depend on untrained position rows."
                ),
                "loss": (
                    f"Training uses {self.config.ce_backend} cross-entropy; "
                    "all reported validation perplexities use exact tiled "
                    "full-vocabulary cross-entropy."
                ),
                "context_metric": (
                    f"Tail perplexity scores the same final "
                    f"{self.config.eval_tail_tokens} target tokens at every "
                    "context length; only the available prefix changes."
                ),
                "optimizer": (
                    "All mechanisms use the same optimizer recipe. Treat close "
                    "results as provisional until each finalist receives a "
                    "small learning-rate sweep."
                ),
                "statistics": (
                    "This is a one-seed screening run. Confirm the leading "
                    "mechanisms with at least two additional seeds."
                ),
            },
        }
        json_path = self.config.output_dir / "report.json"
        json_path.write_text(
            json.dumps(report, indent=2, default=str) + "\n",
            encoding="utf-8",
        )

        perplexity_rows: list[dict[str, Any]] = []
        runtime_rows: list[dict[str, Any]] = []
        for result in results:
            mechanism = result["mechanism"]
            for evaluation in result["evaluations"]:
                row = {
                    "mechanism": mechanism,
                    "sequence_length": evaluation["sequence_length"],
                    "regime": evaluation["regime"],
                    "loss": evaluation["loss"],
                    "perplexity": evaluation["perplexity"],
                    "token_count": evaluation["token_count"],
                    "tail_tokens": evaluation["tail_tokens"],
                    "tail_token_count": evaluation["tail_token_count"],
                    "tail_loss": evaluation["tail_loss"],
                    "tail_perplexity": evaluation["tail_perplexity"],
                }
                for bucket in POSITION_BUCKETS:
                    label = bucket.replace("%", "pct").replace("-", "_to_")
                    row[f"loss_{label}"] = evaluation["position_loss"][bucket]
                    row[f"perplexity_{label}"] = evaluation[
                        "position_perplexity"
                    ][bucket]
                perplexity_rows.append(row)
            training = result["training"]
            runtime_rows.append(
                {
                    "mechanism": mechanism,
                    "parameters": result["parameters"],
                    "delta_vs_multigrid": (
                        result["parameters"] - reference_parameters
                    ),
                    "trainable_parameters": result["trainable_parameters"],
                    "d_ff": result["d_ff"],
                    "ffn_type": result["ffn_type"],
                    "n_quats": result["n_quats"],
                    "ffn_description": (
                        f"{result['n_quats']:,} quats"
                        if result["ffn_type"] == "quatspin"
                        else f"{result['d_ff']:,} hidden"
                    ),
                    "steps": training["steps"],
                    "tokens_seen": training["tokens_seen"],
                    "elapsed_seconds": training["elapsed_seconds"],
                    "tokens_per_second": training["tokens_per_second"],
                    "peak_memory_bytes": training["peak_memory_bytes"],
                }
            )
        self._write_csv(
            self.config.output_dir / "perplexity.csv",
            perplexity_rows,
        )
        self._write_csv(
            self.config.output_dir / "training.csv",
            runtime_rows,
        )
        self._write_csv(
            self.config.output_dir / "learning_curves.csv",
            learning_curves,
        )

        markdown_path = self.config.output_dir / "report.md"
        markdown_path.write_text(
            self._render_markdown(report, runtime_rows),
            encoding="utf-8",
        )
        return markdown_path

    @staticmethod
    def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
        if not rows:
            return
        with path.open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(
                stream,
                fieldnames=list(rows[0]),
                lineterminator="\n",
            )
            writer.writeheader()
            writer.writerows(rows)

    def _render_markdown(
        self,
        report: dict[str, Any],
        runtime_rows: list[dict[str, Any]],
    ) -> str:
        results = report["results"]
        mechanisms = [result["mechanism"] for result in results]
        tokens_per_mechanism = (
            self.config.train_steps
            * self.config.batch_size
            * self.config.train_sequence_length
        )
        lines = [
            "# Multigrid NLP Evaluation Results",
            "",
            "Primary score: **exact WikiText-103 perplexity on aligned target "
            "tails** (lower is better). The best result in each row is bold.",
            "",
            "## Evaluation setup",
            "",
            "| Setting | Value |",
            "| --- | --- |",
            f"| Mechanisms | {', '.join(mechanisms)} |",
            f"| Training steps | {self.config.train_steps:,} |",
            f"| Training context | {self.config.train_sequence_length:,} tokens |",
            f"| Training batch | {self.config.batch_size:,} |",
            (
                "| Tokens per mechanism | "
                f"{tokens_per_mechanism:,} "
                "|"
            ),
            (
                "| Evaluation contexts | "
                + ", ".join(
                    f"{length:,}" for length in self.config.eval_sequence_lengths
                )
                + " |"
            ),
            f"| Aligned scored tail | {self.config.eval_tail_tokens:,} tokens |",
            f"| Evaluation batches | {self.config.eval_batches:,} |",
            f"| FFN type | {self.config.model.ffn_type} |",
            (
                f"| Reference quaternion channels | "
                f"{self.config.model.n_quats or self.config.model.d_model:,} |"
                if self.config.model.ffn_type == "quatspin"
                else f"| Reference FFN width | {self.config.model.d_ff:,} |"
            ),
            f"| Seed | {self.config.seed:,} |",
            "| Tokenizer | cl100k_base |",
            "",
            "## Aligned-tail perplexity by context length",
            "",
            "| Context | Regime | " + " | ".join(mechanisms) + " |",
            "| --- | --- | " + " | ".join("---" for _ in mechanisms) + " |",
        ]
        result_by_mechanism = {
            result["mechanism"]: result for result in results
        }
        for length in self.config.eval_sequence_lengths:
            evaluations = {
                mechanism: next(
                    evaluation
                    for evaluation in result_by_mechanism[mechanism]["evaluations"]
                    if evaluation["sequence_length"] == length
                )
                for mechanism in mechanisms
            }
            best = min(
                evaluation["tail_perplexity"]
                for evaluation in evaluations.values()
            )
            regime = next(iter(evaluations.values()))["regime"]
            values = []
            for mechanism in mechanisms:
                value = evaluations[mechanism]["tail_perplexity"]
                label = f"{value:.2f}"
                if math.isclose(value, best, rel_tol=1e-12, abs_tol=1e-12):
                    label = f"**{label}**"
                values.append(label)
            lines.append(
                f"| {length:,} | {regime.title()} | "
                + " | ".join(values)
                + " |"
            )

        lines.extend(
            [
                "",
                "## Full-sequence perplexity",
                "",
                "| Context | Regime | " + " | ".join(mechanisms) + " |",
                "| --- | --- | "
                + " | ".join("---" for _ in mechanisms)
                + " |",
            ]
        )
        for length in self.config.eval_sequence_lengths:
            evaluations = {
                mechanism: next(
                    evaluation
                    for evaluation in result_by_mechanism[mechanism]["evaluations"]
                    if evaluation["sequence_length"] == length
                )
                for mechanism in mechanisms
            }
            best = min(
                evaluation["perplexity"]
                for evaluation in evaluations.values()
            )
            regime = next(iter(evaluations.values()))["regime"]
            values = []
            for mechanism in mechanisms:
                value = evaluations[mechanism]["perplexity"]
                label = f"{value:.2f}"
                if math.isclose(value, best, rel_tol=1e-12, abs_tol=1e-12):
                    label = f"**{label}**"
                values.append(label)
            lines.append(
                f"| {length:,} | {regime.title()} | "
                + " | ".join(values)
                + " |"
            )

        longest = max(self.config.eval_sequence_lengths)
        lines.extend(
            [
                "",
                f"## Position perplexity at context {longest:,}",
                "",
                "| Mechanism | " + " | ".join(POSITION_BUCKETS) + " |",
                "| --- | " + " | ".join("---" for _ in POSITION_BUCKETS) + " |",
            ]
        )
        for mechanism in mechanisms:
            evaluation = next(
                item
                for item in result_by_mechanism[mechanism]["evaluations"]
                if item["sequence_length"] == longest
            )
            values = [
                f"{evaluation['position_perplexity'][bucket]:.2f}"
                for bucket in POSITION_BUCKETS
            ]
            lines.append(f"| {mechanism} | " + " | ".join(values) + " |")

        lines.extend(
            [
                "",
                "## Training and model cost",
                "",
                "| Mechanism | Parameters | Δ vs. multigrid | Trainable | "
                "FFN shape | Tokens/s | Time (min) | Peak memory (MiB) |",
                "| --- | --- | --- | --- | --- | --- | --- | --- |",
            ]
        )
        for row in runtime_rows:
            lines.append(
                "| {mechanism} | {parameters:,} | {delta_vs_multigrid:+,} | "
                "{trainable_parameters:,} | {ffn_description} | "
                "{tokens_per_second:,.1f} | "
                "{minutes:.1f} | {memory:.1f} |".format(
                    **row,
                    minutes=row["elapsed_seconds"] / 60.0,
                    memory=row["peak_memory_bytes"] / 1024**2,
                )
            )

        lines.extend(
            [
                "",
                "## Notes",
                "",
                "- Every mechanism sees identical deterministic WikiText "
                "token windows.",
                "- All models share the same tied SVD token frontend and "
                "fixed sinusoidal positions.",
                f"- Training uses `{self.config.ce_backend}` cross-entropy; "
                "reported validation perplexity is exact full-vocabulary "
                "cross-entropy.",
                "- Contexts longer than the training context are extrapolation tests.",
                f"- Tail perplexity scores the same final "
                f"{self.config.eval_tail_tokens:,} target tokens at every "
                "context length.",
                "- This one-seed run is a screen; confirm finalists with at "
                "least two additional seeds.",
                "- If results are close, run a small per-mechanism learning-rate "
                "sweep before claiming an architecture winner.",
                "- `learning_curves.csv` contains token-matched optimization curves.",
                "",
            ]
        )
        return "\n".join(lines)


def run_language_evaluation(
    config: LanguageEvaluationConfig,
    *,
    logger: PrettyLogger | None = None,
) -> Path:
    """Convenience entry point for the configuration-only script."""
    return LanguageEvaluator(config, logger).run()
