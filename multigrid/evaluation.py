"""Train and compare sequence mechanisms on diagnostic primitive tasks."""

from __future__ import annotations

import csv
import json
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import torch
from torch import Tensor, nn
from torch.nn import functional as F
from torch.optim import AdamW

from config import COMPILE_BACKENDS, COMPILE_MODES
from training.logger import PrettyLogger
from training.runtime import make_scheduler, resolve_device, resolve_precision

from .benchmarks import (
    ANSWER_TOKEN_COUNT,
    ANSWER_TOKEN_START,
    TASK_NAMES,
    TASK_SUMMARIES,
    BenchmarkBatch,
    generate_primitive_batch,
)
from .mechanisms import (
    MECHANISM_NAMES,
    SymbolicModelConfig,
    SymbolicSequenceModel,
    matched_model_configs,
)
from .triton_memory import triton_memory_available
from .triton_linear_attention import triton_linear_attention_available
from .triton_ssm import triton_ssm_available


PROGRESS_REFRESH_SECONDS = 0.2


@dataclass
class PrimitiveEvaluationConfig:
    """Complete reproducible configuration for the diagnostic evaluation."""

    tasks: tuple[str, ...] = TASK_NAMES
    mechanisms: tuple[str, ...] = MECHANISM_NAMES
    capacities: tuple[int, ...] = (4, 8, 16, 32, 64)
    train_capacity: int = 16
    curriculum_capacities: tuple[int, ...] = (2, 4, 8, 16)
    train_steps: int = 5_000
    batch_size: int = 512
    eval_batches: int = 16
    learning_rate: float = 1e-3
    weight_decay: float = 0.01
    warmup_steps: int = 200
    max_grad_norm: float = 1.0
    seed: int = 42
    device: str = "auto"
    dtype: str = "bf16"
    compile: bool = True
    compile_mode: str = "default"
    compile_backend: str = "auto"
    compile_dynamic: bool = False
    save_every: int = 1_000
    resume: bool = True
    output_dir: Path = Path("checkpoints/multigrid-evaluation-v2")
    runtime_lengths: tuple[int, ...] = (64, 128, 256, 512)
    runtime_repetitions: int = 5
    decode_steps: int = 8
    model: SymbolicModelConfig = field(default_factory=SymbolicModelConfig)

    def __post_init__(self) -> None:
        if isinstance(self.model, dict):
            self.model = SymbolicModelConfig(**self.model)
        self.tasks = tuple(self.tasks)
        self.mechanisms = tuple(self.mechanisms)
        self.capacities = tuple(self.capacities)
        self.curriculum_capacities = tuple(self.curriculum_capacities)
        self.runtime_lengths = tuple(self.runtime_lengths)
        self.output_dir = Path(self.output_dir)
        unknown_tasks = set(self.tasks) - set(TASK_NAMES)
        unknown_mechanisms = set(self.mechanisms) - set(MECHANISM_NAMES)
        if unknown_tasks:
            raise ValueError(f"unknown tasks: {sorted(unknown_tasks)}")
        if unknown_mechanisms:
            raise ValueError(
                f"unknown mechanisms: {sorted(unknown_mechanisms)}"
            )
        positive = (
            self.train_capacity,
            self.train_steps,
            self.batch_size,
            self.eval_batches,
            self.learning_rate,
            self.max_grad_norm,
            self.runtime_repetitions,
            self.decode_steps,
        )
        if any(value <= 0 for value in positive):
            raise ValueError("evaluation dimensions and rates must be positive")
        if not self.tasks or not self.mechanisms or not self.capacities:
            raise ValueError("tasks, mechanisms, and capacities cannot be empty")
        if any(capacity <= 0 for capacity in self.capacities):
            raise ValueError("capacities must be positive")
        if (
            not self.curriculum_capacities
            or any(value <= 0 for value in self.curriculum_capacities)
            or tuple(sorted(set(self.curriculum_capacities)))
            != self.curriculum_capacities
            or self.curriculum_capacities[-1] != self.train_capacity
        ):
            raise ValueError(
                "curriculum_capacities must be sorted, unique, positive, and "
                "end at train_capacity"
            )
        if self.warmup_steps < 0 or self.save_every < 0:
            raise ValueError("warmup_steps and save_every must be non-negative")
        if self.weight_decay < 0:
            raise ValueError("weight_decay must be non-negative")
        if self.compile_mode not in COMPILE_MODES:
            raise ValueError(f"compile_mode must be one of {COMPILE_MODES}")
        if self.compile_backend not in COMPILE_BACKENDS:
            raise ValueError(
                f"compile_backend must be one of {COMPILE_BACKENDS}"
            )


def _masked_loss(logits: Tensor, batch: BenchmarkBatch) -> Tensor:
    mask = batch.answer_mask
    if not torch.any(mask):
        raise ValueError("benchmark batch has no supervised positions")
    answer_logits = logits[
        ..., ANSWER_TOKEN_START : ANSWER_TOKEN_START + ANSWER_TOKEN_COUNT
    ]
    return F.cross_entropy(
        answer_logits[mask], batch.targets[mask] - ANSWER_TOKEN_START
    )


def _move_batch(batch: BenchmarkBatch, device: torch.device) -> BenchmarkBatch:
    return BenchmarkBatch(
        batch.input_ids.to(device, non_blocking=True),
        batch.targets.to(device, non_blocking=True),
    )


def _synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


class PrimitiveTrainer:
    """Fit fresh models per primitive and report capacity/interference curves."""

    def __init__(
        self,
        config: PrimitiveEvaluationConfig,
        logger: PrettyLogger | None = None,
    ) -> None:
        self.config = config
        self.logger = logger or PrettyLogger()
        self.device = resolve_device(config.device)
        if self.device.type == "cuda":
            # Inductor autotuning and vendor libraries sometimes consult the
            # process-wide current device even when tensor arguments live on
            # another GPU. Keep it aligned with the configured evaluator.
            torch.cuda.set_device(self.device)
        self.dtype, self.amp_enabled = resolve_precision(
            config.dtype, self.device
        )
        self.config.output_dir.mkdir(parents=True, exist_ok=True)
        self.model_configs = matched_model_configs(
            config.model, config.mechanisms
        )

    def _autocast(self):
        return torch.autocast(
            device_type=self.device.type,
            dtype=self.dtype,
            enabled=self.amp_enabled,
        )

    def _seed_for(self, mechanism: str, task: str) -> int:
        return (
            self.config.seed
            + 10_000 * self.config.mechanisms.index(mechanism)
            + 100 * self.config.tasks.index(task)
        )

    @staticmethod
    def _mechanism_uses_compile(mechanism: str) -> bool:
        if mechanism == "gru":
            return False
        return not (mechanism == "softmax" and torch.version.hip is not None)

    def _checkpoint_path(self, mechanism: str, task: str) -> Path:
        return self.config.output_dir / mechanism / task / "latest.pt"

    def _enable_compile(self, model: SymbolicSequenceModel) -> None:
        mechanism = model.config.mechanism
        if mechanism == "gru":
            # Dynamo deliberately graph-breaks on nn.GRU. On ROCm the
            # resulting wrapper is slower than the already-fused MIOpen path
            # while still appearing compiled in the status output.
            self.logger.compile_status(
                False,
                warning="GRU uses the fused eager MIOpen/cuDNN implementation",
            )
            return
        if mechanism == "softmax" and torch.version.hip is not None:
            # ROCm SDPA already selects causal Flash Attention. Compiling the
            # surrounding model was consistently slower and adds substantial
            # per-task startup time on gfx1201.
            self.logger.compile_status(
                False,
                warning="softmax uses eager ROCm Flash Attention",
            )
            return
        try:
            backend = model.compile_model(
                mode=self.config.compile_mode,
                backend=self.config.compile_backend,
                dynamic=self.config.compile_dynamic,
            )
            warning = None
            if self.config.compile_backend == "auto" and backend == "aot_eager":
                warning = (
                    "Triton is unavailable for this accelerator; selected "
                    "aot_eager instead of Inductor."
                )
            self.logger.compile_status(
                True,
                backend=backend,
                mode=self.config.compile_mode,
                warning=warning,
            )
        except Exception as error:
            self.logger.compile_status(
                False,
                warning=f"torch.compile unavailable: {error}",
            )

    def _save_checkpoint(
        self,
        path: Path,
        model: SymbolicSequenceModel,
        optimizer: AdamW,
        scheduler: Any,
        generator: torch.Generator,
        step: int,
        task: str,
    ) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(".tmp")
        torch.save(
            {
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
                "generator_state": generator.get_state(),
                "step": step,
                "task": task,
                "model_config": model.config.to_dict(),
                "evaluation_config": {
                    key: str(value) if isinstance(value, Path) else value
                    for key, value in asdict(self.config).items()
                    if key != "model"
                },
            },
            temporary,
        )
        temporary.replace(path)

    def _make_batch(
        self,
        task: str,
        capacity: int,
        batch_size: int,
        generator: torch.Generator,
        pad_to_length: int | None = None,
    ) -> BenchmarkBatch:
        batch = generate_primitive_batch(
            task,
            batch_size,
            capacity,
            self.config.model.vocab_size,
            generator=generator,
        )
        if batch.input_ids.size(1) > self.config.model.max_seq_len:
            raise ValueError(
                f"{task} capacity {capacity} produces length "
                f"{batch.input_ids.size(1)}, above max_seq_len="
                f"{self.config.model.max_seq_len}"
            )
        if pad_to_length is not None:
            if batch.input_ids.size(1) > pad_to_length:
                raise ValueError("batch is longer than the requested padded length")
            padding = pad_to_length - batch.input_ids.size(1)
            if padding:
                batch = BenchmarkBatch(
                    F.pad(batch.input_ids, (0, padding), value=0),
                    F.pad(batch.targets, (0, padding), value=-1),
                )
        return _move_batch(batch, self.device)

    def _training_sequence_length(self, task: str) -> int:
        example = generate_primitive_batch(
            task,
            1,
            self.config.train_capacity,
            self.config.model.vocab_size,
            generator=torch.Generator().manual_seed(self.config.seed),
        )
        return example.input_ids.size(1)

    def _curriculum_capacity(self, step: int) -> int:
        stage = min(
            len(self.config.curriculum_capacities) - 1,
            step * len(self.config.curriculum_capacities)
            // self.config.train_steps,
        )
        return self.config.curriculum_capacities[stage]

    def fit_one(
        self,
        mechanism: str,
        task: str,
    ) -> SymbolicSequenceModel:
        """Train or resume one mechanism/task pair."""
        if self.config.compile and self._mechanism_uses_compile(mechanism):
            # Each task owns an independent model and fixed sequence length.
            # Cached graphs from the preceding task cannot be reused and only
            # consume Dynamo's per-code-object recompilation budget.
            torch.compiler.reset()
        seed = self._seed_for(mechanism, task)
        torch.manual_seed(seed)
        generator = torch.Generator().manual_seed(seed + 1)
        model = SymbolicSequenceModel(self.model_configs[mechanism]).to(
            self.device
        )
        if self.config.compile:
            self._enable_compile(model)
        optimizer = AdamW(
            model.parameters(),
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay,
            betas=(0.9, 0.95),
            # The fused implementation cuts the optimizer portion of a
            # multigrid step from roughly 1.9 ms to 0.6 ms on ROCm. CPU keeps
            # the portable implementation used by tests and smoke runs.
            fused=self.device.type == "cuda",
        )
        scheduler = make_scheduler(
            optimizer, self.config.warmup_steps, self.config.train_steps
        )
        checkpoint_path = self._checkpoint_path(mechanism, task)
        start_step = 0
        if self.config.resume and checkpoint_path.exists():
            checkpoint = torch.load(
                checkpoint_path, map_location=self.device, weights_only=False
            )
            if checkpoint.get("task") != task:
                raise ValueError(f"checkpoint task does not match {task!r}")
            saved_model_config = dict(checkpoint.get("model_config", {}))
            # Checkpoints created before the fused kernel was added use the
            # same parameterization and remain safe to resume.
            saved_model_config.setdefault(
                "use_triton_memory", model.config.use_triton_memory
            )
            if saved_model_config != model.config.to_dict():
                raise ValueError(
                    f"checkpoint model configuration does not match {mechanism!r}"
                )
            model.load_state_dict(checkpoint["model"])
            optimizer.load_state_dict(checkpoint["optimizer"])
            scheduler.load_state_dict(checkpoint["scheduler"])
            if "generator_state" in checkpoint:
                # ``map_location=self.device`` is correct for model and
                # optimizer tensors, but torch.Generator always expects its
                # state to be a CPU ByteTensor.
                generator.set_state(checkpoint["generator_state"].cpu())
            start_step = int(checkpoint["step"])
            self.logger.resumed(checkpoint_path, start_step)

        model.train()
        padded_length = self._training_sequence_length(task)
        last_metrics_at = 0.0
        with self.logger.training_progress(
            self.config.train_steps, initial=start_step
        ) as progress:
            progress_task = progress.add_task(
                f"{mechanism} · {task}",
                total=self.config.train_steps,
                completed=start_step,
            )
            for step in range(start_step, self.config.train_steps):
                capacity = self._curriculum_capacity(step)
                batch = self._make_batch(
                    task,
                    capacity,
                    self.config.batch_size,
                    generator,
                    pad_to_length=padded_length,
                )
                optimizer.zero_grad(set_to_none=True)
                answer_mask = batch.answer_mask
                with self._autocast():
                    answer_logits = model.supervised_logits(
                        batch.input_ids,
                        answer_mask,
                        ANSWER_TOKEN_START,
                        ANSWER_TOKEN_COUNT,
                    )
                    loss = F.cross_entropy(
                        answer_logits,
                        batch.targets[answer_mask] - ANSWER_TOKEN_START,
                    )
                loss.backward()
                nn.utils.clip_grad_norm_(
                    model.parameters(),
                    self.config.max_grad_norm,
                    foreach=self.device.type == "cuda",
                )
                optimizer.step()
                scheduler.step()
                completed = step + 1
                now = time.monotonic()
                if (
                    now - last_metrics_at >= PROGRESS_REFRESH_SECONDS
                    or completed == self.config.train_steps
                ):
                    # loss.item() synchronizes the GPU. Refresh human-facing
                    # metrics at screen rate instead of stalling every step.
                    progress.update(
                        progress_task,
                        epoch=f"{completed}/{self.config.train_steps}",
                        loss=f"{loss.item():.4f}",
                        lr=f"{scheduler.get_last_lr()[0]:.2e}",
                        capacity=str(capacity),
                    )
                    last_metrics_at = now
                progress.advance(progress_task)
                if (
                    self.config.save_every > 0
                    and completed % self.config.save_every == 0
                ):
                    self._save_checkpoint(
                        checkpoint_path,
                        model,
                        optimizer,
                        scheduler,
                        generator,
                        completed,
                        task,
                    )
        self._save_checkpoint(
            checkpoint_path,
            model,
            optimizer,
            scheduler,
            generator,
            self.config.train_steps,
            task,
        )
        return model

    @torch.no_grad()
    def evaluate_one(
        self,
        model: SymbolicSequenceModel,
        mechanism: str,
        task: str,
        capacity: int,
    ) -> dict[str, Any]:
        """Evaluate held-out examples at one capacity."""
        seed = self._seed_for(mechanism, task) + 1_000_000 + capacity
        generator = torch.Generator().manual_seed(seed)
        model.eval()
        loss_sum = torch.zeros((), device=self.device)
        correct = torch.zeros((), device=self.device, dtype=torch.long)
        exact = torch.zeros((), device=self.device, dtype=torch.long)
        answers = 0
        examples = 0
        with self.logger.validation_progress(
            self.config.eval_batches
        ) as progress:
            progress_task = progress.add_task(
                f"{task} · capacity {capacity}",
                total=self.config.eval_batches,
            )
            for _ in range(self.config.eval_batches):
                batch = self._make_batch(
                    task, capacity, self.config.batch_size, generator
                )
                mask = batch.answer_mask
                with self._autocast():
                    answer_logits = model.supervised_logits(
                        batch.input_ids,
                        mask,
                        ANSWER_TOKEN_START,
                        ANSWER_TOKEN_COUNT,
                    )
                    answer_targets = batch.targets[mask]
                    loss = F.cross_entropy(
                        answer_logits,
                        answer_targets - ANSWER_TOKEN_START,
                    )
                predictions = (
                    answer_logits.argmax(dim=-1) + ANSWER_TOKEN_START
                )
                matches = predictions.eq(answer_targets)
                loss_sum += loss.float()
                correct += matches.sum()
                answers += matches.numel()
                batch_size = batch.input_ids.size(0)
                exact += matches.reshape(batch_size, -1).all(dim=1).sum()
                examples += batch_size
                progress.advance(progress_task)
        return {
            "mechanism": mechanism,
            "task": task,
            "capacity": capacity,
            "regime": (
                "interpolation"
                if capacity <= self.config.train_capacity
                else "extrapolation"
            ),
            "sequence_length": int(batch.input_ids.size(1)),
            "loss": loss_sum.item() / self.config.eval_batches,
            "token_accuracy": correct.item() / max(1, answers),
            "exact_match": exact.item() / max(1, examples),
            "parameters": model.num_parameters(),
        }

    @torch.no_grad()
    def benchmark_runtime(
        self,
        model: SymbolicSequenceModel,
        mechanism: str,
    ) -> list[dict[str, Any]]:
        """Measure full-prefix prefill and uncached autoregressive decoding."""
        model.eval()
        rows: list[dict[str, Any]] = []
        for length in self.config.runtime_lengths:
            if (
                length > model.config.max_seq_len
                or length <= self.config.decode_steps
            ):
                continue
            generator = torch.Generator().manual_seed(
                self.config.seed + 2_000_000 + length
            )
            tokens = torch.randint(
                model.config.vocab_size,
                (1, length),
                generator=generator,
            ).to(self.device)
            if self.device.type == "cuda":
                torch.cuda.reset_peak_memory_stats(self.device)
            with self._autocast():
                model(tokens)
            _synchronize(self.device)
            start = time.perf_counter()
            for _ in range(self.config.runtime_repetitions):
                with self._autocast():
                    model(tokens)
            _synchronize(self.device)
            prefill_seconds = (
                time.perf_counter() - start
            ) / self.config.runtime_repetitions

            prefix_length = length - self.config.decode_steps
            start = time.perf_counter()
            for _ in range(self.config.runtime_repetitions):
                with self._autocast():
                    for end in range(prefix_length, length):
                        model(tokens[:, : end + 1])
            _synchronize(self.device)
            decode_seconds = (
                time.perf_counter() - start
            ) / self.config.runtime_repetitions
            peak_memory = (
                torch.cuda.max_memory_allocated(self.device)
                if self.device.type == "cuda"
                else None
            )
            rows.append(
                {
                    "mechanism": mechanism,
                    "sequence_length": length,
                    "prefill_ms": 1000.0 * prefill_seconds,
                    "prefill_tokens_per_second": length / prefill_seconds,
                    "uncached_decode_ms_per_token": (
                        1000.0 * decode_seconds / self.config.decode_steps
                    ),
                    "peak_memory_bytes": peak_memory,
                    "decode_mode": "full-prefix replay",
                }
            )
        return rows

    def run(self) -> Path:
        """Run the complete matrix and write machine-readable reports."""
        accuracy_rows: list[dict[str, Any]] = []
        runtime_rows: list[dict[str, Any]] = []
        reference_config = matched_model_configs(
            self.config.model, ("multigrid",)
        )["multigrid"]
        reference_parameters = SymbolicSequenceModel(
            reference_config
        ).num_parameters()
        parameter_rows = []
        for mechanism, model_config in self.model_configs.items():
            parameters = SymbolicSequenceModel(model_config).num_parameters()
            parameter_rows.append(
                {
                    "name": mechanism,
                    "mechanism": mechanism,
                    "parameters": parameters,
                    "delta_vs_multigrid": parameters - reference_parameters,
                    "d_ff": model_config.d_ff,
                }
            )

        compile_description = (
            f"{self.config.compile_backend} backend · "
            f"{self.config.compile_mode} mode · training updates"
            if self.config.compile
            else "disabled (eager)"
        )
        if self.config.model.use_triton_memory and triton_memory_available():
            compile_description += " · fused Triton episodic memory"
        if (
            "linear_attention" in self.config.mechanisms
            and triton_linear_attention_available()
        ):
            compile_description += " · fused Triton linear attention"
        if "ssm" in self.config.mechanisms and triton_ssm_available():
            compile_description += " · fused Triton SSM"
        self.logger.evaluation_header(
            device=self.device,
            dtype=self.config.dtype,
            compile_description=compile_description,
            seed=self.config.seed,
            output_dir=self.config.output_dir,
            train_capacity=self.config.train_capacity,
            curriculum_capacities=self.config.curriculum_capacities,
            capacities=self.config.capacities,
            train_steps=self.config.train_steps,
            batch_size=self.config.batch_size,
            eval_batches=self.config.eval_batches,
            tasks=tuple(
                (task, TASK_SUMMARIES[task]) for task in self.config.tasks
            ),
            mechanisms=tuple(parameter_rows),
        )

        total_cases = len(self.config.mechanisms) * len(self.config.tasks)
        case_index = 0
        parameters_by_mechanism = {
            row["mechanism"]: row for row in parameter_rows
        }
        for mechanism, model_config in self.model_configs.items():
            runtime_model: SymbolicSequenceModel | None = None
            for task in self.config.tasks:
                case_index += 1
                example = generate_primitive_batch(
                    task,
                    1,
                    self.config.train_capacity,
                    self.config.model.vocab_size,
                    generator=torch.Generator().manual_seed(
                        self._seed_for(mechanism, task)
                    ),
                )
                parameter_row = parameters_by_mechanism[mechanism]
                self.logger.evaluation_case(
                    index=case_index,
                    total=total_cases,
                    mechanism=mechanism,
                    task=task,
                    description=TASK_SUMMARIES[task],
                    parameters=int(parameter_row["parameters"]),
                    d_ff=model_config.d_ff,
                    train_capacity=self.config.train_capacity,
                    sequence_length=example.input_ids.size(1),
                    answers_per_example=int(example.answer_mask.sum()),
                    capacities=self.config.capacities,
                )
                model = self.fit_one(mechanism, task)
                runtime_model = model
                # Held-out capacities deliberately use different shapes.
                # Compiling five one-off validation shapes costs more than
                # running this short sweep eagerly.
                model.disable_compile()
                for capacity in self.config.capacities:
                    metrics = self.evaluate_one(
                        model, mechanism, task, capacity
                    )
                    accuracy_rows.append(metrics)
                    self.logger.console.print(
                        f"{mechanism} · {task} · capacity={capacity}: "
                        f"accuracy={metrics['token_accuracy']:.4f} "
                        f"exact={metrics['exact_match']:.4f}"
                    )
            if runtime_model is not None:
                runtime_rows.extend(
                    self.benchmark_runtime(runtime_model, mechanism)
                )

        report = {
            "config": asdict(self.config),
            "parameters": parameter_rows,
            "accuracy": accuracy_rows,
            "runtime": runtime_rows,
            "notes": {
                "language_modeling": (
                    "WikiText remains a separate secondary evaluation via "
                    "scripts/train_multigrid.py; these results isolate "
                    "mechanism-level computation first."
                ),
                "decode": (
                    "All mechanisms use full-prefix replay because the public "
                    "model APIs do not expose inference caches."
                ),
            },
        }
        report_path = self.config.output_dir / "report.json"
        report_path.write_text(
            json.dumps(report, indent=2, default=str) + "\n",
            encoding="utf-8",
        )
        self._write_csv(self.config.output_dir / "accuracy.csv", accuracy_rows)
        self._write_csv(self.config.output_dir / "runtime.csv", runtime_rows)
        return report_path

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


def run_primitive_evaluation(
    config: PrimitiveEvaluationConfig,
    *,
    logger: PrettyLogger | None = None,
) -> Path:
    """Convenience entry point used by configuration-only scripts."""
    return PrimitiveTrainer(config, logger).run()
