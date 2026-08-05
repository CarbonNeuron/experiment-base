"""Structured terminal logging for training runs, with a plain-text fallback."""

from __future__ import annotations

import math
import sys
from collections.abc import Mapping
from typing import Any

try:
    from rich.color import Color
    from rich.columns import Columns
    from rich.console import Console
    from rich.panel import Panel
    from rich.progress import (
        BarColumn,
        MofNCompleteColumn,
        Progress,
        ProgressColumn,
        SpinnerColumn,
        Task,
        TaskProgressColumn,
        TextColumn,
        TimeElapsedColumn,
        TimeRemainingColumn,
    )
    from rich.table import Table
    from rich.text import Text

    _RICH_AVAILABLE = True
except ImportError:  # pragma: no cover - exercised only without optional Rich
    Columns = Console = Panel = Table = Text = None  # type: ignore[assignment]
    Progress = ProgressColumn = Task = None  # type: ignore[assignment]
    _RICH_AVAILABLE = False


if _RICH_AVAILABLE:
    import colorsys
    import time

    class _GradientLabelColumn(ProgressColumn):
        """Animated rainbow gradient text that shifts hue each frame."""

        def __init__(self, speed: float = 0.5, saturation: float = 0.7,
                     lightness: float = 0.65) -> None:
            super().__init__()
            self.speed = speed
            self.saturation = saturation
            self.lightness = lightness

        @staticmethod
        def _hsl_to_rgb(h: float, s: float, l: float) -> tuple[int, int, int]:
            r, g, b = colorsys.hls_to_rgb(h % 1.0, l, s)
            return int(r * 255), int(g * 255), int(b * 255)

        def render(self, task: Task) -> Text:
            label = task.description
            offset = -time.monotonic() * self.speed
            result = Text()
            for i, ch in enumerate(label):
                hue = offset + i / max(len(label), 1)
                r, g, b = self._hsl_to_rgb(hue, self.saturation, self.lightness)
                result.append(ch, style=f"bold #{r:02x}{g:02x}{b:02x}")
            return result

    class _StepCompleteColumn(ProgressColumn):
        """Render the main task count with an explicit step label."""

        def render(self, task: Task) -> Text:
            total = int(task.total or 0)
            width = len(f"{total:,}")
            completed = f"{int(task.completed):,}".rjust(width)
            result = Text()
            result.append("step ", style="cyan")
            result.append(f"{completed}", style="bold cyan")
            result.append("/", style="dim")
            result.append(f"{total:,}", style="cyan")
            return result


    class _RateColumn(ProgressColumn):
        """Render task throughput with a caller-selected unit."""

        def __init__(self, unit: str, width: int = 10) -> None:
            super().__init__()
            self.unit = unit
            self.width = width

        def render(self, task: Task) -> Text:
            if task.speed is None:
                label = f"-- {self.unit}"
            else:
                label = f"{task.speed:,.2f} {self.unit}"
            return Text(label.rjust(self.width), style="progress.data.speed")


    class _TrainingMetricsColumn(ProgressColumn):
        """Render tqdm-style postfix values stored in Rich task fields."""

        # (field_name, value_width, label_style, value_style)
        _FIELDS = (
            ("epoch", 5, "white", "bold white"),
            ("loss", 6, "yellow", "bold yellow"),
            ("lr", 8, "magenta", "bold magenta"),
            ("hard", 6, "red", "bold red"),
            ("margin", 5, "green", "bold green"),
        )

        def render(self, task: Task) -> Text:
            result = Text()
            first = True
            for name, width, label_style, value_style in self._FIELDS:
                if name not in task.fields:
                    continue
                if not first:
                    result.append("  ")
                first = False
                result.append(f"{name} ", style=label_style)
                val = str(task.fields[name])
                if name == "epoch" and "/" in val:
                    cur, tot = val.split("/", 1)
                    padded = cur.rjust(len(tot))
                    result.append(padded, style=value_style)
                    result.append("/", style="dim")
                    result.append(tot, style=value_style)
                else:
                    result.append(val.rjust(width), style=value_style)
            return result


class _TqdmProgress:
    """Expose the small Rich Progress API used by this project via tqdm."""

    def __init__(
        self,
        total: int,
        description: str,
        unit: str,
        *,
        initial: int = 0,
        transient: bool = False,
    ) -> None:
        self.total = total
        self.description = description
        self.unit = unit
        self.initial = initial
        self.transient = transient
        self._progress: Any | None = None

    def __enter__(self) -> _TqdmProgress:
        from tqdm.auto import tqdm

        self._progress = tqdm(
            total=self.total,
            initial=self.initial,
            desc=self.description,
            unit=self.unit,
            dynamic_ncols=True,
            mininterval=0.2,
            leave=not self.transient,
        )
        return self

    def __exit__(self, *_: object) -> None:
        if self._progress is not None:
            self._progress.close()

    def add_task(
        self,
        description: str,
        *,
        total: int | None = None,
        completed: int = 0,
        **_: object,
    ) -> int:
        if self._progress is None:
            raise RuntimeError("progress context has not been entered")
        self._progress.set_description_str(description, refresh=False)
        if total is not None:
            self._progress.total = total
        self._progress.n = completed
        return 0

    def advance(self, task_id: int, advance: int = 1) -> None:
        del task_id
        if self._progress is None:
            raise RuntimeError("progress context has not been entered")
        self._progress.update(advance)

    def update(self, task_id: int, **fields: object) -> None:
        del task_id
        if self._progress is None:
            raise RuntimeError("progress context has not been entered")
        self._progress.set_postfix(fields, refresh=False)


class _FallbackConsole:
    """Small Console-compatible writer used when Rich is unavailable."""

    def print(
        self,
        *objects: object,
        style: str | None = None,
        **_: object,
    ) -> None:
        del style
        sys.stderr.write(" ".join(str(value) for value in objects) + "\n")


class PrettyLogger:
    """Keep all human-facing training output consistent and easy to call."""

    GOOD_PPL = 50.0
    MEDIOCRE_PPL = 100.0

    def __init__(self, console: Any | None = None) -> None:
        if console is not None:
            self.console = console
        elif _RICH_AVAILABLE:
            self.console = Console(stderr=True)
        else:
            self.console = _FallbackConsole()

    def training_progress(self, total: int, initial: int = 0) -> Progress:
        """Return a Rich Progress context for the main training loop."""
        if not _RICH_AVAILABLE:
            return _TqdmProgress(
                total,
                "Training",
                "step",
                initial=initial,
            )
        return Progress(
            _GradientLabelColumn(),
            TimeElapsedColumn(),
            BarColumn(bar_width=None),
            TaskProgressColumn(),
            _StepCompleteColumn(),
            _RateColumn("steps/s", width=14),
            _TrainingMetricsColumn(),
            TimeRemainingColumn(),
            console=self.console,
            expand=True,
            refresh_per_second=5,
            transient=False,
        )

    def validation_progress(self, total: int) -> Progress:
        """Return a Rich Progress context for validation."""
        if not _RICH_AVAILABLE:
            return _TqdmProgress(
                total,
                "Validating",
                "batch",
                transient=True,
            )
        return Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TaskProgressColumn(),
            _StepCompleteColumn(),
            TimeElapsedColumn(),
            console=self.console,
            refresh_per_second=5,
            transient=True,
        )

    def tokenization_progress(
        self,
        total: int,
        description: str,
    ) -> Progress:
        """Return a Rich Progress context for data tokenization."""
        if not _RICH_AVAILABLE:
            return _TqdmProgress(
                total,
                description,
                "doc",
                transient=True,
            )
        return Progress(
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TaskProgressColumn(),
            MofNCompleteColumn(),
            _RateColumn("docs/s"),
            TimeRemainingColumn(),
            console=self.console,
            refresh_per_second=5,
            transient=True,
        )

    @staticmethod
    def _scalar(value: Any) -> Any:
        item = getattr(value, "item", None)
        if callable(item):
            try:
                return item()
            except (RuntimeError, ValueError):
                return value
        return value

    @classmethod
    def _format_value(cls, value: Any) -> str:
        value = cls._scalar(value)
        if isinstance(value, bool):
            return "yes" if value else "no"
        if isinstance(value, int):
            return f"{value:,}"
        if isinstance(value, float):
            if not math.isfinite(value):
                return str(value)
            return f"{value:.4f}"
        if isinstance(value, (list, tuple)):
            return ", ".join(cls._format_value(item) for item in value)
        return str(value)

    @staticmethod
    def _human_count(value: int) -> str:
        magnitude = abs(value)
        for scale, suffix in (
            (1_000_000_000_000, "T"),
            (1_000_000_000, "B"),
            (1_000_000, "M"),
            (1_000, "K"),
        ):
            if magnitude >= scale:
                return f"{value / scale:.1f}{suffix}"
        return f"{value:,}"

    @staticmethod
    def _human_bytes(value: int) -> str:
        magnitude = float(abs(value))
        for scale, suffix in (
            (1024**4, "TiB"),
            (1024**3, "GiB"),
            (1024**2, "MiB"),
            (1024, "KiB"),
        ):
            if magnitude >= scale:
                return f"{value / scale:.2f} {suffix} ({value:,} bytes)"
        return f"{value:,} bytes"

    @staticmethod
    def _model_config_items(model_config: Any) -> list[tuple[str, Any]]:
        if isinstance(model_config, Mapping):
            values = model_config
        else:
            to_dict = getattr(model_config, "to_dict", None)
            if callable(to_dict):
                values = to_dict()
            else:
                values = {
                    key: value
                    for key, value in vars(model_config).items()
                    if not key.startswith("_")
                }
        return [(str(key), value) for key, value in values.items()]

    @staticmethod
    def _gpu_info(device: Any) -> tuple[str, str] | None:
        """Return (gpu_name, vram_str) if *device* is a CUDA device."""
        try:
            import torch

            dev = torch.device(device) if not hasattr(device, "type") else device
            if dev.type != "cuda":
                return None
            props = torch.cuda.get_device_properties(dev)
            vram_gb = props.total_memory / (1024 ** 3)
            return props.name, f"{vram_gb:.1f} GB"
        except Exception:
            return None

    def experiment_header(
        self,
        device: Any,
        dtype: str,
        vocab_size: int,
        total_params: int,
        trainable_params: int,
        model_config: Any,
    ) -> None:
        """Show runtime, parameter, and architecture details in one panel."""
        config_items = self._model_config_items(model_config)
        gpu = self._gpu_info(device)
        if not _RICH_AVAILABLE:
            gpu_str = f"{gpu[0]} ({gpu[1]})" if gpu else str(device)
            architecture = ", ".join(
                f"{key}={self._format_value(value)}"
                for key, value in config_items
            )
            self.console.print(
                "Experiment | "
                f"device={gpu_str} dtype={dtype} vocab={vocab_size:,} "
                f"parameters={total_params:,} ({self._human_count(total_params)}) "
                f"trainable={trainable_params:,} "
                f"({self._human_count(trainable_params)}) | {architecture}"
            )
            return

        metadata = Table.grid(padding=(0, 1))
        metadata.add_column(style="bold cyan", no_wrap=True)
        metadata.add_column(style="white")
        if gpu:
            metadata.add_row("GPU", f"{gpu[0]}  [dim]({gpu[1]})[/dim]")
            metadata.add_row("Device", str(device))
        else:
            metadata.add_row("Device", str(device))
        metadata.add_row("Precision", dtype)
        metadata.add_row("Vocabulary", f"{vocab_size:,}")
        metadata.add_row(
            "Parameters",
            f"{total_params:,} ({self._human_count(total_params)})",
        )
        metadata.add_row(
            "Trainable",
            f"{trainable_params:,} ({self._human_count(trainable_params)})",
        )

        architecture = Table(
            "Setting",
            "Value",
            header_style="bold cyan",
            box=None,
            padding=(0, 1),
        )
        for key, value in config_items:
            architecture.add_row(
                Text(key, style="cyan"),
                Text(self._format_value(value)),
            )

        layout = Table.grid(padding=(0, 4))
        layout.add_column()
        layout.add_column()
        layout.add_row(metadata, architecture)

        self.console.print(
            Panel(
                layout,
                title=Text("Experiment", style="bold"),
                border_style="cyan",
                expand=False,
            )
        )

    def data_loaded(self, split: str, token_count: int) -> None:
        """Report a cached/tokenized dataset split."""
        if not _RICH_AVAILABLE:
            self.console.print(f"✓ {split}: {token_count:,} tokens")
            return
        self.console.print(
            Text.assemble(
                ("✓ ", "bold green"),
                (split, "bold cyan"),
                (f": {token_count:,} tokens", "white"),
            )
        )

    def compile_status(
        self,
        enabled: bool,
        backend: str | None = None,
        mode: str | None = None,
        warning: str | None = None,
    ) -> None:
        """Report compiler selection or an eager-mode fallback."""
        if not _RICH_AVAILABLE:
            if enabled:
                self.console.print(
                    "✓ torch.compile enabled "
                    f"(backend={backend or 'default'}, mode={mode or 'default'})"
                )
            else:
                self.console.print("⚠ torch.compile disabled; using eager encoder")
            if warning:
                self.console.print(f"⚠ {warning}")
            return

        if enabled:
            line = Text.assemble(
                ("✓ ", "bold green"),
                ("torch.compile enabled", "bold"),
                (
                    f"  backend={backend or 'default'} · mode={mode or 'default'}",
                    "cyan",
                ),
            )
        else:
            line = Text.assemble(
                ("⚠ ", "bold yellow"),
                ("torch.compile disabled; using eager encoder", "yellow"),
            )
        self.console.print(line)
        if warning:
            self.console.print(Text(f"  ⚠ {warning}", style="yellow"))

    def resumed(self, path: Any, step: int) -> None:
        """Report checkpoint restoration."""
        if not _RICH_AVAILABLE:
            self.console.print(f"↻ resumed {path} at step {step:,}")
            return
        self.console.print(
            Text.assemble(
                ("↻ ", "bold cyan"),
                ("Resumed ", "bold"),
                (str(path), "cyan"),
                (f" at step {step:,}", "white"),
            )
        )

    def validation(self, step: int, loss: float, ppl: float) -> None:
        """Report validation loss and color-code perplexity quality."""
        if ppl < self.GOOD_PPL:
            ppl_style = "green"
        elif ppl < self.MEDIOCRE_PPL:
            ppl_style = "yellow"
        else:
            ppl_style = "red"

        if not _RICH_AVAILABLE:
            self.console.print(
                f"step {step:,}: validation loss={loss:.4f} ppl={ppl:.2f}"
            )
            return
        self.console.print(
            Text.assemble(
                ("Validation", "bold cyan"),
                (f"  step {step:,}", "white"),
                ("  loss ", "dim"),
                (f"{loss:.4f}", "bold"),
                ("  perplexity ", "dim"),
                (f"{ppl:.2f}", f"bold {ppl_style}"),
            )
        )

    def hard_negative_enabled(
        self,
        backend: str,
        k: int,
        index_location: Any,
        fingerprint: str,
    ) -> None:
        """Show hard-negative retrieval configuration."""
        short_fingerprint = fingerprint[:12]
        if not _RICH_AVAILABLE:
            self.console.print(
                "Hard-negative retrieval enabled | "
                f"backend={backend} k={k:,} index={index_location} "
                f"fingerprint={short_fingerprint}"
            )
            return

        details = Table.grid(padding=(0, 1))
        details.add_column(style="bold cyan", no_wrap=True)
        details.add_column()
        details.add_row("Backend", backend)
        details.add_row("Neighbors", f"{k:,}")
        details.add_row("Index", str(index_location))
        details.add_row("Fingerprint", short_fingerprint)
        self.console.print(
            Panel(
                details,
                title=Text("Hard-negative retrieval enabled", style="bold green"),
                border_style="cyan",
            )
        )

    @classmethod
    def _format_metric(cls, name: str, value: Any) -> str:
        value = cls._scalar(value)
        if not isinstance(value, (int, float)):
            return str(value)
        numeric = float(value)
        if name.endswith("_seconds"):
            return f"{1000 * numeric:.2f} ms"
        if "orthogonality" in name or name.endswith("orth_error"):
            return f"{numeric:.2e}"
        if name == "mean_valid_hard_negatives":
            return f"{numeric:.1f}"
        if name in {
            "mean_positive_logit",
            "mean_max_hard_logit",
            "mean_hard_margin",
            "hard_error_rate",
        }:
            return f"{numeric:.3f}"
        return f"{numeric:.4f}"

    def hard_negative_recall(self, step: int, k: int, recall: float) -> None:
        """Report exact hard-negative recall diagnostics."""
        if not _RICH_AVAILABLE:
            self.console.print(
                f"step {step:,}: hard-negative exact recall@{k:,}={recall:.4f}"
            )
            return
        self.console.print(
            Text.assemble(
                ("Hard-negative recall", "bold cyan"),
                (f"  step {step:,}", "white"),
                (f"  recall@{k:,} ", "dim"),
                (f"{recall:.4f}", "bold green"),
            )
        )

    def hard_negative_metrics(
        self,
        step: int,
        metrics_dict: Mapping[str, Any],
        step_seconds: float,
        peak_memory: int,
        scale: float,
        orth_error: float,
    ) -> None:
        """Render retrieval loss, timing, memory, and geometry diagnostics."""
        rows = [
            (name, self._format_metric(name, value))
            for name, value in metrics_dict.items()
        ]
        rows.extend(
            [
                ("step_time", f"{1000 * step_seconds:.2f} ms"),
                ("peak_memory", self._human_bytes(int(peak_memory))),
                ("scale", f"{scale:.4f}"),
                ("rotation_orth_error", f"{orth_error:.2e}"),
            ]
        )

        if not _RICH_AVAILABLE:
            values = " ".join(f"{name}={value}" for name, value in rows)
            self.console.print(f"step {step:,}: {values}")
            return

        labels = {
            "sampled_loss": "Sampled loss",
            "hard_loss": "Hard loss",
            "weighted_hard_loss": "Weighted hard loss",
            "total_loss": "Total loss",
            "hard_loss_weight": "Hard-loss weight",
            "mean_positive_logit": "Mean positive logit",
            "mean_max_hard_logit": "Mean max-hard logit",
            "mean_hard_margin": "Mean hard margin",
            "hard_error_rate": "Hard error rate",
            "mean_valid_hard_negatives": "Valid hard negatives",
            "retrieval_seconds": "Retrieval time",
            "candidate_score_seconds": "Candidate-score time",
            "hard_loss_seconds": "Hard-loss time",
            "step_time": "Step time",
            "peak_memory": "Peak memory",
            "scale": "Embedding scale",
            "rotation_orth_error": "Rotation orthogonality error",
        }
        table = Table(
            title=f"Hard-negative diagnostics · step {step:,}",
            title_style="bold",
            header_style="bold cyan",
            border_style="cyan",
        )
        table.add_column("Metric", style="cyan", no_wrap=True)
        table.add_column("Value", justify="right")
        for name, value in rows:
            value_style = "green" if name.endswith("loss") else "white"
            table.add_row(
                labels.get(name, name.replace("_", " ").title()),
                Text(value, style=value_style),
            )
        self.console.print(table)

    def training_finished(self, step: int, checkpoint_path: Any) -> None:
        """Show final step and checkpoint location."""
        if not _RICH_AVAILABLE:
            self.console.print(
                f"✓ training finished at step {step:,}; wrote {checkpoint_path}"
            )
            return
        body = Text.assemble(
            ("Final step  ", "bold cyan"),
            (f"{step:,}\n", "bold"),
            ("Checkpoint  ", "bold cyan"),
            (str(checkpoint_path), "green"),
        )
        self.console.print(
            Panel(
                body,
                title=Text("✓ Training complete", style="bold green"),
                border_style="green",
            )
        )

    def log(self, message: object, style: str | None = None) -> None:
        """Print a one-off message through the same console."""
        if _RICH_AVAILABLE:
            self.console.print(Text(str(message), style=style or ""))
        else:
            self.console.print(message, style=style)
