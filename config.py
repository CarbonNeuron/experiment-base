"""Configuration objects shared by CLI entry points and experiment scripts."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any


COMPILE_BACKENDS = ("auto", "inductor", "aot_eager", "eager")
COMPILE_MODES = ("default", "reduce-overhead", "max-autotune")
DTYPES = ("fp32", "bf16", "fp16")
LOSS_BACKENDS = ("tiled", "sampled", "checkpoint", "full")


@dataclass
class TransformerConfig:
    """Transformer architecture; embedding metadata comes from svd-embeds."""

    d_model: int = 128
    n_heads: int = 8
    n_layers: int = 6
    d_ff: int = 512
    max_seq_len: int = 512
    dropout: float = 0.1
    layer_norm_eps: float = 1e-5

    def __post_init__(self) -> None:
        if self.d_model <= 0 or self.n_heads <= 0 or self.n_layers <= 0:
            raise ValueError("model dimensions and layer counts must be positive")
        if self.d_model % self.n_heads:
            raise ValueError("d_model must be divisible by n_heads")
        if self.d_ff <= 0 or self.max_seq_len <= 0:
            raise ValueError("d_ff and max_seq_len must be positive")
        if not 0.0 <= self.dropout < 1.0:
            raise ValueError("dropout must be in [0, 1)")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class DataConfig:
    """Dataset and DataLoader settings."""

    batch_size: int = 8
    num_workers: int = 2
    val_num_workers: int = 0
    cache_dir: Path = Path("data_cache")

    def __post_init__(self) -> None:
        self.cache_dir = Path(self.cache_dir)
        if self.batch_size <= 0:
            raise ValueError("batch_size must be positive")
        if self.num_workers < 0 or self.val_num_workers < 0:
            raise ValueError("worker counts must be non-negative")


@dataclass
class TrainingConfig:
    """Optimization, evaluation, and checkpoint cadence."""

    epochs: int = 3
    lr: float = 3e-4
    weight_decay: float = 0.1
    warmup_steps: int = 500
    grad_accum_steps: int = 1
    max_grad_norm: float = 1.0
    ce_chunk_size: int = 1024
    ce_backend: str = "tiled"
    ce_negative_samples: int = 4096
    max_steps: int = 0
    eval_every: int = 500
    eval_batches: int = 50
    save_every: int = 1000
    seed: int = 42

    def __post_init__(self) -> None:
        if self.epochs <= 0 or self.grad_accum_steps <= 0:
            raise ValueError("epochs and grad_accum_steps must be positive")
        if self.lr <= 0 or self.max_grad_norm <= 0:
            raise ValueError("lr and max_grad_norm must be positive")
        if self.weight_decay < 0:
            raise ValueError("weight_decay must be non-negative")
        if self.ce_backend not in LOSS_BACKENDS:
            raise ValueError(f"ce_backend must be one of {LOSS_BACKENDS}")
        if self.ce_negative_samples <= 0:
            raise ValueError("ce_negative_samples must be positive")
        for name in (
            "warmup_steps",
            "ce_chunk_size",
            "max_steps",
            "eval_every",
            "eval_batches",
            "save_every",
        ):
            if getattr(self, name) < 0:
                raise ValueError(f"{name} must be non-negative")


@dataclass
class RuntimeConfig:
    """Device, precision, compiler, and checkpoint location settings."""

    device: str = "auto"
    dtype: str = "bf16"
    compile: bool = False
    compile_mode: str = "default"
    compile_backend: str = "auto"
    checkpoint_dir: Path = Path("checkpoints")
    resume: Path | None = None

    def __post_init__(self) -> None:
        self.checkpoint_dir = Path(self.checkpoint_dir)
        self.resume = Path(self.resume) if self.resume is not None else None
        if self.dtype not in DTYPES:
            raise ValueError(f"dtype must be one of {DTYPES}")
        if self.compile_mode not in COMPILE_MODES:
            raise ValueError(f"compile_mode must be one of {COMPILE_MODES}")
        if self.compile_backend not in COMPILE_BACKENDS:
            raise ValueError(f"compile_backend must be one of {COMPILE_BACKENDS}")


@dataclass
class ExperimentConfig:
    """Complete experiment assembled from independently editable sections."""

    model: TransformerConfig = field(default_factory=TransformerConfig)
    data: DataConfig = field(default_factory=DataConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    runtime: RuntimeConfig = field(default_factory=RuntimeConfig)
    embed_path: Path | None = None

    def __post_init__(self) -> None:
        self.embed_path = (
            Path(self.embed_path) if self.embed_path is not None else None
        )
