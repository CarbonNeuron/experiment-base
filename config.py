"""Configuration objects shared by CLI entry points and experiment scripts."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Protocol


COMPILE_BACKENDS = ("auto", "inductor", "aot_eager", "eager")
COMPILE_MODES = ("default", "reduce-overhead", "max-autotune")
DTYPES = ("fp32", "bf16", "fp16")
LOSS_BACKENDS = ("tiled", "sampled", "checkpoint", "full")
HARD_NEGATIVE_BACKENDS = ("ivf", "exact")
HARD_NEGATIVE_LOSSES = ("candidate_ce", "pairwise")


class ArchitectureConfig(Protocol):
    """Minimum config contract understood by the experiment runner."""

    max_seq_len: int

    def to_dict(self) -> dict[str, Any]: ...


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
class CompoundQConfig:
    """Transformer architecture using compound query projections."""

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
class GrowingWidthConfig:
    """Architecture settings for a growing-then-frozen residual stream."""

    d_embed: int = 128
    n_heads: int = 8
    n_layers: int = 12
    d_ff_ratio: float = 4.0
    max_seq_len: int = 512
    dropout: float = 0.1
    layer_norm_eps: float = 1e-5
    embed_cutoff_ratio: float = 0.7

    def __post_init__(self) -> None:
        if min(
            self.d_embed,
            self.n_heads,
            self.n_layers,
            self.max_seq_len,
        ) <= 0:
            raise ValueError("model dimensions and layer counts must be positive")
        if self.d_embed % self.n_heads:
            raise ValueError("d_embed must be divisible by n_heads")
        if self.d_ff_ratio <= 0:
            raise ValueError("d_ff_ratio must be positive")
        if not 0.0 <= self.dropout < 1.0:
            raise ValueError("dropout must be in [0, 1)")
        if not 0.0 <= self.embed_cutoff_ratio <= 1.0:
            raise ValueError("embed_cutoff_ratio must be in [0, 1]")

    @property
    def embed_cutoff_layer(self) -> int:
        """First layer that may reconstruct the decayed embed channels."""
        return round(self.embed_cutoff_ratio * self.n_layers)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class HydraConfig:
    """Three-phase shared, streamed, and wide-merge decoder settings."""

    d_embed: int = 128
    n_streams: int = 8
    n_heads_per_stream: int = 8
    n_intake_layers: int = 2
    n_stream_layers: int = 3
    n_merge_layers: int = 2
    d_ff_ratio: float = 4.0
    max_seq_len: int = 512
    dropout: float = 0.1
    layer_norm_eps: float = 1e-5

    def __post_init__(self) -> None:
        if min(
            self.d_embed,
            self.n_streams,
            self.n_heads_per_stream,
            self.n_intake_layers,
            self.n_stream_layers,
            self.n_merge_layers,
            self.d_ff_ratio,
            self.max_seq_len,
            self.layer_norm_eps,
        ) <= 0:
            raise ValueError("model dimensions and layer counts must be positive")
        if self.d_embed % self.n_heads_per_stream:
            raise ValueError("d_embed must be divisible by n_heads_per_stream")
        if self.d_wide % self.n_heads_wide:
            raise ValueError("d_wide must be divisible by n_heads_wide")
        if not 0.0 <= self.dropout < 1.0:
            raise ValueError("dropout must be in [0, 1)")

    @property
    def d_wide(self) -> int:
        return self.d_embed * self.n_streams

    @property
    def n_heads_wide(self) -> int:
        return self.n_streams

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class ChainedHydraConfig:
    """Settings for chained, recursively nested Hydra blocks."""

    d_embed: int = 128
    n_heads: int = 8
    n_intake_layers: int = 2
    n_blocks: int = 3
    n_streams: int = 2
    depth: int = 2
    n_stream_layers: int = 2
    n_merge_layers: int = 1
    d_ff_ratio: float = 4.0
    max_seq_len: int = 512
    dropout: float = 0.1
    layer_norm_eps: float = 1e-5

    def __post_init__(self) -> None:
        if min(
            self.d_embed,
            self.n_heads,
            self.n_intake_layers,
            self.n_blocks,
            self.n_streams,
            self.depth,
            self.n_stream_layers,
            self.n_merge_layers,
            self.d_ff_ratio,
            self.max_seq_len,
            self.layer_norm_eps,
        ) <= 0:
            raise ValueError("model dimensions and layer counts must be positive")
        if self.d_embed % self.n_heads:
            raise ValueError("d_embed must be divisible by n_heads")
        if self.d_wide % self.n_heads_merge:
            raise ValueError("d_wide must be divisible by n_heads_merge")
        if not 0.0 <= self.dropout < 1.0:
            raise ValueError("dropout must be in [0, 1)")

    @property
    def d_wide(self) -> int:
        return self.d_embed * self.n_streams

    @property
    def n_heads_merge(self) -> int:
        return self.n_streams

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class TournamentHydraConfig:
    """Settings for chained flat-expert tournament Hydra blocks."""

    d_embed: int = 128
    n_heads: int = 8
    n_intake_layers: int = 2
    n_blocks: int = 3
    n_experts: int = 8
    merge_schedule: tuple[int, ...] = (4, 2)
    n_expert_layers: int = 2
    n_merge_layers: int = 1
    merge_mode: str = "full"
    d_ff_ratio: float = 4.0
    max_seq_len: int = 512
    dropout: float = 0.1
    layer_norm_eps: float = 1e-5

    def __post_init__(self) -> None:
        if self.merge_mode not in ("full", "ffn", "compress"):
            raise ValueError(
                "merge_mode must be 'full', 'ffn', or 'compress'"
            )
        if min(
            self.d_embed,
            self.n_heads,
            self.n_intake_layers,
            self.n_blocks,
            self.n_experts,
            self.n_expert_layers,
            self.n_merge_layers,
            self.max_seq_len,
        ) <= 0:
            raise ValueError("model dimensions and layer counts must be positive")
        if self.d_embed % self.n_heads:
            raise ValueError("d_embed must be divisible by n_heads")
        if not isinstance(self.merge_schedule, tuple) or not self.merge_schedule:
            raise ValueError("merge_schedule must be a non-empty tuple")
        if any(
            not isinstance(group_size, int) or group_size <= 0
            for group_size in self.merge_schedule
        ):
            raise ValueError("merge_schedule entries must be positive integers")

        schedule_product = 1
        for group_size in self.merge_schedule:
            schedule_product *= group_size
            if (group_size * self.d_embed) % group_size:
                raise ValueError(
                    "merge width must be divisible by its attention heads"
                )
        if schedule_product != self.n_experts:
            raise ValueError(
                "product of merge_schedule entries must equal n_experts"
            )
        if self.d_ff_ratio <= 0:
            raise ValueError("d_ff_ratio must be positive")
        if not 0.0 <= self.dropout < 1.0:
            raise ValueError("dropout must be in [0, 1)")

    @property
    def n_rounds(self) -> int:
        """Number of scheduled tournament reduction rounds."""
        return len(self.merge_schedule)

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
class HardNegativeIndexConfig:
    """Settings for the derived static output-direction index."""

    path: Path | None = None
    rebuild: bool = False
    num_clusters: int = 512
    nprobe: int = 8
    max_candidates_per_query: int = 2048
    build_batch_size: int = 8192
    kmeans_iterations: int = 8
    vocab_chunk_size: int = 8192
    seed: int = 0

    def __post_init__(self) -> None:
        self.path = Path(self.path) if self.path is not None else None
        for name in (
            "num_clusters",
            "nprobe",
            "max_candidates_per_query",
            "build_batch_size",
            "kmeans_iterations",
            "vocab_chunk_size",
        ):
            if getattr(self, name) <= 0:
                raise ValueError(f"hard-negative index {name} must be positive")
        if self.nprobe > self.num_clusters:
            raise ValueError("hard-negative index nprobe cannot exceed num_clusters")


@dataclass
class HardNegativeDiagnosticsConfig:
    """Cadence for optional retrieval diagnostics."""

    log_interval: int = 100
    exact_recall_interval: int = 0
    exact_recall_query_count: int = 64

    def __post_init__(self) -> None:
        if min(self.log_interval, self.exact_recall_interval) < 0:
            raise ValueError("hard-negative diagnostic intervals must be non-negative")
        if self.exact_recall_query_count <= 0:
            raise ValueError("exact_recall_query_count must be positive")


@dataclass
class HardNegativeRetrievalConfig:
    """Optional auxiliary loss over statically retrieved output directions."""

    enabled: bool = False
    backend: str = "ivf"
    hard_k: int = 32
    retrieve_extra: int = 8
    query_chunk_size: int = 1024
    loss_weight: float = 0.25
    warmup_steps: int = 1000
    loss_type: str = "candidate_ce"
    pairwise_margin: float = 0.0
    normalize_directions: bool = True
    normalize_queries: bool = True
    position_fraction: float = 1.0
    max_positions_per_batch: int | None = None
    ignore_index: int = -1
    invalid_token_ids: tuple[int, ...] = ()
    index: HardNegativeIndexConfig = field(default_factory=HardNegativeIndexConfig)
    diagnostics: HardNegativeDiagnosticsConfig = field(
        default_factory=HardNegativeDiagnosticsConfig
    )

    def __post_init__(self) -> None:
        if isinstance(self.index, dict):
            self.index = HardNegativeIndexConfig(**self.index)
        if isinstance(self.diagnostics, dict):
            self.diagnostics = HardNegativeDiagnosticsConfig(**self.diagnostics)
        self.invalid_token_ids = tuple(self.invalid_token_ids)
        if self.backend not in HARD_NEGATIVE_BACKENDS:
            raise ValueError(
                f"hard-negative backend must be one of {HARD_NEGATIVE_BACKENDS}"
            )
        if self.loss_type not in HARD_NEGATIVE_LOSSES:
            raise ValueError(
                f"hard-negative loss_type must be one of {HARD_NEGATIVE_LOSSES}"
            )
        if self.hard_k <= 0 or self.query_chunk_size <= 0:
            raise ValueError("hard_k and query_chunk_size must be positive")
        if self.retrieve_extra < 0 or self.warmup_steps < 0:
            raise ValueError("retrieve_extra and warmup_steps must be non-negative")
        if self.loss_weight < 0:
            raise ValueError("hard-negative loss_weight must be non-negative")
        if not 0.0 < self.position_fraction <= 1.0:
            raise ValueError("position_fraction must be in (0, 1]")
        if (
            self.max_positions_per_batch is not None
            and self.max_positions_per_batch <= 0
        ):
            raise ValueError("max_positions_per_batch must be positive when set")


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
    hard_negative_retrieval: HardNegativeRetrievalConfig = field(
        default_factory=HardNegativeRetrievalConfig
    )

    def __post_init__(self) -> None:
        if isinstance(self.hard_negative_retrieval, dict):
            self.hard_negative_retrieval = HardNegativeRetrievalConfig(
                **self.hard_negative_retrieval
            )
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

    # Architecture packages own their config types. The experiment runner
    # resolves them through its registry instead of growing this union forever.
    model: ArchitectureConfig = field(default_factory=TransformerConfig)
    data: DataConfig = field(default_factory=DataConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    runtime: RuntimeConfig = field(default_factory=RuntimeConfig)
    embed_path: Path | None = None

    def __post_init__(self) -> None:
        self.embed_path = (
            Path(self.embed_path) if self.embed_path is not None else None
        )
