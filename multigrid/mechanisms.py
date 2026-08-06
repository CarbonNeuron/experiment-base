"""Parameter-comparable sequence mechanisms for primitive evaluation."""

from __future__ import annotations

import math
import warnings
from dataclasses import asdict, dataclass, replace
from typing import Any

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from models.base import resolve_compile_backend

from .config import MultigridMemoryConfig
from .layers import MemoryState, MultigridMemoryBlock
from .triton_linear_attention import (
    fused_linear_attention,
    triton_linear_attention_available,
)
from .triton_ssm import fused_diagonal_ssm, triton_ssm_available


MECHANISM_NAMES = ("multigrid", "softmax", "linear_attention", "ssm", "gru")


@dataclass(frozen=True)
class SymbolicModelConfig:
    """Shared shape settings for a primitive-evaluation model."""

    mechanism: str = "multigrid"
    vocab_size: int = 512
    d_model: int = 128
    n_layers: int = 4
    d_ff: int = 256
    n_heads: int = 8
    max_seq_len: int = 512
    dropout: float = 0.0
    n_cycles: int = 1
    n_memory_slots: int = 64
    d_memory: int = 64
    d_key: int = 64
    memory_addressing: str = "softmax"
    n_hash_bits: int = 32
    hash_top_k: int = 8
    use_triton_memory: bool = True
    position_scale: float = 0.02

    def __post_init__(self) -> None:
        if self.mechanism not in MECHANISM_NAMES:
            raise ValueError(f"mechanism must be one of {MECHANISM_NAMES}")
        dimensions = (
            self.vocab_size,
            self.d_model,
            self.n_layers,
            self.d_ff,
            self.n_heads,
            self.max_seq_len,
        )
        if any(value <= 0 for value in dimensions):
            raise ValueError("model dimensions must be positive")
        if self.d_model % self.n_heads:
            raise ValueError("d_model must be divisible by n_heads")
        if not 0.0 <= self.dropout < 1.0:
            raise ValueError("dropout must be in [0, 1)")
        if self.position_scale < 0.0:
            raise ValueError("position_scale must be non-negative")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _sinusoidal_positions(length: int, width: int) -> Tensor:
    positions = torch.arange(length, dtype=torch.float32).unsqueeze(1)
    frequencies = torch.exp(
        torch.arange(0, width, 2, dtype=torch.float32)
        * (-math.log(10_000.0) / width)
    )
    table = torch.zeros(length, width)
    table[:, 0::2] = torch.sin(positions * frequencies)
    odd_width = table[:, 1::2].size(1)
    table[:, 1::2] = torch.cos(positions * frequencies[:odd_width])
    return table


class _FeedForward(nn.Module):
    def __init__(self, config: SymbolicModelConfig) -> None:
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(config.d_model, config.d_ff),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.d_ff, config.d_model),
            nn.Dropout(config.dropout),
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.layers(x)


class _SoftmaxAttention(nn.Module):
    def __init__(self, config: SymbolicModelConfig) -> None:
        super().__init__()
        self.n_heads = config.n_heads
        self.head_width = config.d_model // config.n_heads
        self.qkv = nn.Linear(config.d_model, 3 * config.d_model, bias=False)
        self.output = nn.Linear(config.d_model, config.d_model, bias=False)
        self.dropout = config.dropout

    def forward(self, x: Tensor) -> Tensor:
        batch_size, seq_len, width = x.shape
        query, key, value = self.qkv(x).chunk(3, dim=-1)

        def heads(tensor: Tensor) -> Tensor:
            return tensor.reshape(
                batch_size, seq_len, self.n_heads, self.head_width
            ).transpose(1, 2)

        attended = F.scaled_dot_product_attention(
            heads(query),
            heads(key),
            heads(value),
            dropout_p=self.dropout if self.training else 0.0,
            is_causal=True,
        )
        attended = attended.transpose(1, 2).reshape(batch_size, seq_len, width)
        return self.output(attended)


class _LinearAttention(nn.Module):
    """Causal gated linear attention with a decaying key/value state."""

    def __init__(self, config: SymbolicModelConfig) -> None:
        super().__init__()
        self.n_heads = config.n_heads
        self.head_width = config.d_model // config.n_heads
        self.qkv = nn.Linear(config.d_model, 3 * config.d_model, bias=False)
        self.decay = nn.Linear(config.d_model, config.n_heads)
        self.gate = nn.Linear(config.d_model, config.d_model)
        self.output = nn.Linear(config.d_model, config.d_model, bias=False)
        self._triton_failed = False

    @torch.compiler.disable
    def _run_recurrence(
        self,
        query: Tensor,
        key: Tensor,
        value: Tensor,
        decays: Tensor,
    ) -> Tensor:
        if (
            query.is_cuda
            and not self._triton_failed
            and triton_linear_attention_available()
            and query.size(-1) <= 128
        ):
            try:
                return fused_linear_attention(
                    query, key, value, decays
                )
            except Exception as error:
                self._triton_failed = True
                warnings.warn(
                    "fused Triton linear attention failed; using the "
                    f"PyTorch recurrence: {type(error).__name__}: {error}",
                    RuntimeWarning,
                    stacklevel=2,
                )
        batch_size, _, seq_len, _ = query.shape
        state = query.new_zeros(
            batch_size, self.n_heads, self.head_width, self.head_width
        )
        normalizer = query.new_zeros(
            batch_size, self.n_heads, self.head_width
        )
        outputs: list[Tensor] = []
        for position in range(seq_len):
            decay = decays[:, :, position, None]
            key_t = key[:, :, position]
            value_t = value[:, :, position]
            state = (
                decay.unsqueeze(-1) * state
                + key_t.unsqueeze(-1) * value_t.unsqueeze(-2)
            )
            normalizer = decay * normalizer + key_t
            query_t = query[:, :, position]
            numerator = torch.matmul(query_t.unsqueeze(-2), state).squeeze(-2)
            denominator = (query_t * normalizer).sum(-1, keepdim=True)
            outputs.append(numerator / denominator.clamp_min(1.0e-6))
        return torch.stack(outputs, dim=2)

    def forward(self, x: Tensor) -> Tensor:
        batch_size, seq_len, width = x.shape
        query, key, value = self.qkv(x).chunk(3, dim=-1)

        def heads(tensor: Tensor) -> Tensor:
            return tensor.reshape(
                batch_size, seq_len, self.n_heads, self.head_width
            ).transpose(1, 2)

        query = F.elu(heads(query)) + 1.0
        key = F.elu(heads(key)) + 1.0
        value = heads(value)
        decays = torch.sigmoid(self.decay(x)).transpose(1, 2)
        attended = self._run_recurrence(
            query, key, value, decays
        ).transpose(1, 2).reshape(
            batch_size, seq_len, width
        )
        return self.output(attended * torch.sigmoid(self.gate(x)))


class _DiagonalSSM(nn.Module):
    """A stable input-dependent diagonal state-space layer."""

    def __init__(self, config: SymbolicModelConfig) -> None:
        super().__init__()
        self.input = nn.Linear(config.d_model, 2 * config.d_model)
        self.logit_decay = nn.Parameter(torch.zeros(config.d_model))
        self.output = nn.Linear(config.d_model, config.d_model)
        self._triton_failed = False

    @torch.compiler.disable
    def _run_recurrence(
        self,
        candidate: Tensor,
        gate: Tensor,
        decay: Tensor,
    ) -> Tensor:
        if (
            candidate.is_cuda
            and not self._triton_failed
            and triton_ssm_available()
        ):
            try:
                return fused_diagonal_ssm(candidate, gate, decay)
            except Exception as error:
                self._triton_failed = True
                warnings.warn(
                    "fused Triton SSM failed; using the PyTorch recurrence: "
                    f"{type(error).__name__}: {error}",
                    RuntimeWarning,
                    stacklevel=2,
                )
        state = torch.zeros_like(candidate[:, 0])
        outputs: list[Tensor] = []
        for position in range(candidate.size(1)):
            state = decay * state + (1.0 - decay) * candidate[:, position]
            outputs.append(state * torch.sigmoid(gate[:, position]))
        return torch.stack(outputs, dim=1)

    def forward(self, x: Tensor) -> Tensor:
        candidate, gate = self.input(x).chunk(2, dim=-1)
        decay = torch.sigmoid(self.logit_decay).to(x.dtype)
        return self.output(self._run_recurrence(candidate, gate, decay))


class _ResidualBlock(nn.Module):
    def __init__(
        self,
        config: SymbolicModelConfig,
        mixer: nn.Module,
    ) -> None:
        super().__init__()
        self.mixer_norm = nn.LayerNorm(config.d_model)
        self.mixer = mixer
        self.ffn_norm = nn.LayerNorm(config.d_model)
        self.ffn = _FeedForward(config)

    def forward(self, x: Tensor) -> Tensor:
        x = x + self.mixer(self.mixer_norm(x))
        return x + self.ffn(self.ffn_norm(x))


class _GRUBlock(nn.Module):
    def __init__(self, config: SymbolicModelConfig) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(config.d_model)
        self.gru = nn.GRU(config.d_model, config.d_model, batch_first=True)
        self.ffn_norm = nn.LayerNorm(config.d_model)
        self.ffn = _FeedForward(config)

    def forward(self, x: Tensor) -> Tensor:
        recurrent, _ = self.gru(self.norm(x))
        x = x + recurrent
        return x + self.ffn(self.ffn_norm(x))


class SymbolicSequenceModel(nn.Module):
    """Small-vocabulary model used to isolate a sequence mechanism."""

    def __init__(self, config: SymbolicModelConfig) -> None:
        super().__init__()
        self.config = config
        self.embedding = nn.Embedding(config.vocab_size, config.d_model)
        self.register_buffer(
            "positions",
            config.position_scale
            * _sinusoidal_positions(config.max_seq_len, config.d_model),
            persistent=False,
        )
        self.dropout = nn.Dropout(config.dropout)
        if config.mechanism == "multigrid":
            multigrid_config = MultigridMemoryConfig(
                d_model=config.d_model,
                n_layers=config.n_layers,
                n_cycles=config.n_cycles,
                d_ff=config.d_ff,
                max_seq_len=config.max_seq_len,
                dropout=config.dropout,
                n_memory_slots=config.n_memory_slots,
                d_memory=config.d_memory,
                d_key=config.d_key,
                memory_addressing=config.memory_addressing,
                n_hash_bits=config.n_hash_bits,
                hash_top_k=config.hash_top_k,
                use_triton_memory=config.use_triton_memory,
            )
            self.blocks = nn.ModuleList(
                MultigridMemoryBlock(multigrid_config)
                for _ in range(config.n_layers)
            )
        elif config.mechanism == "softmax":
            self.blocks = nn.ModuleList(
                _ResidualBlock(config, _SoftmaxAttention(config))
                for _ in range(config.n_layers)
            )
        elif config.mechanism == "linear_attention":
            self.blocks = nn.ModuleList(
                _ResidualBlock(config, _LinearAttention(config))
                for _ in range(config.n_layers)
            )
        elif config.mechanism == "ssm":
            self.blocks = nn.ModuleList(
                _ResidualBlock(config, _DiagonalSSM(config))
                for _ in range(config.n_layers)
            )
        else:
            self.blocks = nn.ModuleList(
                _GRUBlock(config) for _ in range(config.n_layers)
            )
        self.final_norm = nn.LayerNorm(config.d_model)
        self.output = nn.Linear(config.d_model, config.vocab_size, bias=False)
        self.apply(self._initialize)
        self._compiled_encode = None
        self._compile_backend: str | None = None

    @staticmethod
    def _initialize(module: nn.Module) -> None:
        if isinstance(module, (nn.Linear, nn.Embedding, nn.Conv1d)):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if getattr(module, "bias", None) is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.LayerNorm):
            nn.init.ones_(module.weight)
            nn.init.zeros_(module.bias)

    def _encode_eager(self, input_ids: Tensor) -> Tensor:
        if input_ids.ndim != 2:
            raise ValueError("input_ids must have shape [batch, sequence]")
        if input_ids.size(1) > self.config.max_seq_len:
            raise ValueError("sequence exceeds configured max_seq_len")
        hidden = self.embedding(input_ids)
        hidden = self.dropout(hidden + self.positions[: input_ids.size(1)])
        if self.config.mechanism == "multigrid":
            state: MemoryState | None = None
            for block in self.blocks:
                hidden, state = block(hidden, state)
        else:
            for block in self.blocks:
                hidden = block(hidden)
        return self.final_norm(hidden)

    def _forward_eager(self, input_ids: Tensor) -> Tensor:
        return self.output(self._encode_eager(input_ids))

    def encode(self, input_ids: Tensor) -> Tensor:
        """Return final hidden states, using the compiled encoder when enabled."""
        if self._compiled_encode is None:
            return self._encode_eager(input_ids)
        try:
            return self._compiled_encode(input_ids)
        except Exception as error:
            summary = str(error).splitlines()[0] or type(error).__name__
            warnings.warn(
                "compiled symbolic model failed; falling back to eager mode: "
                f"{type(error).__name__}: {summary}",
                RuntimeWarning,
                stacklevel=2,
            )
            self.disable_compile()
            return self._encode_eager(input_ids)

    def supervised_logits(
        self,
        input_ids: Tensor,
        mask: Tensor,
        output_start: int,
        output_count: int,
    ) -> Tensor:
        """Project only supervised positions into a contiguous output range.

        Primitive batches supervise a handful of positions and only 96 of the
        512 vocabulary items can be answers. Avoiding the otherwise dense
        [batch, time, vocabulary] projection saves work without changing the
        model parameters or checkpoint format.
        """
        hidden = self.encode(input_ids)
        if mask.shape != input_ids.shape:
            raise ValueError("supervision mask must match input_ids")
        output_end = output_start + output_count
        if output_start < 0 or output_end > self.config.vocab_size:
            raise ValueError("supervised output range exceeds vocabulary")
        return F.linear(hidden[mask], self.output.weight[output_start:output_end])

    def forward(self, input_ids: Tensor) -> Tensor:
        return self.output(self.encode(input_ids))

    def compile_model(
        self,
        *,
        mode: str = "default",
        dynamic: bool = False,
        backend: str = "auto",
    ) -> str:
        """Compile stateless work while stateful boundaries remain eager."""
        selected_backend = resolve_compile_backend(
            backend, self.embedding.weight.device.type
        )
        kwargs: dict[str, Any] = {
            "backend": selected_backend,
            "dynamic": dynamic,
        }
        if selected_backend == "inductor":
            kwargs["mode"] = mode
        self._compiled_encode = torch.compile(self._encode_eager, **kwargs)
        self._compile_backend = selected_backend
        return selected_backend

    def disable_compile(self) -> None:
        self._compiled_encode = None
        self._compile_backend = None

    def num_parameters(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters())


def matched_model_configs(
    base: SymbolicModelConfig,
    mechanisms: tuple[str, ...],
) -> dict[str, SymbolicModelConfig]:
    """Match baselines to the multigrid parameter count by adjusting FFN width."""
    reference_config = replace(base, mechanism="multigrid")
    target = SymbolicSequenceModel(reference_config).num_parameters()
    configs: dict[str, SymbolicModelConfig] = {}
    for mechanism in mechanisms:
        if mechanism == "multigrid":
            configs[mechanism] = reference_config
            continue
        probe = replace(base, mechanism=mechanism, d_ff=8)
        low_count = SymbolicSequenceModel(probe).num_parameters()
        wider = replace(probe, d_ff=9)
        per_unit = SymbolicSequenceModel(wider).num_parameters() - low_count
        width = max(8, round(8 + (target - low_count) / max(1, per_unit)))
        configs[mechanism] = replace(probe, d_ff=width)
    return configs
