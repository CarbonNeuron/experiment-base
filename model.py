"""Decoder-only transformer variants with frozen SVD token directions."""

from __future__ import annotations

import importlib.util
import warnings
from pathlib import Path
from typing import TYPE_CHECKING, Any

import torch
from torch import Tensor, nn
from torch.nn import functional as F
from svd_embeds import OpenAIEmbedding

from config import (
    ChainedHydraConfig,
    COMPILE_BACKENDS,
    CompoundQConfig,
    GrowingWidthConfig,
    HydraConfig,
    TournamentHydraConfig,
    TransformerConfig,
)

if TYPE_CHECKING:
    from output_retrieval.hard_negative_loss import HardNegativeTrainer


def _combine_hard_negative_loss(
    model: nn.Module,
    hidden: Tensor,
    targets: Tensor,
    sampled_loss: Tensor,
    trainer: "HardNegativeTrainer | None",
    hard_loss_weight: float,
) -> Tensor:
    """Add the auxiliary hard loss without changing the sampled estimator."""
    if trainer is None:
        model.last_hard_negative_metrics = None
        return sampled_loss
    embeddings = model.embeddings
    hard_loss, metrics = trainer.compute(
        hidden,
        targets,
        embeddings.rotation.matrix,
        embeddings.magnitude,
    )
    total = sampled_loss + hard_loss_weight * hard_loss
    metrics["sampled_loss"] = sampled_loss.detach()
    metrics["weighted_hard_loss"] = (hard_loss_weight * hard_loss).detach()
    metrics["hard_loss_weight"] = torch.tensor(
        hard_loss_weight, device=sampled_loss.device
    )
    metrics["total_loss"] = total.detach()
    model.last_hard_negative_metrics = metrics
    return total


def resolve_compile_backend(requested: str, device_type: str) -> str:
    """Choose a usable compiler backend before the first training batch."""
    if requested not in COMPILE_BACKENDS:
        choices = ", ".join(COMPILE_BACKENDS)
        raise ValueError(f"unknown compile backend {requested!r}; choose: {choices}")
    if requested != "auto":
        return requested
    if device_type == "cuda":
        try:
            has_triton = importlib.util.find_spec("triton") is not None
        except (ImportError, ValueError):
            has_triton = False
        if not has_triton:
            return "aot_eager"
    if device_type not in {"cpu", "cuda"}:
        return "aot_eager"
    return "inductor"


class CausalSelfAttention(nn.Module):
    """Standard multi-head causal self-attention using PyTorch SDPA."""

    def __init__(self, config: TransformerConfig) -> None:
        super().__init__()
        self.n_heads = config.n_heads
        self.head_dim = config.d_model // config.n_heads
        self.d_model = config.d_model
        self.qkv = nn.Linear(config.d_model, 3 * config.d_model, bias=False)
        self.out_proj = nn.Linear(config.d_model, config.d_model, bias=False)
        self.dropout = config.dropout
        self.resid_dropout = nn.Dropout(config.dropout)

    def forward(self, x: Tensor) -> Tensor:
        batch_size, seq_len, width = x.shape
        query, key, value = self.qkv(x).split(self.d_model, dim=-1)

        def split_heads(tensor: Tensor) -> Tensor:
            return tensor.view(
                batch_size, seq_len, self.n_heads, self.head_dim
            ).transpose(1, 2)

        query, key, value = map(split_heads, (query, key, value))
        attended = F.scaled_dot_product_attention(
            query,
            key,
            value,
            dropout_p=self.dropout if self.training else 0.0,
            is_causal=True,
        )
        attended = attended.transpose(1, 2).contiguous().view(
            batch_size, seq_len, width
        )
        return self.resid_dropout(self.out_proj(attended))


class FeedForward(nn.Module):
    def __init__(self, config: TransformerConfig) -> None:
        super().__init__()
        self.up = nn.Linear(config.d_model, config.d_ff)
        self.down = nn.Linear(config.d_ff, config.d_model)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x: Tensor) -> Tensor:
        return self.dropout(self.down(F.gelu(self.up(x))))


class TransformerBlock(nn.Module):
    """A pre-norm transformer block."""

    def __init__(self, config: TransformerConfig) -> None:
        super().__init__()
        self.attn_norm = nn.LayerNorm(
            config.d_model, eps=config.layer_norm_eps
        )
        self.attn = CausalSelfAttention(config)
        self.ffn_norm = nn.LayerNorm(
            config.d_model, eps=config.layer_norm_eps
        )
        self.ffn = FeedForward(config)

    def forward(self, x: Tensor) -> Tensor:
        x = x + self.attn(self.attn_norm(x))
        return x + self.ffn(self.ffn_norm(x))


class CompoundQAttention(nn.Module):
    """Causal attention with conjunctive and disjunctive query projections."""

    def __init__(
        self, config: TransformerConfig | CompoundQConfig
    ) -> None:
        super().__init__()
        self.n_heads = config.n_heads
        self.head_dim = config.d_model // config.n_heads
        self.d_model = config.d_model
        self.q1_proj = nn.Linear(config.d_model, config.d_model, bias=False)
        self.q2_proj = nn.Linear(config.d_model, config.d_model, bias=False)
        self.q3_proj = nn.Linear(config.d_model, config.d_model, bias=False)
        self.k_proj = nn.Linear(config.d_model, config.d_model, bias=False)
        self.v_proj = nn.Linear(config.d_model, config.d_model, bias=False)
        self.out_proj = nn.Linear(config.d_model, config.d_model, bias=False)
        self.dropout = config.dropout
        self.resid_dropout = nn.Dropout(config.dropout)

    def forward(self, x: Tensor) -> Tensor:
        batch_size, seq_len, width = x.shape
        query1 = self.q1_proj(x)
        query2 = self.q2_proj(x)
        query3 = self.q3_proj(x)
        key = self.k_proj(x)
        value = self.v_proj(x)

        def split_heads(tensor: Tensor) -> Tensor:
            return tensor.view(
                batch_size, seq_len, self.n_heads, self.head_dim
            ).transpose(1, 2)

        query1, query2, query3, key, value = map(
            split_heads, (query1, query2, query3, key, value)
        )
        query_and = torch.minimum(query1, query2)
        query_or = torch.maximum(query1, query3)
        query = query_and + query_or
        attended = F.scaled_dot_product_attention(
            query,
            key,
            value,
            dropout_p=self.dropout if self.training else 0.0,
            is_causal=True,
        )
        attended = attended.transpose(1, 2).contiguous().view(
            batch_size, seq_len, width
        )
        return self.resid_dropout(self.out_proj(attended))


class CompoundQBlock(nn.Module):
    """A pre-norm transformer block using compound query attention."""

    def __init__(self, config: CompoundQConfig) -> None:
        super().__init__()
        self.attn_norm = nn.LayerNorm(
            config.d_model, eps=config.layer_norm_eps
        )
        self.attn = CompoundQAttention(config)
        self.ffn_norm = nn.LayerNorm(
            config.d_model, eps=config.layer_norm_eps
        )
        self.ffn = FeedForward(config)

    def forward(self, x: Tensor) -> Tensor:
        x = x + self.attn(self.attn_norm(x))
        return x + self.ffn(self.ffn_norm(x))


class HydraAttention(nn.Module):
    """Width-parameterized causal self-attention for Hydra blocks."""

    def __init__(self, width: int, n_heads: int, dropout: float) -> None:
        super().__init__()
        self.n_heads = n_heads
        self.head_dim = width // n_heads
        self.d_model = width
        self.qkv = nn.Linear(width, 3 * width, bias=False)
        self.out_proj = nn.Linear(width, width, bias=False)
        self.dropout = dropout
        self.resid_dropout = nn.Dropout(dropout)

    def forward(self, x: Tensor) -> Tensor:
        batch_size, seq_len, width = x.shape
        query, key, value = self.qkv(x).split(self.d_model, dim=-1)

        def split_heads(tensor: Tensor) -> Tensor:
            return tensor.view(
                batch_size, seq_len, self.n_heads, self.head_dim
            ).transpose(1, 2)

        query, key, value = map(split_heads, (query, key, value))
        attended = F.scaled_dot_product_attention(
            query,
            key,
            value,
            dropout_p=self.dropout if self.training else 0.0,
            is_causal=True,
        )
        attended = attended.transpose(1, 2).contiguous().view(
            batch_size, seq_len, width
        )
        return self.resid_dropout(self.out_proj(attended))


class HydraFeedForward(nn.Module):
    """Width-parameterized feed-forward network for Hydra blocks."""

    def __init__(
        self, width: int, d_ff_ratio: float, dropout: float
    ) -> None:
        super().__init__()
        d_ff = int(width * d_ff_ratio)
        self.up = nn.Linear(width, d_ff)
        self.down = nn.Linear(d_ff, width)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: Tensor) -> Tensor:
        return self.dropout(self.down(F.gelu(self.up(x))))


class FFNMergeBlock(nn.Module):
    """Pre-norm FFN-only block for intra-token merge. No attention."""

    def __init__(
        self,
        width: int,
        d_ff_ratio: float,
        dropout: float,
        layer_norm_eps: float,
    ) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(width, eps=layer_norm_eps)
        self.ffn = HydraFeedForward(width, d_ff_ratio, dropout)

    def forward(self, x: Tensor) -> Tensor:
        return x + self.ffn(self.norm(x))


class CompressMergeBlock(nn.Module):
    """Attention at full width for inter-token mixing, then learned compression.

    Unlike HydraBlock which does attn+FFN at same width then relies on external
    slice, this does attention at d_wide then projects down to d_out via an
    FFN-style down-projection. The compression IS the block -- no external
    slice needed.
    """

    def __init__(
        self,
        d_wide: int,
        d_out: int,
        n_heads: int,
        d_ff_ratio: float,
        dropout: float,
        layer_norm_eps: float,
    ) -> None:
        super().__init__()
        self.attn_norm = nn.LayerNorm(d_wide, eps=layer_norm_eps)
        self.attn = HydraAttention(d_wide, n_heads, dropout)
        self.compress_norm = nn.LayerNorm(d_wide, eps=layer_norm_eps)
        d_ff = int(d_wide * d_ff_ratio)
        self.compress = nn.Sequential(
            nn.Linear(d_wide, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_out),
            nn.Dropout(dropout),
        )

    def forward(self, x: Tensor) -> Tensor:
        x = x + self.attn(self.attn_norm(x))
        return self.compress(self.compress_norm(x))


class HydraBlock(nn.Module):
    """A width-parameterized pre-norm transformer block."""

    def __init__(
        self,
        width: int,
        n_heads: int,
        d_ff_ratio: float,
        dropout: float,
        layer_norm_eps: float,
    ) -> None:
        super().__init__()
        self.attn_norm = nn.LayerNorm(width, eps=layer_norm_eps)
        self.attn = HydraAttention(width, n_heads, dropout)
        self.ffn_norm = nn.LayerNorm(width, eps=layer_norm_eps)
        self.ffn = HydraFeedForward(width, d_ff_ratio, dropout)

    def forward(self, x: Tensor) -> Tensor:
        x = x + self.attn(self.attn_norm(x))
        return x + self.ffn(self.ffn_norm(x))


class RecursiveHydraBlock(nn.Module):
    """A width-preserving Hydra block that can nest recursively."""

    def __init__(
        self,
        d_embed: int,
        n_streams: int,
        depth: int,
        n_stream_layers: int,
        n_merge_layers: int,
        n_heads: int,
        d_ff_ratio: float,
        dropout: float,
        layer_norm_eps: float,
    ) -> None:
        super().__init__()
        self.d_embed = d_embed
        self.n_streams = n_streams
        d_wide = d_embed * n_streams
        n_heads_merge = n_streams

        if depth <= 1:
            self.streams = nn.ModuleList(
                nn.ModuleList(
                    HydraBlock(
                        d_embed,
                        n_heads,
                        d_ff_ratio,
                        dropout,
                        layer_norm_eps,
                    )
                    for _ in range(n_stream_layers)
                )
                for _ in range(n_streams)
            )
            self.nested = False
        else:
            self.streams = nn.ModuleList(
                RecursiveHydraBlock(
                    d_embed=d_embed,
                    n_streams=n_streams,
                    depth=depth - 1,
                    n_stream_layers=n_stream_layers,
                    n_merge_layers=n_merge_layers,
                    n_heads=n_heads,
                    d_ff_ratio=d_ff_ratio,
                    dropout=dropout,
                    layer_norm_eps=layer_norm_eps,
                )
                for _ in range(n_streams)
            )
            self.nested = True

        self.merge_blocks = nn.ModuleList(
            HydraBlock(
                d_wide,
                n_heads_merge,
                d_ff_ratio,
                dropout,
                layer_norm_eps,
            )
            for _ in range(n_merge_layers)
        )
        self.slice_norm = nn.LayerNorm(d_embed, eps=layer_norm_eps)

    def forward(self, x: Tensor) -> Tensor:
        if self.nested:
            stream_outputs = [stream(x) for stream in self.streams]
        else:
            stream_outputs = []
            for stream_blocks in self.streams:
                hidden = x
                for block in stream_blocks:
                    hidden = block(hidden)
                stream_outputs.append(hidden)

        wide = torch.cat(stream_outputs, dim=-1)
        for block in self.merge_blocks:
            wide = block(wide)
        return self.slice_norm(wide[:, :, : self.d_embed])


class TournamentRound(nn.Module):
    """Merge adjacent stream groups of a configurable size independently."""

    def __init__(
        self,
        n_groups: int,
        group_size: int,
        d_embed: int,
        n_merge_layers: int,
        d_ff_ratio: float,
        dropout: float,
        layer_norm_eps: float,
        merge_mode: str = "full",
    ) -> None:
        super().__init__()
        d_wide = d_embed * group_size
        n_heads_merge = group_size
        self.group_size = group_size
        self.d_embed = d_embed
        self.merge_mode = merge_mode
        self.group_mergers = nn.ModuleList()
        self.group_norms = nn.ModuleList()
        for _ in range(n_groups):
            if merge_mode == "ffn":
                merger = nn.ModuleList(
                    FFNMergeBlock(
                        d_wide,
                        d_ff_ratio,
                        dropout,
                        layer_norm_eps,
                    )
                    for _ in range(n_merge_layers)
                )
            elif merge_mode == "compress":
                merger = nn.ModuleList(
                    HydraBlock(
                        d_wide,
                        n_heads_merge,
                        d_ff_ratio,
                        dropout,
                        layer_norm_eps,
                    )
                    for _ in range(n_merge_layers - 1)
                )
                merger.append(
                    CompressMergeBlock(
                        d_wide,
                        d_embed,
                        n_heads_merge,
                        d_ff_ratio,
                        dropout,
                        layer_norm_eps,
                    )
                )
            else:
                merger = nn.ModuleList(
                    HydraBlock(
                        d_wide,
                        n_heads_merge,
                        d_ff_ratio,
                        dropout,
                        layer_norm_eps,
                    )
                    for _ in range(n_merge_layers)
                )
            self.group_mergers.append(merger)
            self.group_norms.append(
                nn.LayerNorm(d_embed, eps=layer_norm_eps)
            )

    def forward(self, streams: list[Tensor]) -> list[Tensor]:
        outputs = []
        for i, (merger, norm) in enumerate(
            zip(self.group_mergers, self.group_norms)
        ):
            group_streams = streams[
                i * self.group_size : (i + 1) * self.group_size
            ]
            wide = torch.cat(group_streams, dim=-1)
            for block in merger:
                wide = block(wide)
            if self.merge_mode == "compress":
                outputs.append(norm(wide))
            else:
                outputs.append(norm(wide[:, :, : self.d_embed]))
        return outputs


class TournamentBlock(nn.Module):
    """Reduce flat independent leaf experts through scheduled merge rounds."""

    def __init__(
        self,
        d_embed: int,
        n_experts: int,
        merge_schedule: tuple[int, ...],
        n_expert_layers: int,
        n_merge_layers: int,
        n_heads: int,
        d_ff_ratio: float,
        dropout: float,
        layer_norm_eps: float,
        merge_mode: str = "full",
    ) -> None:
        super().__init__()
        self.experts = nn.ModuleList(
            nn.ModuleList(
                HydraBlock(
                    d_embed,
                    n_heads,
                    d_ff_ratio,
                    dropout,
                    layer_norm_eps,
                )
                for _ in range(n_expert_layers)
            )
            for _ in range(n_experts)
        )

        self.rounds = nn.ModuleList()
        n_remaining = n_experts
        for group_size in merge_schedule:
            n_groups = n_remaining // group_size
            self.rounds.append(
                TournamentRound(
                    n_groups=n_groups,
                    group_size=group_size,
                    d_embed=d_embed,
                    n_merge_layers=n_merge_layers,
                    d_ff_ratio=d_ff_ratio,
                    dropout=dropout,
                    layer_norm_eps=layer_norm_eps,
                    merge_mode=merge_mode,
                )
            )
            n_remaining = n_groups

    def forward(self, x: Tensor) -> Tensor:
        streams = []
        for expert_blocks in self.experts:
            hidden = x
            for block in expert_blocks:
                hidden = block(hidden)
            streams.append(hidden)

        for round_module in self.rounds:
            streams = round_module(streams)

        return streams[0]


class GenericTransformer(nn.Module):
    """GPT-style LM with cosine-preserving SVD token adaptation.

    The effective tied input/output embedding table is::

        E = orthogonal_rotation(global_magnitude * directions)

    ``directions`` is a frozen buffer. One positive global magnitude and an
    identity-initialized orthogonal rotation are learned, preserving all token
    cosine similarities. Learned absolute position vectors are initialized so
    their mean L2 norm equals the source table's mean L2 norm.
    """

    def __init__(
        self,
        config: TransformerConfig | dict[str, Any] | None = None,
        embed_path: str | Path | None = None,
    ) -> None:
        super().__init__()
        if config is None:
            config = TransformerConfig()
        elif isinstance(config, dict):
            config = TransformerConfig(**config)
        self.config = config

        self.embedding_dropout = nn.Dropout(config.dropout)
        self.blocks = nn.ModuleList(
            TransformerBlock(config) for _ in range(config.n_layers)
        )
        self.final_norm = nn.LayerNorm(
            config.d_model, eps=config.layer_norm_eps
        )

        self.apply(self._init_weights)
        embedding_kwargs = (
            {"embedding_path": embed_path} if embed_path is not None else {}
        )
        self.embeddings = OpenAIEmbedding(
            config.d_model,
            max_seq_len=config.max_seq_len,
            **embedding_kwargs,
        )
        self._compiled_encoder = None
        self._compile_backend: str | None = None
        self.last_hard_negative_metrics: dict[str, Tensor] | None = None

    @staticmethod
    def _init_weights(module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, nn.LayerNorm):
            nn.init.ones_(module.weight)
            nn.init.zeros_(module.bias)

    def effective_embeddings(self) -> Tensor:
        """Return the tied, trainable embedding table."""
        return self.embeddings.weight

    @property
    def vocab_size(self) -> int:
        """Vocabulary size supplied by the embedding artifact."""
        return self.embeddings.num_embeddings

    def _encode_hidden(self, input_ids: Tensor) -> Tensor:
        hidden = self.embedding_dropout(self.embeddings(input_ids))
        for block in self.blocks:
            hidden = block(hidden)
        return self.final_norm(hidden)

    def encode(self, input_ids: Tensor) -> Tensor:
        """Return final hidden states without materializing vocabulary logits."""
        if input_ids.ndim != 2:
            raise ValueError("input_ids must have shape [batch, sequence]")
        seq_len = input_ids.size(1)
        if seq_len > self.config.max_seq_len:
            raise ValueError(
                f"sequence length {seq_len} exceeds max_seq_len="
                f"{self.config.max_seq_len}"
            )

        if self._compiled_encoder is None:
            return self._encode_hidden(input_ids)
        try:
            return self._compiled_encoder(input_ids)
        except Exception as error:
            summary = str(error).splitlines()[0] or type(error).__name__
            warnings.warn(
                "compiled encoder failed; falling back to eager mode: "
                f"{type(error).__name__}: {summary}",
                RuntimeWarning,
                stacklevel=2,
            )
            self._compiled_encoder = None
            self._compile_backend = None
            return self._encode_hidden(input_ids)

    def compile_encoder(
        self,
        *,
        mode: str = "default",
        dynamic: bool = False,
        backend: str = "auto",
    ) -> str:
        """Compile the hidden-state path while keeping chunked loss eager.

        Compilation is lazy. If the selected backend fails on its first real
        batch, :meth:`encode` warns once and transparently returns to eager
        execution.
        """
        selected_backend = resolve_compile_backend(
            backend, self.embeddings.directions.device.type
        )
        kwargs: dict[str, Any] = {
            "backend": selected_backend,
            "dynamic": dynamic,
        }
        if selected_backend == "inductor":
            kwargs["mode"] = mode
        self._compiled_encoder = torch.compile(self._encode_hidden, **kwargs)
        self._compile_backend = selected_backend
        return selected_backend

    def disable_compile(self) -> None:
        """Return the encoder to eager execution."""
        self._compiled_encoder = None
        self._compile_backend = None

    def logits(self, hidden: Tensor) -> Tensor:
        """Project hidden states with the tied effective embedding table."""
        return self.embeddings.project(hidden)

    def transform_hidden_to_fixed_space(self, hidden: Tensor) -> Tensor:
        """Return row-batch queries ``hidden @ Q`` for frozen directions."""
        return hidden @ self.embeddings.rotation.matrix

    def forward(
        self,
        input_ids: Tensor,
        targets: Tensor | None = None,
        *,
        loss_chunk_size: int = 0,
        loss_backend: str = "tiled",
        loss_negative_samples: int = 4096,
        hard_negative_trainer: "HardNegativeTrainer | None" = None,
        hard_loss_weight: float = 0.0,
    ) -> tuple[Tensor | None, Tensor | None]:
        hidden = self.encode(input_ids)
        loss = None
        if targets is not None:
            if targets.shape != input_ids.shape:
                raise ValueError("targets must have the same shape as input_ids")
            if loss_chunk_size > 0:
                sampled_loss = self.embeddings.cross_entropy(
                    hidden,
                    targets,
                    chunk_size=loss_chunk_size,
                    backend=loss_backend,
                    num_negative_samples=loss_negative_samples,
                )
                loss = _combine_hard_negative_loss(
                    self,
                    hidden,
                    targets,
                    sampled_loss,
                    hard_negative_trainer,
                    hard_loss_weight,
                )
                return None, loss

        logits = self.embeddings.project(hidden)
        if targets is not None:
            loss = F.cross_entropy(
                logits.reshape(-1, logits.size(-1)),
                targets.reshape(-1),
                ignore_index=-1,
            )
        return logits, loss

    def num_parameters(self, trainable_only: bool = False) -> int:
        parameters = sum(
            parameter.numel()
            for parameter in self.parameters()
            if not trainable_only or parameter.requires_grad
        )
        if not trainable_only:
            parameters += self.embeddings.directions.numel()
        return parameters


class CompoundQTransformer(GenericTransformer):
    """GPT-style LM whose blocks use three compound query projections."""

    def __init__(
        self,
        config: CompoundQConfig | dict[str, Any] | None = None,
        embed_path: str | Path | None = None,
    ) -> None:
        nn.Module.__init__(self)
        if config is None:
            config = CompoundQConfig()
        elif isinstance(config, dict):
            config = CompoundQConfig(**config)
        self.config = config

        self.embedding_dropout = nn.Dropout(config.dropout)
        self.blocks = nn.ModuleList(
            CompoundQBlock(config) for _ in range(config.n_layers)
        )
        self.final_norm = nn.LayerNorm(
            config.d_model, eps=config.layer_norm_eps
        )

        self.apply(self._init_weights)
        embedding_kwargs = (
            {"embedding_path": embed_path} if embed_path is not None else {}
        )
        self.embeddings = OpenAIEmbedding(
            config.d_model,
            max_seq_len=config.max_seq_len,
            **embedding_kwargs,
        )
        self._compiled_encoder = None
        self._compile_backend: str | None = None
        self.last_hard_negative_metrics: dict[str, Tensor] | None = None


class ChainedHydraTransformer(nn.Module):
    """Transformer with chained, recursively nested Hydra blocks."""

    def __init__(
        self,
        config: ChainedHydraConfig,
        embed_path: str | Path | None = None,
    ) -> None:
        super().__init__()
        self.config = config

        embedding_kwargs = (
            {"embedding_path": embed_path} if embed_path is not None else {}
        )
        self.embeddings = OpenAIEmbedding(
            config.d_embed,
            max_seq_len=config.max_seq_len,
            **embedding_kwargs,
        )
        self.embedding_dropout = nn.Dropout(config.dropout)

        self.intake_blocks = nn.ModuleList(
            HydraBlock(
                config.d_embed,
                config.n_heads,
                config.d_ff_ratio,
                config.dropout,
                config.layer_norm_eps,
            )
            for _ in range(config.n_intake_layers)
        )
        self.intake_norm = nn.LayerNorm(
            config.d_embed, eps=config.layer_norm_eps
        )
        self.hydra_blocks = nn.ModuleList(
            RecursiveHydraBlock(
                d_embed=config.d_embed,
                n_streams=config.n_streams,
                depth=config.depth,
                n_stream_layers=config.n_stream_layers,
                n_merge_layers=config.n_merge_layers,
                n_heads=config.n_heads,
                d_ff_ratio=config.d_ff_ratio,
                dropout=config.dropout,
                layer_norm_eps=config.layer_norm_eps,
            )
            for _ in range(config.n_blocks)
        )
        self.final_norm = nn.LayerNorm(
            config.d_embed, eps=config.layer_norm_eps
        )

        self.apply(self._init_weights)
        self._compiled_encoder = None
        self._compile_backend: str | None = None
        self.last_hard_negative_metrics: dict[str, Tensor] | None = None

    @staticmethod
    def _init_weights(module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, nn.LayerNorm):
            nn.init.ones_(module.weight)
            nn.init.zeros_(module.bias)

    def effective_embeddings(self) -> Tensor:
        """Return the tied, trainable embedding table."""
        return self.embeddings.weight

    @property
    def vocab_size(self) -> int:
        """Vocabulary size supplied by the embedding artifact."""
        return self.embeddings.num_embeddings

    def _encode_hidden(self, input_ids: Tensor) -> Tensor:
        hidden = self.embedding_dropout(self.embeddings(input_ids))

        for block in self.intake_blocks:
            hidden = block(hidden)
        hidden = self.intake_norm(hidden)

        for hydra_block in self.hydra_blocks:
            hidden = hydra_block(hidden)

        return self.final_norm(hidden)

    def encode(self, input_ids: Tensor) -> Tensor:
        """Return final hidden states without materializing vocabulary logits."""
        if input_ids.ndim != 2:
            raise ValueError("input_ids must have shape [batch, sequence]")
        seq_len = input_ids.size(1)
        if seq_len > self.config.max_seq_len:
            raise ValueError(
                f"sequence length {seq_len} exceeds max_seq_len="
                f"{self.config.max_seq_len}"
            )

        if self._compiled_encoder is None:
            return self._encode_hidden(input_ids)
        try:
            return self._compiled_encoder(input_ids)
        except Exception as error:
            summary = str(error).splitlines()[0] or type(error).__name__
            warnings.warn(
                "compiled encoder failed; falling back to eager mode: "
                f"{type(error).__name__}: {summary}",
                RuntimeWarning,
                stacklevel=2,
            )
            self._compiled_encoder = None
            self._compile_backend = None
            return self._encode_hidden(input_ids)

    def compile_encoder(
        self,
        *,
        mode: str = "default",
        dynamic: bool = False,
        backend: str = "auto",
    ) -> str:
        """Compile the hidden-state path while keeping chunked loss eager.

        Compilation is lazy. If the selected backend fails on its first real
        batch, :meth:`encode` warns once and transparently returns to eager
        execution.
        """
        selected_backend = resolve_compile_backend(
            backend, self.embeddings.directions.device.type
        )
        kwargs: dict[str, Any] = {
            "backend": selected_backend,
            "dynamic": dynamic,
        }
        if selected_backend == "inductor":
            kwargs["mode"] = mode
        self._compiled_encoder = torch.compile(self._encode_hidden, **kwargs)
        self._compile_backend = selected_backend
        return selected_backend

    def disable_compile(self) -> None:
        """Return the encoder to eager execution."""
        self._compiled_encoder = None
        self._compile_backend = None

    def logits(self, hidden: Tensor) -> Tensor:
        """Project hidden states with the tied effective embedding table."""
        return self.embeddings.project(hidden)

    def transform_hidden_to_fixed_space(self, hidden: Tensor) -> Tensor:
        """Return row-batch queries ``hidden @ Q`` for frozen directions."""
        return hidden @ self.embeddings.rotation.matrix

    def forward(
        self,
        input_ids: Tensor,
        targets: Tensor | None = None,
        *,
        loss_chunk_size: int = 0,
        loss_backend: str = "tiled",
        loss_negative_samples: int = 4096,
        hard_negative_trainer: "HardNegativeTrainer | None" = None,
        hard_loss_weight: float = 0.0,
    ) -> tuple[Tensor | None, Tensor | None]:
        hidden = self.encode(input_ids)
        loss = None
        if targets is not None:
            if targets.shape != input_ids.shape:
                raise ValueError("targets must have the same shape as input_ids")
            if loss_chunk_size > 0:
                sampled_loss = self.embeddings.cross_entropy(
                    hidden,
                    targets,
                    chunk_size=loss_chunk_size,
                    backend=loss_backend,
                    num_negative_samples=loss_negative_samples,
                )
                loss = _combine_hard_negative_loss(
                    self,
                    hidden,
                    targets,
                    sampled_loss,
                    hard_negative_trainer,
                    hard_loss_weight,
                )
                return None, loss

        logits = self.embeddings.project(hidden)
        if targets is not None:
            loss = F.cross_entropy(
                logits.reshape(-1, logits.size(-1)),
                targets.reshape(-1),
                ignore_index=-1,
            )
        return logits, loss

    def num_parameters(self, trainable_only: bool = False) -> int:
        parameters = sum(
            parameter.numel()
            for parameter in self.parameters()
            if not trainable_only or parameter.requires_grad
        )
        if not trainable_only:
            parameters += self.embeddings.directions.numel()
        return parameters


class TournamentHydraTransformer(nn.Module):
    """Transformer with chained tournament Hydra blocks."""

    def __init__(
        self,
        config: TournamentHydraConfig,
        embed_path: str | Path | None = None,
    ) -> None:
        super().__init__()
        self.config = config

        embedding_kwargs = (
            {"embedding_path": embed_path} if embed_path is not None else {}
        )
        self.embeddings = OpenAIEmbedding(
            config.d_embed,
            max_seq_len=config.max_seq_len,
            **embedding_kwargs,
        )
        self.embedding_dropout = nn.Dropout(config.dropout)

        self.intake_blocks = nn.ModuleList(
            HydraBlock(
                config.d_embed,
                config.n_heads,
                config.d_ff_ratio,
                config.dropout,
                config.layer_norm_eps,
            )
            for _ in range(config.n_intake_layers)
        )
        self.intake_norm = nn.LayerNorm(
            config.d_embed, eps=config.layer_norm_eps
        )
        self.tournament_blocks = nn.ModuleList(
            TournamentBlock(
                d_embed=config.d_embed,
                n_experts=config.n_experts,
                merge_schedule=config.merge_schedule,
                n_expert_layers=config.n_expert_layers,
                n_merge_layers=config.n_merge_layers,
                n_heads=config.n_heads,
                d_ff_ratio=config.d_ff_ratio,
                dropout=config.dropout,
                layer_norm_eps=config.layer_norm_eps,
                merge_mode=config.merge_mode,
            )
            for _ in range(config.n_blocks)
        )
        self.final_norm = nn.LayerNorm(
            config.d_embed, eps=config.layer_norm_eps
        )

        self.apply(self._init_weights)
        self._compiled_encoder = None
        self._compile_backend: str | None = None
        self.last_hard_negative_metrics: dict[str, Tensor] | None = None

    @staticmethod
    def _init_weights(module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, nn.LayerNorm):
            nn.init.ones_(module.weight)
            nn.init.zeros_(module.bias)

    def effective_embeddings(self) -> Tensor:
        """Return the tied, trainable embedding table."""
        return self.embeddings.weight

    @property
    def vocab_size(self) -> int:
        """Vocabulary size supplied by the embedding artifact."""
        return self.embeddings.num_embeddings

    def _encode_hidden(self, input_ids: Tensor) -> Tensor:
        hidden = self.embedding_dropout(self.embeddings(input_ids))

        for block in self.intake_blocks:
            hidden = block(hidden)
        hidden = self.intake_norm(hidden)

        for tournament_block in self.tournament_blocks:
            hidden = tournament_block(hidden)

        return self.final_norm(hidden)

    def encode(self, input_ids: Tensor) -> Tensor:
        """Return final hidden states without materializing vocabulary logits."""
        if input_ids.ndim != 2:
            raise ValueError("input_ids must have shape [batch, sequence]")
        seq_len = input_ids.size(1)
        if seq_len > self.config.max_seq_len:
            raise ValueError(
                f"sequence length {seq_len} exceeds max_seq_len="
                f"{self.config.max_seq_len}"
            )

        if self._compiled_encoder is None:
            return self._encode_hidden(input_ids)
        try:
            return self._compiled_encoder(input_ids)
        except Exception as error:
            summary = str(error).splitlines()[0] or type(error).__name__
            warnings.warn(
                "compiled encoder failed; falling back to eager mode: "
                f"{type(error).__name__}: {summary}",
                RuntimeWarning,
                stacklevel=2,
            )
            self._compiled_encoder = None
            self._compile_backend = None
            return self._encode_hidden(input_ids)

    def compile_encoder(
        self,
        *,
        mode: str = "default",
        dynamic: bool = False,
        backend: str = "auto",
    ) -> str:
        """Compile the hidden-state path while keeping chunked loss eager.

        Compilation is lazy. If the selected backend fails on its first real
        batch, :meth:`encode` warns once and transparently returns to eager
        execution.
        """
        selected_backend = resolve_compile_backend(
            backend, self.embeddings.directions.device.type
        )
        kwargs: dict[str, Any] = {
            "backend": selected_backend,
            "dynamic": dynamic,
        }
        if selected_backend == "inductor":
            kwargs["mode"] = mode
        self._compiled_encoder = torch.compile(self._encode_hidden, **kwargs)
        self._compile_backend = selected_backend
        return selected_backend

    def disable_compile(self) -> None:
        """Return the encoder to eager execution."""
        self._compiled_encoder = None
        self._compile_backend = None

    def logits(self, hidden: Tensor) -> Tensor:
        """Project hidden states with the tied effective embedding table."""
        return self.embeddings.project(hidden)

    def transform_hidden_to_fixed_space(self, hidden: Tensor) -> Tensor:
        """Return row-batch queries ``hidden @ Q`` for frozen directions."""
        return hidden @ self.embeddings.rotation.matrix

    def forward(
        self,
        input_ids: Tensor,
        targets: Tensor | None = None,
        *,
        loss_chunk_size: int = 0,
        loss_backend: str = "tiled",
        loss_negative_samples: int = 4096,
        hard_negative_trainer: "HardNegativeTrainer | None" = None,
        hard_loss_weight: float = 0.0,
    ) -> tuple[Tensor | None, Tensor | None]:
        hidden = self.encode(input_ids)
        loss = None
        if targets is not None:
            if targets.shape != input_ids.shape:
                raise ValueError("targets must have the same shape as input_ids")
            if loss_chunk_size > 0:
                sampled_loss = self.embeddings.cross_entropy(
                    hidden,
                    targets,
                    chunk_size=loss_chunk_size,
                    backend=loss_backend,
                    num_negative_samples=loss_negative_samples,
                )
                loss = _combine_hard_negative_loss(
                    self,
                    hidden,
                    targets,
                    sampled_loss,
                    hard_negative_trainer,
                    hard_loss_weight,
                )
                return None, loss

        logits = self.embeddings.project(hidden)
        if targets is not None:
            loss = F.cross_entropy(
                logits.reshape(-1, logits.size(-1)),
                targets.reshape(-1),
                ignore_index=-1,
            )
        return logits, loss

    def num_parameters(self, trainable_only: bool = False) -> int:
        parameters = sum(
            parameter.numel()
            for parameter in self.parameters()
            if not trainable_only or parameter.requires_grad
        )
        if not trainable_only:
            parameters += self.embeddings.directions.numel()
        return parameters


class HydraTransformer(nn.Module):
    """Three-phase decoder with shared intake, parallel streams, and merge."""

    def __init__(
        self,
        config: HydraConfig,
        embed_path: str | Path | None = None,
    ) -> None:
        super().__init__()
        self.config = config

        embedding_kwargs = (
            {"embedding_path": embed_path} if embed_path is not None else {}
        )
        self.embeddings = OpenAIEmbedding(
            config.d_embed,
            max_seq_len=config.max_seq_len,
            **embedding_kwargs,
        )
        self.embedding_dropout = nn.Dropout(config.dropout)

        self.intake_blocks = nn.ModuleList(
            HydraBlock(
                config.d_embed,
                config.n_heads_per_stream,
                config.d_ff_ratio,
                config.dropout,
                config.layer_norm_eps,
            )
            for _ in range(config.n_intake_layers)
        )
        self.intake_norm = nn.LayerNorm(
            config.d_embed, eps=config.layer_norm_eps
        )
        self.streams = nn.ModuleList(
            nn.ModuleList(
                HydraBlock(
                    config.d_embed,
                    config.n_heads_per_stream,
                    config.d_ff_ratio,
                    config.dropout,
                    config.layer_norm_eps,
                )
                for _ in range(config.n_stream_layers)
            )
            for _ in range(config.n_streams)
        )
        self.merge_blocks = nn.ModuleList(
            HydraBlock(
                config.d_wide,
                config.n_heads_wide,
                config.d_ff_ratio,
                config.dropout,
                config.layer_norm_eps,
            )
            for _ in range(config.n_merge_layers)
        )
        self.final_norm = nn.LayerNorm(
            config.d_embed, eps=config.layer_norm_eps
        )

        self.apply(self._init_weights)
        self._compiled_encoder = None
        self._compile_backend: str | None = None
        self.last_hard_negative_metrics: dict[str, Tensor] | None = None

    @staticmethod
    def _init_weights(module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, nn.LayerNorm):
            nn.init.ones_(module.weight)
            nn.init.zeros_(module.bias)

    def effective_embeddings(self) -> Tensor:
        """Return the tied, trainable embedding table."""
        return self.embeddings.weight

    @property
    def vocab_size(self) -> int:
        """Vocabulary size supplied by the embedding artifact."""
        return self.embeddings.num_embeddings

    def _encode_hidden(self, input_ids: Tensor) -> Tensor:
        hidden = self.embedding_dropout(self.embeddings(input_ids))

        for block in self.intake_blocks:
            hidden = block(hidden)
        hidden = self.intake_norm(hidden)

        stream_outputs = []
        for stream_blocks in self.streams:
            stream_hidden = hidden
            for block in stream_blocks:
                stream_hidden = block(stream_hidden)
            stream_outputs.append(stream_hidden)

        wide = torch.cat(stream_outputs, dim=-1)
        for block in self.merge_blocks:
            wide = block(wide)

        output = wide[:, :, : self.config.d_embed]
        return self.final_norm(output)

    def encode(self, input_ids: Tensor) -> Tensor:
        """Return final hidden states without materializing vocabulary logits."""
        if input_ids.ndim != 2:
            raise ValueError("input_ids must have shape [batch, sequence]")
        seq_len = input_ids.size(1)
        if seq_len > self.config.max_seq_len:
            raise ValueError(
                f"sequence length {seq_len} exceeds max_seq_len="
                f"{self.config.max_seq_len}"
            )

        if self._compiled_encoder is None:
            return self._encode_hidden(input_ids)
        try:
            return self._compiled_encoder(input_ids)
        except Exception as error:
            summary = str(error).splitlines()[0] or type(error).__name__
            warnings.warn(
                "compiled encoder failed; falling back to eager mode: "
                f"{type(error).__name__}: {summary}",
                RuntimeWarning,
                stacklevel=2,
            )
            self._compiled_encoder = None
            self._compile_backend = None
            return self._encode_hidden(input_ids)

    def compile_encoder(
        self,
        *,
        mode: str = "default",
        dynamic: bool = False,
        backend: str = "auto",
    ) -> str:
        """Compile the hidden-state path while keeping chunked loss eager.

        Compilation is lazy. If the selected backend fails on its first real
        batch, :meth:`encode` warns once and transparently returns to eager
        execution.
        """
        selected_backend = resolve_compile_backend(
            backend, self.embeddings.directions.device.type
        )
        kwargs: dict[str, Any] = {
            "backend": selected_backend,
            "dynamic": dynamic,
        }
        if selected_backend == "inductor":
            kwargs["mode"] = mode
        self._compiled_encoder = torch.compile(self._encode_hidden, **kwargs)
        self._compile_backend = selected_backend
        return selected_backend

    def disable_compile(self) -> None:
        """Return the encoder to eager execution."""
        self._compiled_encoder = None
        self._compile_backend = None

    def logits(self, hidden: Tensor) -> Tensor:
        """Project hidden states with the tied effective embedding table."""
        return self.embeddings.project(hidden)

    def transform_hidden_to_fixed_space(self, hidden: Tensor) -> Tensor:
        """Return row-batch queries ``hidden @ Q`` for frozen directions."""
        return hidden @ self.embeddings.rotation.matrix

    def forward(
        self,
        input_ids: Tensor,
        targets: Tensor | None = None,
        *,
        loss_chunk_size: int = 0,
        loss_backend: str = "tiled",
        loss_negative_samples: int = 4096,
        hard_negative_trainer: "HardNegativeTrainer | None" = None,
        hard_loss_weight: float = 0.0,
    ) -> tuple[Tensor | None, Tensor | None]:
        hidden = self.encode(input_ids)
        loss = None
        if targets is not None:
            if targets.shape != input_ids.shape:
                raise ValueError("targets must have the same shape as input_ids")
            if loss_chunk_size > 0:
                sampled_loss = self.embeddings.cross_entropy(
                    hidden,
                    targets,
                    chunk_size=loss_chunk_size,
                    backend=loss_backend,
                    num_negative_samples=loss_negative_samples,
                )
                loss = _combine_hard_negative_loss(
                    self,
                    hidden,
                    targets,
                    sampled_loss,
                    hard_negative_trainer,
                    hard_loss_weight,
                )
                return None, loss

        logits = self.embeddings.project(hidden)
        if targets is not None:
            loss = F.cross_entropy(
                logits.reshape(-1, logits.size(-1)),
                targets.reshape(-1),
                ignore_index=-1,
            )
        return logits, loss

    def num_parameters(self, trainable_only: bool = False) -> int:
        parameters = sum(
            parameter.numel()
            for parameter in self.parameters()
            if not trainable_only or parameter.requires_grad
        )
        if not trainable_only:
            parameters += self.embeddings.directions.numel()
        return parameters


class ScratchBlock(nn.Module):
    """Read the full stream and produce a fixed-width block output."""

    def __init__(self, layer_index: int, config: GrowingWidthConfig) -> None:
        super().__init__()
        self.layer_index = layer_index
        stream_width = config.d_embed * (
            min(layer_index, config.embed_cutoff_layer) + 1
        )
        self.input_proj = nn.Linear(
            stream_width, config.d_embed, bias=False
        )

        internal_config = TransformerConfig(
            d_model=config.d_embed,
            n_heads=config.n_heads,
            n_layers=1,
            d_ff=round(config.d_ff_ratio * config.d_embed),
            max_seq_len=config.max_seq_len,
            dropout=config.dropout,
            layer_norm_eps=config.layer_norm_eps,
        )
        self.attn_norm = nn.LayerNorm(
            config.d_embed, eps=config.layer_norm_eps
        )
        self.attn = CausalSelfAttention(internal_config)
        self.ffn_norm = nn.LayerNorm(
            config.d_embed, eps=config.layer_norm_eps
        )
        self.ffn = FeedForward(internal_config)
        self.embed_write = (
            nn.Linear(config.d_embed, config.d_embed, bias=False)
            if layer_index >= config.embed_cutoff_layer
            else None
        )

    def forward(self, stream: Tensor) -> Tensor:
        hidden = self.input_proj(stream)
        hidden = hidden + self.attn(self.attn_norm(hidden))
        return hidden + self.ffn(self.ffn_norm(hidden))


class GrowingWidthTransformer(nn.Module):
    """Decoder with a growing-then-frozen, fixed-chunk residual stream.

    Pre-cutoff blocks append one ``d_embed``-wide scratch chunk. Post-cutoff
    blocks read that frozen stream and accumulate learned reconstruction
    writes in the embedding channels without appending further chunks.
    """

    def __init__(
        self,
        config: GrowingWidthConfig | dict[str, Any] | None = None,
        embed_path: str | Path | None = None,
    ) -> None:
        super().__init__()
        if config is None:
            config = GrowingWidthConfig()
        elif isinstance(config, dict):
            config = GrowingWidthConfig(**config)
        self.config = config

        self.embedding_dropout = nn.Dropout(config.dropout)
        self.blocks = nn.ModuleList(
            ScratchBlock(layer_index, config)
            for layer_index in range(config.n_layers)
        )
        self.final_norm = nn.LayerNorm(
            config.d_embed, eps=config.layer_norm_eps
        )

        self.apply(GenericTransformer._init_weights)
        embedding_kwargs = (
            {"embedding_path": embed_path} if embed_path is not None else {}
        )
        self.embeddings = OpenAIEmbedding(
            config.d_embed,
            max_seq_len=config.max_seq_len,
            **embedding_kwargs,
        )
        self._compiled_encoder = None
        self._compile_backend: str | None = None
        self.last_hard_negative_metrics: dict[str, Tensor] | None = None

    @property
    def layer_widths(self) -> list[int]:
        """Full stream widths read by successive blocks."""
        return [block.input_proj.in_features for block in self.blocks]

    def embed_decay(self, layer_index: int) -> float:
        """Return the original embedding signal retained at one layer."""
        cutoff = self.config.embed_cutoff_layer
        if layer_index < 0 or layer_index >= self.config.n_layers:
            raise IndexError("layer index out of range")
        if cutoff == 0 or layer_index >= cutoff:
            return 0.0
        return 1.0 - layer_index / cutoff

    def effective_embeddings(self) -> Tensor:
        """Return the tied, trainable embedding table."""
        return self.embeddings.weight

    @property
    def vocab_size(self) -> int:
        """Vocabulary size supplied by the embedding artifact."""
        return self.embeddings.num_embeddings

    def _encode_hidden(self, input_ids: Tensor) -> Tensor:
        embedded = self.embedding_dropout(self.embeddings(input_ids))
        embed_updates = torch.zeros_like(embedded)
        scratch_chunks: list[Tensor] = []

        for layer_index, block in enumerate(self.blocks):
            # Track the decaying source separately so learned late-layer writes
            # persist instead of being multiplied by the source decay again.
            embed_channels = (
                embedded * self.embed_decay(layer_index) + embed_updates
            )
            stream = torch.cat((embed_channels, *scratch_chunks), dim=-1)
            scratch = block(stream)
            if layer_index < self.config.embed_cutoff_layer:
                scratch_chunks.append(scratch)
            else:
                assert block.embed_write is not None
                embed_updates = embed_updates + block.embed_write(scratch)

        final_embed = (
            embedded * self.embed_decay(self.config.n_layers - 1)
            + embed_updates
        )
        return self.final_norm(final_embed)

    def encode(self, input_ids: Tensor) -> Tensor:
        """Return final hidden states in the tied embedding space."""
        if input_ids.ndim != 2:
            raise ValueError("input_ids must have shape [batch, sequence]")
        seq_len = input_ids.size(1)
        if seq_len > self.config.max_seq_len:
            raise ValueError(
                f"sequence length {seq_len} exceeds max_seq_len="
                f"{self.config.max_seq_len}"
            )

        if self._compiled_encoder is None:
            return self._encode_hidden(input_ids)
        try:
            return self._compiled_encoder(input_ids)
        except Exception as error:
            summary = str(error).splitlines()[0] or type(error).__name__
            warnings.warn(
                "compiled encoder failed; falling back to eager mode: "
                f"{type(error).__name__}: {summary}",
                RuntimeWarning,
                stacklevel=2,
            )
            self.disable_compile()
            return self._encode_hidden(input_ids)

    def compile_encoder(
        self,
        *,
        mode: str = "default",
        dynamic: bool = False,
        backend: str = "auto",
    ) -> str:
        """Compile the hidden-state path while keeping embedding loss eager."""
        selected_backend = resolve_compile_backend(
            backend, self.embeddings.directions.device.type
        )
        kwargs: dict[str, Any] = {
            "backend": selected_backend,
            "dynamic": dynamic,
        }
        if selected_backend == "inductor":
            kwargs["mode"] = mode
        self._compiled_encoder = torch.compile(self._encode_hidden, **kwargs)
        self._compile_backend = selected_backend
        return selected_backend

    def disable_compile(self) -> None:
        """Return the encoder to eager execution."""
        self._compiled_encoder = None
        self._compile_backend = None

    def logits(self, hidden: Tensor) -> Tensor:
        """Project embedding-space hidden states to tied token logits."""
        return self.embeddings.project(hidden)

    def transform_hidden_to_fixed_space(self, hidden: Tensor) -> Tensor:
        """Return row-batch queries ``hidden @ Q`` for frozen directions."""
        return hidden @ self.embeddings.rotation.matrix

    def forward(
        self,
        input_ids: Tensor,
        targets: Tensor | None = None,
        *,
        loss_chunk_size: int = 0,
        loss_backend: str = "tiled",
        loss_negative_samples: int = 4096,
        hard_negative_trainer: "HardNegativeTrainer | None" = None,
        hard_loss_weight: float = 0.0,
    ) -> tuple[Tensor | None, Tensor | None]:
        hidden = self.encode(input_ids)
        loss = None
        if targets is not None:
            if targets.shape != input_ids.shape:
                raise ValueError("targets must have the same shape as input_ids")
            if loss_chunk_size > 0:
                sampled_loss = self.embeddings.cross_entropy(
                    hidden,
                    targets,
                    chunk_size=loss_chunk_size,
                    backend=loss_backend,
                    num_negative_samples=loss_negative_samples,
                )
                loss = _combine_hard_negative_loss(
                    self,
                    hidden,
                    targets,
                    sampled_loss,
                    hard_negative_trainer,
                    hard_loss_weight,
                )
                return None, loss

        logits = self.embeddings.project(hidden)
        if targets is not None:
            loss = F.cross_entropy(
                logits.reshape(-1, logits.size(-1)),
                targets.reshape(-1),
                ignore_index=-1,
            )
        return logits, loss

    def num_parameters(self, trainable_only: bool = False) -> int:
        """Count parameters, including growing stream input projections."""
        parameters = sum(
            parameter.numel()
            for parameter in self.parameters()
            if not trainable_only or parameter.requires_grad
        )
        if not trainable_only:
            parameters += self.embeddings.directions.numel()
        return parameters
