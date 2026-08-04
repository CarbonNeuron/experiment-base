"""Decoder-only transformer variants with frozen SVD token directions."""

from __future__ import annotations

import importlib.util
import warnings
from pathlib import Path
from typing import Any

import torch
from torch import Tensor, nn
from torch.nn import functional as F
from svd_embeds import OpenAIEmbedding

from config import COMPILE_BACKENDS, GrowingWidthConfig, TransformerConfig


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

    def forward(
        self,
        input_ids: Tensor,
        targets: Tensor | None = None,
        *,
        loss_chunk_size: int = 0,
        loss_backend: str = "tiled",
        loss_negative_samples: int = 4096,
    ) -> tuple[Tensor | None, Tensor | None]:
        hidden = self.encode(input_ids)
        loss = None
        if targets is not None:
            if targets.shape != input_ids.shape:
                raise ValueError("targets must have the same shape as input_ids")
            if loss_chunk_size > 0:
                loss = self.embeddings.cross_entropy(
                    hidden,
                    targets,
                    chunk_size=loss_chunk_size,
                    backend=loss_backend,
                    num_negative_samples=loss_negative_samples,
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
    """Read the full stream and append one fixed-width scratch chunk."""

    def __init__(self, layer_index: int, config: GrowingWidthConfig) -> None:
        super().__init__()
        self.layer_index = layer_index
        stream_width = config.d_embed * (layer_index + 1)
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
    """Decoder with an append-only, fixed-chunk residual stream.

    Every block reads the embedding channels and all earlier scratch chunks,
    then appends one ``d_embed``-wide chunk. The original embedding signal
    decays linearly; late blocks accumulate learned writes in its place.
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
            scratch_chunks.append(scratch)
            if block.embed_write is not None:
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

    def forward(
        self,
        input_ids: Tensor,
        targets: Tensor | None = None,
        *,
        loss_chunk_size: int = 0,
        loss_backend: str = "tiled",
        loss_negative_samples: int = 4096,
    ) -> tuple[Tensor | None, Tensor | None]:
        hidden = self.encode(input_ids)
        loss = None
        if targets is not None:
            if targets.shape != input_ids.shape:
                raise ValueError("targets must have the same shape as input_ids")
            if loss_chunk_size > 0:
                loss = self.embeddings.cross_entropy(
                    hidden,
                    targets,
                    chunk_size=loss_chunk_size,
                    backend=loss_backend,
                    num_negative_samples=loss_negative_samples,
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
