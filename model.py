"""A conventional decoder-only transformer with frozen SVD token directions."""

from __future__ import annotations

import warnings
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch
from torch import Tensor, nn
from torch.nn import functional as F
from svd_embeds import OpenAIEmbedding


@dataclass
class TransformerConfig:
    """Architecture settings for :class:`GenericTransformer`."""

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
    """GPT-style LM with frozen SVD directions and trainable norms/rotation.

    The effective tied input/output embedding table is::

        E = rotation(norms[:, None] * directions)

    ``directions`` is a frozen buffer. Per-token ``norms`` and the shared,
    identity-initialized ``rotation`` are learned. Learned absolute position
    vectors are initialized so their mean L2 norm equals the mean L2 norm of
    the source SVD token embeddings.
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
            warnings.warn(
                f"compiled encoder failed; falling back to eager mode: {error}",
                RuntimeWarning,
                stacklevel=2,
            )
            self._compiled_encoder = None
            return self._encode_hidden(input_ids)

    def compile_encoder(
        self,
        *,
        mode: str = "default",
        dynamic: bool = False,
        backend: str | None = None,
    ) -> None:
        """Compile the hidden-state path while keeping chunked loss eager.

        Compilation is lazy. If the selected backend fails on its first real
        batch, :meth:`encode` warns once and transparently returns to eager
        execution.
        """
        kwargs: dict[str, Any] = {"dynamic": dynamic}
        if backend is None:
            kwargs["mode"] = mode
        else:
            kwargs["backend"] = backend
        self._compiled_encoder = torch.compile(self._encode_hidden, **kwargs)

    def disable_compile(self) -> None:
        """Return the encoder to eager execution."""
        self._compiled_encoder = None

    def logits(self, hidden: Tensor) -> Tensor:
        """Project hidden states with the tied effective embedding table."""
        return self.embeddings.project(hidden)

    def forward(
        self,
        input_ids: Tensor,
        targets: Tensor | None = None,
        *,
        loss_chunk_size: int = 0,
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
