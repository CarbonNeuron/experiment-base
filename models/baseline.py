"""Baseline and compound-query transformer components."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from config import CompoundQConfig, TransformerConfig
from .base import SVDLanguageModel


class CausalSelfAttention(nn.Module):
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
    def __init__(self, config: TransformerConfig | CompoundQConfig) -> None:
        super().__init__()
        self.up = nn.Linear(config.d_model, config.d_ff)
        self.down = nn.Linear(config.d_ff, config.d_model)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x: Tensor) -> Tensor:
        return self.dropout(self.down(F.gelu(self.up(x))))


class TransformerBlock(nn.Module):
    def __init__(self, config: TransformerConfig) -> None:
        super().__init__()
        self.attn_norm = nn.LayerNorm(config.d_model, eps=config.layer_norm_eps)
        self.attn = CausalSelfAttention(config)
        self.ffn_norm = nn.LayerNorm(config.d_model, eps=config.layer_norm_eps)
        self.ffn = FeedForward(config)

    def forward(self, x: Tensor) -> Tensor:
        x = x + self.attn(self.attn_norm(x))
        return x + self.ffn(self.ffn_norm(x))


class CompoundQAttention(nn.Module):
    """Causal attention with conjunctive and disjunctive query projections."""

    def __init__(self, config: TransformerConfig | CompoundQConfig) -> None:
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

        def split_heads(tensor: Tensor) -> Tensor:
            return tensor.view(
                batch_size, seq_len, self.n_heads, self.head_dim
            ).transpose(1, 2)

        query1, query2, query3, key, value = map(
            split_heads,
            (
                self.q1_proj(x),
                self.q2_proj(x),
                self.q3_proj(x),
                self.k_proj(x),
                self.v_proj(x),
            ),
        )
        query = torch.minimum(query1, query2) + torch.maximum(query1, query3)
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
    def __init__(self, config: CompoundQConfig) -> None:
        super().__init__()
        self.attn_norm = nn.LayerNorm(config.d_model, eps=config.layer_norm_eps)
        self.attn = CompoundQAttention(config)
        self.ffn_norm = nn.LayerNorm(config.d_model, eps=config.layer_norm_eps)
        self.ffn = FeedForward(config)

    def forward(self, x: Tensor) -> Tensor:
        x = x + self.attn(self.attn_norm(x))
        return x + self.ffn(self.ffn_norm(x))


class GenericTransformer(SVDLanguageModel):
    """GPT-style LM with cosine-preserving SVD token adaptation."""

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
        self.final_norm = nn.LayerNorm(config.d_model, eps=config.layer_norm_eps)
        self._finish_initialization(
            config.d_model, config.max_seq_len, embed_path
        )

    def _encode_hidden(self, input_ids: Tensor) -> Tensor:
        hidden = self.embedding_dropout(self.embeddings(input_ids))
        for block in self.blocks:
            hidden = block(hidden)
        return self.final_norm(hidden)


class CompoundQTransformer(GenericTransformer):
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
        self.final_norm = nn.LayerNorm(config.d_model, eps=config.layer_norm_eps)
        self._finish_initialization(
            config.d_model, config.max_seq_len, embed_path
        )
