"""Reusable width-scaled components for Hydra architecture families."""

from __future__ import annotations

import torch
from torch import Tensor, nn
from torch.nn import functional as F


class HydraAttention(nn.Module):
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
    def __init__(self, width: int, d_ff_ratio: float, dropout: float) -> None:
        super().__init__()
        d_ff = int(width * d_ff_ratio)
        self.up = nn.Linear(width, d_ff)
        self.down = nn.Linear(d_ff, width)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: Tensor) -> Tensor:
        return self.dropout(self.down(F.gelu(self.up(x))))


class FFNMergeBlock(nn.Module):
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
        if depth <= 1:
            self.streams = nn.ModuleList(
                nn.ModuleList(
                    HydraBlock(
                        d_embed, n_heads, d_ff_ratio, dropout, layer_norm_eps
                    )
                    for _ in range(n_stream_layers)
                )
                for _ in range(n_streams)
            )
            self.nested = False
        else:
            self.streams = nn.ModuleList(
                RecursiveHydraBlock(
                    d_embed,
                    n_streams,
                    depth - 1,
                    n_stream_layers,
                    n_merge_layers,
                    n_heads,
                    d_ff_ratio,
                    dropout,
                    layer_norm_eps,
                )
                for _ in range(n_streams)
            )
            self.nested = True
        self.merge_blocks = nn.ModuleList(
            HydraBlock(
                d_wide, n_streams, d_ff_ratio, dropout, layer_norm_eps
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
        self.group_size = group_size
        self.d_embed = d_embed
        self.merge_mode = merge_mode
        self.group_mergers = nn.ModuleList()
        self.group_norms = nn.ModuleList()
        for _ in range(n_groups):
            if merge_mode == "ffn":
                merger = nn.ModuleList(
                    FFNMergeBlock(
                        d_wide, d_ff_ratio, dropout, layer_norm_eps
                    )
                    for _ in range(n_merge_layers)
                )
            elif merge_mode == "compress":
                merger = nn.ModuleList(
                    HydraBlock(
                        d_wide,
                        group_size,
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
                        group_size,
                        d_ff_ratio,
                        dropout,
                        layer_norm_eps,
                    )
                )
            else:
                merger = nn.ModuleList(
                    HydraBlock(
                        d_wide,
                        group_size,
                        d_ff_ratio,
                        dropout,
                        layer_norm_eps,
                    )
                    for _ in range(n_merge_layers)
                )
            self.group_mergers.append(merger)
            self.group_norms.append(nn.LayerNorm(d_embed, eps=layer_norm_eps))

    def forward(self, streams: list[Tensor]) -> list[Tensor]:
        outputs = []
        for index, (merger, norm) in enumerate(
            zip(self.group_mergers, self.group_norms)
        ):
            start = index * self.group_size
            wide = torch.cat(streams[start : start + self.group_size], dim=-1)
            for block in merger:
                wide = block(wide)
            if self.merge_mode == "compress":
                outputs.append(norm(wide))
            else:
                outputs.append(norm(wide[:, :, : self.d_embed]))
        return outputs


class TournamentBlock(nn.Module):
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
                    d_embed, n_heads, d_ff_ratio, dropout, layer_norm_eps
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
                    n_groups,
                    group_size,
                    d_embed,
                    n_merge_layers,
                    d_ff_ratio,
                    dropout,
                    layer_norm_eps,
                    merge_mode,
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
