"""Hydra, recursive Hydra, and tournament Hydra language models."""

from __future__ import annotations

from pathlib import Path

import torch
from torch import Tensor, nn

from config import ChainedHydraConfig, HydraConfig, TournamentHydraConfig
from .base import SVDLanguageModel
from .hydra_layers import HydraBlock, RecursiveHydraBlock, TournamentBlock


class HydraTransformer(SVDLanguageModel):
    def __init__(
        self, config: HydraConfig, embed_path: str | Path | None = None
    ) -> None:
        super().__init__()
        self.config = config
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
        self._finish_initialization(
            config.d_embed, config.max_seq_len, embed_path
        )

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
        return self.final_norm(wide[:, :, : self.config.d_embed])


class ChainedHydraTransformer(SVDLanguageModel):
    def __init__(
        self, config: ChainedHydraConfig, embed_path: str | Path | None = None
    ) -> None:
        super().__init__()
        self.config = config
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
                config.d_embed,
                config.n_streams,
                config.depth,
                config.n_stream_layers,
                config.n_merge_layers,
                config.n_heads,
                config.d_ff_ratio,
                config.dropout,
                config.layer_norm_eps,
            )
            for _ in range(config.n_blocks)
        )
        self.final_norm = nn.LayerNorm(
            config.d_embed, eps=config.layer_norm_eps
        )
        self._finish_initialization(
            config.d_embed, config.max_seq_len, embed_path
        )

    def _encode_hidden(self, input_ids: Tensor) -> Tensor:
        hidden = self.embedding_dropout(self.embeddings(input_ids))
        for block in self.intake_blocks:
            hidden = block(hidden)
        hidden = self.intake_norm(hidden)
        for block in self.hydra_blocks:
            hidden = block(hidden)
        return self.final_norm(hidden)


class TournamentHydraTransformer(SVDLanguageModel):
    def __init__(
        self,
        config: TournamentHydraConfig,
        embed_path: str | Path | None = None,
    ) -> None:
        super().__init__()
        self.config = config
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
                config.d_embed,
                config.n_experts,
                config.merge_schedule,
                config.n_expert_layers,
                config.n_merge_layers,
                config.n_heads,
                config.d_ff_ratio,
                config.dropout,
                config.layer_norm_eps,
                config.merge_mode,
            )
            for _ in range(config.n_blocks)
        )
        self.final_norm = nn.LayerNorm(
            config.d_embed, eps=config.layer_norm_eps
        )
        self._finish_initialization(
            config.d_embed, config.max_seq_len, embed_path
        )

    def _encode_hidden(self, input_ids: Tensor) -> Tensor:
        hidden = self.embedding_dropout(self.embeddings(input_ids))
        for block in self.intake_blocks:
            hidden = block(hidden)
        hidden = self.intake_norm(hidden)
        for block in self.tournament_blocks:
            hidden = block(hidden)
        return self.final_norm(hidden)
