"""A conventional decoder-only transformer with frozen SVD token directions."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch
from torch import Tensor, nn
from torch.nn import functional as F


DEFAULT_EMBED_PATH = Path("embeddings/openai_svd_embeddings_128d.pt")
EMBED_REPO_ID = "Carbun1/FixingEmbeds"
EMBED_REPO_FILENAME = "openai/openai_svd_embeddings_128d.pt"


@dataclass
class TransformerConfig:
    """Architecture settings for :class:`GenericTransformer`."""

    d_model: int = 128
    n_heads: int = 8
    n_layers: int = 6
    d_ff: int = 512
    vocab_size: int = 100_277  # cl100k_base
    max_seq_len: int = 512
    dropout: float = 0.1
    layer_norm_eps: float = 1e-5

    def __post_init__(self) -> None:
        if self.d_model <= 0 or self.n_heads <= 0 or self.n_layers <= 0:
            raise ValueError("model dimensions and layer counts must be positive")
        if self.d_model % self.n_heads:
            raise ValueError("d_model must be divisible by n_heads")
        if self.d_ff <= 0 or self.vocab_size <= 0 or self.max_seq_len <= 0:
            raise ValueError("d_ff, vocab_size, and max_seq_len must be positive")
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
        embed_path: str | Path = DEFAULT_EMBED_PATH,
    ) -> None:
        super().__init__()
        if config is None:
            config = TransformerConfig()
        elif isinstance(config, dict):
            config = TransformerConfig(**config)
        self.config = config

        source = self._load_embedding_tensor(embed_path)
        source_norms = source.norm(dim=-1).clamp_min(1e-8)
        self.register_buffer(
            "directions", (source / source_norms.unsqueeze(-1)).contiguous()
        )
        self.norms = nn.Parameter(source_norms)
        self.rotation = nn.Linear(config.d_model, config.d_model, bias=False)

        self.position_embedding = nn.Embedding(
            config.max_seq_len, config.d_model
        )
        self.embedding_dropout = nn.Dropout(config.dropout)
        self.blocks = nn.ModuleList(
            TransformerBlock(config) for _ in range(config.n_layers)
        )
        self.final_norm = nn.LayerNorm(
            config.d_model, eps=config.layer_norm_eps
        )

        self.apply(self._init_weights)
        nn.init.eye_(self.rotation.weight)
        self._init_position_embeddings(source_norms.mean())

    def _load_embedding_tensor(self, embed_path: str | Path) -> Tensor:
        path = Path(embed_path).expanduser()
        if not path.exists() and path == DEFAULT_EMBED_PATH:
            try:
                from huggingface_hub import hf_hub_download
            except ImportError as error:
                raise FileNotFoundError(
                    f"{path} was not found. Install requirements.txt to enable "
                    "automatic embedding downloads."
                ) from error
            path = Path(
                hf_hub_download(
                    repo_id=EMBED_REPO_ID,
                    filename=EMBED_REPO_FILENAME,
                )
            )
        if not path.exists():
            raise FileNotFoundError(f"embedding file not found: {path}")
        embeddings = torch.load(path, map_location="cpu", weights_only=True)
        if not isinstance(embeddings, Tensor) or embeddings.ndim != 2:
            raise TypeError("embedding file must contain a rank-2 torch.Tensor")
        expected = (self.config.vocab_size, self.config.d_model)
        if tuple(embeddings.shape) != expected:
            raise ValueError(
                f"embedding shape {tuple(embeddings.shape)} does not match {expected}"
            )
        if not embeddings.is_floating_point():
            raise TypeError("embedding tensor must have a floating-point dtype")
        return embeddings.float().contiguous()

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

    @torch.no_grad()
    def _init_position_embeddings(self, target_norm: Tensor) -> None:
        """Give every initial position vector the mean source-token norm."""
        weights = self.position_embedding.weight
        nn.init.normal_(weights, mean=0.0, std=1.0)
        weights.div_(weights.norm(dim=-1, keepdim=True).clamp_min(1e-8))
        weights.mul_(target_norm.to(device=weights.device, dtype=weights.dtype))

    def effective_embeddings(self) -> Tensor:
        """Return the tied, trainable embedding table."""
        return self.rotation(self.norms.unsqueeze(-1) * self.directions)

    def _encode_with_embeddings(
        self, input_ids: Tensor, embeddings: Tensor
    ) -> Tensor:
        if input_ids.ndim != 2:
            raise ValueError("input_ids must have shape [batch, sequence]")
        seq_len = input_ids.size(1)
        if seq_len > self.config.max_seq_len:
            raise ValueError(
                f"sequence length {seq_len} exceeds max_seq_len="
                f"{self.config.max_seq_len}"
            )

        positions = torch.arange(seq_len, device=input_ids.device)
        hidden = F.embedding(input_ids, embeddings)
        hidden = hidden + self.position_embedding(positions)
        hidden = self.embedding_dropout(hidden)
        for block in self.blocks:
            hidden = block(hidden)
        return self.final_norm(hidden)

    def encode(self, input_ids: Tensor) -> Tensor:
        """Return final hidden states without materializing vocabulary logits."""
        return self._encode_with_embeddings(
            input_ids, self.effective_embeddings()
        )

    def logits(self, hidden: Tensor) -> Tensor:
        """Project hidden states with the tied effective embedding table."""
        return F.linear(hidden, self.effective_embeddings())

    def forward(
        self, input_ids: Tensor, targets: Tensor | None = None
    ) -> tuple[Tensor, Tensor | None]:
        embeddings = self.effective_embeddings()
        hidden = self._encode_with_embeddings(input_ids, embeddings)
        logits = F.linear(hidden, embeddings)
        loss = None
        if targets is not None:
            if targets.shape != input_ids.shape:
                raise ValueError("targets must have the same shape as input_ids")
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
            parameters += self.directions.numel()
        return parameters
