"""WikiText token caching and DataLoader construction."""

from __future__ import annotations

from pathlib import Path

import torch
from datasets import load_dataset
from svd_embeds import get_tokenizer
from torch import Tensor
from torch.utils.data import DataLoader, Dataset

from config import DataConfig
from training.logger import PrettyLogger


class TokenDataset(Dataset[Tensor]):
    """Contiguous next-token examples cut from one token stream."""

    def __init__(self, tokens: Tensor, seq_len: int) -> None:
        self.tokens = tokens
        self.seq_len = seq_len
        self.num_sequences = max(0, (tokens.numel() - 1) // seq_len)

    def __len__(self) -> int:
        return self.num_sequences

    def __getitem__(self, index: int) -> Tensor:
        start = index * self.seq_len
        return self.tokens[start : start + self.seq_len + 1]


def load_wikitext(
    cache_dir: Path,
    seq_len: int,
    split: str,
    logger: PrettyLogger | None = None,
) -> TokenDataset:
    """Load and tokenize one WikiText-103 split, reusing its disk cache."""
    if logger is None:
        logger = PrettyLogger()
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_path = cache_dir / f"wikitext103_{split}_cl100k.pt"
    if cache_path.exists():
        tokens = torch.load(cache_path, map_location="cpu", weights_only=True)
    else:
        raw = load_dataset(
            "Salesforce/wikitext", "wikitext-103-raw-v1", split=split
        )
        tokenizer = get_tokenizer()
        token_ids: list[int] = []
        description = f"Tokenizing {split}"
        with logger.tokenization_progress(
            len(raw["text"]),
            description,
        ) as progress:
            task = progress.add_task(description, total=len(raw["text"]))
            for text in raw["text"]:
                if text:
                    token_ids.extend(tokenizer.encode_ordinary(text))
                    token_ids.append(tokenizer.eot_token)
                progress.advance(task)
        tokens = torch.tensor(token_ids, dtype=torch.long)
        torch.save(tokens, cache_path)
    logger.data_loaded(split, tokens.numel())
    return TokenDataset(tokens, seq_len)


def build_dataloaders(
    config: DataConfig,
    *,
    seq_len: int,
    device: torch.device,
    logger: PrettyLogger | None = None,
) -> tuple[DataLoader[Tensor], DataLoader[Tensor]]:
    """Build train/validation loaders without knowing model or optimizer details."""
    if logger is None:
        logger = PrettyLogger()
    train_data = load_wikitext(config.cache_dir, seq_len, "train", logger)
    val_data = load_wikitext(config.cache_dir, seq_len, "validation", logger)
    common_args = {
        "batch_size": config.batch_size,
        "pin_memory": device.type == "cuda",
    }
    train_loader = DataLoader(
        train_data,
        shuffle=True,
        drop_last=True,
        num_workers=config.num_workers,
        persistent_workers=config.num_workers > 0,
        **common_args,
    )
    val_loader = DataLoader(
        val_data,
        shuffle=False,
        num_workers=config.val_num_workers,
        persistent_workers=config.val_num_workers > 0,
        **common_args,
    )
    return train_loader, val_loader
