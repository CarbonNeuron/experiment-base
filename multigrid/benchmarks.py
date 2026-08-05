"""Synthetic sequence tasks for evaluating long-range discrete recall."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor


@dataclass(frozen=True)
class BenchmarkBatch:
    """Token sequences with sparse next-token targets (``-1`` means ignored)."""

    input_ids: Tensor
    targets: Tensor

    @property
    def answer_mask(self) -> Tensor:
        return self.targets.ne(-1)


def _randint(
    high: int,
    shape: tuple[int, ...],
    *,
    device: torch.device | str | None,
    generator: torch.Generator | None,
) -> Tensor:
    return torch.randint(high, shape, device=device, generator=generator)


def associative_recall(
    batch_size: int = 32,
    n_pairs: int = 8,
    key_vocab_size: int = 64,
    value_vocab_size: int = 64,
    *,
    device: torch.device | str | None = None,
    generator: torch.Generator | None = None,
) -> BenchmarkBatch:
    """Create ``k1 v1 ... kN vN QUERY kq`` examples targeting ``vq``."""
    if min(batch_size, n_pairs, key_vocab_size, value_vocab_size) <= 0:
        raise ValueError("benchmark dimensions must be positive")
    if n_pairs > key_vocab_size:
        raise ValueError("n_pairs cannot exceed key_vocab_size")

    keys = torch.stack(
        [
            torch.randperm(
                key_vocab_size, device=device, generator=generator
            )[:n_pairs]
            for _ in range(batch_size)
        ]
    )
    values = _randint(
        value_vocab_size,
        (batch_size, n_pairs),
        device=device,
        generator=generator,
    ) + key_vocab_size
    query_indices = _randint(
        n_pairs, (batch_size,), device=device, generator=generator
    )
    row_indices = torch.arange(batch_size, device=device)
    query_keys = keys[row_indices, query_indices]
    answers = values[row_indices, query_indices]
    query_token = key_vocab_size + value_vocab_size

    sequence = torch.empty(
        batch_size,
        2 * n_pairs + 2,
        dtype=torch.long,
        device=device,
    )
    sequence[:, 0 : 2 * n_pairs : 2] = keys
    sequence[:, 1 : 2 * n_pairs : 2] = values
    sequence[:, -2] = query_token
    sequence[:, -1] = query_keys
    targets = torch.full_like(sequence, -1)
    targets[:, -1] = answers
    return BenchmarkBatch(sequence, targets)


def induction(
    batch_size: int = 32,
    context_length: int = 16,
    vocab_size: int = 128,
    *,
    device: torch.device | str | None = None,
    generator: torch.Generator | None = None,
) -> BenchmarkBatch:
    """Create contexts containing ``A B ... A`` whose answer is ``B``."""
    if context_length < 4 or vocab_size < 2 or batch_size <= 0:
        raise ValueError("induction needs batch_size > 0, length >= 4, vocab >= 2")
    sequence = _randint(
        vocab_size,
        (batch_size, context_length),
        device=device,
        generator=generator,
    )
    first = _randint(
        vocab_size, (batch_size,), device=device, generator=generator
    )
    second = _randint(
        vocab_size, (batch_size,), device=device, generator=generator
    )
    sequence[:, 0] = first
    sequence[:, 1] = second
    sequence[:, -1] = first
    targets = torch.full_like(sequence, -1)
    targets[:, -1] = second
    return BenchmarkBatch(sequence, targets)


def copying(
    batch_size: int = 32,
    copy_length: int = 8,
    vocab_size: int = 128,
    *,
    device: torch.device | str | None = None,
    generator: torch.Generator | None = None,
) -> BenchmarkBatch:
    """Create ``payload DELIMITER payload`` examples with copy targets."""
    if min(batch_size, copy_length, vocab_size) <= 0:
        raise ValueError("benchmark dimensions must be positive")
    payload = _randint(
        vocab_size,
        (batch_size, copy_length),
        device=device,
        generator=generator,
    )
    delimiter = torch.full(
        (batch_size, 1), vocab_size, dtype=torch.long, device=device
    )
    sequence = torch.cat((payload, delimiter, payload), dim=1)
    targets = torch.full_like(sequence, -1)
    targets[:, copy_length + 1 :] = payload
    return BenchmarkBatch(sequence, targets)


def state_tracking(
    batch_size: int = 32,
    n_updates: int = 8,
    n_states: int = 16,
    *,
    device: torch.device | str | None = None,
    generator: torch.Generator | None = None,
) -> BenchmarkBatch:
    """Create state-update streams whose final query targets the latest state."""
    if min(batch_size, n_updates, n_states) <= 0:
        raise ValueError("benchmark dimensions must be positive")
    states = _randint(
        n_states,
        (batch_size, n_updates),
        device=device,
        generator=generator,
    )
    update_token = n_states
    query_token = n_states + 1
    sequence = torch.empty(
        batch_size, 2 * n_updates + 1, dtype=torch.long, device=device
    )
    sequence[:, 0 : 2 * n_updates : 2] = update_token
    sequence[:, 1 : 2 * n_updates : 2] = states
    sequence[:, -1] = query_token
    targets = torch.full_like(sequence, -1)
    targets[:, -1] = states[:, -1]
    return BenchmarkBatch(sequence, targets)


generate_associative_recall = associative_recall
generate_induction = induction
generate_copying = copying
generate_state_tracking = state_tracking
