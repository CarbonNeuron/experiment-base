"""Synthetic sequence tasks for evaluating long-range discrete computation.

The target at a position is predicted from the input at that same position.
Generators therefore never place an answer token at its supervised position;
doing so would let a token embedding solve the benchmark without using context.
"""

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


TASK_NAMES = (
    "associative_recall",
    "multi_query_recall",
    "induction",
    "copying",
    "nested_scopes",
    "state_tracking",
    "paraphrased_retrieval",
)

TASK_SUMMARIES = {
    "associative_recall": "Retrieve one exact value from stored key/value pairs.",
    "multi_query_recall": "Answer several queries without memory overwriting.",
    "induction": "Repeat a learned follower after an arbitrary distractor delay.",
    "copying": "Reproduce an eight-token payload after a distractor delay.",
    "nested_scopes": "Resolve shadowed variable bindings while scopes unwind.",
    "state_tracking": "Track eight entity states through interleaved distractors.",
    "paraphrased_retrieval": "Retrieve a needle through three query templates.",
}

ANSWER_TOKEN_START = 128
ANSWER_TOKEN_COUNT = 96

# A shared symbolic vocabulary. Values and keys occupy disjoint ranges so a
# model cannot answer by copying the query key.
_KEY_BASE = 32
_KEY_COUNT = 96
_VALUE_BASE = ANSWER_TOKEN_START
_VALUE_COUNT = ANSWER_TOKEN_COUNT
_DISTRACTOR_BASE = 224

STORE = 1
QUERY = 2
COPY = 3
OPEN = 4
CLOSE = 5
BIND = 6
UPDATE = 7
DISTRACTOR = 8
FACT = 9
ASK = 10
VALUE = 11
PLEASE = 12
FIND = 13
WHAT = 14
FOR = 15


def _randint(
    high: int,
    shape: tuple[int, ...],
    *,
    device: torch.device | str | None,
    generator: torch.Generator | None,
) -> Tensor:
    return torch.randint(high, shape, device=device, generator=generator)


def _remove_adjacent_repeats(values: Tensor, base: int, count: int) -> Tensor:
    """Ensure a shifted copy target never equals its current input token."""
    if count < 2:
        raise ValueError("copy vocabulary must contain at least two symbols")
    for position in range(1, values.size(1)):
        repeated = values[:, position].eq(values[:, position - 1])
        replacement = (values[:, position] - base + 1).remainder(count) + base
        values[:, position] = torch.where(
            repeated, replacement, values[:, position]
        )
    return values


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
    """Create autoregressive ``payload DELIMITER payload`` examples.

    The first copied token is predicted at the delimiter and later tokens are
    teacher-forced from the preceding copied token. This avoids supervising a
    token with itself.
    """
    if min(batch_size, copy_length, vocab_size) <= 0:
        raise ValueError("benchmark dimensions must be positive")
    payload = _randint(
        vocab_size,
        (batch_size, copy_length),
        device=device,
        generator=generator,
    )
    payload = _remove_adjacent_repeats(payload, 0, vocab_size)
    delimiter = torch.full(
        (batch_size, 1), vocab_size, dtype=torch.long, device=device
    )
    sequence = torch.cat((payload, delimiter, payload[:, :-1]), dim=1)
    targets = torch.full_like(sequence, -1)
    targets[:, copy_length:] = payload
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


def _validate_symbolic(
    batch_size: int,
    capacity: int,
    vocab_size: int,
) -> None:
    if batch_size <= 0 or capacity <= 0:
        raise ValueError("batch_size and capacity must be positive")
    if vocab_size < _DISTRACTOR_BASE + 2:
        raise ValueError(f"vocab_size must be at least {_DISTRACTOR_BASE + 2}")


def _symbol_range(
    base: int,
    count: int,
    shape: tuple[int, ...],
    generator: torch.Generator | None,
) -> Tensor:
    return _randint(count, shape, device=None, generator=generator) + base


def _unique_keys(
    batch_size: int,
    count: int,
    generator: torch.Generator | None,
) -> Tensor:
    if count > _KEY_COUNT:
        raise ValueError(f"capacity cannot exceed {_KEY_COUNT}")
    # One batched sort is substantially cheaper than launching a separate
    # randperm for every sample, especially once training batches reach the
    # hundreds. The first ``count`` indices are still unique within each row.
    priorities = torch.rand(batch_size, _KEY_COUNT, generator=generator)
    return priorities.argsort(dim=1)[:, :count] + _KEY_BASE


def exact_associative_recall(
    batch_size: int,
    capacity: int,
    vocab_size: int = 512,
    *,
    generator: torch.Generator | None = None,
) -> BenchmarkBatch:
    """Store ``capacity`` key/value pairs and query one random key."""
    _validate_symbolic(batch_size, capacity, vocab_size)
    keys = _unique_keys(batch_size, capacity, generator)
    values = _symbol_range(
        _VALUE_BASE, _VALUE_COUNT, (batch_size, capacity), generator
    )
    query_indices = _randint(
        capacity, (batch_size,), device=None, generator=generator
    )
    rows = torch.arange(batch_size)
    sequence = torch.empty(batch_size, 3 * capacity + 2, dtype=torch.long)
    sequence[:, 0 : 3 * capacity : 3] = STORE
    sequence[:, 1 : 3 * capacity : 3] = keys
    sequence[:, 2 : 3 * capacity : 3] = values
    sequence[:, -2] = QUERY
    sequence[:, -1] = keys[rows, query_indices]
    targets = torch.full_like(sequence, -1)
    targets[:, -1] = values[rows, query_indices]
    return BenchmarkBatch(sequence, targets)


def multi_query_recall(
    batch_size: int,
    capacity: int,
    vocab_size: int = 512,
    *,
    generator: torch.Generator | None = None,
) -> BenchmarkBatch:
    """Query several stored pairs, exposing overwrite and interference."""
    _validate_symbolic(batch_size, capacity, vocab_size)
    keys = _unique_keys(batch_size, capacity, generator)
    values = _symbol_range(
        _VALUE_BASE, _VALUE_COUNT, (batch_size, capacity), generator
    )
    n_queries = min(8, max(2, capacity))
    query_indices = _randint(
        capacity, (batch_size, n_queries), device=None, generator=generator
    )
    rows = torch.arange(batch_size).unsqueeze(1)
    sequence = torch.empty(
        batch_size, 3 * capacity + 2 * n_queries, dtype=torch.long
    )
    sequence[:, 0 : 3 * capacity : 3] = STORE
    sequence[:, 1 : 3 * capacity : 3] = keys
    sequence[:, 2 : 3 * capacity : 3] = values
    query_start = 3 * capacity
    sequence[:, query_start::2] = QUERY
    sequence[:, query_start + 1 :: 2] = keys[rows, query_indices]
    targets = torch.full_like(sequence, -1)
    targets[:, query_start + 1 :: 2] = values[rows, query_indices]
    return BenchmarkBatch(sequence, targets)


def delayed_induction(
    batch_size: int,
    capacity: int,
    vocab_size: int = 512,
    *,
    generator: torch.Generator | None = None,
) -> BenchmarkBatch:
    """Repeat a cue after ``capacity`` distractors and predict its follower."""
    _validate_symbolic(batch_size, capacity, vocab_size)
    cue = _symbol_range(_KEY_BASE, _KEY_COUNT, (batch_size,), generator)
    follower = _symbol_range(
        _VALUE_BASE, _VALUE_COUNT, (batch_size,), generator
    )
    distractors = _symbol_range(
        _DISTRACTOR_BASE,
        vocab_size - _DISTRACTOR_BASE,
        (batch_size, capacity),
        generator,
    )
    sequence = torch.cat(
        (cue[:, None], follower[:, None], distractors, cue[:, None]), dim=1
    )
    targets = torch.full_like(sequence, -1)
    targets[:, -1] = follower
    return BenchmarkBatch(sequence, targets)


def delayed_copying(
    batch_size: int,
    capacity: int,
    vocab_size: int = 512,
    *,
    generator: torch.Generator | None = None,
) -> BenchmarkBatch:
    """Copy an eight-token payload after ``capacity`` distractor tokens."""
    _validate_symbolic(batch_size, capacity, vocab_size)
    copy_length = 8
    payload = _symbol_range(
        _VALUE_BASE, _VALUE_COUNT, (batch_size, copy_length), generator
    )
    payload = _remove_adjacent_repeats(
        payload, _VALUE_BASE, _VALUE_COUNT
    )
    distractors = _symbol_range(
        _DISTRACTOR_BASE,
        vocab_size - _DISTRACTOR_BASE,
        (batch_size, capacity),
        generator,
    )
    marker = torch.full((batch_size, 1), COPY, dtype=torch.long)
    sequence = torch.cat(
        (payload, distractors, marker, payload[:, :-1]), dim=1
    )
    targets = torch.full_like(sequence, -1)
    targets[:, copy_length + capacity :] = payload
    return BenchmarkBatch(sequence, targets)


def nested_scope_binding(
    batch_size: int,
    capacity: int,
    vocab_size: int = 512,
    *,
    generator: torch.Generator | None = None,
) -> BenchmarkBatch:
    """Shadow one variable in nested scopes, then query while unwinding."""
    _validate_symbolic(batch_size, capacity, vocab_size)
    if capacity > 64:
        raise ValueError("nested scope capacity cannot exceed 64")
    variable = _symbol_range(_KEY_BASE, _KEY_COUNT, (batch_size,), generator)
    values = _symbol_range(
        _VALUE_BASE, _VALUE_COUNT, (batch_size, capacity), generator
    )
    parts: list[Tensor] = []
    target_parts: list[Tensor] = []
    ignored = lambda width: torch.full(  # noqa: E731
        (batch_size, width), -1, dtype=torch.long
    )
    for depth in range(capacity):
        if depth:
            parts.append(torch.full((batch_size, 1), OPEN, dtype=torch.long))
            target_parts.append(ignored(1))
        binding = torch.stack(
            (
                torch.full_like(variable, BIND),
                variable,
                values[:, depth],
            ),
            dim=1,
        )
        parts.append(binding)
        target_parts.append(ignored(3))
    for depth in range(capacity - 1, -1, -1):
        query = torch.stack(
            (torch.full_like(variable, QUERY), variable), dim=1
        )
        query_targets = ignored(2)
        query_targets[:, -1] = values[:, depth]
        parts.append(query)
        target_parts.append(query_targets)
        if depth:
            parts.append(torch.full((batch_size, 1), CLOSE, dtype=torch.long))
            target_parts.append(ignored(1))
    return BenchmarkBatch(torch.cat(parts, dim=1), torch.cat(target_parts, dim=1))


def distracted_state_tracking(
    batch_size: int,
    capacity: int,
    vocab_size: int = 512,
    *,
    generator: torch.Generator | None = None,
) -> BenchmarkBatch:
    """Track several entity states through updates separated by distractors."""
    _validate_symbolic(batch_size, capacity, vocab_size)
    n_entities = min(8, capacity)
    entities = _unique_keys(batch_size, n_entities, generator)
    update_entities = _randint(
        n_entities, (batch_size, capacity), device=None, generator=generator
    )
    initial_priorities = torch.rand(
        batch_size, n_entities, generator=generator
    )
    update_entities[:, :n_entities] = initial_priorities.argsort(dim=1)
    states = _symbol_range(
        _VALUE_BASE, _VALUE_COUNT, (batch_size, capacity), generator
    )
    rows = torch.arange(batch_size).unsqueeze(1)
    update_keys = entities[rows, update_entities]
    stream = torch.empty(batch_size, 5 * capacity, dtype=torch.long)
    stream[:, 0::5] = UPDATE
    stream[:, 1::5] = update_keys
    stream[:, 2::5] = states
    stream[:, 3::5] = DISTRACTOR
    stream[:, 4::5] = _symbol_range(
        _DISTRACTOR_BASE,
        vocab_size - _DISTRACTOR_BASE,
        (batch_size, capacity),
        generator,
    )
    # Locate each entity's final update in parallel. Every entity is guaranteed
    # to occur in the initial permutation above, so all positions are valid.
    entity_ids = torch.arange(n_entities).view(1, 1, n_entities)
    update_positions = torch.arange(1, capacity + 1).view(1, capacity, 1)
    matching_positions = torch.where(
        update_entities.unsqueeze(-1).eq(entity_ids), update_positions, 0
    )
    latest_positions = matching_positions.amax(dim=1) - 1
    latest = states.gather(1, latest_positions)
    queries = torch.empty(batch_size, 2 * n_entities, dtype=torch.long)
    queries[:, 0::2] = QUERY
    queries[:, 1::2] = entities
    sequence = torch.cat((stream, queries), dim=1)
    targets = torch.full_like(sequence, -1)
    targets[:, 5 * capacity + 1 :: 2] = latest
    return BenchmarkBatch(sequence, targets)


def paraphrased_needle_retrieval(
    batch_size: int,
    capacity: int,
    vocab_size: int = 512,
    *,
    generator: torch.Generator | None = None,
) -> BenchmarkBatch:
    """Retrieve one fact using one of three equivalent query templates."""
    _validate_symbolic(batch_size, capacity, vocab_size)
    keys = _unique_keys(batch_size, capacity, generator)
    values = _symbol_range(
        _VALUE_BASE, _VALUE_COUNT, (batch_size, capacity), generator
    )
    facts = torch.empty(batch_size, 3 * capacity, dtype=torch.long)
    facts[:, 0::3] = FACT
    facts[:, 1::3] = keys
    facts[:, 2::3] = values
    needle_indices = _randint(
        capacity, (batch_size,), device=None, generator=generator
    )
    rows = torch.arange(batch_size)
    templates = torch.tensor(
        ((ASK, VALUE), (PLEASE, FIND), (WHAT, FOR)), dtype=torch.long
    )
    template_ids = _randint(3, (batch_size,), device=None, generator=generator)
    query_prefix = templates[template_ids]
    query = torch.cat(
        (query_prefix, keys[rows, needle_indices][:, None]), dim=1
    )
    sequence = torch.cat((facts, query), dim=1)
    targets = torch.full_like(sequence, -1)
    targets[:, -1] = values[rows, needle_indices]
    return BenchmarkBatch(sequence, targets)


def generate_primitive_batch(
    task: str,
    batch_size: int,
    capacity: int,
    vocab_size: int = 512,
    *,
    generator: torch.Generator | None = None,
) -> BenchmarkBatch:
    """Generate one batch from the named primitive benchmark."""
    generators = {
        "associative_recall": exact_associative_recall,
        "multi_query_recall": multi_query_recall,
        "induction": delayed_induction,
        "copying": delayed_copying,
        "nested_scopes": nested_scope_binding,
        "state_tracking": distracted_state_tracking,
        "paraphrased_retrieval": paraphrased_needle_retrieval,
    }
    try:
        factory = generators[task]
    except KeyError as error:
        choices = ", ".join(TASK_NAMES)
        raise ValueError(f"unknown task {task!r}; choose from: {choices}") from error
    return factory(batch_size, capacity, vocab_size, generator=generator)
