"""GPU-parallel factorization of softmax-addressed episodic memory."""

from __future__ import annotations

import torch
from torch import Tensor
import triton
import triton.language as tl


@triton.jit
def _schedule_kernel(
    write_priorities,
    selected,
    overwrites,
    priority_trace,
    writer_priority_trace,
    final_priorities,
    T: tl.constexpr,
    W: tl.constexpr,
    S: tl.constexpr,
    BS: tl.constexpr,
    TEMPERATURE: tl.constexpr,
):
    batch = tl.program_id(0)
    slot = tl.arange(0, BS)
    valid_slot = slot < S
    priorities = tl.zeros((BS,), tl.float32)
    for position in range(T):
        tl.store(
            priority_trace + (batch * T + position) * S + slot,
            priorities,
            mask=valid_slot,
        )
        for writer in range(W):
            event = (batch * W + writer) * T + position
            tl.store(
                writer_priority_trace + event * S + slot,
                priorities,
                mask=valid_slot,
            )
            probability = tl.load(write_priorities + event).to(tl.float32)
            logits = tl.where(
                valid_slot,
                -priorities / TEMPERATURE,
                -float("inf"),
            )
            logits -= tl.max(logits, axis=0)
            soft_slots = tl.exp(logits)
            soft_slots /= tl.sum(soft_slots, axis=0)
            replacement = tl.argmax(soft_slots, axis=0)
            hard_slots = (slot == replacement) & valid_slot
            lowest = tl.sum(soft_slots * priorities, axis=0)
            overwrite = tl.sigmoid(
                (probability - lowest) / TEMPERATURE
            )
            tl.store(selected + event, replacement)
            tl.store(overwrites + event, overwrite)
            update = hard_slots * overwrite
            priorities = (
                priorities * (1.0 - update) + probability * update
            )
    tl.store(
        final_priorities + batch * S + slot,
        priorities,
        mask=valid_slot,
    )


@triton.jit
def _state_kernel(
    write_values,
    write_keys,
    selected,
    overwrites,
    value_trace,
    key_trace,
    final_values,
    final_keys,
    T: tl.constexpr,
    W: tl.constexpr,
    S: tl.constexpr,
    M: tl.constexpr,
    K: tl.constexpr,
    BM: tl.constexpr,
    BK: tl.constexpr,
):
    program = tl.program_id(0)
    batch = program // S
    slot_id = program - batch * S
    memory = tl.arange(0, BM)
    key_dim = tl.arange(0, BK)
    valid_memory = memory < M
    valid_key = key_dim < K
    values = tl.zeros((BM,), tl.float32)
    keys = tl.zeros((BK,), tl.float32)
    for position in range(T):
        tl.store(
            value_trace
            + ((batch * T + position) * S + slot_id) * M
            + memory,
            values,
            mask=valid_memory,
        )
        tl.store(
            key_trace
            + ((batch * T + position) * S + slot_id) * K
            + key_dim,
            keys,
            mask=valid_key,
        )
        for writer in range(W):
            event = (batch * W + writer) * T + position
            chosen = tl.load(selected + event)
            overwrite = tl.load(overwrites + event).to(tl.float32)
            update = tl.where(chosen == slot_id, overwrite, 0.0)
            write_value = tl.load(
                write_values + event * M + memory,
                mask=valid_memory,
                other=0.0,
            ).to(tl.float32)
            write_key = tl.load(
                write_keys + event * K + key_dim,
                mask=valid_key,
                other=0.0,
            ).to(tl.float32)
            values = values * (1.0 - update) + write_value * update
            keys = keys * (1.0 - update) + write_key * update
    tl.store(
        final_values + (batch * S + slot_id) * M + memory,
        values,
        mask=valid_memory,
    )
    tl.store(
        final_keys + (batch * S + slot_id) * K + key_dim,
        keys,
        mask=valid_key,
    )


@triton.jit
def _read_kernel(
    queries,
    value_trace,
    key_trace,
    priority_trace,
    reads,
    T: tl.constexpr,
    S: tl.constexpr,
    M: tl.constexpr,
    K: tl.constexpr,
    BS: tl.constexpr,
    BM: tl.constexpr,
    BK: tl.constexpr,
):
    program = tl.program_id(0)
    batch = program // T
    position = program - batch * T
    slot = tl.arange(0, BS)
    memory = tl.arange(0, BM)
    key_dim = tl.arange(0, BK)
    valid_slot = slot < S
    valid_memory = memory < M
    valid_key = key_dim < K
    values = tl.load(
        value_trace
        + ((batch * T + position) * S) * M
        + slot[:, None] * M
        + memory[None, :],
        mask=valid_slot[:, None] & valid_memory[None, :],
        other=0.0,
    ).to(tl.float32)
    keys = tl.load(
        key_trace
        + ((batch * T + position) * S) * K
        + slot[:, None] * K
        + key_dim[None, :],
        mask=valid_slot[:, None] & valid_key[None, :],
        other=0.0,
    ).to(tl.float32)
    priorities = tl.load(
        priority_trace + (batch * T + position) * S + slot,
        mask=valid_slot,
        other=0.0,
    ).to(tl.float32)
    query = tl.load(
        queries + (batch * T + position) * K + key_dim,
        mask=valid_key,
        other=0.0,
    ).to(tl.float32)
    scores = tl.sum(keys * query[None, :], axis=1) / (K**0.5)
    occupied = valid_slot & (priorities > 0.0)
    scores = tl.where(occupied, scores, -1.0e4)
    maximum = tl.max(scores, axis=0)
    weights = tl.exp(scores - maximum) * occupied
    weights /= tl.maximum(tl.sum(weights, axis=0), 1.0e-8)
    read = tl.sum(weights[:, None] * values, axis=0)
    tl.store(
        reads + (batch * T + position) * M + memory,
        read,
        mask=valid_memory,
    )


@triton.jit
def _read_backward_kernel(
    queries,
    value_trace,
    key_trace,
    priority_trace,
    grad_reads,
    grad_queries,
    read_weights,
    grad_scores_out,
    T: tl.constexpr,
    S: tl.constexpr,
    M: tl.constexpr,
    K: tl.constexpr,
    BS: tl.constexpr,
    BM: tl.constexpr,
    BK: tl.constexpr,
):
    program = tl.program_id(0)
    batch = program // T
    position = program - batch * T
    slot = tl.arange(0, BS)
    memory = tl.arange(0, BM)
    key_dim = tl.arange(0, BK)
    valid_slot = slot < S
    valid_memory = memory < M
    valid_key = key_dim < K
    values = tl.load(
        value_trace
        + ((batch * T + position) * S) * M
        + slot[:, None] * M
        + memory[None, :],
        mask=valid_slot[:, None] & valid_memory[None, :],
        other=0.0,
    ).to(tl.float32)
    keys = tl.load(
        key_trace
        + ((batch * T + position) * S) * K
        + slot[:, None] * K
        + key_dim[None, :],
        mask=valid_slot[:, None] & valid_key[None, :],
        other=0.0,
    ).to(tl.float32)
    priorities = tl.load(
        priority_trace + (batch * T + position) * S + slot,
        mask=valid_slot,
        other=0.0,
    ).to(tl.float32)
    query = tl.load(
        queries + (batch * T + position) * K + key_dim,
        mask=valid_key,
        other=0.0,
    ).to(tl.float32)
    grad_read = tl.load(
        grad_reads + (batch * T + position) * M + memory,
        mask=valid_memory,
        other=0.0,
    ).to(tl.float32)
    inverse_sqrt_key: tl.constexpr = 1.0 / (K**0.5)
    scores = tl.sum(keys * query[None, :], axis=1) * inverse_sqrt_key
    occupied = valid_slot & (priorities > 0.0)
    scores = tl.where(occupied, scores, -1.0e4)
    maximum = tl.max(scores, axis=0)
    weights = tl.exp(scores - maximum) * occupied
    weights /= tl.maximum(tl.sum(weights, axis=0), 1.0e-8)
    grad_weights = tl.sum(values * grad_read[None, :], axis=1)
    weight_inner = tl.sum(weights * grad_weights, axis=0)
    grad_scores = weights * (grad_weights - weight_inner)
    grad_query = tl.sum(
        grad_scores[:, None] * keys, axis=0
    ) * inverse_sqrt_key
    trace_offset = (batch * T + position) * S + slot
    tl.store(read_weights + trace_offset, weights, mask=valid_slot)
    tl.store(grad_scores_out + trace_offset, grad_scores, mask=valid_slot)
    tl.store(
        grad_queries + (batch * T + position) * K + key_dim,
        grad_query,
        mask=valid_key,
    )


@triton.jit
def _state_backward_kernel(
    queries,
    write_values,
    write_keys,
    selected,
    overwrites,
    value_trace,
    key_trace,
    grad_reads,
    read_weights,
    grad_scores,
    grad_final_values,
    grad_final_keys,
    grad_write_values,
    grad_write_keys,
    grad_updates,
    T: tl.constexpr,
    W: tl.constexpr,
    S: tl.constexpr,
    M: tl.constexpr,
    K: tl.constexpr,
    BM: tl.constexpr,
    BK: tl.constexpr,
):
    program = tl.program_id(0)
    batch = program // S
    slot_id = program - batch * S
    memory = tl.arange(0, BM)
    key_dim = tl.arange(0, BK)
    valid_memory = memory < M
    valid_key = key_dim < K
    grad_values = tl.load(
        grad_final_values + (batch * S + slot_id) * M + memory,
        mask=valid_memory,
        other=0.0,
    ).to(tl.float32)
    grad_keys = tl.load(
        grad_final_keys + (batch * S + slot_id) * K + key_dim,
        mask=valid_key,
        other=0.0,
    ).to(tl.float32)
    inverse_sqrt_key: tl.constexpr = 1.0 / (K**0.5)
    for reverse_position in range(T):
        position: tl.constexpr = T - 1 - reverse_position
        for reverse_writer in range(W):
            writer: tl.constexpr = W - 1 - reverse_writer
            old_values = tl.load(
                value_trace
                + ((batch * T + position) * S + slot_id) * M
                + memory,
                mask=valid_memory,
                other=0.0,
            ).to(tl.float32)
            old_keys = tl.load(
                key_trace
                + ((batch * T + position) * S + slot_id) * K
                + key_dim,
                mask=valid_key,
                other=0.0,
            ).to(tl.float32)
            for replay_writer in range(writer):
                replay_event = (
                    (batch * W + replay_writer) * T + position
                )
                replay_slot = tl.load(selected + replay_event)
                replay_overwrite = tl.load(
                    overwrites + replay_event
                ).to(tl.float32)
                replay_update = tl.where(
                    replay_slot == slot_id, replay_overwrite, 0.0
                )
                replay_value = tl.load(
                    write_values + replay_event * M + memory,
                    mask=valid_memory,
                    other=0.0,
                ).to(tl.float32)
                replay_key = tl.load(
                    write_keys + replay_event * K + key_dim,
                    mask=valid_key,
                    other=0.0,
                ).to(tl.float32)
                old_values = (
                    old_values * (1.0 - replay_update)
                    + replay_value * replay_update
                )
                old_keys = (
                    old_keys * (1.0 - replay_update)
                    + replay_key * replay_update
                )

            event = (batch * W + writer) * T + position
            chosen = tl.load(selected + event)
            overwrite = tl.load(overwrites + event).to(tl.float32)
            update = tl.where(chosen == slot_id, overwrite, 0.0)
            write_value = tl.load(
                write_values + event * M + memory,
                mask=valid_memory,
                other=0.0,
            ).to(tl.float32)
            write_key = tl.load(
                write_keys + event * K + key_dim,
                mask=valid_key,
                other=0.0,
            ).to(tl.float32)
            grad_update = tl.sum(
                grad_values * (write_value - old_values), axis=0
            )
            grad_update += tl.sum(
                grad_keys * (write_key - old_keys), axis=0
            )
            tl.store(
                grad_updates + event * S + slot_id,
                grad_update,
            )
            selected_slot = chosen == slot_id
            tl.store(
                grad_write_values + event * M + memory,
                grad_values * update,
                mask=selected_slot & valid_memory,
            )
            tl.store(
                grad_write_keys + event * K + key_dim,
                grad_keys * update,
                mask=selected_slot & valid_key,
            )
            grad_values *= 1.0 - update
            grad_keys *= 1.0 - update

        trace_offset = (batch * T + position) * S + slot_id
        weight = tl.load(read_weights + trace_offset).to(tl.float32)
        grad_score = tl.load(grad_scores + trace_offset).to(tl.float32)
        grad_read = tl.load(
            grad_reads + (batch * T + position) * M + memory,
            mask=valid_memory,
            other=0.0,
        ).to(tl.float32)
        query = tl.load(
            queries + (batch * T + position) * K + key_dim,
            mask=valid_key,
            other=0.0,
        ).to(tl.float32)
        grad_values += weight * grad_read
        grad_keys += grad_score * query * inverse_sqrt_key


@triton.jit
def _priority_backward_kernel(
    write_priorities,
    writer_priority_trace,
    grad_updates,
    grad_final_priorities,
    grad_write_priorities,
    T: tl.constexpr,
    W: tl.constexpr,
    S: tl.constexpr,
    BS: tl.constexpr,
    TEMPERATURE: tl.constexpr,
):
    batch = tl.program_id(0)
    slot = tl.arange(0, BS)
    valid_slot = slot < S
    grad_priorities = tl.load(
        grad_final_priorities + batch * S + slot,
        mask=valid_slot,
        other=0.0,
    ).to(tl.float32)
    for reverse_position in range(T):
        position: tl.constexpr = T - 1 - reverse_position
        for reverse_writer in range(W):
            writer: tl.constexpr = W - 1 - reverse_writer
            event = (batch * W + writer) * T + position
            old_priorities = tl.load(
                writer_priority_trace + event * S + slot,
                mask=valid_slot,
                other=0.0,
            ).to(tl.float32)
            probability = tl.load(write_priorities + event).to(tl.float32)
            logits = tl.where(
                valid_slot,
                -old_priorities / TEMPERATURE,
                -float("inf"),
            )
            logits -= tl.max(logits, axis=0)
            soft_slots = tl.exp(logits)
            soft_slots /= tl.sum(soft_slots, axis=0)
            replacement = tl.argmax(soft_slots, axis=0)
            hard_slots = (slot == replacement) & valid_slot
            lowest = tl.sum(soft_slots * old_priorities, axis=0)
            overwrite = tl.sigmoid(
                (probability - lowest) / TEMPERATURE
            )
            update = hard_slots * overwrite
            grad_update = tl.load(
                grad_updates + event * S + slot,
                mask=valid_slot,
                other=0.0,
            ).to(tl.float32)
            grad_update += grad_priorities * (
                probability - old_priorities
            )
            grad_probability = tl.sum(grad_priorities * update, axis=0)
            grad_old = grad_priorities * (1.0 - update)
            grad_slot_weights = grad_update * overwrite
            grad_overwrite = tl.sum(grad_update * hard_slots, axis=0)
            grad_activation = (
                grad_overwrite
                * overwrite
                * (1.0 - overwrite)
                / TEMPERATURE
            )
            grad_probability += grad_activation
            grad_lowest = -grad_activation
            grad_soft = grad_slot_weights + grad_lowest * old_priorities
            grad_old += grad_lowest * soft_slots
            soft_inner = tl.sum(soft_slots * grad_soft, axis=0)
            grad_logits = soft_slots * (grad_soft - soft_inner)
            grad_old += -grad_logits / TEMPERATURE
            tl.store(grad_write_priorities + event, grad_probability)
            grad_priorities = grad_old


def _factorized_memory_forward(
    queries: Tensor,
    write_values: Tensor,
    write_keys: Tensor,
    write_priorities: Tensor,
    n_slots: int,
    temperature: float,
) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    batch_size, n_writers, sequence_length, d_memory = write_values.shape
    d_key = queries.size(-1)
    block_slots = triton.next_power_of_2(n_slots)
    block_memory = triton.next_power_of_2(d_memory)
    block_key = triton.next_power_of_2(d_key)
    selected = torch.empty_like(write_priorities, dtype=torch.int32)
    overwrites = torch.empty_like(write_priorities)
    priority_trace = torch.empty(
        batch_size,
        sequence_length,
        n_slots,
        device=queries.device,
        dtype=write_priorities.dtype,
    )
    writer_priority_trace = torch.empty(
        batch_size,
        n_writers,
        sequence_length,
        n_slots,
        device=queries.device,
        dtype=write_priorities.dtype,
    )
    value_trace = torch.empty(
        batch_size,
        sequence_length,
        n_slots,
        d_memory,
        device=queries.device,
        dtype=write_values.dtype,
    )
    key_trace = torch.empty(
        batch_size,
        sequence_length,
        n_slots,
        d_key,
        device=queries.device,
        dtype=write_keys.dtype,
    )
    reads = torch.empty(
        batch_size,
        sequence_length,
        d_memory,
        device=queries.device,
        dtype=queries.dtype,
    )
    final_values = torch.empty(
        batch_size,
        n_slots,
        d_memory,
        device=queries.device,
        dtype=write_values.dtype,
    )
    final_keys = torch.empty(
        batch_size,
        n_slots,
        d_key,
        device=queries.device,
        dtype=write_keys.dtype,
    )
    final_priorities = torch.empty(
        batch_size,
        n_slots,
        device=queries.device,
        dtype=write_priorities.dtype,
    )
    _schedule_kernel[(batch_size,)](
        write_priorities,
        selected,
        overwrites,
        priority_trace,
        writer_priority_trace,
        final_priorities,
        T=sequence_length,
        W=n_writers,
        S=n_slots,
        BS=block_slots,
        TEMPERATURE=temperature,
        num_warps=4,
    )
    _state_kernel[(batch_size * n_slots,)](
        write_values,
        write_keys,
        selected,
        overwrites,
        value_trace,
        key_trace,
        final_values,
        final_keys,
        T=sequence_length,
        W=n_writers,
        S=n_slots,
        M=d_memory,
        K=d_key,
        BM=block_memory,
        BK=block_key,
        num_warps=1,
    )
    _read_kernel[(batch_size * sequence_length,)](
        queries,
        value_trace,
        key_trace,
        priority_trace,
        reads,
        T=sequence_length,
        S=n_slots,
        M=d_memory,
        K=d_key,
        BS=block_slots,
        BM=block_memory,
        BK=block_key,
        num_warps=4,
    )
    outputs = (reads, final_values, final_keys, final_priorities)
    saved = (
        selected,
        overwrites,
        priority_trace,
        writer_priority_trace,
        value_trace,
        key_trace,
    )
    return outputs, saved


def factorized_memory_forward(
    queries: Tensor,
    write_values: Tensor,
    write_keys: Tensor,
    write_priorities: Tensor,
    n_slots: int,
    temperature: float,
) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    with torch.cuda.device(queries.device):
        outputs, _ = _factorized_memory_forward(
            queries,
            write_values,
            write_keys,
            write_priorities,
            n_slots,
            temperature,
        )
    return outputs


class _FactorizedMemoryFunction(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        queries: Tensor,
        write_values: Tensor,
        write_keys: Tensor,
        write_priorities: Tensor,
        n_slots: int,
        temperature: float,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        with torch.cuda.device(queries.device):
            outputs, saved = _factorized_memory_forward(
                queries,
                write_values,
                write_keys,
                write_priorities,
                int(n_slots),
                float(temperature),
            )
        ctx.n_slots = int(n_slots)
        ctx.temperature = float(temperature)
        ctx.save_for_backward(
            queries,
            write_values,
            write_keys,
            write_priorities,
            *saved,
        )
        return outputs

    @staticmethod
    def backward(
        ctx,
        grad_reads: Tensor | None,
        grad_final_values: Tensor | None,
        grad_final_keys: Tensor | None,
        grad_final_priorities: Tensor | None,
    ) -> tuple[Tensor | None, ...]:
        (
            queries,
            write_values,
            write_keys,
            write_priorities,
            selected,
            overwrites,
            priority_trace,
            writer_priority_trace,
            value_trace,
            key_trace,
        ) = ctx.saved_tensors
        batch_size, n_writers, sequence_length, d_memory = (
            write_values.shape
        )
        d_key = queries.size(-1)
        n_slots = ctx.n_slots
        block_slots = triton.next_power_of_2(n_slots)
        block_memory = triton.next_power_of_2(d_memory)
        block_key = triton.next_power_of_2(d_key)
        grad_reads = (
            torch.zeros(
                batch_size,
                sequence_length,
                d_memory,
                device=queries.device,
                dtype=write_values.dtype,
            )
            if grad_reads is None
            else grad_reads.contiguous()
        )
        grad_final_values = (
            torch.zeros(
                batch_size,
                n_slots,
                d_memory,
                device=queries.device,
                dtype=write_values.dtype,
            )
            if grad_final_values is None
            else grad_final_values.contiguous()
        )
        grad_final_keys = (
            torch.zeros(
                batch_size,
                n_slots,
                d_key,
                device=queries.device,
                dtype=write_keys.dtype,
            )
            if grad_final_keys is None
            else grad_final_keys.contiguous()
        )
        grad_final_priorities = (
            torch.zeros(
                batch_size,
                n_slots,
                device=queries.device,
                dtype=write_priorities.dtype,
            )
            if grad_final_priorities is None
            else grad_final_priorities.contiguous()
        )
        grad_queries = torch.empty_like(queries)
        grad_write_values = torch.empty_like(write_values)
        grad_write_keys = torch.empty_like(write_keys)
        grad_write_priorities = torch.empty_like(write_priorities)
        read_weights = torch.empty(
            batch_size,
            sequence_length,
            n_slots,
            device=queries.device,
            dtype=torch.float32,
        )
        grad_scores = torch.empty_like(read_weights)
        grad_updates = torch.empty(
            batch_size,
            n_writers,
            sequence_length,
            n_slots,
            device=queries.device,
            dtype=torch.float32,
        )
        _read_backward_kernel[(batch_size * sequence_length,)](
            queries,
            value_trace,
            key_trace,
            priority_trace,
            grad_reads,
            grad_queries,
            read_weights,
            grad_scores,
            T=sequence_length,
            S=n_slots,
            M=d_memory,
            K=d_key,
            BS=block_slots,
            BM=block_memory,
            BK=block_key,
            num_warps=4,
        )
        _state_backward_kernel[(batch_size * n_slots,)](
            queries,
            write_values,
            write_keys,
            selected,
            overwrites,
            value_trace,
            key_trace,
            grad_reads,
            read_weights,
            grad_scores,
            grad_final_values,
            grad_final_keys,
            grad_write_values,
            grad_write_keys,
            grad_updates,
            T=sequence_length,
            W=n_writers,
            S=n_slots,
            M=d_memory,
            K=d_key,
            BM=block_memory,
            BK=block_key,
            num_warps=1,
        )
        _priority_backward_kernel[(batch_size,)](
            write_priorities,
            writer_priority_trace,
            grad_updates,
            grad_final_priorities,
            grad_write_priorities,
            T=sequence_length,
            W=n_writers,
            S=n_slots,
            BS=block_slots,
            TEMPERATURE=ctx.temperature,
            num_warps=4,
        )
        return (
            grad_queries,
            grad_write_values,
            grad_write_keys,
            grad_write_priorities,
            None,
            None,
        )


def factorized_softmax_memory(
    queries: Tensor,
    write_values: Tensor,
    write_keys: Tensor,
    write_priorities: Tensor,
    n_slots: int,
    temperature: float,
) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    """Run the exact recurrence as parallel schedule/state/read stages."""
    return _FactorizedMemoryFunction.apply(
        queries.contiguous(),
        write_values.contiguous(),
        write_keys.contiguous(),
        write_priorities.contiguous(),
        int(n_slots),
        float(temperature),
    )


__all__ = ["factorized_softmax_memory"]
