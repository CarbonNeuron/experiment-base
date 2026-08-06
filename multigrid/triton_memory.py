"""Fused Triton recurrence for softmax-addressed episodic memory."""

from __future__ import annotations

import torch
from torch import Tensor

try:
    import triton
    import triton.language as tl

    _TRITON_AVAILABLE = True
except ImportError:  # pragma: no cover - depends on the accelerator environment
    triton = tl = None  # type: ignore[assignment]
    _TRITON_AVAILABLE = False


def triton_memory_available() -> bool:
    """Return whether the optional Triton package can be imported."""
    return _TRITON_AVAILABLE


if _TRITON_AVAILABLE:

    @triton.jit
    def _forward_kernel(
        queries,
        write_values,
        write_keys,
        write_priorities,
        reads,
        final_values,
        final_keys,
        final_priorities,
        trace_values,
        trace_keys,
        trace_priorities,
        T: tl.constexpr,
        W: tl.constexpr,
        S: tl.constexpr,
        M: tl.constexpr,
        K: tl.constexpr,
        BS: tl.constexpr,
        BM: tl.constexpr,
        BK: tl.constexpr,
        TEMPERATURE: tl.constexpr,
        SAVE_TRACE: tl.constexpr,
        HARD_OVERWRITE: tl.constexpr,
    ):
        batch = tl.program_id(0)
        slot = tl.arange(0, BS)
        memory = tl.arange(0, BM)
        key_dim = tl.arange(0, BK)
        slot_memory = slot[:, None] * M + memory[None, :]
        slot_key = slot[:, None] * K + key_dim[None, :]
        valid_slot = slot < S
        valid_memory = memory < M
        valid_key = key_dim < K
        values = tl.zeros((BS, BM), tl.float32)
        keys = tl.zeros((BS, BK), tl.float32)
        priorities = tl.zeros((BS,), tl.float32)
        inv_sqrt_key: tl.constexpr = 1.0 / (K**0.5)

        for position in range(T):
            query_base = (batch * T + position) * K
            query = tl.load(
                queries + query_base + key_dim,
                mask=valid_key,
                other=0.0,
            ).to(tl.float32)
            scores = tl.sum(keys * query[None, :], axis=1) * inv_sqrt_key
            occupied = valid_slot & (priorities > 0.0)
            scores = tl.where(occupied, scores, -1.0e4)
            maximum = tl.max(scores, axis=0)
            weights = tl.exp(scores - maximum) * occupied
            weights = weights / tl.maximum(tl.sum(weights, axis=0), 1.0e-8)
            read = tl.sum(weights[:, None] * values, axis=0)
            read_base = (batch * T + position) * M
            tl.store(
                reads + read_base + memory,
                read,
                mask=valid_memory,
            )

            for writer in range(W):
                if SAVE_TRACE:
                    # Backward needs the state immediately before each
                    # writer. Saving it here uses more memory than one trace
                    # per token, but avoids replaying writers 0..writer-1 for
                    # every reverse step (quadratic work in the layer count).
                    trace_value_base = (
                        ((batch * T + position) * W + writer) * S
                    ) * M
                    trace_key_base = (
                        ((batch * T + position) * W + writer) * S
                    ) * K
                    trace_priority_base = (
                        (batch * T + position) * W + writer
                    ) * S
                    tl.store(
                        trace_values + trace_value_base + slot_memory,
                        values,
                        mask=valid_slot[:, None] & valid_memory[None, :],
                    )
                    tl.store(
                        trace_keys + trace_key_base + slot_key,
                        keys,
                        mask=valid_slot[:, None] & valid_key[None, :],
                    )
                    tl.store(
                        trace_priorities + trace_priority_base + slot,
                        priorities,
                        mask=valid_slot,
                    )

                write_value_base = ((batch * W + writer) * T + position) * M
                write_key_base = ((batch * W + writer) * T + position) * K
                write_priority_index = (batch * W + writer) * T + position
                write_value = tl.load(
                    write_values + write_value_base + memory,
                    mask=valid_memory,
                    other=0.0,
                ).to(tl.float32)
                write_key = tl.load(
                    write_keys + write_key_base + key_dim,
                    mask=valid_key,
                    other=0.0,
                ).to(tl.float32)
                write_priority = tl.load(
                    write_priorities + write_priority_index
                ).to(tl.float32)

                logits = tl.where(
                    valid_slot, -priorities / TEMPERATURE, -float("inf")
                )
                logits = logits - tl.max(logits, axis=0)
                soft_slots = tl.exp(logits)
                soft_slots = soft_slots / tl.sum(soft_slots, axis=0)
                replacement = tl.argmax(soft_slots, axis=0)
                hard_slots = (slot == replacement) & valid_slot
                if HARD_OVERWRITE:
                    lowest_priority = tl.min(
                        tl.where(valid_slot, priorities, float("inf")), axis=0
                    )
                    overwrite = (write_priority > lowest_priority).to(
                        tl.float32
                    )
                else:
                    lowest_priority = tl.sum(
                        soft_slots * priorities, axis=0
                    )
                    overwrite = tl.sigmoid(
                        (write_priority - lowest_priority) / TEMPERATURE
                    )
                update = hard_slots * overwrite
                values = (
                    values * (1.0 - update[:, None])
                    + write_value[None, :] * update[:, None]
                )
                keys = (
                    keys * (1.0 - update[:, None])
                    + write_key[None, :] * update[:, None]
                )
                priorities = (
                    priorities * (1.0 - update)
                    + write_priority * update
                )

        final_value_base = (batch * S) * M
        final_key_base = (batch * S) * K
        final_priority_base = batch * S
        tl.store(
            final_values + final_value_base + slot_memory,
            values,
            mask=valid_slot[:, None] & valid_memory[None, :],
        )
        tl.store(
            final_keys + final_key_base + slot_key,
            keys,
            mask=valid_slot[:, None] & valid_key[None, :],
        )
        tl.store(
            final_priorities + final_priority_base + slot,
            priorities,
            mask=valid_slot,
        )


    @triton.jit
    def _backward_kernel(
        queries,
        write_values,
        write_keys,
        write_priorities,
        grad_reads,
        grad_final_values,
        grad_final_keys,
        grad_final_priorities,
        trace_values,
        trace_keys,
        trace_priorities,
        grad_queries,
        grad_write_values,
        grad_write_keys,
        grad_write_priorities,
        T: tl.constexpr,
        W: tl.constexpr,
        S: tl.constexpr,
        M: tl.constexpr,
        K: tl.constexpr,
        BS: tl.constexpr,
        BM: tl.constexpr,
        BK: tl.constexpr,
        TEMPERATURE: tl.constexpr,
    ):
        batch = tl.program_id(0)
        slot = tl.arange(0, BS)
        memory = tl.arange(0, BM)
        key_dim = tl.arange(0, BK)
        slot_memory = slot[:, None] * M + memory[None, :]
        slot_key = slot[:, None] * K + key_dim[None, :]
        valid_slot = slot < S
        valid_memory = memory < M
        valid_key = key_dim < K
        final_value_base = (batch * S) * M
        final_key_base = (batch * S) * K
        final_priority_base = batch * S
        grad_values = tl.load(
            grad_final_values + final_value_base + slot_memory,
            mask=valid_slot[:, None] & valid_memory[None, :],
            other=0.0,
        ).to(tl.float32)
        grad_keys = tl.load(
            grad_final_keys + final_key_base + slot_key,
            mask=valid_slot[:, None] & valid_key[None, :],
            other=0.0,
        ).to(tl.float32)
        grad_priorities = tl.load(
            grad_final_priorities + final_priority_base + slot,
            mask=valid_slot,
            other=0.0,
        ).to(tl.float32)
        inv_sqrt_key: tl.constexpr = 1.0 / (K**0.5)

        for reverse_position in range(T):
            position: tl.constexpr = T - 1 - reverse_position
            for reverse_writer in range(W):
                writer: tl.constexpr = W - 1 - reverse_writer
                trace_value_base = (
                    ((batch * T + position) * W + writer) * S
                ) * M
                trace_key_base = (
                    ((batch * T + position) * W + writer) * S
                ) * K
                trace_priority_base = (
                    (batch * T + position) * W + writer
                ) * S
                old_values = tl.load(
                    trace_values + trace_value_base + slot_memory,
                    mask=valid_slot[:, None] & valid_memory[None, :],
                    other=0.0,
                ).to(tl.float32)
                old_keys = tl.load(
                    trace_keys + trace_key_base + slot_key,
                    mask=valid_slot[:, None] & valid_key[None, :],
                    other=0.0,
                ).to(tl.float32)
                old_priorities = tl.load(
                    trace_priorities + trace_priority_base + slot,
                    mask=valid_slot,
                    other=0.0,
                ).to(tl.float32)

                write_value_base = ((batch * W + writer) * T + position) * M
                write_key_base = ((batch * W + writer) * T + position) * K
                write_priority_index = (batch * W + writer) * T + position
                write_value = tl.load(
                    write_values + write_value_base + memory,
                    mask=valid_memory,
                    other=0.0,
                ).to(tl.float32)
                write_key = tl.load(
                    write_keys + write_key_base + key_dim,
                    mask=valid_key,
                    other=0.0,
                ).to(tl.float32)
                write_priority = tl.load(
                    write_priorities + write_priority_index
                ).to(tl.float32)

                logits = tl.where(
                    valid_slot,
                    -old_priorities / TEMPERATURE,
                    -float("inf"),
                )
                logits = logits - tl.max(logits, axis=0)
                soft_slots = tl.exp(logits)
                soft_slots = soft_slots / tl.sum(soft_slots, axis=0)
                replacement = tl.argmax(soft_slots, axis=0)
                hard_slots = (slot == replacement) & valid_slot
                lowest_priority = tl.sum(
                    soft_slots * old_priorities, axis=0
                )
                overwrite = tl.sigmoid(
                    (write_priority - lowest_priority) / TEMPERATURE
                )
                update = hard_slots * overwrite

                grad_update = tl.sum(
                    grad_values * (write_value[None, :] - old_values),
                    axis=1,
                )
                grad_update += tl.sum(
                    grad_keys * (write_key[None, :] - old_keys),
                    axis=1,
                )
                grad_update += grad_priorities * (
                    write_priority - old_priorities
                )
                grad_write_value = tl.sum(
                    grad_values * update[:, None], axis=0
                )
                grad_write_key = tl.sum(
                    grad_keys * update[:, None], axis=0
                )
                grad_write_priority = tl.sum(
                    grad_priorities * update, axis=0
                )
                grad_old_values = grad_values * (1.0 - update[:, None])
                grad_old_keys = grad_keys * (1.0 - update[:, None])
                grad_old_priorities = grad_priorities * (1.0 - update)

                grad_slot_weights = grad_update * overwrite
                grad_overwrite = tl.sum(
                    grad_update * hard_slots, axis=0
                )
                grad_activation = (
                    grad_overwrite
                    * overwrite
                    * (1.0 - overwrite)
                    / TEMPERATURE
                )
                grad_write_priority += grad_activation
                grad_lowest = -grad_activation
                grad_soft = (
                    grad_slot_weights + grad_lowest * old_priorities
                )
                grad_old_priorities += grad_lowest * soft_slots
                soft_inner = tl.sum(soft_slots * grad_soft, axis=0)
                grad_logits = soft_slots * (grad_soft - soft_inner)
                grad_old_priorities += -grad_logits / TEMPERATURE

                tl.store(
                    grad_write_values + write_value_base + memory,
                    grad_write_value,
                    mask=valid_memory,
                )
                tl.store(
                    grad_write_keys + write_key_base + key_dim,
                    grad_write_key,
                    mask=valid_key,
                )
                tl.store(
                    grad_write_priorities + write_priority_index,
                    grad_write_priority,
                )
                grad_values = grad_old_values
                grad_keys = grad_old_keys
                grad_priorities = grad_old_priorities

            # Writer zero's pre-update state is the state used by the read.
            read_value_base = ((batch * T + position) * W * S) * M
            read_key_base = ((batch * T + position) * W * S) * K
            read_priority_base = (batch * T + position) * W * S
            read_values = tl.load(
                trace_values + read_value_base + slot_memory,
                mask=valid_slot[:, None] & valid_memory[None, :],
                other=0.0,
            ).to(tl.float32)
            read_keys = tl.load(
                trace_keys + read_key_base + slot_key,
                mask=valid_slot[:, None] & valid_key[None, :],
                other=0.0,
            ).to(tl.float32)
            read_priorities = tl.load(
                trace_priorities + read_priority_base + slot,
                mask=valid_slot,
                other=0.0,
            ).to(tl.float32)
            query_base = (batch * T + position) * K
            query = tl.load(
                queries + query_base + key_dim,
                mask=valid_key,
                other=0.0,
            ).to(tl.float32)
            scores = tl.sum(read_keys * query[None, :], axis=1) * inv_sqrt_key
            occupied = valid_slot & (read_priorities > 0.0)
            scores = tl.where(occupied, scores, -1.0e4)
            maximum = tl.max(scores, axis=0)
            weights = tl.exp(scores - maximum) * occupied
            weights = weights / tl.maximum(tl.sum(weights, axis=0), 1.0e-8)
            grad_read_base = (batch * T + position) * M
            grad_read = tl.load(
                grad_reads + grad_read_base + memory,
                mask=valid_memory,
                other=0.0,
            ).to(tl.float32)
            grad_values += weights[:, None] * grad_read[None, :]
            grad_weights = tl.sum(
                read_values * grad_read[None, :], axis=1
            )
            weight_inner = tl.sum(weights * grad_weights, axis=0)
            grad_scores = weights * (grad_weights - weight_inner)
            grad_keys += (
                grad_scores[:, None] * query[None, :] * inv_sqrt_key
            )
            grad_query = tl.sum(
                grad_scores[:, None] * read_keys, axis=0
            ) * inv_sqrt_key
            tl.store(
                grad_queries + query_base + key_dim,
                grad_query,
                mask=valid_key,
            )


class _SoftmaxMemoryFunction(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        queries: Tensor,
        write_values: Tensor,
        write_keys: Tensor,
        write_priorities: Tensor,
        n_slots: int,
        temperature: float,
        hard_overwrite: bool,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        batch_size, n_writers, seq_len, d_memory = write_values.shape
        d_key = queries.size(-1)
        n_slots = ctx.n_slots = int(n_slots)
        block_slots = triton.next_power_of_2(n_slots)
        block_memory = triton.next_power_of_2(d_memory)
        block_key = triton.next_power_of_2(d_key)
        reads = torch.empty(
            batch_size, seq_len, d_memory, device=queries.device, dtype=queries.dtype
        )
        final_values = torch.empty(
            batch_size, n_slots, d_memory,
            device=queries.device,
            dtype=write_values.dtype,
        )
        final_keys = torch.empty(
            batch_size, n_slots, d_key,
            device=queries.device,
            dtype=write_keys.dtype,
        )
        final_priorities = torch.empty(
            batch_size, n_slots,
            device=queries.device,
            dtype=write_priorities.dtype,
        )
        save_trace = not hard_overwrite
        if save_trace:
            trace_values = torch.empty(
                batch_size, seq_len, n_writers, n_slots, d_memory,
                device=queries.device,
                dtype=write_values.dtype,
            )
            trace_keys = torch.empty(
                batch_size, seq_len, n_writers, n_slots, d_key,
                device=queries.device,
                dtype=write_keys.dtype,
            )
            trace_priorities = torch.empty(
                batch_size, seq_len, n_writers, n_slots,
                device=queries.device,
                dtype=write_priorities.dtype,
            )
        else:
            # The inference kernel never touches these placeholders. Avoid
            # allocating multi-gigabyte backward traces during capacity and
            # long-context evaluation.
            trace_values = write_values.new_empty(1)
            trace_keys = write_keys.new_empty(1)
            trace_priorities = write_priorities.new_empty(1)
        with torch.cuda.device(queries.device):
            _forward_kernel[(batch_size,)](
                queries,
                write_values,
                write_keys,
                write_priorities,
                reads,
                final_values,
                final_keys,
                final_priorities,
                trace_values,
                trace_keys,
                trace_priorities,
                T=seq_len,
                W=n_writers,
                S=n_slots,
                M=d_memory,
                K=d_key,
                BS=block_slots,
                BM=block_memory,
                BK=block_key,
                TEMPERATURE=temperature,
                SAVE_TRACE=save_trace,
                HARD_OVERWRITE=hard_overwrite,
                num_warps=4,
            )
        ctx.temperature = temperature
        ctx.hard_overwrite = hard_overwrite
        if save_trace:
            ctx.save_for_backward(
                queries,
                write_values,
                write_keys,
                write_priorities,
                trace_values,
                trace_keys,
                trace_priorities,
            )
        return reads, final_values, final_keys, final_priorities

    @staticmethod
    def backward(
        ctx,
        grad_reads: Tensor | None,
        grad_final_values: Tensor | None,
        grad_final_keys: Tensor | None,
        grad_final_priorities: Tensor | None,
    ) -> tuple[Tensor | None, ...]:
        if ctx.hard_overwrite:
            raise RuntimeError(
                "hard-overwrite fused memory is inference-only"
            )
        (
            queries,
            write_values,
            write_keys,
            write_priorities,
            trace_values,
            trace_keys,
            trace_priorities,
        ) = ctx.saved_tensors
        batch_size, n_writers, seq_len, d_memory = write_values.shape
        d_key = queries.size(-1)
        n_slots = ctx.n_slots
        grad_reads = (
            torch.zeros(batch_size, seq_len, d_memory, device=queries.device,
                        dtype=queries.dtype)
            if grad_reads is None else grad_reads.contiguous()
        )
        grad_final_values = (
            torch.zeros(batch_size, n_slots, d_memory, device=queries.device,
                        dtype=write_values.dtype)
            if grad_final_values is None else grad_final_values.contiguous()
        )
        grad_final_keys = (
            torch.zeros(batch_size, n_slots, d_key, device=queries.device,
                        dtype=write_keys.dtype)
            if grad_final_keys is None else grad_final_keys.contiguous()
        )
        grad_final_priorities = (
            torch.zeros(batch_size, n_slots, device=queries.device,
                        dtype=write_priorities.dtype)
            if grad_final_priorities is None else grad_final_priorities.contiguous()
        )
        grad_queries = torch.empty_like(queries)
        grad_write_values = torch.empty_like(write_values)
        grad_write_keys = torch.empty_like(write_keys)
        grad_write_priorities = torch.empty_like(write_priorities)
        with torch.cuda.device(queries.device):
            _backward_kernel[(batch_size,)](
                queries,
                write_values,
                write_keys,
                write_priorities,
                grad_reads,
                grad_final_values,
                grad_final_keys,
                grad_final_priorities,
                trace_values,
                trace_keys,
                trace_priorities,
                grad_queries,
                grad_write_values,
                grad_write_keys,
                grad_write_priorities,
                T=seq_len,
                W=n_writers,
                S=n_slots,
                M=d_memory,
                K=d_key,
                BS=triton.next_power_of_2(n_slots),
                BM=triton.next_power_of_2(d_memory),
                BK=triton.next_power_of_2(d_key),
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
            None,
        )


def fused_softmax_memory(
    queries: Tensor,
    write_values: Tensor,
    write_keys: Tensor,
    write_priorities: Tensor,
    n_slots: int,
    temperature: float,
    hard_overwrite: bool = False,
) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    """Run a fused read-before-write recurrence.

    Write tensors use ``[batch, writer, time, feature]`` layout, where prior
    block histories precede the current block's writes.
    """
    if not _TRITON_AVAILABLE:
        raise RuntimeError("Triton is not installed")
    if not queries.is_cuda:
        raise ValueError("fused memory requires a CUDA/ROCm tensor")
    return _SoftmaxMemoryFunction.apply(
        queries.contiguous(),
        write_values.contiguous(),
        write_keys.contiguous(),
        write_priorities.contiguous(),
        int(n_slots),
        float(temperature),
        bool(hard_overwrite),
    )


__all__ = ["fused_softmax_memory", "triton_memory_available"]
