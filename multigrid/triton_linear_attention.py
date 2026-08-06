"""Fused Triton recurrence for causal gated linear attention.

The public function in this module intentionally lives separately from the
mechanism using the eager recurrence.  This keeps the optional Triton import
safe on CPU-only installations and makes it possible to compare both paths.
"""

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


_MIN_DENOMINATOR = 1.0e-6
_MAX_HEAD_WIDTH = 128


def triton_linear_attention_available() -> bool:
    """Return whether the optional Triton package can be imported."""
    return _TRITON_AVAILABLE


if _TRITON_AVAILABLE:

    @triton.jit
    def _linear_attention_forward_kernel(
        query,
        key,
        value,
        decays,
        output,
        state_trace,
        normalizer_trace,
        T: tl.constexpr,
        D: tl.constexpr,
        BD: tl.constexpr,
        SAVE_TRACE: tl.constexpr,
        EPSILON: tl.constexpr,
    ):
        batch_head = tl.program_id(0)
        row = tl.arange(0, BD)
        column = tl.arange(0, BD)
        matrix_offset = row[:, None] * D + column[None, :]
        valid = row < D
        valid_matrix = valid[:, None] & (column[None, :] < D)
        state = tl.zeros((BD, BD), dtype=tl.float32)
        normalizer = tl.zeros((BD,), dtype=tl.float32)

        for position in range(T):
            vector_base = (batch_head * T + position) * D
            if SAVE_TRACE:
                trace_base = (batch_head * T + position) * D * D
                normalizer_base = (batch_head * T + position) * D
                tl.store(
                    state_trace + trace_base + matrix_offset,
                    state,
                    mask=valid_matrix,
                )
                tl.store(
                    normalizer_trace + normalizer_base + row,
                    normalizer,
                    mask=valid,
                )

            query_t = tl.load(
                query + vector_base + row, mask=valid, other=0.0
            ).to(tl.float32)
            key_t = tl.load(
                key + vector_base + row, mask=valid, other=0.0
            ).to(tl.float32)
            value_t = tl.load(
                value + vector_base + column,
                mask=column < D,
                other=0.0,
            ).to(tl.float32)
            decay_t = tl.load(decays + batch_head * T + position).to(
                tl.float32
            )
            state = decay_t * state + key_t[:, None] * value_t[None, :]
            normalizer = decay_t * normalizer + key_t
            numerator = tl.sum(query_t[:, None] * state, axis=0)
            denominator = tl.sum(query_t * normalizer, axis=0)
            denominator = tl.maximum(denominator, EPSILON)
            tl.store(
                output + vector_base + column,
                numerator / denominator,
                mask=column < D,
            )


    @triton.jit
    def _linear_attention_backward_kernel(
        query,
        key,
        value,
        decays,
        grad_output,
        state_trace,
        normalizer_trace,
        grad_query,
        grad_key,
        grad_value,
        grad_decays,
        T: tl.constexpr,
        D: tl.constexpr,
        BD: tl.constexpr,
        EPSILON: tl.constexpr,
    ):
        batch_head = tl.program_id(0)
        row = tl.arange(0, BD)
        column = tl.arange(0, BD)
        matrix_offset = row[:, None] * D + column[None, :]
        valid = row < D
        valid_column = column < D
        valid_matrix = valid[:, None] & valid_column[None, :]
        carried_state_gradient = tl.zeros((BD, BD), dtype=tl.float32)
        carried_normalizer_gradient = tl.zeros((BD,), dtype=tl.float32)

        for reverse_position in range(T):
            position = T - 1 - reverse_position
            vector_base = (batch_head * T + position) * D
            trace_base = (batch_head * T + position) * D * D
            normalizer_base = (batch_head * T + position) * D
            previous_state = tl.load(
                state_trace + trace_base + matrix_offset,
                mask=valid_matrix,
                other=0.0,
            ).to(tl.float32)
            previous_normalizer = tl.load(
                normalizer_trace + normalizer_base + row,
                mask=valid,
                other=0.0,
            ).to(tl.float32)
            query_t = tl.load(
                query + vector_base + row, mask=valid, other=0.0
            ).to(tl.float32)
            key_t = tl.load(
                key + vector_base + row, mask=valid, other=0.0
            ).to(tl.float32)
            value_t = tl.load(
                value + vector_base + column,
                mask=valid_column,
                other=0.0,
            ).to(tl.float32)
            output_gradient = tl.load(
                grad_output + vector_base + column,
                mask=valid_column,
                other=0.0,
            ).to(tl.float32)
            decay_t = tl.load(decays + batch_head * T + position).to(
                tl.float32
            )

            state = (
                decay_t * previous_state
                + key_t[:, None] * value_t[None, :]
            )
            normalizer = decay_t * previous_normalizer + key_t
            numerator = tl.sum(query_t[:, None] * state, axis=0)
            raw_denominator = tl.sum(query_t * normalizer, axis=0)
            denominator = tl.maximum(raw_denominator, EPSILON)
            numerator_gradient = output_gradient / denominator
            denominator_gradient = -tl.sum(
                output_gradient * numerator, axis=0
            ) / (denominator * denominator)
            denominator_gradient = tl.where(
                raw_denominator > EPSILON, denominator_gradient, 0.0
            )

            output_state_gradient = (
                query_t[:, None] * numerator_gradient[None, :]
            )
            state_gradient = (
                carried_state_gradient + output_state_gradient
            )
            normalizer_gradient = (
                carried_normalizer_gradient
                + denominator_gradient * query_t
            )

            query_gradient = tl.sum(
                state * numerator_gradient[None, :], axis=1
            ) + denominator_gradient * normalizer
            key_gradient = tl.sum(
                state_gradient * value_t[None, :], axis=1
            ) + normalizer_gradient
            value_gradient = tl.sum(
                state_gradient * key_t[:, None], axis=0
            )
            state_decay_gradient = tl.sum(
                tl.sum(state_gradient * previous_state, axis=1), axis=0
            )
            normalizer_decay_gradient = tl.sum(
                normalizer_gradient * previous_normalizer, axis=0
            )

            tl.store(
                grad_query + vector_base + row,
                query_gradient,
                mask=valid,
            )
            tl.store(
                grad_key + vector_base + row,
                key_gradient,
                mask=valid,
            )
            tl.store(
                grad_value + vector_base + column,
                value_gradient,
                mask=valid_column,
            )
            tl.store(
                grad_decays + batch_head * T + position,
                state_decay_gradient + normalizer_decay_gradient,
            )
            carried_state_gradient = decay_t * state_gradient
            carried_normalizer_gradient = decay_t * normalizer_gradient


class _LinearAttentionFunction(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        query: Tensor,
        key: Tensor,
        value: Tensor,
        decays: Tensor,
        save_trace: bool,
    ) -> Tensor:
        batch_size, n_heads, seq_len, head_width = query.shape
        batch_heads = batch_size * n_heads
        block_width = triton.next_power_of_2(head_width)
        output = torch.empty_like(query)
        if save_trace:
            # FP32 traces preserve the kernel's accumulator precision.  They
            # are allocated only when a backward pass can actually occur.
            state_trace = torch.empty(
                batch_size,
                n_heads,
                seq_len,
                head_width,
                head_width,
                device=query.device,
                dtype=torch.float32,
            )
            normalizer_trace = torch.empty(
                batch_size,
                n_heads,
                seq_len,
                head_width,
                device=query.device,
                dtype=torch.float32,
            )
        else:
            state_trace = query.new_empty(1)
            normalizer_trace = query.new_empty(1)

        with torch.cuda.device(query.device):
            _linear_attention_forward_kernel[(batch_heads,)](
                query,
                key,
                value,
                decays,
                output,
                state_trace,
                normalizer_trace,
                T=seq_len,
                D=head_width,
                BD=block_width,
                SAVE_TRACE=save_trace,
                EPSILON=_MIN_DENOMINATOR,
                num_warps=4,
            )
        ctx.save_trace = save_trace
        ctx.set_materialize_grads(False)
        if save_trace:
            ctx.save_for_backward(
                query,
                key,
                value,
                decays,
                state_trace,
                normalizer_trace,
            )
        return output

    @staticmethod
    def backward(
        ctx, grad_output: Tensor | None
    ) -> tuple[Tensor | None, ...]:
        if not ctx.save_trace:
            raise RuntimeError("linear-attention forward did not save a trace")
        if grad_output is None:
            return None, None, None, None, None
        (
            query,
            key,
            value,
            decays,
            state_trace,
            normalizer_trace,
        ) = ctx.saved_tensors
        batch_size, n_heads, seq_len, head_width = query.shape
        batch_heads = batch_size * n_heads
        grad_query = torch.empty_like(query)
        grad_key = torch.empty_like(key)
        grad_value = torch.empty_like(value)
        grad_decays = torch.empty_like(decays)
        with torch.cuda.device(query.device):
            _linear_attention_backward_kernel[(batch_heads,)](
                query,
                key,
                value,
                decays,
                grad_output.contiguous(),
                state_trace,
                normalizer_trace,
                grad_query,
                grad_key,
                grad_value,
                grad_decays,
                T=seq_len,
                D=head_width,
                BD=triton.next_power_of_2(head_width),
                EPSILON=_MIN_DENOMINATOR,
                num_warps=4,
            )
        return grad_query, grad_key, grad_value, grad_decays, None


def _validate_inputs(
    query: Tensor, key: Tensor, value: Tensor, decays: Tensor
) -> None:
    if not _TRITON_AVAILABLE:
        raise RuntimeError("Triton is not installed")
    if not query.is_cuda:
        raise ValueError("fused linear attention requires a CUDA/ROCm tensor")
    if query.ndim != 4:
        raise ValueError("query must have shape [batch, head, time, feature]")
    if key.shape != query.shape or value.shape != query.shape:
        raise ValueError("query, key, and value must have identical shapes")
    if decays.shape != query.shape[:3]:
        raise ValueError("decays must have shape [batch, head, time]")
    tensors = (query, key, value, decays)
    if any(tensor.device != query.device for tensor in tensors[1:]):
        raise ValueError("all inputs must be on the same device")
    if any(tensor.dtype != query.dtype for tensor in tensors[1:]):
        raise ValueError("all inputs must have the same dtype")
    if query.dtype not in (torch.float16, torch.bfloat16, torch.float32):
        raise TypeError("inputs must use float16, bfloat16, or float32")
    if any(dimension == 0 for dimension in query.shape):
        raise ValueError("input dimensions must be non-zero")
    if query.size(-1) > _MAX_HEAD_WIDTH:
        raise ValueError(
            f"head width must be at most {_MAX_HEAD_WIDTH} for the fused kernel"
        )


def fused_linear_attention(
    query: Tensor,
    key: Tensor,
    value: Tensor,
    decays: Tensor,
) -> Tensor:
    """Run the causal linear-attention recurrence with a fused Triton kernel.

    Inputs use ``[batch, head, time, feature]`` layout and decays use
    ``[batch, head, time]``. Accumulation is performed in FP32 and the result
    has the same dtype as ``query``.
    """
    _validate_inputs(query, key, value, decays)
    save_trace = torch.is_grad_enabled() and any(
        tensor.requires_grad for tensor in (query, key, value, decays)
    )
    return _LinearAttentionFunction.apply(
        query.contiguous(),
        key.contiguous(),
        value.contiguous(),
        decays.contiguous(),
        save_trace,
    )


__all__ = ["fused_linear_attention", "triton_linear_attention_available"]
