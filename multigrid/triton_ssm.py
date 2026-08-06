"""Fused Triton kernels for the diagonal SSM recurrence."""

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


_BLOCK_SIZE = 256
_SUPPORTED_DTYPES = (torch.float16, torch.bfloat16, torch.float32)


def triton_ssm_available() -> bool:
    """Return whether the optional Triton package can be imported."""
    return _TRITON_AVAILABLE


if _TRITON_AVAILABLE:

    @triton.jit
    def _ssm_forward_kernel(
        candidate,
        gate,
        decay,
        output,
        state_trace,
        T: tl.constexpr,
        D: tl.constexpr,
        BLOCK: tl.constexpr,
        SAVE_TRACE: tl.constexpr,
    ):
        batch = tl.program_id(0)
        feature = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
        feature_mask = feature < D
        decay_value = tl.load(
            decay + feature, mask=feature_mask, other=0.0
        ).to(tl.float32)
        state = tl.zeros((BLOCK,), dtype=tl.float32)

        for position in range(T):
            offset = (batch * T + position) * D + feature
            candidate_value = tl.load(
                candidate + offset, mask=feature_mask, other=0.0
            ).to(tl.float32)
            gate_value = tl.load(
                gate + offset, mask=feature_mask, other=0.0
            ).to(tl.float32)
            state = (
                decay_value * state
                + (1.0 - decay_value) * candidate_value
            )
            tl.store(
                output + offset,
                state * tl.sigmoid(gate_value),
                mask=feature_mask,
            )
            if SAVE_TRACE:
                tl.store(state_trace + offset, state, mask=feature_mask)


    @triton.jit
    def _ssm_backward_kernel(
        candidate,
        gate,
        decay,
        state_trace,
        grad_output,
        grad_candidate,
        grad_gate,
        grad_decay_partial,
        T: tl.constexpr,
        D: tl.constexpr,
        BLOCK: tl.constexpr,
        WRITE_CANDIDATE: tl.constexpr,
        WRITE_GATE: tl.constexpr,
        WRITE_DECAY: tl.constexpr,
    ):
        batch = tl.program_id(0)
        feature = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
        feature_mask = feature < D
        decay_value = tl.load(
            decay + feature, mask=feature_mask, other=0.0
        ).to(tl.float32)
        decay_gradient = tl.zeros((BLOCK,), dtype=tl.float32)

        # Batch rows are independent reverse scans, so give each one its own
        # program. A tiny second kernel reduces their partial decay gradients.
        state_gradient = tl.zeros((BLOCK,), dtype=tl.float32)
        for reverse_position in range(T):
            position = T - 1 - reverse_position
            offset = (batch * T + position) * D + feature
            output_gradient = tl.load(
                grad_output + offset, mask=feature_mask, other=0.0
            ).to(tl.float32)
            gate_value = tl.load(
                gate + offset, mask=feature_mask, other=0.0
            ).to(tl.float32)
            gate_activation = tl.sigmoid(gate_value)
            state = tl.load(
                state_trace + offset, mask=feature_mask, other=0.0
            ).to(tl.float32)
            state_gradient += output_gradient * gate_activation

            if WRITE_GATE:
                tl.store(
                    grad_gate + offset,
                    output_gradient
                    * state
                    * gate_activation
                    * (1.0 - gate_activation),
                    mask=feature_mask,
                )
            if WRITE_CANDIDATE:
                tl.store(
                    grad_candidate + offset,
                    state_gradient * (1.0 - decay_value),
                    mask=feature_mask,
                )
            if WRITE_DECAY:
                candidate_value = tl.load(
                    candidate + offset,
                    mask=feature_mask,
                    other=0.0,
                ).to(tl.float32)
                if position == 0:
                    previous_state = tl.zeros(
                        (BLOCK,), dtype=tl.float32
                    )
                else:
                    previous_offset = offset - D
                    previous_state = tl.load(
                        state_trace + previous_offset,
                        mask=feature_mask,
                        other=0.0,
                    ).to(tl.float32)
                decay_gradient += state_gradient * (
                    previous_state - candidate_value
                )

            state_gradient *= decay_value

        if WRITE_DECAY:
            tl.store(
                grad_decay_partial + batch * D + feature,
                decay_gradient,
                mask=feature_mask,
            )


    @triton.jit
    def _ssm_decay_reduce_kernel(
        grad_decay_partial,
        grad_decay,
        B: tl.constexpr,
        D: tl.constexpr,
        BLOCK: tl.constexpr,
    ):
        feature = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
        feature_mask = feature < D
        total = tl.zeros((BLOCK,), dtype=tl.float32)
        for batch in range(B):
            total += tl.load(
                grad_decay_partial + batch * D + feature,
                mask=feature_mask,
                other=0.0,
            )
        tl.store(grad_decay + feature, total, mask=feature_mask)


class _DiagonalSSMFunction(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        candidate: Tensor,
        gate: Tensor,
        decay: Tensor,
        save_trace: bool,
    ) -> Tensor:
        batch_size, seq_len, width = candidate.shape
        output = torch.empty_like(candidate)

        # The trace is needed only for the reverse recurrence. Keeping it in
        # FP32 matches the accumulator used by both kernels and avoids any
        # trace allocation at all under inference_mode/no_grad.
        if save_trace:
            state_trace = torch.empty(
                candidate.shape,
                device=candidate.device,
                dtype=torch.float32,
            )
        else:
            state_trace = torch.empty(
                0, device=candidate.device, dtype=torch.float32
            )

        grid = (batch_size, triton.cdiv(width, _BLOCK_SIZE))
        with torch.cuda.device(candidate.device):
            _ssm_forward_kernel[grid](
                candidate,
                gate,
                decay,
                output,
                state_trace,
                T=seq_len,
                D=width,
                BLOCK=_BLOCK_SIZE,
                SAVE_TRACE=save_trace,
                num_warps=4,
            )

        ctx.save_trace = save_trace
        if save_trace:
            ctx.save_for_backward(candidate, gate, decay, state_trace)
            ctx.input_needs_grad = tuple(ctx.needs_input_grad[:3])
        return output

    @staticmethod
    def backward(
        ctx, grad_output: Tensor | None
    ) -> tuple[Tensor | None, Tensor | None, Tensor | None, None]:
        if not ctx.save_trace:
            raise RuntimeError("SSM backward requested without a state trace")
        candidate, gate, decay, state_trace = ctx.saved_tensors
        batch_size, seq_len, width = candidate.shape
        need_candidate, need_gate, need_decay = ctx.input_needs_grad
        if grad_output is None:
            grad_output = torch.zeros_like(candidate)
        else:
            grad_output = grad_output.contiguous()

        grad_candidate = (
            torch.empty_like(candidate)
            if need_candidate
            else candidate.new_empty(0)
        )
        grad_gate = (
            torch.empty_like(gate) if need_gate else gate.new_empty(0)
        )
        grad_decay = (
            torch.empty_like(decay) if need_decay else decay.new_empty(0)
        )
        grad_decay_partial = (
            torch.empty(
                batch_size,
                width,
                device=decay.device,
                dtype=torch.float32,
            )
            if need_decay
            else decay.new_empty(0)
        )

        grid = (batch_size, triton.cdiv(width, _BLOCK_SIZE))
        with torch.cuda.device(candidate.device):
            _ssm_backward_kernel[grid](
                candidate,
                gate,
                decay,
                state_trace,
                grad_output,
                grad_candidate,
                grad_gate,
                grad_decay_partial,
                T=seq_len,
                D=width,
                BLOCK=_BLOCK_SIZE,
                WRITE_CANDIDATE=need_candidate,
                WRITE_GATE=need_gate,
                WRITE_DECAY=need_decay,
                num_warps=4,
            )
            if need_decay:
                _ssm_decay_reduce_kernel[
                    (triton.cdiv(width, _BLOCK_SIZE),)
                ](
                    grad_decay_partial,
                    grad_decay,
                    B=batch_size,
                    D=width,
                    BLOCK=_BLOCK_SIZE,
                    num_warps=4,
                )
        return (
            grad_candidate if need_candidate else None,
            grad_gate if need_gate else None,
            grad_decay if need_decay else None,
            None,
        )


def fused_diagonal_ssm(
    candidate: Tensor,
    gate: Tensor,
    decay: Tensor,
) -> Tensor:
    """Run a fused causal diagonal SSM recurrence.

    ``candidate`` and ``gate`` use ``[batch, time, feature]`` layout and
    ``decay`` has one value per feature. FP16, BF16, and FP32 inputs are
    supported on CUDA and ROCm devices.
    """
    if not _TRITON_AVAILABLE:
        raise RuntimeError("Triton is not installed")
    if candidate.ndim != 3:
        raise ValueError("candidate must have shape [batch, time, feature]")
    if gate.shape != candidate.shape:
        raise ValueError("gate must have the same shape as candidate")
    if decay.ndim != 1 or decay.numel() != candidate.size(2):
        raise ValueError("decay must have shape [feature]")
    if candidate.size(0) == 0 or candidate.size(1) == 0 or candidate.size(2) == 0:
        raise ValueError("candidate dimensions must be non-zero")
    if not candidate.is_cuda:
        raise ValueError("fused diagonal SSM requires a CUDA/ROCm tensor")
    if gate.device != candidate.device or decay.device != candidate.device:
        raise ValueError("candidate, gate, and decay must be on the same device")
    if candidate.dtype not in _SUPPORTED_DTYPES:
        raise TypeError("candidate must use FP16, BF16, or FP32")
    if gate.dtype != candidate.dtype:
        raise TypeError("gate must have the same dtype as candidate")
    if decay.dtype not in _SUPPORTED_DTYPES:
        raise TypeError("decay must use FP16, BF16, or FP32")

    save_trace = torch.is_grad_enabled() and (
        candidate.requires_grad or gate.requires_grad or decay.requires_grad
    )
    return _DiagonalSSMFunction.apply(
        candidate.contiguous(),
        gate.contiguous(),
        decay.contiguous(),
        save_trace,
    )


__all__ = ["fused_diagonal_ssm", "triton_ssm_available"]
