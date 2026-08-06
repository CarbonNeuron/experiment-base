"""Fused Triton kernels for QuatSpin's Hamilton product and normalization."""

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


def triton_quatspin_available() -> bool:
    """Return whether the optional Triton package can be imported."""
    return _TRITON_AVAILABLE


if _TRITON_AVAILABLE:

    @triton.jit
    def _quatspin_forward_kernel(
        multipliers,
        activations,
        scale,
        output,
        N: tl.constexpr,
        C: tl.constexpr,
        EPSILON: tl.constexpr,
        BLOCK: tl.constexpr,
    ):
        quaternion = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
        mask = quaternion < N
        base = quaternion * 4

        mw = tl.load(multipliers + base, mask=mask, other=0.0).to(tl.float32)
        mx = tl.load(multipliers + base + 1, mask=mask, other=0.0).to(
            tl.float32
        )
        my = tl.load(multipliers + base + 2, mask=mask, other=0.0).to(
            tl.float32
        )
        mz = tl.load(multipliers + base + 3, mask=mask, other=0.0).to(
            tl.float32
        )
        aw = tl.load(activations + base, mask=mask, other=0.0).to(tl.float32)
        ax = tl.load(activations + base + 1, mask=mask, other=0.0).to(
            tl.float32
        )
        ay = tl.load(activations + base + 2, mask=mask, other=0.0).to(
            tl.float32
        )
        az = tl.load(activations + base + 3, mask=mask, other=0.0).to(
            tl.float32
        )

        qw = mw * aw - mx * ax - my * ay - mz * az
        qx = mw * ax + mx * aw + my * az - mz * ay
        qy = mw * ay - mx * az + my * aw + mz * ax
        qz = mw * az + mx * ay - my * ax + mz * aw
        inverse_magnitude = tl.rsqrt(
            tl.maximum(qw * qw + qx * qx + qy * qy + qz * qz, EPSILON * EPSILON)
        )
        channel_scale = tl.load(
            scale + quaternion % C, mask=mask, other=0.0
        ).to(tl.float32)
        factor = inverse_magnitude * channel_scale

        tl.store(output + base, qw * factor, mask=mask)
        tl.store(output + base + 1, qx * factor, mask=mask)
        tl.store(output + base + 2, qy * factor, mask=mask)
        tl.store(output + base + 3, qz * factor, mask=mask)


    @triton.jit
    def _quatspin_input_backward_kernel(
        multipliers,
        activations,
        scale,
        grad_output,
        grad_multipliers,
        grad_activations,
        N: tl.constexpr,
        C: tl.constexpr,
        EPSILON: tl.constexpr,
        BLOCK: tl.constexpr,
        WRITE_MULTIPLIERS: tl.constexpr,
        WRITE_ACTIVATIONS: tl.constexpr,
    ):
        quaternion = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
        mask = quaternion < N
        base = quaternion * 4

        mw = tl.load(multipliers + base, mask=mask, other=0.0).to(tl.float32)
        mx = tl.load(multipliers + base + 1, mask=mask, other=0.0).to(
            tl.float32
        )
        my = tl.load(multipliers + base + 2, mask=mask, other=0.0).to(
            tl.float32
        )
        mz = tl.load(multipliers + base + 3, mask=mask, other=0.0).to(
            tl.float32
        )
        aw = tl.load(activations + base, mask=mask, other=0.0).to(tl.float32)
        ax = tl.load(activations + base + 1, mask=mask, other=0.0).to(
            tl.float32
        )
        ay = tl.load(activations + base + 2, mask=mask, other=0.0).to(
            tl.float32
        )
        az = tl.load(activations + base + 3, mask=mask, other=0.0).to(
            tl.float32
        )
        gw = tl.load(grad_output + base, mask=mask, other=0.0).to(tl.float32)
        gx = tl.load(grad_output + base + 1, mask=mask, other=0.0).to(
            tl.float32
        )
        gy = tl.load(grad_output + base + 2, mask=mask, other=0.0).to(
            tl.float32
        )
        gz = tl.load(grad_output + base + 3, mask=mask, other=0.0).to(
            tl.float32
        )

        qw = mw * aw - mx * ax - my * ay - mz * az
        qx = mw * ax + mx * aw + my * az - mz * ay
        qy = mw * ay - mx * az + my * aw + mz * ax
        qz = mw * az + mx * ay - my * ax + mz * aw
        magnitude_squared = qw * qw + qx * qx + qy * qy + qz * qz
        clamped_squared = tl.maximum(magnitude_squared, EPSILON * EPSILON)
        inverse_magnitude = tl.rsqrt(clamped_squared)
        channel_scale = tl.load(
            scale + quaternion % C, mask=mask, other=0.0
        ).to(tl.float32)
        dot = gw * qw + gx * qx + gy * qy + gz * qz
        projection = tl.where(
            magnitude_squared > EPSILON * EPSILON,
            dot / clamped_squared,
            0.0,
        )
        gradient_factor = channel_scale * inverse_magnitude
        dqw = gradient_factor * (gw - qw * projection)
        dqx = gradient_factor * (gx - qx * projection)
        dqy = gradient_factor * (gy - qy * projection)
        dqz = gradient_factor * (gz - qz * projection)

        if WRITE_MULTIPLIERS:
            dmw = dqw * aw + dqx * ax + dqy * ay + dqz * az
            dmx = -dqw * ax + dqx * aw - dqy * az + dqz * ay
            dmy = -dqw * ay + dqx * az + dqy * aw - dqz * ax
            dmz = -dqw * az - dqx * ay + dqy * ax + dqz * aw
            tl.store(grad_multipliers + base, dmw, mask=mask)
            tl.store(grad_multipliers + base + 1, dmx, mask=mask)
            tl.store(grad_multipliers + base + 2, dmy, mask=mask)
            tl.store(grad_multipliers + base + 3, dmz, mask=mask)

        if WRITE_ACTIVATIONS:
            daw = dqw * mw + dqx * mx + dqy * my + dqz * mz
            dax = -dqw * mx + dqx * mw + dqy * mz - dqz * my
            day = -dqw * my - dqx * mz + dqy * mw + dqz * mx
            daz = -dqw * mz + dqx * my - dqy * mx + dqz * mw
            tl.store(grad_activations + base, daw, mask=mask)
            tl.store(grad_activations + base + 1, dax, mask=mask)
            tl.store(grad_activations + base + 2, day, mask=mask)
            tl.store(grad_activations + base + 3, daz, mask=mask)


    @triton.jit
    def _quatspin_scale_backward_kernel(
        multipliers,
        activations,
        grad_output,
        grad_scale,
        T: tl.constexpr,
        C: tl.constexpr,
        EPSILON: tl.constexpr,
        BLOCK: tl.constexpr,
    ):
        channel = tl.program_id(0)
        token = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
        mask = token < T
        quaternion = token * C + channel
        base = quaternion * 4

        mw = tl.load(multipliers + base, mask=mask, other=0.0).to(tl.float32)
        mx = tl.load(multipliers + base + 1, mask=mask, other=0.0).to(
            tl.float32
        )
        my = tl.load(multipliers + base + 2, mask=mask, other=0.0).to(
            tl.float32
        )
        mz = tl.load(multipliers + base + 3, mask=mask, other=0.0).to(
            tl.float32
        )
        aw = tl.load(activations + base, mask=mask, other=0.0).to(tl.float32)
        ax = tl.load(activations + base + 1, mask=mask, other=0.0).to(
            tl.float32
        )
        ay = tl.load(activations + base + 2, mask=mask, other=0.0).to(
            tl.float32
        )
        az = tl.load(activations + base + 3, mask=mask, other=0.0).to(
            tl.float32
        )
        gw = tl.load(grad_output + base, mask=mask, other=0.0).to(tl.float32)
        gx = tl.load(grad_output + base + 1, mask=mask, other=0.0).to(
            tl.float32
        )
        gy = tl.load(grad_output + base + 2, mask=mask, other=0.0).to(
            tl.float32
        )
        gz = tl.load(grad_output + base + 3, mask=mask, other=0.0).to(
            tl.float32
        )

        qw = mw * aw - mx * ax - my * ay - mz * az
        qx = mw * ax + mx * aw + my * az - mz * ay
        qy = mw * ay - mx * az + my * aw + mz * ax
        qz = mw * az + mx * ay - my * ax + mz * aw
        inverse_magnitude = tl.rsqrt(
            tl.maximum(qw * qw + qx * qx + qy * qy + qz * qz, EPSILON * EPSILON)
        )
        contribution = (
            gw * qw + gx * qx + gy * qy + gz * qz
        ) * inverse_magnitude
        partial = tl.sum(tl.where(mask, contribution, 0.0), axis=0)
        tl.atomic_add(grad_scale + channel, partial)


class _QuatSpinFunction(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        multipliers: Tensor,
        activations: Tensor,
        scale: Tensor,
        eps: float,
        save_inputs: bool,
    ) -> Tensor:
        n_quaternions = multipliers.numel() // 4
        n_channels = scale.numel()
        output_dtype = torch.promote_types(multipliers.dtype, scale.dtype)
        output = torch.empty_like(multipliers, dtype=output_dtype)
        with torch.cuda.device(multipliers.device):
            _quatspin_forward_kernel[
                (triton.cdiv(n_quaternions, _BLOCK_SIZE),)
            ](
                multipliers,
                activations,
                scale,
                output,
                N=n_quaternions,
                C=n_channels,
                EPSILON=eps,
                BLOCK=_BLOCK_SIZE,
                num_warps=4,
            )

        ctx.save_inputs = save_inputs
        ctx.eps = eps
        ctx.set_materialize_grads(False)
        if save_inputs:
            ctx.save_for_backward(multipliers, activations, scale)
            ctx.input_needs_grad = tuple(ctx.needs_input_grad[:3])
        return output

    @staticmethod
    def backward(
        ctx,
        grad_output: Tensor | None,
    ) -> tuple[Tensor | None, Tensor | None, Tensor | None, None, None]:
        if not ctx.save_inputs:
            raise RuntimeError("QuatSpin backward requested without saved inputs")
        if grad_output is None:
            return None, None, None, None, None
        multipliers, activations, scale = ctx.saved_tensors
        need_multipliers, need_activations, need_scale = ctx.input_needs_grad
        n_quaternions = multipliers.numel() // 4
        n_channels = scale.numel()
        n_tokens = n_quaternions // n_channels
        grad_multipliers = (
            torch.empty_like(multipliers)
            if need_multipliers
            else multipliers.new_empty(0)
        )
        grad_activations = (
            torch.empty_like(activations)
            if need_activations
            else activations.new_empty(0)
        )
        grad_output = grad_output.contiguous()

        with torch.cuda.device(multipliers.device):
            if need_multipliers or need_activations:
                _quatspin_input_backward_kernel[
                    (triton.cdiv(n_quaternions, _BLOCK_SIZE),)
                ](
                    multipliers,
                    activations,
                    scale,
                    grad_output,
                    grad_multipliers,
                    grad_activations,
                    N=n_quaternions,
                    C=n_channels,
                    EPSILON=ctx.eps,
                    BLOCK=_BLOCK_SIZE,
                    WRITE_MULTIPLIERS=need_multipliers,
                    WRITE_ACTIVATIONS=need_activations,
                    num_warps=4,
                )

            if need_scale:
                grad_scale_float = torch.zeros(
                    n_channels,
                    device=scale.device,
                    dtype=torch.float32,
                )
                _quatspin_scale_backward_kernel[
                    (n_channels, triton.cdiv(n_tokens, _BLOCK_SIZE))
                ](
                    multipliers,
                    activations,
                    grad_output,
                    grad_scale_float,
                    T=n_tokens,
                    C=n_channels,
                    EPSILON=ctx.eps,
                    BLOCK=_BLOCK_SIZE,
                    num_warps=4,
                )
                grad_scale = grad_scale_float.to(scale.dtype)
            else:
                grad_scale = None

        return (
            grad_multipliers if need_multipliers else None,
            grad_activations if need_activations else None,
            grad_scale,
            None,
            None,
        )


def _validate_inputs(
    multipliers: Tensor,
    activations: Tensor,
    scale: Tensor,
    eps: float,
) -> None:
    if not _TRITON_AVAILABLE:
        raise RuntimeError("Triton is not installed")
    if not multipliers.is_cuda:
        raise ValueError("fused QuatSpin requires a CUDA/ROCm tensor")
    if multipliers.ndim < 2 or multipliers.size(-1) != 4:
        raise ValueError("multipliers must have shape [..., channels, 4]")
    if activations.shape != multipliers.shape:
        raise ValueError("activations must have the same shape as multipliers")
    if scale.ndim != 1 or scale.numel() != multipliers.size(-2):
        raise ValueError("scale must have one value per quaternion channel")
    if any(dimension == 0 for dimension in multipliers.shape):
        raise ValueError("input dimensions must be non-zero")
    if activations.device != multipliers.device or scale.device != multipliers.device:
        raise ValueError("multipliers, activations, and scale must share a device")
    if multipliers.dtype not in _SUPPORTED_DTYPES:
        raise TypeError("multipliers must use FP16, BF16, or FP32")
    if activations.dtype != multipliers.dtype:
        raise TypeError("activations must have the same dtype as multipliers")
    if scale.dtype not in _SUPPORTED_DTYPES:
        raise TypeError("scale must use FP16, BF16, or FP32")
    if eps <= 0:
        raise ValueError("eps must be positive")


def fused_quat_mul_norm(
    multipliers: Tensor,
    activations: Tensor,
    scale: Tensor,
    eps: float = 1e-6,
) -> Tensor:
    """Fuse Hamilton product, quaternion RMS normalization, and scaling.

    Inputs have shape ``[..., channels, 4]``. Arithmetic and normalization
    accumulate in FP32; the result follows PyTorch dtype promotion between the
    quaternion inputs and the learned scale. Forward and backward both run as
    Triton kernels on CUDA or ROCm devices.
    """
    _validate_inputs(multipliers, activations, scale, eps)
    save_inputs = torch.is_grad_enabled() and any(
        tensor.requires_grad for tensor in (multipliers, activations, scale)
    )
    return _QuatSpinFunction.apply(
        multipliers.contiguous(),
        activations.contiguous(),
        scale.contiguous(),
        float(eps),
        save_inputs,
    )


__all__ = ["fused_quat_mul_norm", "triton_quatspin_available"]
