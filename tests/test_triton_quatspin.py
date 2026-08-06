from __future__ import annotations

from contextlib import contextmanager
from collections.abc import Iterator

import pytest
import torch
from torch import Tensor

from models.quatspin import quat_mul
from models.triton_quatspin import (
    fused_quat_mul_norm,
    triton_quatspin_available,
)


def _accelerator_available() -> bool:
    return torch.cuda.is_available() and triton_quatspin_available()


@contextmanager
def _target_device() -> Iterator[torch.device]:
    original = torch.cuda.current_device()
    target_index = torch.cuda.device_count() - 1
    current_index = 0 if target_index != 0 else target_index
    torch.cuda.set_device(current_index)
    try:
        yield torch.device(f"cuda:{target_index}")
    finally:
        torch.cuda.set_device(original)


def _eager_quat_mul_norm(
    multipliers: Tensor,
    activations: Tensor,
    scale: Tensor,
    eps: float = 1e-6,
) -> Tensor:
    product = quat_mul(multipliers, activations)
    magnitude = torch.linalg.vector_norm(
        product, dim=-1, keepdim=True
    ).clamp_min(eps)
    scale_shape = (1,) * (product.ndim - 2) + (scale.numel(), 1)
    return product / magnitude * scale.reshape(scale_shape)


def test_availability_probe_returns_bool() -> None:
    assert isinstance(triton_quatspin_available(), bool)


def test_fused_quatspin_rejects_cpu_inputs() -> None:
    multipliers = torch.randn(2, 3, 4)
    activations = torch.randn_like(multipliers)
    scale = torch.ones(3)
    expected = "CUDA/ROCm" if triton_quatspin_available() else "not installed"
    with pytest.raises((ValueError, RuntimeError), match=expected):
        fused_quat_mul_norm(multipliers, activations, scale)


@pytest.mark.skipif(
    not _accelerator_available(),
    reason="fused QuatSpin requires CUDA/ROCm and Triton",
)
@pytest.mark.parametrize(
    ("dtype", "tolerance"),
    [
        (torch.float32, 2.0e-5),
        (torch.float16, 4.0e-3),
        (torch.bfloat16, 3.0e-2),
    ],
)
def test_fused_quatspin_matches_eager_forward(
    dtype: torch.dtype,
    tolerance: float,
) -> None:
    if dtype is torch.bfloat16 and not torch.cuda.is_bf16_supported():
        pytest.skip("device does not support bfloat16")
    with _target_device() as device:
        generator = torch.Generator(device=device).manual_seed(173)
        shape = (2, 7, 13, 4)
        multipliers = torch.randn(
            shape, device=device, dtype=dtype, generator=generator
        )
        activations = torch.randn(
            shape, device=device, dtype=dtype, generator=generator
        )
        scale = torch.randn(13, device=device, generator=generator)

        actual = fused_quat_mul_norm(multipliers, activations, scale)
        expected = _eager_quat_mul_norm(multipliers, activations, scale)

        assert actual.device == device
        assert actual.dtype == expected.dtype
        torch.testing.assert_close(
            actual, expected, atol=tolerance, rtol=tolerance
        )


@pytest.mark.skipif(
    not _accelerator_available(),
    reason="fused QuatSpin requires CUDA/ROCm and Triton",
)
@pytest.mark.parametrize(
    ("dtype", "tolerance"),
    [
        (torch.float32, 1.0e-4),
        (torch.float16, 1.0e-2),
        (torch.bfloat16, 6.0e-2),
    ],
)
def test_fused_quatspin_matches_eager_gradients(
    dtype: torch.dtype,
    tolerance: float,
) -> None:
    if dtype is torch.bfloat16 and not torch.cuda.is_bf16_supported():
        pytest.skip("device does not support bfloat16")
    with _target_device() as device:
        generator = torch.Generator(device=device).manual_seed(179)
        shape = (2, 5, 11, 4)
        values = (
            torch.randn(shape, device=device, dtype=dtype, generator=generator),
            torch.randn(shape, device=device, dtype=dtype, generator=generator),
            torch.randn(11, device=device, generator=generator),
        )
        fused_inputs = tuple(
            value.detach().requires_grad_() for value in values
        )
        eager_inputs = tuple(
            value.detach().requires_grad_() for value in values
        )
        upstream = torch.randn(
            shape, device=device, dtype=torch.float32, generator=generator
        )

        actual = fused_quat_mul_norm(*fused_inputs)
        expected = _eager_quat_mul_norm(*eager_inputs)
        actual_gradients = torch.autograd.grad(
            actual, fused_inputs, upstream, retain_graph=True
        )
        expected_gradients = torch.autograd.grad(
            expected, eager_inputs, upstream
        )

        torch.testing.assert_close(
            actual, expected, atol=tolerance, rtol=tolerance
        )
        for actual_gradient, expected_gradient in zip(
            actual_gradients, expected_gradients, strict=True
        ):
            torch.testing.assert_close(
                actual_gradient,
                expected_gradient,
                atol=tolerance,
                rtol=tolerance,
            )


@pytest.mark.skipif(
    not _accelerator_available(),
    reason="fused QuatSpin requires CUDA/ROCm and Triton",
)
def test_fused_quatspin_inference_has_no_autograd_state() -> None:
    with _target_device() as device:
        multipliers = torch.randn(
            2, 3, 5, 4, device=device, requires_grad=True
        )
        activations = torch.randn_like(multipliers, requires_grad=True)
        scale = torch.ones(5, device=device, requires_grad=True)

        with torch.inference_mode():
            output = fused_quat_mul_norm(multipliers, activations, scale)

        assert output.grad_fn is None
        assert not output.requires_grad
