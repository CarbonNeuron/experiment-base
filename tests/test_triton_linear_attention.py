from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager

import pytest
import torch
from torch import Tensor

from multigrid.triton_linear_attention import (
    fused_linear_attention,
    triton_linear_attention_available,
)


def _eager_linear_attention(
    query: Tensor, key: Tensor, value: Tensor, decays: Tensor
) -> Tensor:
    """Reference with the same FP32 accumulation used by the kernel."""
    batch_size, n_heads, seq_len, head_width = query.shape
    state = torch.zeros(
        batch_size,
        n_heads,
        head_width,
        head_width,
        device=query.device,
        dtype=torch.float32,
    )
    normalizer = torch.zeros(
        batch_size,
        n_heads,
        head_width,
        device=query.device,
        dtype=torch.float32,
    )
    outputs: list[Tensor] = []
    for position in range(seq_len):
        query_t = query[:, :, position].float()
        key_t = key[:, :, position].float()
        value_t = value[:, :, position].float()
        decay_t = decays[:, :, position, None].float()
        state = (
            decay_t.unsqueeze(-1) * state
            + key_t.unsqueeze(-1) * value_t.unsqueeze(-2)
        )
        normalizer = decay_t * normalizer + key_t
        numerator = torch.matmul(query_t.unsqueeze(-2), state).squeeze(-2)
        denominator = (query_t * normalizer).sum(-1, keepdim=True)
        outputs.append(numerator / denominator.clamp_min(1.0e-6))
    return torch.stack(outputs, dim=2).to(query.dtype)


def _accelerator_available() -> bool:
    return torch.cuda.is_available() and triton_linear_attention_available()


@contextmanager
def _target_device() -> Iterator[torch.device]:
    """Exercise launches while a possibly different GPU is current."""
    original = torch.cuda.current_device()
    target_index = torch.cuda.device_count() - 1
    current_index = 0 if target_index != 0 else target_index
    torch.cuda.set_device(current_index)
    try:
        yield torch.device(f"cuda:{target_index}")
    finally:
        torch.cuda.set_device(original)


def test_availability_probe_returns_bool() -> None:
    assert isinstance(triton_linear_attention_available(), bool)


@pytest.mark.skipif(
    not _accelerator_available(),
    reason="fused linear attention requires CUDA/ROCm and Triton",
)
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
def test_fused_forward_matches_eager(dtype: torch.dtype) -> None:
    if dtype is torch.bfloat16 and not torch.cuda.is_bf16_supported():
        pytest.skip("device does not support bfloat16")
    with _target_device() as device:
        generator = torch.Generator(device=device).manual_seed(1701)
        shape = (2, 3, 7, 7)
        # Positive queries and keys match the ELU+1 values supplied by the
        # mechanism and keep the comparison away from the denominator clamp.
        query = (
            torch.rand(
                shape, device=device, dtype=dtype, generator=generator
            )
            + 0.2
        )
        key = (
            torch.rand(
                shape, device=device, dtype=dtype, generator=generator
            )
            + 0.2
        )
        value = torch.randn(
            shape, device=device, dtype=dtype, generator=generator
        )
        decays = torch.sigmoid(
            torch.randn(
                shape[:3],
                device=device,
                dtype=dtype,
                generator=generator,
            )
        )

        actual = fused_linear_attention(query, key, value, decays)
        expected = _eager_linear_attention(query, key, value, decays)

        tolerance = 3.0e-5 if dtype is torch.float32 else 8.0e-3
        torch.testing.assert_close(
            actual, expected, atol=tolerance, rtol=tolerance
        )


@pytest.mark.skipif(
    not _accelerator_available(),
    reason="fused linear attention requires CUDA/ROCm and Triton",
)
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
def test_fused_backward_matches_eager(dtype: torch.dtype) -> None:
    if dtype is torch.bfloat16 and not torch.cuda.is_bf16_supported():
        pytest.skip("device does not support bfloat16")
    with _target_device() as device:
        generator = torch.Generator(device=device).manual_seed(90210)
        shape = (2, 2, 5, 5)
        inputs = (
            torch.rand(
                shape, device=device, dtype=dtype, generator=generator
            )
            + 0.25,
            torch.rand(
                shape, device=device, dtype=dtype, generator=generator
            )
            + 0.25,
            torch.randn(
                shape, device=device, dtype=dtype, generator=generator
            ),
            torch.sigmoid(
                torch.randn(
                    shape[:3],
                    device=device,
                    dtype=dtype,
                    generator=generator,
                )
            ),
        )
        fused_inputs = tuple(
            tensor.detach().requires_grad_() for tensor in inputs
        )
        eager_inputs = tuple(
            tensor.detach().requires_grad_() for tensor in inputs
        )
        weights = torch.randn(
            shape, device=device, dtype=dtype, generator=generator
        )

        fused_loss = (fused_linear_attention(*fused_inputs) * weights).sum()
        eager_loss = (_eager_linear_attention(*eager_inputs) * weights).sum()
        fused_gradients = torch.autograd.grad(fused_loss, fused_inputs)
        eager_gradients = torch.autograd.grad(eager_loss, eager_inputs)

        tolerance = 8.0e-5 if dtype is torch.float32 else 3.0e-2
        for actual, expected in zip(fused_gradients, eager_gradients):
            torch.testing.assert_close(
                actual, expected, atol=tolerance, rtol=tolerance
            )


@pytest.mark.skipif(
    not _accelerator_available(),
    reason="fused linear attention requires CUDA/ROCm and Triton",
)
def test_fused_recurrence_is_causal() -> None:
    with _target_device() as device:
        generator = torch.Generator(device=device).manual_seed(41)
        shape = (1, 2, 8, 8)
        query = torch.rand(shape, device=device, generator=generator) + 0.2
        key = torch.rand(shape, device=device, generator=generator) + 0.2
        value = torch.randn(shape, device=device, generator=generator)
        decays = torch.sigmoid(
            torch.randn(shape[:3], device=device, generator=generator)
        )
        boundary = 4
        changed = [tensor.clone() for tensor in (query, key, value, decays)]
        for tensor in changed:
            tensor[:, :, boundary:] = torch.randn_like(tensor[:, :, boundary:])

        baseline = fused_linear_attention(query, key, value, decays)
        perturbed = fused_linear_attention(*changed)

        torch.testing.assert_close(
            perturbed[:, :, :boundary],
            baseline[:, :, :boundary],
            atol=0.0,
            rtol=0.0,
        )


@pytest.mark.skipif(
    not _accelerator_available(),
    reason="fused linear attention requires CUDA/ROCm and Triton",
)
def test_no_grad_inference_does_not_allocate_state_trace(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    with _target_device() as device:
        shape = (1, 2, 5, 4)
        query = (torch.rand(shape, device=device) + 0.2).requires_grad_()
        key = torch.rand(shape, device=device) + 0.2
        value = torch.randn(shape, device=device)
        decays = torch.sigmoid(torch.randn(shape[:3], device=device))
        allocations: list[tuple[int, ...]] = []
        original_empty = torch.empty

        def recording_empty(*args: object, **kwargs: object) -> Tensor:
            if args and isinstance(args[0], (tuple, list)):
                allocations.append(tuple(args[0]))
            elif args and all(isinstance(arg, int) for arg in args):
                allocations.append(tuple(args))  # type: ignore[arg-type]
            return original_empty(*args, **kwargs)

        monkeypatch.setattr(torch, "empty", recording_empty)
        with torch.no_grad():
            result = fused_linear_attention(query, key, value, decays)

        assert result.shape == query.shape
        assert shape + (shape[-1],) not in allocations
