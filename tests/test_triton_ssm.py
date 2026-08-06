from __future__ import annotations

import pytest
import torch
from torch import Tensor

from multigrid.triton_ssm import (
    fused_diagonal_ssm,
    triton_ssm_available,
)


pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or not triton_ssm_available(),
    reason="fused diagonal SSM requires CUDA/ROCm and Triton",
)


def _eager_recurrence(
    candidate: Tensor,
    gate: Tensor,
    decay: Tensor,
) -> Tensor:
    state = torch.zeros_like(candidate[:, 0])
    outputs: list[Tensor] = []
    for position in range(candidate.size(1)):
        state = (
            decay * state
            + (1.0 - decay) * candidate[:, position]
        )
        outputs.append(state * torch.sigmoid(gate[:, position]))
    return torch.stack(outputs, dim=1)


def _tensor_device() -> torch.device:
    return torch.device(f"cuda:{torch.cuda.device_count() - 1}")


@pytest.mark.parametrize(
    ("dtype", "atol", "rtol"),
    [
        (torch.float32, 2.0e-5, 2.0e-5),
        (torch.float16, 4.0e-3, 4.0e-3),
        (torch.bfloat16, 2.0e-2, 2.0e-2),
    ],
)
def test_fused_diagonal_ssm_matches_eager_forward(
    dtype: torch.dtype,
    atol: float,
    rtol: float,
) -> None:
    if dtype == torch.bfloat16 and not torch.cuda.is_bf16_supported():
        pytest.skip("device does not support BF16")
    device = _tensor_device()
    torch.manual_seed(91)
    candidate = torch.randn(3, 13, 67, device=device, dtype=dtype)
    gate = torch.randn_like(candidate)
    decay = torch.sigmoid(torch.randn(67, device=device, dtype=dtype))

    expected = _eager_recurrence(candidate, gate, decay)
    with torch.no_grad():
        actual = fused_diagonal_ssm(candidate, gate, decay)

    assert actual.dtype == dtype
    torch.testing.assert_close(actual, expected, atol=atol, rtol=rtol)


@pytest.mark.parametrize(
    ("dtype", "atol", "rtol"),
    [
        (torch.float32, 3.0e-5, 3.0e-5),
        (torch.float16, 8.0e-3, 8.0e-3),
        (torch.bfloat16, 6.0e-2, 6.0e-2),
    ],
)
def test_fused_diagonal_ssm_matches_eager_gradients(
    dtype: torch.dtype,
    atol: float,
    rtol: float,
) -> None:
    if dtype == torch.bfloat16 and not torch.cuda.is_bf16_supported():
        pytest.skip("device does not support BF16")
    device = _tensor_device()
    torch.manual_seed(97)
    candidate = torch.randn(
        2, 9, 37, device=device, dtype=dtype, requires_grad=True
    )
    gate = torch.randn_like(candidate, requires_grad=True)
    decay = torch.sigmoid(
        torch.randn(37, device=device, dtype=dtype)
    ).requires_grad_()
    upstream = torch.randn_like(candidate)

    fused = fused_diagonal_ssm(candidate, gate, decay)
    fused_gradients = torch.autograd.grad(
        fused, (candidate, gate, decay), upstream, retain_graph=True
    )
    expected = _eager_recurrence(candidate, gate, decay)
    eager_gradients = torch.autograd.grad(
        expected, (candidate, gate, decay), upstream
    )

    torch.testing.assert_close(fused, expected, atol=atol, rtol=rtol)
    for actual_gradient, expected_gradient in zip(
        fused_gradients, eager_gradients
    ):
        torch.testing.assert_close(
            actual_gradient,
            expected_gradient,
            atol=atol,
            rtol=rtol,
        )


def test_fused_diagonal_ssm_is_causal_and_uses_tensor_device() -> None:
    original_device = torch.cuda.current_device()
    target_index = torch.cuda.device_count() - 1
    active_index = 0 if target_index != 0 else target_index
    torch.cuda.set_device(active_index)
    device = torch.device(f"cuda:{target_index}")
    try:
        torch.manual_seed(101)
        candidate = torch.randn(2, 12, 41, device=device)
        gate = torch.randn_like(candidate)
        decay = torch.sigmoid(torch.randn(41, device=device))
        boundary = 7
        changed_candidate = candidate.clone()
        changed_gate = gate.clone()
        changed_candidate[:, boundary:] = torch.randn_like(
            changed_candidate[:, boundary:]
        )
        changed_gate[:, boundary:] = torch.randn_like(
            changed_gate[:, boundary:]
        )

        with torch.inference_mode():
            original = fused_diagonal_ssm(candidate, gate, decay)
            changed = fused_diagonal_ssm(
                changed_candidate, changed_gate, decay
            )

        assert original.device == device
        torch.testing.assert_close(
            original[:, :boundary],
            changed[:, :boundary],
            atol=0.0,
            rtol=0.0,
        )
    finally:
        torch.cuda.set_device(original_device)


def test_fused_diagonal_ssm_inference_does_not_save_a_trace() -> None:
    device = _tensor_device()
    candidate = torch.randn(1, 5, 7, device=device, requires_grad=True)
    gate = torch.randn_like(candidate, requires_grad=True)
    decay = torch.full((7,), 0.75, device=device, requires_grad=True)

    with torch.inference_mode():
        output = fused_diagonal_ssm(candidate, gate, decay)

    assert output.grad_fn is None
    assert not output.requires_grad
