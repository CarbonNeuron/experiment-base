"""QuatSpin feed-forward layers for transformer blocks.

Adapted from https://github.com/CarbonNeuron/quatspin (MIT, commit b2e1dad).
The Hamilton product is the only nonlinearity in :class:`QuatSpinFFN`.
"""

from __future__ import annotations

import warnings

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from .triton_quatspin import fused_quat_mul_norm, triton_quatspin_available


def quat_mul(a: Tensor, b: Tensor) -> Tensor:
    """Return the Hamilton product of quaternion tensors shaped ``[..., 4]``."""
    if a.ndim < 1 or b.ndim < 1 or a.shape != b.shape or a.size(-1) != 4:
        raise ValueError("quaternion inputs must have matching shapes ending in 4")

    a_w, a_x = a[..., 0:1], a[..., 1:2]
    a_y, a_z = a[..., 2:3], a[..., 3:4]
    b_w, b_x = b[..., 0:1], b[..., 1:2]
    b_y, b_z = b[..., 2:3], b[..., 3:4]
    return torch.cat(
        (
            a_w * b_w - a_x * b_x - a_y * b_y - a_z * b_z,
            a_w * b_x + a_x * b_w + a_y * b_z - a_z * b_y,
            a_w * b_y - a_x * b_z + a_y * b_w + a_z * b_x,
            a_w * b_z + a_x * b_y - a_y * b_x + a_z * b_w,
        ),
        dim=-1,
    )


class QuatRMSNorm(nn.Module):
    """Normalize each quaternion magnitude with a learned channel scale."""

    def __init__(self, n_quats: int, eps: float = 1e-6) -> None:
        super().__init__()
        if n_quats <= 0:
            raise ValueError("n_quats must be positive")
        if eps <= 0:
            raise ValueError("eps must be positive")
        self.n_quats = n_quats
        self.eps = eps
        self.scale = nn.Parameter(torch.ones(n_quats))

    def forward(self, x: Tensor) -> Tensor:
        if x.ndim < 1 or x.size(-1) != self.n_quats * 4:
            received = x.size(-1) if x.ndim else "a scalar"
            raise ValueError(
                f"expected a final dimension of {self.n_quats * 4}, "
                f"got {received}"
            )
        quaternions = x.reshape(*x.shape[:-1], self.n_quats, 4)
        magnitude = torch.linalg.vector_norm(
            quaternions, dim=-1, keepdim=True
        ).clamp_min(self.eps)
        scale_shape = (1,) * (quaternions.ndim - 2) + (self.n_quats, 1)
        normalized = quaternions / magnitude
        normalized = normalized * self.scale.reshape(scale_shape)
        return normalized.reshape_as(x)


class QuatSpinFFN(nn.Module):
    """Transformer FFN using a Hamilton product instead of an activation.

    Two independent projections produce quaternion activations and
    multipliers. Their Hamilton product is magnitude-normalized and projected
    back to the residual width. By default, one quaternion is used per input
    feature, matching the upstream QuatFormer implementation.
    """

    def __init__(
        self,
        d_model: int,
        n_quats: int | None = None,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        if d_model <= 0:
            raise ValueError("d_model must be positive")
        if n_quats is None:
            n_quats = d_model
        if n_quats <= 0:
            raise ValueError("n_quats must be positive")
        if not 0.0 <= dropout < 1.0:
            raise ValueError("dropout must be in [0, 1)")

        self.d_model = d_model
        self.n_quats = n_quats
        quaternion_width = n_quats * 4
        self.W_act = nn.Linear(d_model, quaternion_width)
        self.W_mul = nn.Linear(d_model, quaternion_width)
        self.norm = QuatRMSNorm(n_quats)
        self.down = nn.Linear(quaternion_width, d_model)
        self.dropout = nn.Dropout(dropout)
        self._triton_failed = False

    def forward(self, x: Tensor) -> Tensor:
        if x.ndim < 1 or x.size(-1) != self.d_model:
            received = x.size(-1) if x.ndim else "a scalar"
            raise ValueError(
                f"expected a final dimension of {self.d_model}, got {received}"
            )
        quaternion_shape = (*x.shape[:-1], self.n_quats, 4)
        activations = F.linear(x, self.W_act.weight, self.W_act.bias).reshape(
            quaternion_shape
        )
        multipliers = F.linear(x, self.W_mul.weight, self.W_mul.bias).reshape(
            quaternion_shape
        )
        if (
            multipliers.is_cuda
            and multipliers.dtype
            in (torch.float16, torch.bfloat16, torch.float32)
            and self.norm.scale.dtype
            in (torch.float16, torch.bfloat16, torch.float32)
            and not self._triton_failed
            and triton_quatspin_available()
        ):
            try:
                product = fused_quat_mul_norm(
                    multipliers,
                    activations,
                    self.norm.scale,
                    self.norm.eps,
                ).flatten(-2)
            except Exception as error:
                self._triton_failed = True
                warnings.warn(
                    "fused Triton QuatSpin failed; using PyTorch operations: "
                    f"{type(error).__name__}: {error}",
                    RuntimeWarning,
                    stacklevel=2,
                )
                product = self.norm(
                    quat_mul(multipliers, activations).flatten(-2)
                )
        else:
            product = self.norm(
                quat_mul(multipliers, activations).flatten(-2)
            )
        return self.dropout(F.linear(product, self.down.weight, self.down.bias))


__all__ = [
    "QuatRMSNorm",
    "QuatSpinFFN",
    "fused_quat_mul_norm",
    "quat_mul",
    "triton_quatspin_available",
]
