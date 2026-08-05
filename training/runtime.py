"""Device, precision, and learning-rate policy helpers."""

from __future__ import annotations

import math

import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR


def _load_default_device() -> str | None:
    """Read DEVICE from the repo-root ``defaults.py`` if it exists."""
    try:
        from defaults import DEVICE  # type: ignore[import-untyped]

        return DEVICE
    except (ImportError, AttributeError):
        return None


def resolve_device(requested: str) -> torch.device:
    """Resolve ``auto`` without leaking device policy into other modules.

    Resolution order for ``"auto"``:
    1. ``defaults.py`` ``DEVICE`` (site-local, gitignored)
    2. ``"cuda"`` if available, else ``"cpu"``
    """
    if requested == "auto":
        site_default = _load_default_device()
        if site_default is not None:
            requested = site_default
        else:
            requested = "cuda" if torch.cuda.is_available() else "cpu"
    return torch.device(requested)


def resolve_precision(
    dtype_name: str, device: torch.device
) -> tuple[torch.dtype, bool]:
    dtype = {
        "fp32": torch.float32,
        "bf16": torch.bfloat16,
        "fp16": torch.float16,
    }[dtype_name]
    amp_enabled = dtype_name != "fp32" and device.type in {"cuda", "cpu"}
    if device.type == "cpu" and dtype == torch.float16:
        amp_enabled = False
    return dtype, amp_enabled


def make_scheduler(
    optimizer: AdamW, warmup_steps: int, total_steps: int
) -> LambdaLR:
    """Linear warmup followed by cosine decay."""

    def multiplier(step: int) -> float:
        if step < warmup_steps:
            return (step + 1) / max(1, warmup_steps)
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        progress = min(max(progress, 0.0), 1.0)
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    return LambdaLR(optimizer, multiplier)
