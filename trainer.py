"""Compatibility exports for reusable components in :mod:`training`."""

from training.runtime import make_scheduler, resolve_device, resolve_precision
from training.trainer import Trainer

__all__ = [
    "Trainer",
    "make_scheduler",
    "resolve_device",
    "resolve_precision",
]
