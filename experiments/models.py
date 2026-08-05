"""Backward-compatible access to the default model registry."""

from . import DEFAULT_REGISTRY, build_model, objective_for, register_model
from .registry import ModelFactory, ModelRegistration, ModelRegistry

__all__ = [
    "DEFAULT_REGISTRY",
    "ModelFactory",
    "ModelRegistration",
    "ModelRegistry",
    "build_model",
    "objective_for",
    "register_model",
]
