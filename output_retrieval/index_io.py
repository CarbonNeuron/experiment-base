"""Fingerprinting and persistent construction of static output indexes."""

from __future__ import annotations

import hashlib
import os
from pathlib import Path
from typing import TYPE_CHECKING

import torch
from torch import Tensor
from torch.nn import functional as F

from .base import StaticOutputIndex
from .exact import ExactStaticOutputIndex
from .ivf import IVFStaticOutputIndex

if TYPE_CHECKING:
    from config import HardNegativeRetrievalConfig


@torch.no_grad()
def directions_fingerprint(directions: Tensor) -> str:
    """Return a stable SHA-256 fingerprint including shape, dtype, and bytes."""
    tensor = directions.detach().cpu().contiguous()
    digest = hashlib.sha256()
    digest.update(str(tuple(tensor.shape)).encode())
    digest.update(str(tensor.dtype).encode())
    digest.update(tensor.numpy().tobytes())
    return digest.hexdigest()


def _index_path(config: "HardNegativeRetrievalConfig", fingerprint: str) -> Path | None:
    if config.index.path is None:
        return None
    path = Path(config.index.path)
    if path.suffix:
        return path
    return path / (
        f"ivf-{fingerprint[:16]}-c{config.index.num_clusters}"
        f"-p{config.index.nprobe}-m{config.index.max_candidates_per_query}.pt"
    )


def _atomic_torch_save(payload: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    torch.save(payload, temporary)
    os.replace(temporary, path)


@torch.no_grad()
def build_or_load_index(
    directions: Tensor,
    config: "HardNegativeRetrievalConfig",
) -> tuple[StaticOutputIndex, str, Path | None]:
    """Build/load a validated index; never silently replace IVF with exact."""
    fingerprint = directions_fingerprint(directions)
    vectors = directions.detach()
    if config.normalize_directions:
        vectors = F.normalize(vectors.float(), dim=-1)

    if config.backend == "exact":
        return (
            ExactStaticOutputIndex(
                vectors, vocab_chunk_size=config.index.vocab_chunk_size
            ),
            fingerprint,
            None,
        )
    if config.backend != "ivf":
        raise ValueError(f"unknown hard-negative backend: {config.backend}")

    path = _index_path(config, fingerprint)
    if path is not None and path.is_file() and not config.index.rebuild:
        payload = torch.load(path, map_location="cpu", weights_only=True)
        if payload.get("fingerprint") != fingerprint:
            raise ValueError(f"static index fingerprint mismatch: {path}")
        if payload.get("backend") != "ivf":
            raise ValueError(f"static index backend mismatch: {path}")
        saved = payload.get("settings", {})
        expected = {
            "num_clusters": config.index.num_clusters,
            "nprobe": config.index.nprobe,
            "max_candidates_per_query": config.index.max_candidates_per_query,
        }
        if saved != expected:
            raise ValueError(
                f"static index settings mismatch: saved={saved}, expected={expected}"
            )
        return (
            IVFStaticOutputIndex.from_state_dict(vectors, payload["state"]),
            fingerprint,
            path,
        )

    index = IVFStaticOutputIndex.build(
        vectors,
        num_clusters=min(config.index.num_clusters, vectors.size(0)),
        nprobe=min(config.index.nprobe, config.index.num_clusters, vectors.size(0)),
        max_candidates_per_query=config.index.max_candidates_per_query,
        build_batch_size=config.index.build_batch_size,
        kmeans_iterations=config.index.kmeans_iterations,
        seed=config.index.seed,
    )
    if path is not None:
        payload = {
            "version": 1,
            "backend": "ivf",
            "fingerprint": fingerprint,
            "settings": {
                "num_clusters": config.index.num_clusters,
                "nprobe": config.index.nprobe,
                "max_candidates_per_query": config.index.max_candidates_per_query,
            },
            "state": index.state_dict(),
        }
        _atomic_torch_save(payload, path)
    return index, fingerprint, path
