"""A dependency-free static IVF index implemented with PyTorch."""

from __future__ import annotations

import torch
from torch import Tensor
from torch.nn import functional as F


class IVFStaticOutputIndex:
    """Static inverted-file retrieval over fixed direction vectors."""

    def __init__(
        self,
        vectors: Tensor,
        centroids: Tensor,
        sorted_token_ids: Tensor,
        cluster_offsets: Tensor,
        *,
        nprobe: int = 8,
        max_candidates_per_query: int = 2048,
    ) -> None:
        if vectors.ndim != 2 or centroids.ndim != 2:
            raise ValueError("vectors and centroids must be rank-2")
        if vectors.size(1) != centroids.size(1):
            raise ValueError("vector and centroid dimensions differ")
        if not 0 < nprobe <= centroids.size(0):
            raise ValueError("nprobe must be between 1 and num_clusters")
        if max_candidates_per_query <= 0:
            raise ValueError("max_candidates_per_query must be positive")
        self.vectors = vectors.detach()
        self.centroids = centroids.detach().to(vectors.device)
        # Ragged membership stays on CPU. Search transfers one bounded packed
        # candidate matrix per query chunk, avoiding C * max_cluster_size
        # padding and per-candidate accelerator synchronizations.
        self.sorted_token_ids = sorted_token_ids.detach().cpu()
        self.cluster_offsets = cluster_offsets.detach().cpu()
        self.nprobe = nprobe
        self.max_candidates_per_query = max_candidates_per_query
        self.max_scored_candidates = 0

    @property
    def size(self) -> int:
        return self.vectors.size(0)

    @classmethod
    @torch.no_grad()
    def build(
        cls,
        vectors: Tensor,
        *,
        num_clusters: int = 512,
        nprobe: int = 8,
        max_candidates_per_query: int = 2048,
        build_batch_size: int = 8192,
        kmeans_iterations: int = 8,
        seed: int = 0,
    ) -> "IVFStaticOutputIndex":
        if vectors.ndim != 2 or not vectors.is_floating_point():
            raise ValueError("vectors must be a floating-point [V, D] tensor")
        if not 0 < num_clusters <= vectors.size(0):
            raise ValueError("num_clusters must be between 1 and vocabulary size")
        if build_batch_size <= 0 or kmeans_iterations <= 0:
            raise ValueError("build_batch_size and kmeans_iterations must be positive")
        work = vectors.detach().float()
        generator = torch.Generator(device=work.device)
        generator.manual_seed(seed)
        initial_ids = torch.randperm(
            work.size(0), device=work.device, generator=generator
        )[:num_clusters]
        centroids = work[initial_ids].clone()
        assignments = torch.empty(work.size(0), dtype=torch.long, device=work.device)

        for _ in range(kmeans_iterations):
            sums = torch.zeros_like(centroids)
            counts = torch.zeros(num_clusters, device=work.device)
            for start in range(0, work.size(0), build_batch_size):
                end = min(start + build_batch_size, work.size(0))
                batch_assignments = (
                    work[start:end] @ centroids.transpose(0, 1)
                ).argmax(dim=-1)
                assignments[start:end] = batch_assignments
                sums.index_add_(0, batch_assignments, work[start:end])
                counts.index_add_(
                    0, batch_assignments, torch.ones(end - start, device=work.device)
                )
            populated = counts > 0
            centroids[populated] = sums[populated] / counts[populated, None]
            centroids = F.normalize(centroids, dim=-1)

        # Reassign once against the final centroids before constructing lists.
        for start in range(0, work.size(0), build_batch_size):
            end = min(start + build_batch_size, work.size(0))
            assignments[start:end] = (
                work[start:end] @ centroids.transpose(0, 1)
            ).argmax(dim=-1)
        order = assignments.argsort(stable=True)
        counts = torch.bincount(assignments, minlength=num_clusters)
        offsets = torch.cat(
            (torch.zeros(1, dtype=torch.long, device=work.device), counts.cumsum(0))
        )
        return cls(
            vectors,
            centroids,
            order,
            offsets,
            nprobe=nprobe,
            max_candidates_per_query=max_candidates_per_query,
        )

    def state_dict(self) -> dict[str, Tensor | int]:
        return {
            "centroids": self.centroids.cpu(),
            "sorted_token_ids": self.sorted_token_ids.cpu(),
            "cluster_offsets": self.cluster_offsets.cpu(),
            "nprobe": self.nprobe,
            "max_candidates_per_query": self.max_candidates_per_query,
        }

    @classmethod
    def from_state_dict(
        cls, vectors: Tensor, state: dict[str, Tensor | int]
    ) -> "IVFStaticOutputIndex":
        return cls(
            vectors,
            state["centroids"],
            state["sorted_token_ids"],
            state["cluster_offsets"],
            nprobe=int(state["nprobe"]),
            max_candidates_per_query=int(state["max_candidates_per_query"]),
        )

    @torch.no_grad()
    def search(self, queries: Tensor, k: int) -> tuple[Tensor, Tensor]:
        if queries.ndim != 2 or queries.size(1) != self.vectors.size(1):
            raise ValueError("queries must have shape [N, D] matching the index")
        if not 0 < k <= self.size:
            raise ValueError(f"k must be between 1 and {self.size}")
        queries = queries.to(device=self.vectors.device, dtype=torch.float32)
        centroid_scores = queries @ self.centroids.float().transpose(0, 1)
        probe_ids = centroid_scores.topk(self.nprobe, dim=-1).indices
        scores = torch.full((queries.size(0), k), -torch.inf, device=queries.device)
        token_ids = torch.full(
            (queries.size(0), k), -1, dtype=torch.long, device=queries.device
        )
        self.max_scored_candidates = 0
        packed_cpu = torch.full(
            (queries.size(0), self.max_candidates_per_query),
            -1,
            dtype=torch.long,
        )
        for row, row_probes in enumerate(probe_ids.detach().cpu()):
            count = 0
            for cluster in row_probes.tolist():
                start = int(self.cluster_offsets[cluster])
                end = int(self.cluster_offsets[cluster + 1])
                take = min(end - start, self.max_candidates_per_query - count)
                if take > 0:
                    packed_cpu[row, count : count + take] = self.sorted_token_ids[
                        start : start + take
                    ]
                    count += take
                if count == self.max_candidates_per_query:
                    break
        packed_candidates = packed_cpu.to(queries.device)
        for row in range(queries.size(0)):
            candidates = packed_candidates[row]
            candidates = candidates[candidates.ge(0)]
            if candidates.numel() == 0:
                continue
            self.max_scored_candidates = max(
                self.max_scored_candidates, candidates.numel()
            )
            candidate_scores = self.vectors[candidates].float() @ queries[row]
            row_k = min(k, candidates.numel())
            row_scores, row_order = candidate_scores.topk(row_k)
            scores[row, :row_k] = row_scores
            token_ids[row, :row_k] = candidates[row_order]
        return scores, token_ids
