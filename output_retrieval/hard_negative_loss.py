"""Filtering, retrieval orchestration, and differentiable candidate loss."""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch
from torch import Tensor
from torch.nn import functional as F

from .base import StaticOutputIndex
from .exact import ExactStaticOutputIndex

if TYPE_CHECKING:
    from config import HardNegativeRetrievalConfig


@dataclass
class HardNegativeBatch:
    token_ids: Tensor
    valid_mask: Tensor
    retrieval_scores: Tensor | None = None


def hard_negative_loss_from_logits(
    positive_logits: Tensor,
    hard_logits: Tensor,
    valid_mask: Tensor,
    *,
    loss_type: str = "candidate_ce",
    pairwise_margin: float = 0.0,
) -> Tensor:
    """Return stable per-position losses for exact selected logits."""
    if hard_logits.shape != valid_mask.shape:
        raise ValueError("hard_logits and valid_mask shapes must match")
    if positive_logits.shape != hard_logits.shape[:1]:
        raise ValueError("positive_logits must have shape [N]")
    relative = (hard_logits - positive_logits[:, None]).float()
    if loss_type == "candidate_ce":
        relative = relative.masked_fill(~valid_mask, -torch.inf)
        zeros = torch.zeros(
            (relative.size(0), 1), dtype=relative.dtype, device=relative.device
        )
        return torch.logsumexp(torch.cat((zeros, relative), dim=-1), dim=-1)
    if loss_type == "pairwise":
        terms = F.softplus(pairwise_margin + relative)
        terms = terms.masked_fill(~valid_mask, 0)
        counts = valid_mask.sum(dim=-1)
        losses = terms.sum(dim=-1) / counts.clamp_min(1)
        return losses.masked_fill(counts.eq(0), 0)
    raise ValueError(f"unknown hard-negative loss: {loss_type}")


@torch.no_grad()
def filter_hard_negatives(
    token_ids: Tensor,
    scores: Tensor | None,
    targets: Tensor,
    *,
    hard_k: int,
    vocab_size: int,
    invalid_token_ids: tuple[int, ...] = (),
) -> HardNegativeBatch:
    """Remove positives/invalids and stable-deduplicate each result row."""
    if token_ids.ndim != 2 or targets.shape != token_ids.shape[:1]:
        raise ValueError("token_ids must be [N, K] and targets must be [N]")
    if scores is not None and scores.shape != token_ids.shape:
        raise ValueError("scores shape must match token_ids")
    rows, columns = token_ids.shape
    valid = token_ids.ge(0) & token_ids.lt(vocab_size)
    valid &= token_ids.ne(targets.unsqueeze(-1))
    for invalid_id in invalid_token_ids:
        valid &= token_ids.ne(invalid_id)

    # Result columns are already score ordered. Mask every repeated ID after
    # its first occurrence, then sort valid source positions to preserve order.
    duplicates = token_ids.unsqueeze(-1).eq(token_ids.unsqueeze(-2))
    seen_before = torch.tril(duplicates, diagonal=-1).any(dim=-1)
    valid &= ~seen_before
    source_positions = torch.arange(columns, device=token_ids.device).expand(rows, -1)
    ordered_positions = torch.where(valid, source_positions, columns).sort(
        dim=-1
    ).values
    selected_columns = min(hard_k, columns)
    selected_positions = ordered_positions[:, :selected_columns]
    selected_valid = selected_positions.lt(columns)
    safe_positions = selected_positions.clamp_max(max(0, columns - 1))

    output_ids = torch.zeros((rows, hard_k), dtype=torch.long, device=token_ids.device)
    output_mask = torch.zeros_like(output_ids, dtype=torch.bool)
    output_ids[:, :selected_columns] = token_ids.gather(-1, safe_positions).masked_fill(
        ~selected_valid, 0
    )
    output_mask[:, :selected_columns] = selected_valid
    output_scores = None
    if scores is not None:
        output_scores = torch.full(
            output_ids.shape,
            -torch.inf,
            dtype=scores.dtype,
            device=scores.device,
        )
        output_scores[:, :selected_columns] = scores.gather(
            -1, safe_positions
        ).masked_fill(~selected_valid, -torch.inf)
    return HardNegativeBatch(output_ids, output_mask, output_scores)


class HardNegativeTrainer:
    """Retrieve detached candidates and score them with differentiable tensors."""

    def __init__(
        self,
        index: StaticOutputIndex,
        fixed_directions: Tensor,
        config: "HardNegativeRetrievalConfig",
    ) -> None:
        self.index = index
        self.fixed_directions = fixed_directions.detach()
        self.config = config
        self.last_retrieval_queries: Tensor | None = None
        self.last_retrieval_targets: Tensor | None = None
        self.last_hard_batch: HardNegativeBatch | None = None

    def _select_positions(
        self, hidden: Tensor, targets: Tensor
    ) -> tuple[Tensor, Tensor]:
        flat_hidden = hidden.reshape(-1, hidden.size(-1))
        flat_targets = targets.reshape(-1)
        valid = flat_targets.ne(self.config.ignore_index)
        hidden = flat_hidden[valid]
        targets = flat_targets[valid]
        count = targets.numel()
        requested = count
        if self.config.position_fraction < 1.0:
            requested = min(
                requested,
                max(1, round(count * self.config.position_fraction)),
            )
        if self.config.max_positions_per_batch is not None:
            requested = min(requested, self.config.max_positions_per_batch)
        if requested < count:
            selection = torch.randperm(count, device=targets.device)[:requested]
            hidden = hidden[selection]
            targets = targets[selection]
        return hidden, targets

    @torch.no_grad()
    def retrieve(self, queries: Tensor, targets: Tensor) -> HardNegativeBatch:
        score_parts: list[Tensor] = []
        id_parts: list[Tensor] = []
        mask_parts: list[Tensor] = []
        search_k = min(
            self.index.size, self.config.hard_k + self.config.retrieve_extra
        )
        for start in range(0, queries.size(0), self.config.query_chunk_size):
            end = min(start + self.config.query_chunk_size, queries.size(0))
            scores, token_ids = self.index.search(queries[start:end], search_k)
            filtered = filter_hard_negatives(
                token_ids,
                scores,
                targets[start:end],
                hard_k=self.config.hard_k,
                vocab_size=self.index.size,
                invalid_token_ids=self.config.invalid_token_ids,
            )
            id_parts.append(filtered.token_ids)
            mask_parts.append(filtered.valid_mask)
            assert filtered.retrieval_scores is not None
            score_parts.append(filtered.retrieval_scores)
        if not id_parts:
            shape = (0, self.config.hard_k)
            return HardNegativeBatch(
                torch.empty(shape, dtype=torch.long, device=queries.device),
                torch.empty(shape, dtype=torch.bool, device=queries.device),
                queries.new_empty(shape),
            )
        return HardNegativeBatch(
            torch.cat(id_parts), torch.cat(mask_parts), torch.cat(score_parts)
        )

    def compute(
        self,
        hidden: Tensor,
        targets: Tensor,
        rotation_matrix: Tensor,
        scale: Tensor,
    ) -> tuple[Tensor, dict[str, Tensor]]:
        valid_hidden, valid_targets = self._select_positions(hidden, targets)
        if valid_targets.numel() == 0:
            zero = hidden.sum() * 0
            return zero, {"hard_loss": zero.detach(), "hard_positions": zero.detach()}

        # Effective rows are s * D * Q^T, hence row-batch queries are h * Q.
        q = valid_hidden @ rotation_matrix
        retrieval_start = time.perf_counter()
        with torch.no_grad():
            retrieval_q = q.detach().float()
            if self.config.normalize_queries:
                retrieval_q = F.normalize(retrieval_q, dim=-1)
            hard_batch = self.retrieve(retrieval_q, valid_targets)
            retained = min(
                retrieval_q.size(0),
                self.config.diagnostics.exact_recall_query_count,
            )
            self.last_retrieval_queries = retrieval_q[:retained].detach()
            self.last_retrieval_targets = valid_targets[:retained].detach()
            self.last_hard_batch = HardNegativeBatch(
                hard_batch.token_ids[:retained].detach(),
                hard_batch.valid_mask[:retained].detach(),
                None,
            )
        retrieval_seconds = time.perf_counter() - retrieval_start

        score_start = time.perf_counter()
        safe_ids = hard_batch.token_ids.masked_fill(~hard_batch.valid_mask, 0)
        directions = self.fixed_directions
        hard_vectors = directions[safe_ids]
        positive_vectors = directions[valid_targets]
        positive_logits = scale * (q * positive_vectors).sum(dim=-1)
        hard_logits = scale * torch.einsum("nd,nkd->nk", q, hard_vectors)
        score_seconds = time.perf_counter() - score_start

        loss_start = time.perf_counter()
        per_position = hard_negative_loss_from_logits(
            positive_logits,
            hard_logits,
            hard_batch.valid_mask,
            loss_type=self.config.loss_type,
            pairwise_margin=self.config.pairwise_margin,
        )
        loss = per_position.mean()
        loss_seconds = time.perf_counter() - loss_start

        masked_hard = hard_logits.float().masked_fill(
            ~hard_batch.valid_mask, -torch.inf
        )
        hardest = masked_hard.max(dim=-1).values
        has_hard = hard_batch.valid_mask.any(dim=-1)
        safe_hardest = torch.where(has_hard, hardest, positive_logits.float())
        metrics = {
            "hard_loss": loss.detach(),
            "hard_positions": torch.tensor(
                float(valid_targets.numel()), device=loss.device
            ),
            "mean_positive_logit": positive_logits.float().mean().detach(),
            "mean_max_hard_logit": safe_hardest.mean().detach(),
            "mean_hard_margin": (
                positive_logits.float() - safe_hardest
            ).mean().detach(),
            "hard_error_rate": (
                (safe_hardest > positive_logits.float()) & has_hard
            ).float().mean().detach(),
            "mean_valid_hard_negatives": hard_batch.valid_mask.sum(
                dim=-1
            ).float().mean().detach(),
            "retrieval_seconds": torch.tensor(retrieval_seconds, device=loss.device),
            "candidate_score_seconds": torch.tensor(score_seconds, device=loss.device),
            "hard_loss_seconds": torch.tensor(loss_seconds, device=loss.device),
        }
        return loss, metrics

    @torch.no_grad()
    def exact_recall(self) -> Tensor | None:
        """Measure filtered hard-negative recall for the latest query sample."""
        if (
            self.last_retrieval_queries is None
            or self.last_retrieval_targets is None
            or self.last_hard_batch is None
        ):
            return None
        exact = ExactStaticOutputIndex(
            self.index.vectors,
            vocab_chunk_size=self.config.index.vocab_chunk_size,
        )
        search_k = min(
            exact.size, self.config.hard_k + self.config.retrieve_extra
        )
        scores, ids = exact.search(self.last_retrieval_queries, search_k)
        expected = filter_hard_negatives(
            ids,
            scores,
            self.last_retrieval_targets,
            hard_k=self.config.hard_k,
            vocab_size=exact.size,
            invalid_token_ids=self.config.invalid_token_ids,
        )
        actual = self.last_hard_batch
        matches = actual.token_ids.unsqueeze(-1).eq(expected.token_ids.unsqueeze(-2))
        matches &= actual.valid_mask.unsqueeze(-1)
        matches &= expected.valid_mask.unsqueeze(-2)
        hits = matches.any(dim=-2).sum(dim=-1).float()
        denominator = expected.valid_mask.sum(dim=-1).clamp_min(1)
        return (hits / denominator).mean()
