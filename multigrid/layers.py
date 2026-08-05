"""Causal multigrid and surprise-indexed episodic-memory layers."""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from .config import MultigridMemoryConfig


def _next_power_of_two(length: int) -> int:
    if length <= 0:
        raise ValueError("sequence length must be positive")
    return 1 << (length - 1).bit_length()


class CausalRestriction(nn.Module):
    """Merge adjacent fine tokens into learned coarse summaries."""

    def __init__(self, d_model: int) -> None:
        super().__init__()
        self.projection = nn.Linear(2 * d_model, d_model)

    def forward(self, x: Tensor) -> Tensor:
        if x.ndim != 3:
            raise ValueError("restriction input must have shape [batch, time, width]")
        if x.size(1) % 2:
            x = F.pad(x, (0, 0, 0, 1))
        batch_size, seq_len, width = x.shape
        pairs = x.reshape(batch_size, seq_len // 2, 2 * width)
        return self.projection(pairs)


class CausalProlongation(nn.Module):
    """Lift completed coarse summaries without leaking into earlier tokens.

    A coarse item for fine pair ``(2i, 2i+1)`` contains token ``2i+1``.
    Consequently its corrections are aligned to positions ``2i+1`` and
    ``2i+2``, never to the earlier position ``2i``. This one-token boundary
    alignment is what makes the complete restrict/prolong path causal.
    """

    def __init__(self, d_model: int) -> None:
        super().__init__()
        self.projection = nn.Linear(d_model, 2 * d_model)

    def forward(self, x: Tensor, target_length: int | None = None) -> Tensor:
        if x.ndim != 3:
            raise ValueError("prolongation input must have shape [batch, time, width]")
        batch_size, coarse_len, width = x.shape
        full_length = 2 * coarse_len
        if target_length is None:
            target_length = full_length
        if target_length <= 0 or target_length > full_length:
            raise ValueError("target_length must be in [1, 2 * coarse_length]")

        lifted = self.projection(x).reshape(batch_size, coarse_len, 2, width)
        output = x.new_zeros(batch_size, full_length, width)
        output[:, 1::2] = lifted[:, :, 1]
        if coarse_len > 1:
            output[:, 2::2] = lifted[:, :-1, 0]
        return output[:, :target_length]


class LocalRefinement(nn.Module):
    """Small causal convolution used for horizontal local computation."""

    def __init__(self, d_model: int, kernel_size: int, dropout: float) -> None:
        super().__init__()
        if kernel_size <= 0:
            raise ValueError("kernel_size must be positive")
        self.kernel_size = kernel_size
        self.conv = nn.Conv1d(d_model, d_model, kernel_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: Tensor) -> Tensor:
        channels_first = x.transpose(1, 2)
        padded = F.pad(channels_first, (self.kernel_size - 1, 0))
        refined = self.conv(padded).transpose(1, 2)
        return self.dropout(F.gelu(refined))


class _VCyclePass(nn.Module):
    def __init__(self, config: MultigridMemoryConfig, max_levels: int) -> None:
        super().__init__()
        self.refinements = nn.ModuleList(
            LocalRefinement(
                config.d_model,
                config.refinement_kernel_size,
                config.dropout,
            )
            for _ in range(max_levels + 1)
        )
        self.restrictions = nn.ModuleList(
            CausalRestriction(config.d_model) for _ in range(max_levels)
        )
        self.prolongations = nn.ModuleList(
            CausalProlongation(config.d_model) for _ in range(max_levels)
        )

    def forward(self, x: Tensor, n_levels: int) -> Tensor:
        pyramid: list[Tensor] = []
        current = x
        for level in range(n_levels + 1):
            current = current + self.refinements[level](current)
            pyramid.append(current)
            if level < n_levels:
                current = self.restrictions[level](current)

        correction = pyramid[-1]
        for level in range(n_levels - 1, -1, -1):
            lifted = self.prolongations[level](
                correction, target_length=pyramid[level].size(1)
            )
            correction = pyramid[level] + lifted
        return correction


class VCycle(nn.Module):
    """One or more causal multigrid V-cycles over a power-of-two pyramid."""

    def __init__(self, config: MultigridMemoryConfig) -> None:
        super().__init__()
        self.max_seq_len = config.max_seq_len
        self.max_padded_len = _next_power_of_two(config.max_seq_len)
        self.max_levels = int(math.log2(self.max_padded_len))
        self.cycles = nn.ModuleList(
            _VCyclePass(config, self.max_levels)
            for _ in range(config.n_cycles)
        )

    def forward(self, x: Tensor) -> Tensor:
        if x.ndim != 3:
            raise ValueError("V-cycle input must have shape [batch, time, width]")
        original_length = x.size(1)
        if original_length > self.max_seq_len:
            raise ValueError(
                f"sequence length {original_length} exceeds max_seq_len="
                f"{self.max_seq_len}"
            )
        padded_length = _next_power_of_two(original_length)
        if padded_length != original_length:
            x = F.pad(x, (0, 0, 0, padded_length - original_length))
        n_levels = int(math.log2(padded_length))
        for cycle in self.cycles:
            x = cycle(x, n_levels)
        return x[:, :original_length]


@dataclass(frozen=True)
class MemoryWrites:
    """One block's precomputed write events, ordered by sequence position."""

    values: Tensor
    keys: Tensor
    priorities: Tensor


@dataclass(frozen=True)
class MemoryState:
    """Batched differentiable memory bank and causal cross-block history."""

    values: Tensor
    keys: Tensor
    priorities: Tensor
    history: tuple[MemoryWrites, ...] = ()


class EpisodicMemory(nn.Module):
    """Sequential read-before-write residual memory."""

    def __init__(self, config: MultigridMemoryConfig) -> None:
        super().__init__()
        self.n_slots = config.n_memory_slots
        self.d_memory = config.d_memory
        self.d_key = config.d_key
        self.addressing_mode = config.memory_addressing
        self.n_hash_bits = config.n_hash_bits
        self.hash_top_k = config.hash_top_k
        self.key_width = (
            config.d_key
            if self.addressing_mode == "softmax"
            else config.n_hash_bits
        )
        self.temperature = 0.1

        self.write_gate = nn.Linear(2 * config.d_model, 1)
        self.compress = nn.Linear(config.d_model, config.d_memory)
        if self.addressing_mode == "softmax":
            self.hash_proj = nn.Linear(config.d_model, config.d_key)
            self.query_proj = nn.Linear(config.d_model, config.d_key)
        else:
            self.hash_encode = nn.Linear(config.d_model, config.n_hash_bits)
        self.value_proj = nn.Linear(config.d_memory, config.d_model)
        self.read_gate = nn.Linear(2 * config.d_model, config.d_model)
        self.dropout = nn.Dropout(config.dropout)

    def initial_state(
        self, batch_size: int, *, device: torch.device, dtype: torch.dtype
    ) -> MemoryState:
        return MemoryState(
            values=torch.zeros(
                batch_size, self.n_slots, self.d_memory, device=device, dtype=dtype
            ),
            keys=torch.zeros(
                batch_size, self.n_slots, self.key_width, device=device, dtype=dtype
            ),
            priorities=torch.zeros(
                batch_size, self.n_slots, device=device, dtype=dtype
            ),
        )

    def compute_surprise(self, x: Tensor, prediction: Tensor) -> tuple[Tensor, Tensor]:
        error = x - prediction
        novelty = torch.linalg.vector_norm(error, dim=-1, keepdim=True)
        return error, novelty

    def compute_write_prob(self, x: Tensor, error: Tensor) -> Tensor:
        return torch.sigmoid(self.write_gate(torch.cat((x, error), dim=-1)))

    def _hash_code(self, x: Tensor) -> Tensor:
        soft_code = torch.tanh(self.hash_encode(x))
        hard_code = torch.sign(soft_code)
        if self.training:
            return hard_code + soft_code - soft_code.detach()
        return hard_code

    def _read_from_state_softmax(
        self, query: Tensor, state: MemoryState
    ) -> Tensor:
        """Read using a pre-projected query vector (softmax mode)."""
        scores = torch.bmm(query.unsqueeze(1), state.keys.transpose(1, 2))
        scores = scores.squeeze(1) / math.sqrt(self.d_key)
        valid = state.priorities.gt(0)
        weights = torch.softmax(scores.masked_fill(~valid, -1.0e4), dim=-1)
        weights = weights * valid.to(weights.dtype)
        weights = weights / weights.sum(dim=-1, keepdim=True).clamp_min(1.0e-8)
        value = torch.bmm(weights.unsqueeze(1), state.values).squeeze(1)
        return self.value_proj(value)

    def _read_from_state_hash(
        self, query_code: Tensor, state: MemoryState
    ) -> Tensor:
        """Read using a pre-projected hash code (hash mode)."""
        similarities = torch.bmm(
            query_code.unsqueeze(1), state.keys.transpose(1, 2)
        ).squeeze(1)
        valid = state.priorities.gt(0)
        similarities = similarities.masked_fill(~valid, -1.0e4)

        top_k = min(self.hash_top_k, self.n_slots)
        top_similarities, top_indices = torch.topk(
            similarities, k=top_k, dim=-1
        )
        top_valid = valid.gather(1, top_indices)
        value_indices = top_indices.unsqueeze(-1).expand(
            -1, -1, self.d_memory
        )
        top_values = state.values.gather(1, value_indices)
        weights = torch.softmax(
            top_similarities / math.sqrt(self.n_hash_bits), dim=-1
        )
        weights = weights * top_valid.to(weights.dtype)
        weights = weights / weights.sum(dim=-1, keepdim=True).clamp_min(1.0e-8)
        value = torch.bmm(weights.unsqueeze(1), top_values).squeeze(1)
        return self.value_proj(value)

    def _read_from_state(self, query: Tensor, state: MemoryState) -> Tensor:
        """Read using a pre-projected query/code vector."""
        if self.addressing_mode == "hash":
            return self._read_from_state_hash(query, state)
        return self._read_from_state_softmax(query, state)

    def _apply_write(
        self,
        value: Tensor,
        key: Tensor,
        probability: Tensor,
        state: MemoryState,
    ) -> MemoryState:
        write_prob = probability.unsqueeze(-1)

        if self.training:
            soft_slots = torch.softmax(
                -state.priorities / self.temperature, dim=-1
            )
            hard_slots = F.one_hot(
                torch.argmax(soft_slots, dim=-1), self.n_slots
            ).to(value.dtype)
            slot_weights = hard_slots + soft_slots - soft_slots.detach()
            lowest_priority = (soft_slots * state.priorities).sum(
                dim=-1, keepdim=True
            )
            overwrite = torch.sigmoid(
                (write_prob - lowest_priority) / self.temperature
            )
            update = slot_weights * overwrite
        else:
            indices = torch.argmin(state.priorities, dim=-1)
            slot_weights = F.one_hot(indices, self.n_slots).to(value.dtype)
            lowest_priority = state.priorities.gather(1, indices.unsqueeze(1))
            overwrite = (
                probability.unsqueeze(1).gt(lowest_priority).to(value.dtype)
            )
            update = slot_weights * overwrite

        update_vector = update.unsqueeze(-1)
        values = (
            state.values * (1.0 - update_vector)
            + value.unsqueeze(1) * update_vector
        )
        keys = state.keys * (1.0 - update_vector) + key.unsqueeze(1) * update_vector
        priorities = (
            state.priorities * (1.0 - update)
            + probability.unsqueeze(1) * update
        )
        return MemoryState(values, keys, priorities, state.history)

    def forward(
        self,
        x: Tensor,
        prediction: Tensor,
        state: MemoryState | None = None,
        *,
        query: Tensor | None = None,
        replay_history: bool = False,
    ) -> tuple[Tensor, MemoryState]:
        if x.shape != prediction.shape or x.ndim != 3:
            raise ValueError("x and prediction must share [batch, time, width] shape")
        if query is None:
            query = prediction
        elif query.shape != prediction.shape:
            raise ValueError("query must have the same shape as prediction")
        batch_size, seq_len, _ = x.shape
        prior_history = () if state is None else state.history
        if state is None or replay_history:
            working_state = self.initial_state(
                batch_size, device=x.device, dtype=x.dtype
            )
        else:
            working_state = state

        # --- Precompute ALL projections in parallel (batch × seq_len) ---
        # These are the expensive linear projections that don't depend on
        # sequential memory state.  Doing them once avoids seq_len separate
        # matmuls inside the loop.

        # Surprise / write gate
        error_all = x - prediction  # [B, T, D]
        write_prob_all = torch.sigmoid(
            self.write_gate(torch.cat((x, error_all), dim=-1))
        )  # [B, T, 1]

        # Compressed write values
        compressed_all = self.compress(error_all) * write_prob_all  # [B, T, d_memory]

        # Write keys
        if self.addressing_mode == "hash":
            write_keys_all = self._hash_code(
                x.reshape(-1, x.size(-1))
            ).reshape(batch_size, seq_len, -1)  # [B, T, n_hash_bits]
        else:
            write_keys_all = self.hash_proj(x)  # [B, T, d_key]

        # Read queries
        if self.addressing_mode == "hash":
            read_queries_all = self._hash_code(
                query.reshape(-1, query.size(-1))
            ).reshape(batch_size, seq_len, -1)  # [B, T, n_hash_bits]
        else:
            read_queries_all = self.query_proj(query)  # [B, T, d_key]

        # Write priorities (squeezed)
        priorities_all = write_prob_all.squeeze(-1)  # [B, T]

        # Precompute history writes if replaying (just index slicing later)
        # No projections needed — history already has values/keys/priorities.

        # --- Sequential loop: only lightweight state bookkeeping ---
        corrections: list[Tensor] = []
        w_vals: list[Tensor] = []
        w_keys: list[Tensor] = []
        w_pris: list[Tensor] = []
        for t in range(seq_len):
            # Read from current memory state (tiny ops over n_slots)
            query_t = query[:, t]
            read_value = self._read_from_state(
                read_queries_all[:, t], working_state
            )
            gate = torch.sigmoid(
                self.read_gate(
                    torch.cat((query_t, read_value), dim=-1)
                )
            )
            corrections.append(self.dropout(gate * read_value))

            # Replay lower-block writes at this position
            if replay_history:
                for writes in prior_history:
                    working_state = self._apply_write(
                        writes.values[:, t],
                        writes.keys[:, t],
                        writes.priorities[:, t],
                        working_state,
                    )

            # Apply this position's write (precomputed)
            val_t = compressed_all[:, t]
            key_t = write_keys_all[:, t]
            pri_t = priorities_all[:, t]
            working_state = self._apply_write(
                val_t, key_t, pri_t, working_state
            )
            w_vals.append(val_t)
            w_keys.append(key_t)
            w_pris.append(pri_t)

        if replay_history:
            current_writes = MemoryWrites(
                torch.stack(w_vals, dim=1),
                torch.stack(w_keys, dim=1),
                torch.stack(w_pris, dim=1),
            )
            working_state = MemoryState(
                working_state.values,
                working_state.keys,
                working_state.priorities,
                prior_history + (current_writes,),
            )
        return torch.stack(corrections, dim=1), working_state


class MultigridMemoryBlock(nn.Module):
    """Pre-norm multigrid, episodic correction, and FFN block."""

    def __init__(self, config: MultigridMemoryConfig) -> None:
        super().__init__()
        self.multigrid_norm = nn.LayerNorm(
            config.d_model, eps=config.layer_norm_eps
        )
        self.multigrid = VCycle(config)
        self.memory_norm = nn.LayerNorm(
            config.d_model, eps=config.layer_norm_eps
        )
        self.memory = EpisodicMemory(config)
        self.ffn_norm = nn.LayerNorm(config.d_model, eps=config.layer_norm_eps)
        self.ffn = nn.Sequential(
            nn.Linear(config.d_model, config.d_ff),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.d_ff, config.d_model),
            nn.Dropout(config.dropout),
        )

    def forward(
        self, x: Tensor, state: MemoryState | None = None
    ) -> tuple[Tensor, MemoryState]:
        multigrid_input = self.multigrid_norm(x)
        prediction = self.multigrid(multigrid_input)
        x = x + prediction
        memory_query = self.memory_norm(x)
        correction, state = self.memory(
            multigrid_input,
            prediction,
            state,
            query=memory_query,
            replay_history=True,
        )
        x = x + correction
        x = x + self.ffn(self.ffn_norm(x))
        return x, state
