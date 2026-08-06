"""Causal multigrid and surprise-indexed episodic-memory layers."""

from __future__ import annotations

import math
import warnings
from dataclasses import dataclass

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from .config import MultigridMemoryConfig
from .triton_memory import fused_softmax_memory, triton_memory_available


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
        # MIOpen lowers these very short causal convolutions through thousands
        # of im2col/col2im kernels. Materializing the three shifted views and
        # applying one GEMM is algebraically identical and substantially faster
        # at the multigrid pyramid's sequence lengths.
        padded = F.pad(x, (0, 0, self.kernel_size - 1, 0))
        context = torch.cat(
            [
                padded[:, offset : offset + x.size(1)]
                for offset in range(self.kernel_size)
            ],
            dim=-1,
        )
        weight = self.conv.weight.permute(0, 2, 1).reshape(
            self.conv.out_channels, -1
        )
        refined = F.linear(context, weight, self.conv.bias)
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
        self.use_triton_memory = config.use_triton_memory
        self._triton_failed = False
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
        """Read a compressed value using a pre-projected softmax query."""
        scores = torch.bmm(query.unsqueeze(1), state.keys.transpose(1, 2))
        scores = scores.squeeze(1) / math.sqrt(self.d_key)
        valid = state.priorities.gt(0)
        weights = torch.softmax(scores.masked_fill(~valid, -1.0e4), dim=-1)
        weights = weights * valid.to(weights.dtype)
        weights = weights / weights.sum(dim=-1, keepdim=True).clamp_min(1.0e-8)
        return torch.bmm(weights.unsqueeze(1), state.values).squeeze(1)

    def _read_from_state_hash(
        self, query_code: Tensor, state: MemoryState
    ) -> Tensor:
        """Read a compressed value using a pre-projected hash code."""
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
        return torch.bmm(weights.unsqueeze(1), top_values).squeeze(1)

    def _read_from_state(self, query: Tensor, state: MemoryState) -> Tensor:
        """Read a compressed value using a pre-projected query/code."""
        if self.addressing_mode == "hash":
            return self._read_from_state_hash(query, state)
        return self._read_from_state_softmax(query, state)

    @torch.compiler.disable
    def _run_recurrence(
        self,
        read_queries: Tensor,
        state: MemoryState,
        history: tuple[MemoryWrites, ...],
        write_values: Tensor,
        write_keys: Tensor,
        write_priorities: Tensor,
        replay_history: bool,
    ) -> tuple[Tensor, MemoryState]:
        """Run only the state-dependent part of the memory update.

        Keeping this small recurrence eager prevents ``torch.compile`` from
        unrolling a sequence-length-sized graph. All learned projections run
        outside it over the complete batch and sequence dimensions.
        """
        reads: list[Tensor] = []
        for position in range(read_queries.size(1)):
            reads.append(
                self._read_from_state(read_queries[:, position], state)
            )

            if replay_history:
                for writes in history:
                    state = self._apply_write(
                        writes.values[:, position],
                        writes.keys[:, position],
                        writes.priorities[:, position],
                        state,
                    )

            state = self._apply_write(
                write_values[:, position],
                write_keys[:, position],
                write_priorities[:, position],
                state,
            )

        return torch.stack(reads, dim=1), state

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

    @torch.compiler.disable
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

        # Project every position together. Only slot reads and writes depend
        # on the state produced by the preceding position.
        error = x - prediction
        write_probabilities = torch.sigmoid(
            self.write_gate(torch.cat((x, error), dim=-1))
        )
        write_values = self.compress(error) * write_probabilities

        # Write keys
        if self.addressing_mode == "hash":
            write_keys = self._hash_code(
                x.reshape(-1, x.size(-1))
            ).reshape(batch_size, seq_len, -1)
        else:
            write_keys = self.hash_proj(x)

        # Read queries
        if self.addressing_mode == "hash":
            read_queries = self._hash_code(
                query.reshape(-1, query.size(-1))
            ).reshape(batch_size, seq_len, -1)
        else:
            read_queries = self.query_proj(query)
        write_priorities = write_probabilities.squeeze(-1)
        use_fused_recurrence = (
            self.use_triton_memory
            and not self._triton_failed
            and triton_memory_available()
            and x.is_cuda
            and self.addressing_mode == "softmax"
            and replay_history
            and self.n_slots <= 128
            and self.d_memory <= 128
            and self.d_key <= 128
            and x.dtype in (torch.float16, torch.bfloat16, torch.float32)
            and (self.training or not torch.is_grad_enabled())
        )
        if use_fused_recurrence:
            try:
                writers = prior_history + (
                    MemoryWrites(write_values, write_keys, write_priorities),
                )
                all_write_values = torch.stack(
                    [writes.values for writes in writers], dim=1
                )
                all_write_keys = torch.stack(
                    [writes.keys for writes in writers], dim=1
                )
                all_write_priorities = torch.stack(
                    [writes.priorities for writes in writers], dim=1
                )
                (
                    compressed_reads,
                    final_values,
                    final_keys,
                    final_priorities,
                ) = fused_softmax_memory(
                    read_queries,
                    all_write_values,
                    all_write_keys,
                    all_write_priorities,
                    self.n_slots,
                    self.temperature,
                    hard_overwrite=not self.training,
                )
                working_state = MemoryState(
                    final_values,
                    final_keys,
                    final_priorities,
                )
            except Exception as error:
                self._triton_failed = True
                use_fused_recurrence = False
                warnings.warn(
                    "fused Triton memory failed; using the PyTorch "
                    f"recurrence: {type(error).__name__}: {error}",
                    RuntimeWarning,
                    stacklevel=2,
                )
        if not use_fused_recurrence:
            compressed_reads, working_state = self._run_recurrence(
                read_queries,
                working_state,
                prior_history,
                write_values,
                write_keys,
                write_priorities,
                replay_history,
            )

        # These used to be one small matrix multiplication per position. They
        # are independent once the raw reads are known, so execute them as two
        # large compiler-friendly projections over [batch, sequence].
        read_values = self.value_proj(compressed_reads)
        read_gates = torch.sigmoid(
            self.read_gate(torch.cat((query, read_values), dim=-1))
        )
        corrections = self.dropout(read_gates * read_values)

        if replay_history:
            current_writes = MemoryWrites(
                write_values,
                write_keys,
                write_priorities,
            )
            working_state = MemoryState(
                working_state.values,
                working_state.keys,
                working_state.priorities,
                prior_history + (current_writes,),
            )
        return corrections, working_state


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
