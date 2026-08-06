from __future__ import annotations

from pathlib import Path

import pytest
import torch
from torch import nn
from torch.nn import functional as F

from multigrid import (
    CausalProlongation,
    CausalRestriction,
    EpisodicMemory,
    MultigridMemoryConfig,
    MultigridMemoryTransformer,
    VCycle,
    associative_recall,
)
from multigrid.layers import MemoryWrites
from multigrid.triton_memory import (
    fused_softmax_memory,
    triton_memory_available,
)


@pytest.fixture(scope="module")
def embed_path(tmp_path_factory: pytest.TempPathFactory) -> Path:
    path = tmp_path_factory.mktemp("multigrid") / "embeddings.pt"
    generator = torch.Generator().manual_seed(31)
    torch.save(torch.randn(100_277, 8, generator=generator) * 0.2, path)
    return path


def tiny_config(**overrides: object) -> MultigridMemoryConfig:
    values = dict(
        d_model=8,
        n_layers=1,
        n_cycles=1,
        d_ff=16,
        max_seq_len=16,
        dropout=0.0,
        refinement_kernel_size=3,
        n_memory_slots=4,
        d_memory=4,
        d_key=4,
        n_hash_bits=4,
        hash_top_k=2,
    )
    values.update(overrides)
    return MultigridMemoryConfig(**values)


def test_output_shape(embed_path: Path) -> None:
    model = MultigridMemoryTransformer(tiny_config(), embed_path)
    input_ids = torch.randint(model.vocab_size, (2, 7))
    with torch.no_grad():
        logits = model(input_ids)
    assert logits.shape == (2, 7, model.vocab_size)


def test_hash_addressing_output_shape(embed_path: Path) -> None:
    model = MultigridMemoryTransformer(
        tiny_config(memory_addressing="hash"), embed_path
    )
    input_ids = torch.randint(model.vocab_size, (2, 7))
    with torch.no_grad():
        logits = model(input_ids)
    assert logits.shape == (2, 7, model.vocab_size)


def test_causal_restriction() -> None:
    torch.manual_seed(1)
    restriction = CausalRestriction(4)
    original = torch.randn(1, 7, 4)
    changed = original.clone()
    changed[:, 4:] = torch.randn_like(changed[:, 4:])
    first = restriction(original)
    second = restriction(changed)
    torch.testing.assert_close(first[:, :2], second[:, :2])
    assert not torch.allclose(first[:, 2:], second[:, 2:])


def test_causal_prolongation() -> None:
    torch.manual_seed(2)
    prolongation = CausalProlongation(4)
    original = torch.randn(1, 4, 4)
    changed = original.clone()
    changed[:, 2:] = torch.randn_like(changed[:, 2:])
    first = prolongation(original)
    second = prolongation(changed)
    torch.testing.assert_close(first[:, :5], second[:, :5])
    assert not torch.allclose(first[:, 5:], second[:, 5:])


def test_vcycle_output_shape_and_strict_causality() -> None:
    torch.manual_seed(3)
    cycle = VCycle(tiny_config(n_cycles=2))
    original = torch.randn(1, 11, 8)
    changed = original.clone()
    changed[:, 7:] = torch.randn_like(changed[:, 7:])
    first = cycle(original)
    second = cycle(changed)
    assert first.shape == original.shape
    torch.testing.assert_close(first[:, :7], second[:, :7], atol=1e-6, rtol=1e-6)


def test_cross_block_memory_remains_causal(embed_path: Path) -> None:
    torch.manual_seed(29)
    model = MultigridMemoryTransformer(tiny_config(n_layers=2), embed_path)
    model.eval()
    original = torch.randint(model.vocab_size, (1, 8))
    changed = original.clone()
    changed[:, 5:] = torch.randint(model.vocab_size, (1, 3))
    with torch.no_grad():
        first = model.encode(original)
        second = model.encode(changed)
    torch.testing.assert_close(first[:, :5], second[:, :5], atol=1e-6, rtol=1e-6)


def test_hash_addressing_causality(embed_path: Path) -> None:
    torch.manual_seed(37)
    model = MultigridMemoryTransformer(
        tiny_config(n_layers=2, memory_addressing="hash"), embed_path
    )
    model.eval()
    original = torch.randint(model.vocab_size, (1, 8))
    changed = original.clone()
    changed[:, 5:] = torch.randint(model.vocab_size, (1, 3))
    with torch.no_grad():
        first = model.encode(original)
        second = model.encode(changed)
    torch.testing.assert_close(first[:, :5], second[:, :5], atol=1e-6, rtol=1e-6)


def test_episodic_memory_causality() -> None:
    torch.manual_seed(4)
    memory = EpisodicMemory(tiny_config())
    source = torch.randn(2, 6, 8)
    prediction = torch.randn(2, 6, 8)
    changed = source.clone()
    changed[:, 3:] = torch.randn_like(changed[:, 3:])
    first, first_state = memory(source, prediction)
    second, second_state = memory(changed, prediction)
    torch.testing.assert_close(first[:, :4], second[:, :4])
    assert not torch.allclose(first_state.values, second_state.values)
    assert not torch.allclose(first_state.keys, second_state.keys)


def test_episodic_memory_batches_state_independent_projections() -> None:
    memory = EpisodicMemory(tiny_config())
    source = torch.randn(3, 6, 8)
    prediction = torch.randn(3, 6, 8)
    value_shapes: list[torch.Size] = []
    gate_shapes: list[torch.Size] = []
    value_hook = memory.value_proj.register_forward_pre_hook(
        lambda _module, inputs: value_shapes.append(inputs[0].shape)
    )
    gate_hook = memory.read_gate.register_forward_pre_hook(
        lambda _module, inputs: gate_shapes.append(inputs[0].shape)
    )
    try:
        memory(source, prediction)
    finally:
        value_hook.remove()
        gate_hook.remove()

    assert value_shapes == [torch.Size((3, 6, 4))]
    assert gate_shapes == [torch.Size((3, 6, 16))]


def test_gradient_flow(embed_path: Path) -> None:
    torch.manual_seed(5)
    model = MultigridMemoryTransformer(tiny_config(max_seq_len=8), embed_path)
    input_ids = torch.randint(model.vocab_size, (2, 8))
    logits = model(input_ids)
    loss = logits[..., :64].square().mean()
    loss.backward()
    missing = [
        name
        for name, parameter in model.named_parameters()
        if parameter.requires_grad and parameter.grad is None
    ]
    assert missing == []
    assert model.embeddings.directions.grad is None


def test_compiled_encoder_supports_batched_memory(embed_path: Path) -> None:
    model = MultigridMemoryTransformer(tiny_config(max_seq_len=8), embed_path)
    model.eval()
    input_ids = torch.randint(model.vocab_size, (3, 8))
    with torch.no_grad():
        eager = model.encode(input_ids)
        selected = model.compile_encoder(backend="eager")
        compiled = model.encode(input_ids)

    assert selected == "eager"
    torch.testing.assert_close(compiled, eager)


@pytest.mark.parametrize("seq_len", [1, 2, 3, 5, 9, 16])
def test_different_seq_lengths(embed_path: Path, seq_len: int) -> None:
    model = MultigridMemoryTransformer(tiny_config(), embed_path)
    input_ids = torch.randint(model.vocab_size, (1, seq_len))
    with torch.no_grad():
        logits = model(input_ids)
    assert logits.shape == (1, seq_len, model.vocab_size)


def test_memory_write_gate() -> None:
    memory = EpisodicMemory(tiny_config())
    source = torch.randn(3, 8)
    error = torch.randn(3, 8)
    write_prob = memory.compute_write_prob(source, error)
    assert torch.all(write_prob >= 0)
    assert torch.all(write_prob <= 1)


def test_associative_recall(embed_path: Path) -> None:
    """A tiny fixed recall set can be learned through the model encoder."""
    torch.manual_seed(17)
    config = tiny_config(max_seq_len=8, n_memory_slots=4)
    model = MultigridMemoryTransformer(config, embed_path)
    head = nn.Linear(config.d_model, 17)
    batch = associative_recall(
        batch_size=8,
        n_pairs=2,
        key_vocab_size=8,
        value_vocab_size=8,
        generator=torch.Generator().manual_seed(19),
    )
    labels = batch.targets[:, -1]
    optimizer = torch.optim.AdamW(
        list(model.parameters()) + list(head.parameters()), lr=8e-3
    )
    with torch.no_grad():
        initial_scores = head(model.encode(batch.input_ids)[:, -1])
        initial_loss = F.cross_entropy(initial_scores, labels)
    for _ in range(80):
        optimizer.zero_grad(set_to_none=True)
        scores = head(model.encode(batch.input_ids)[:, -1])
        loss = F.cross_entropy(scores, labels)
        loss.backward()
        optimizer.step()
    with torch.no_grad():
        scores = head(model.encode(batch.input_ids)[:, -1])
        final_loss = F.cross_entropy(scores, labels)
        accuracy = scores.argmax(dim=-1).eq(labels).float().mean()
    assert final_loss < initial_loss * 0.35
    assert accuracy >= 0.75


def test_hash_associative_recall(embed_path: Path) -> None:
    """The fixed recall set is also learnable with sparse hash reads."""
    torch.manual_seed(41)
    config = tiny_config(
        max_seq_len=8,
        n_memory_slots=4,
        memory_addressing="hash",
        hash_top_k=2,
    )
    model = MultigridMemoryTransformer(config, embed_path)
    head = nn.Linear(config.d_model, 17)
    batch = associative_recall(
        batch_size=8,
        n_pairs=2,
        key_vocab_size=8,
        value_vocab_size=8,
        generator=torch.Generator().manual_seed(43),
    )
    labels = batch.targets[:, -1]
    optimizer = torch.optim.AdamW(
        list(model.parameters()) + list(head.parameters()), lr=8e-3
    )
    with torch.no_grad():
        initial_scores = head(model.encode(batch.input_ids)[:, -1])
        initial_loss = F.cross_entropy(initial_scores, labels)
    for _ in range(80):
        optimizer.zero_grad(set_to_none=True)
        scores = head(model.encode(batch.input_ids)[:, -1])
        loss = F.cross_entropy(scores, labels)
        loss.backward()
        optimizer.step()
    with torch.no_grad():
        scores = head(model.encode(batch.input_ids)[:, -1])
        final_loss = F.cross_entropy(scores, labels)
        accuracy = scores.argmax(dim=-1).eq(labels).float().mean()
    assert final_loss < initial_loss * 0.35
    assert accuracy >= 0.75


def test_hash_vs_softmax_both_train() -> None:
    torch.manual_seed(47)
    source = torch.randn(4, 6, 8)
    prediction = torch.randn(4, 6, 8)
    target = torch.randn(4, 6, 8)

    for addressing_mode in ("softmax", "hash"):
        memory = EpisodicMemory(
            tiny_config(memory_addressing=addressing_mode)
        )
        optimizer = torch.optim.Adam(memory.parameters(), lr=2e-2)
        with torch.no_grad():
            initial_output, _ = memory(source, prediction)
            initial_loss = F.mse_loss(initial_output, target)
        for _ in range(20):
            optimizer.zero_grad(set_to_none=True)
            output, _ = memory(source, prediction)
            loss = F.mse_loss(output, target)
            loss.backward()
            optimizer.step()
        with torch.no_grad():
            final_output, _ = memory(source, prediction)
            final_loss = F.mse_loss(final_output, target)
        assert final_loss < initial_loss


def test_num_parameters(embed_path: Path) -> None:
    model = MultigridMemoryTransformer(tiny_config(), embed_path)
    trainable = sum(parameter.numel() for parameter in model.parameters())
    total = trainable + model.embeddings.directions.numel()
    assert model.num_parameters(trainable_only=True) == trainable
    assert model.num_parameters() == total


@pytest.mark.skipif(
    not torch.cuda.is_available() or not triton_memory_available(),
    reason="fused episodic memory requires CUDA/ROCm and Triton",
)
def test_triton_memory_matches_gradients_on_tensor_device() -> None:
    original_device = torch.cuda.current_device()
    target_index = torch.cuda.device_count() - 1
    active_index = 0 if target_index != 0 else target_index
    torch.cuda.set_device(active_index)
    device = torch.device(f"cuda:{target_index}")
    try:
        torch.manual_seed(83)
        batch_size, seq_len, n_writers = 2, 3, 4
        memory = EpisodicMemory(
            tiny_config(n_memory_slots=4, d_memory=4, d_key=4)
        ).to(device)
        memory.train()
        queries = torch.randn(
            batch_size, seq_len, 4, device=device, requires_grad=True
        )
        write_values = torch.randn(
            batch_size,
            n_writers,
            seq_len,
            4,
            device=device,
            requires_grad=True,
        )
        write_keys = torch.randn_like(write_values, requires_grad=True)
        write_priorities = torch.sigmoid(
            torch.randn(batch_size, n_writers, seq_len, device=device)
        ).requires_grad_()
        state = memory.initial_state(
            batch_size, device=device, dtype=queries.dtype
        )
        history = tuple(
            MemoryWrites(
                write_values[:, writer],
                write_keys[:, writer],
                write_priorities[:, writer],
            )
            for writer in range(n_writers - 1)
        )
        eager_reads, eager_state = memory._run_recurrence(
            queries,
            state,
            history,
            write_values[:, -1],
            write_keys[:, -1],
            write_priorities[:, -1],
            True,
        )
        fused_reads, fused_values, fused_keys, fused_priorities = (
            fused_softmax_memory(
                queries,
                write_values,
                write_keys,
                write_priorities,
                4,
                memory.temperature,
            )
        )
        torch.testing.assert_close(fused_reads, eager_reads, atol=1e-6, rtol=1e-6)
        torch.testing.assert_close(
            fused_values, eager_state.values, atol=1e-6, rtol=1e-6
        )
        torch.testing.assert_close(
            fused_keys, eager_state.keys, atol=1e-6, rtol=1e-6
        )
        torch.testing.assert_close(
            fused_priorities,
            eager_state.priorities,
            atol=1e-6,
            rtol=1e-6,
        )

        fused_loss = sum(
            tensor.square().sum()
            for tensor in (
                fused_reads,
                fused_values,
                fused_keys,
                fused_priorities,
            )
        )
        eager_loss = sum(
            tensor.square().sum()
            for tensor in (
                eager_reads,
                eager_state.values,
                eager_state.keys,
                eager_state.priorities,
            )
        )
        inputs = (queries, write_values, write_keys, write_priorities)
        fused_gradients = torch.autograd.grad(
            fused_loss, inputs, retain_graph=True
        )
        eager_gradients = torch.autograd.grad(eager_loss, inputs)
        for fused_gradient, eager_gradient in zip(
            fused_gradients, eager_gradients
        ):
            torch.testing.assert_close(
                fused_gradient,
                eager_gradient,
                atol=2e-5,
                rtol=2e-5,
            )

        # Change every query and write in the future while keeping the same
        # row and tensor shape. Earlier reads must remain bit-identical; using
        # a different batch row can introduce unrelated GEMM rounding noise.
        boundary = 2
        changed_queries = queries.detach().clone()
        changed_values = write_values.detach().clone()
        changed_keys = write_keys.detach().clone()
        changed_priorities = write_priorities.detach().clone()
        changed_queries[:, boundary:] = torch.randn_like(
            changed_queries[:, boundary:]
        )
        changed_values[:, :, boundary:] = torch.randn_like(
            changed_values[:, :, boundary:]
        )
        changed_keys[:, :, boundary:] = torch.randn_like(
            changed_keys[:, :, boundary:]
        )
        changed_priorities[:, :, boundary:] = torch.rand_like(
            changed_priorities[:, :, boundary:]
        )
        with torch.no_grad():
            changed_reads, _, _, _ = fused_softmax_memory(
                changed_queries,
                changed_values,
                changed_keys,
                changed_priorities,
                4,
                memory.temperature,
            )
        torch.testing.assert_close(
            changed_reads[:, :boundary],
            fused_reads[:, :boundary],
            atol=0.0,
            rtol=0.0,
        )

        memory.eval()
        eval_state = memory.initial_state(
            batch_size, device=device, dtype=queries.dtype
        )
        with torch.no_grad():
            eval_reads, eval_state = memory._run_recurrence(
                queries,
                eval_state,
                history,
                write_values[:, -1],
                write_keys[:, -1],
                write_priorities[:, -1],
                True,
            )
            (
                fused_eval_reads,
                fused_eval_values,
                fused_eval_keys,
                fused_eval_priorities,
            ) = fused_softmax_memory(
                queries,
                write_values,
                write_keys,
                write_priorities,
                4,
                memory.temperature,
                hard_overwrite=True,
            )
        torch.testing.assert_close(
            fused_eval_reads, eval_reads, atol=1e-6, rtol=1e-6
        )
        torch.testing.assert_close(fused_eval_values, eval_state.values)
        torch.testing.assert_close(fused_eval_keys, eval_state.keys)
        torch.testing.assert_close(
            fused_eval_priorities, eval_state.priorities
        )
        assert fused_reads.device == device
        assert torch.cuda.current_device() == active_index
    finally:
        torch.cuda.set_device(original_device)
