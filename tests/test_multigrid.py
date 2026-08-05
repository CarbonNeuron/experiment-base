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
