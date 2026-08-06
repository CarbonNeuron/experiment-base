from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import pytest
import torch

from multigrid import (
    ANSWER_TOKEN_COUNT,
    MECHANISM_NAMES,
    TASK_NAMES,
    PrimitiveEvaluationConfig,
    PrimitiveTrainer,
    SymbolicModelConfig,
    SymbolicSequenceModel,
    generate_primitive_batch,
    matched_model_configs,
)
from multigrid.evaluation import _masked_loss


def tiny_model_config(**overrides: object) -> SymbolicModelConfig:
    values = dict(
        vocab_size=256,
        d_model=8,
        n_layers=1,
        d_ff=12,
        n_heads=1,
        max_seq_len=64,
        dropout=0.0,
        n_memory_slots=4,
        d_memory=4,
        d_key=4,
        n_hash_bits=4,
        hash_top_k=2,
    )
    values.update(overrides)
    return SymbolicModelConfig(**values)


@pytest.mark.parametrize("task", TASK_NAMES)
def test_primitive_batches_are_deterministic_and_supervised(task: str) -> None:
    first = generate_primitive_batch(
        task, 3, 4, 256, generator=torch.Generator().manual_seed(71)
    )
    second = generate_primitive_batch(
        task, 3, 4, 256, generator=torch.Generator().manual_seed(71)
    )

    torch.testing.assert_close(first.input_ids, second.input_ids)
    torch.testing.assert_close(first.targets, second.targets)
    assert first.input_ids.shape == first.targets.shape
    assert first.input_ids.dtype == torch.long
    assert first.answer_mask.any(dim=1).all()
    assert first.input_ids.min() >= 0
    assert first.input_ids.max() < 256


@pytest.mark.parametrize("task", TASK_NAMES)
def test_supervised_position_never_contains_its_answer(task: str) -> None:
    batch = generate_primitive_batch(
        task, 256, 16, 256, generator=torch.Generator().manual_seed(72)
    )

    assert torch.all(
        batch.input_ids[batch.answer_mask]
        != batch.targets[batch.answer_mask]
    )


def test_copy_targets_are_shifted_instead_of_leaking_answers() -> None:
    batch = generate_primitive_batch(
        "copying", 2, 5, 256, generator=torch.Generator().manual_seed(73)
    )
    # Eight payload tokens, five distractors, a copy marker, and seven
    # teacher-forced previous outputs.
    assert batch.input_ids.shape == (2, 21)
    assert batch.answer_mask.sum(dim=1).tolist() == [8, 8]
    first_answer = batch.answer_mask.to(torch.int64).argmax(dim=1)
    assert torch.all(batch.input_ids.gather(1, first_answer[:, None]) == 3)


def test_answer_loss_ignores_tokens_that_cannot_be_answers() -> None:
    batch = generate_primitive_batch(
        "associative_recall",
        2,
        2,
        256,
        generator=torch.Generator().manual_seed(74),
    )
    logits = torch.zeros(2, batch.input_ids.size(1), 256)
    logits[..., :128] = 100.0
    loss = _masked_loss(logits, batch)
    torch.testing.assert_close(loss, torch.tensor(ANSWER_TOKEN_COUNT).log())


def test_sparse_supervised_projection_matches_full_vocabulary() -> None:
    model = SymbolicSequenceModel(tiny_model_config())
    inputs = torch.randint(0, 256, (2, 7))
    mask = torch.zeros_like(inputs, dtype=torch.bool)
    mask[:, (2, 6)] = True

    full = model(inputs)[..., 128 : 128 + ANSWER_TOKEN_COUNT][mask]
    sparse = model.supervised_logits(
        inputs, mask, 128, ANSWER_TOKEN_COUNT
    )

    torch.testing.assert_close(sparse, full)


def test_curriculum_uses_one_padded_training_shape(tmp_path: Path) -> None:
    config = PrimitiveEvaluationConfig(
        tasks=("associative_recall",),
        mechanisms=("gru",),
        capacities=(4, 8, 16),
        train_capacity=16,
        curriculum_capacities=(2, 4, 8, 16),
        train_steps=4,
        batch_size=2,
        eval_batches=1,
        device="cpu",
        dtype="fp32",
        compile=False,
        output_dir=tmp_path,
        runtime_lengths=(8,),
        runtime_repetitions=1,
        decode_steps=2,
        model=tiny_model_config(max_seq_len=64),
    )
    trainer = PrimitiveTrainer(config)
    padded_length = trainer._training_sequence_length("associative_recall")
    capacities = [trainer._curriculum_capacity(step) for step in range(4)]
    lengths = [
        trainer._make_batch(
            "associative_recall",
            capacity,
            2,
            torch.Generator().manual_seed(75 + capacity),
            pad_to_length=padded_length,
        ).input_ids.size(1)
        for capacity in capacities
    ]

    assert capacities == [2, 4, 8, 16]
    assert lengths == [padded_length] * 4


def test_matched_mechanisms_have_close_counts_and_train() -> None:
    base = tiny_model_config()
    configs = matched_model_configs(base, MECHANISM_NAMES)
    target = SymbolicSequenceModel(configs["multigrid"]).num_parameters()

    for mechanism, config in configs.items():
        model = SymbolicSequenceModel(config)
        inputs = torch.randint(0, config.vocab_size, (2, 9))
        logits = model(inputs)
        logits.square().mean().backward()
        assert logits.shape == (2, 9, config.vocab_size)
        assert abs(model.num_parameters() - target) / target < 0.01, mechanism


def test_symbolic_model_is_causal() -> None:
    torch.manual_seed(79)
    for mechanism in MECHANISM_NAMES:
        model = SymbolicSequenceModel(
            replace(tiny_model_config(), mechanism=mechanism)
        )
        model.eval()
        original = torch.randint(0, 256, (1, 9))
        changed = original.clone()
        changed[:, 6:] = torch.randint(0, 256, (1, 3))
        with torch.no_grad():
            first = model(original)
            second = model(changed)
        torch.testing.assert_close(
            first[:, :6], second[:, :6], atol=1e-5, rtol=1e-5
        )


@pytest.mark.parametrize("training", [False, True])
def test_multigrid_is_causal_at_every_boundary_and_length(
    training: bool,
) -> None:
    torch.manual_seed(80)
    model = SymbolicSequenceModel(
        replace(
            tiny_model_config(max_seq_len=32, n_layers=2),
            mechanism="multigrid",
        )
    )
    model.train(training)
    original = torch.randint(0, 256, (1, 17))
    with torch.no_grad():
        reference = model.encode(original)

    for boundary in range(1, original.size(1)):
        changed = original.clone()
        changed[:, boundary:] = torch.randint(
            0, 256, changed[:, boundary:].shape
        )
        with torch.no_grad():
            candidate = model.encode(changed)
        torch.testing.assert_close(
            candidate[:, :boundary],
            reference[:, :boundary],
            atol=0.0,
            rtol=0.0,
        )

    extended = torch.cat((original, torch.randint(0, 256, (1, 15))), dim=1)
    with torch.no_grad():
        extended_output = model.encode(extended)
    torch.testing.assert_close(
        extended_output[:, : original.size(1)],
        reference,
        atol=0.0,
        rtol=0.0,
    )


@pytest.mark.parametrize("mechanism", MECHANISM_NAMES)
def test_symbolic_models_can_compile(mechanism: str) -> None:
    model = SymbolicSequenceModel(
        replace(tiny_model_config(), mechanism=mechanism)
    )
    model.eval()
    inputs = torch.randint(0, 256, (2, 9))
    with torch.no_grad():
        eager = model(inputs)
        selected = model.compile_model(backend="eager")
        compiled = model(inputs)

    assert selected == "eager"
    torch.testing.assert_close(compiled, eager)


def test_primitive_trainer_writes_report(tmp_path: Path) -> None:
    config = PrimitiveEvaluationConfig(
        tasks=("associative_recall",),
        mechanisms=("gru",),
        capacities=(2,),
        train_capacity=2,
        curriculum_capacities=(2,),
        train_steps=1,
        batch_size=2,
        eval_batches=1,
        warmup_steps=0,
        save_every=0,
        device="cpu",
        dtype="fp32",
        compile=False,
        output_dir=tmp_path,
        runtime_lengths=(8,),
        runtime_repetitions=1,
        decode_steps=2,
        model=tiny_model_config(max_seq_len=16),
    )
    report_path = PrimitiveTrainer(config).run()
    report = json.loads(report_path.read_text(encoding="utf-8"))

    assert report["accuracy"][0]["capacity"] == 2
    assert report["accuracy"][0]["regime"] == "interpolation"
    assert report["runtime"][0]["decode_mode"] == "full-prefix replay"
    assert (tmp_path / "accuracy.csv").is_file()
    assert (tmp_path / "runtime.csv").is_file()
    assert (tmp_path / "gru" / "associative_recall" / "latest.pt").is_file()
