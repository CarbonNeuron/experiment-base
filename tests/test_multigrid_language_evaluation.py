from __future__ import annotations

import json
import math
from dataclasses import replace
from pathlib import Path

import pytest
import torch

from models import QuatSpinFFN
from multigrid import (
    MECHANISM_NAMES,
    LanguageEvaluationConfig,
    LanguageEvaluator,
    MechanismLanguageModel,
    SymbolicModelConfig,
    matched_model_configs,
)
from multigrid.language_evaluation import (
    _sample_training_batch,
    _validation_starts,
)


@pytest.fixture(scope="module")
def embed_path(tmp_path_factory: pytest.TempPathFactory) -> Path:
    path = tmp_path_factory.mktemp("multigrid-language") / "embeddings.pt"
    generator = torch.Generator().manual_seed(91)
    torch.save(
        torch.randn(100_277, 8, generator=generator) * 0.2,
        path,
    )
    return path


def tiny_model_config(**overrides: object) -> SymbolicModelConfig:
    values = dict(
        vocab_size=100_277,
        d_model=8,
        n_layers=1,
        d_ff=12,
        n_heads=1,
        max_seq_len=16,
        dropout=0.0,
        n_memory_slots=4,
        d_memory=4,
        d_key=4,
        n_hash_bits=4,
        hash_top_k=2,
    )
    values.update(overrides)
    return SymbolicModelConfig(**values)


def test_training_windows_are_deterministic_across_resume() -> None:
    tokens = torch.arange(100)
    first = _sample_training_batch(
        tokens,
        sequence_length=8,
        batch_size=3,
        seed=42,
        step=7,
    )
    resumed = _sample_training_batch(
        tokens,
        sequence_length=8,
        batch_size=3,
        seed=42,
        step=7,
    )
    another_step = _sample_training_batch(
        tokens,
        sequence_length=8,
        batch_size=3,
        seed=42,
        step=8,
    )

    torch.testing.assert_close(first, resumed)
    assert not torch.equal(first, another_step)
    assert first.shape == (3, 9)
    assert torch.all(first[:, 1:] - first[:, :-1] == 1)


def test_validation_windows_span_the_split() -> None:
    starts = _validation_starts(
        101,
        sequence_length=10,
        example_count=5,
    )

    assert starts.tolist() == [0, 22, 45, 68, 90]


def test_validation_targets_align_across_context_lengths() -> None:
    short = _validation_starts(
        101,
        sequence_length=10,
        maximum_sequence_length=20,
        example_count=5,
    )
    long = _validation_starts(
        101,
        sequence_length=20,
        maximum_sequence_length=20,
        example_count=5,
    )

    torch.testing.assert_close(short + 10, long + 20)


@pytest.mark.parametrize("mechanism", MECHANISM_NAMES)
def test_language_frontend_is_causal(
    mechanism: str,
    embed_path: Path,
) -> None:
    torch.manual_seed(92)
    model = MechanismLanguageModel(
        replace(tiny_model_config(), mechanism=mechanism),
        embed_path,
    )
    model.eval()
    original = torch.randint(0, model.vocab_size, (1, 9))
    changed = original.clone()
    changed[:, 6:] = torch.randint(0, model.vocab_size, (1, 3))

    with torch.no_grad():
        first = model.encode(original)
        second = model.encode(changed)

    torch.testing.assert_close(
        first[:, :6],
        second[:, :6],
        atol=1e-5,
        rtol=1e-5,
    )
    assert not model.embeddings.position_embedding.weight.requires_grad


def test_language_config_rejects_context_beyond_model() -> None:
    with pytest.raises(ValueError, match="exceeds model.max_seq_len"):
        LanguageEvaluationConfig(
            mechanisms=("gru",),
            train_sequence_length=8,
            eval_sequence_lengths=(8, 32),
            eval_tail_tokens=8,
            model=tiny_model_config(max_seq_len=16),
        )


def test_language_frontends_remain_parameter_matched(
    embed_path: Path,
) -> None:
    configs = matched_model_configs(tiny_model_config(), MECHANISM_NAMES)
    counts = {}
    for mechanism, config in configs.items():
        model = MechanismLanguageModel(config, embed_path)
        counts[mechanism] = model.num_parameters()

    target = counts["multigrid"]
    for mechanism, count in counts.items():
        assert abs(count - target) / target < 0.01, mechanism


def test_quatspin_language_frontends_remain_parameter_matched(
    embed_path: Path,
) -> None:
    base = tiny_model_config(ffn_type="quatspin", n_quats=4)
    configs = matched_model_configs(base, MECHANISM_NAMES)
    counts = {}
    for mechanism, config in configs.items():
        model = MechanismLanguageModel(config, embed_path)
        counts[mechanism] = model.num_parameters()
        block = model.blocks[0]
        ffn = block.ffn if mechanism == "multigrid" else block.ffn.layers
        assert isinstance(ffn, QuatSpinFFN)

    target = counts["multigrid"]
    for mechanism, count in counts.items():
        assert abs(count - target) / target < 0.01, mechanism


def test_nlp_script_builds_distinct_quatspin_run_config() -> None:
    from scripts.evaluate_multigrid_nlp import config_from_args, parse_args

    args = parse_args(["--ffn-type", "quatspin", "--n-quats", "85"])
    config = config_from_args(args)

    assert config.model.ffn_type == "quatspin"
    assert config.model.n_quats == 85
    assert config.output_dir.name.endswith("-quatspin-q85")


def test_short_nlp_context_updates_every_multigrid_parameter(
    embed_path: Path,
) -> None:
    model = MechanismLanguageModel(
        replace(
            tiny_model_config(max_seq_len=16),
            mechanism="multigrid",
        ),
        embed_path,
    )
    inputs = torch.randint(0, model.vocab_size, (2, 8))
    probe = torch.randn(2, 8, model.config.d_model)

    (model.encode(inputs) * probe).sum().backward()

    missing = [
        name
        for name, parameter in model.named_parameters()
        if parameter.requires_grad and parameter.grad is None
    ]
    assert missing == []


def test_language_evaluator_writes_complete_report(
    tmp_path: Path,
    embed_path: Path,
) -> None:
    config = LanguageEvaluationConfig(
        mechanisms=("gru",),
        train_sequence_length=4,
        eval_sequence_lengths=(4, 8),
        eval_tail_tokens=4,
        train_steps=1,
        batch_size=2,
        eval_batch_size=1,
        eval_batches=1,
        warmup_steps=0,
        ce_chunk_size=8,
        ce_backend="sampled",
        ce_negative_samples=16,
        log_every=1,
        save_every=0,
        device="cpu",
        dtype="fp32",
        compile=False,
        resume=False,
        output_dir=tmp_path,
        embed_path=embed_path,
        model=tiny_model_config(max_seq_len=8),
    )
    generator = torch.Generator().manual_seed(93)
    train_tokens = torch.randint(100_277, (64,), generator=generator)
    validation_tokens = torch.randint(100_277, (40,), generator=generator)

    report_path = LanguageEvaluator(config).run(
        train_tokens=train_tokens,
        validation_tokens=validation_tokens,
    )
    report = json.loads(
        (tmp_path / "report.json").read_text(encoding="utf-8")
    )

    assert report_path == tmp_path / "report.md"
    assert report["results"][0]["mechanism"] == "gru"
    assert [
        evaluation["regime"]
        for evaluation in report["results"][0]["evaluations"]
    ] == ["trained", "extrapolation"]
    assert set(
        report["results"][0]["evaluations"][0]["position_loss"]
    ) == {"0-25%", "25-50%", "50-75%", "75-100%"}
    assert report["results"][0]["evaluations"][0]["tail_tokens"] == 4
    assert report["results"][0]["evaluations"][0]["tail_token_count"] == 4
    assert (tmp_path / "perplexity.csv").is_file()
    assert (tmp_path / "training.csv").is_file()
    assert (tmp_path / "learning_curves.csv").is_file()
    assert (tmp_path / "gru" / "latest.pt").is_file()

    config.resume = True
    LanguageEvaluator(config).run(
        train_tokens=train_tokens,
        validation_tokens=validation_tokens,
    )
    resumed_report = json.loads(
        (tmp_path / "report.json").read_text(encoding="utf-8")
    )
    assert math.isfinite(
        resumed_report["results"][0]["training"]["last_training_loss"]
    )
