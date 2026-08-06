"""Compare matched sequence mechanisms on WikiText-103 language modeling."""

from __future__ import annotations

import argparse
import sys
from dataclasses import replace
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from config import FFN_TYPES  # noqa: E402
from multigrid import (  # noqa: E402
    LanguageEvaluationConfig,
    SymbolicModelConfig,
    run_language_evaluation,
)


CONFIG = LanguageEvaluationConfig(
    mechanisms=(
        "multigrid",
        "softmax",
        "linear_attention",
        "ssm"
    ),
    train_sequence_length=512,
    eval_sequence_lengths=(128, 256, 512, 1024, 2048),
    eval_tail_tokens=128,
    train_steps=10_000,
    batch_size=8,
    eval_batch_size=2,
    eval_batches=16,
    learning_rate=3e-4,
    warmup_steps=100,
    ce_backend="sampled",
    ce_negative_samples=4096,
    log_every=10,
    save_every=250,
    device="cuda:1",
    compile=True,
    compile_mode="default",
    compile_backend="auto",
    output_dir=Path("checkpoints/multigrid-nlp-evaluation-v2-2"),
    model=SymbolicModelConfig(
        vocab_size=100_277,
        d_model=128,
        n_layers=8,
        d_ff=1024,
        n_heads=8,
        max_seq_len=2048,
        dropout=0.0,
        n_memory_slots=64,
        d_memory=64,
        d_key=64,
        ffn_type="gelu",
        n_quats=None,
    ),
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--ffn-type",
        choices=FFN_TYPES,
        default=CONFIG.model.ffn_type,
        help="FFN shared by every compared sequence mechanism",
    )
    parser.add_argument(
        "--n-quats",
        type=int,
        default=CONFIG.model.n_quats,
        help="Reference Multigrid quaternion channels (default: d-model)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Report/checkpoint directory (QuatSpin gets a distinct default)",
    )
    return parser.parse_args(argv)


def config_from_args(args: argparse.Namespace) -> LanguageEvaluationConfig:
    model = replace(
        CONFIG.model,
        ffn_type=args.ffn_type,
        n_quats=args.n_quats,
    )
    output_dir = args.output_dir
    if output_dir is None:
        output_dir = CONFIG.output_dir
        if model.ffn_type == "quatspin":
            channels = model.n_quats or model.d_model
            output_dir = output_dir.with_name(
                f"{output_dir.name}-quatspin-q{channels}"
            )
    return replace(CONFIG, model=model, output_dir=output_dir)


if __name__ == "__main__":
    report = run_language_evaluation(config_from_args(parse_args()))
    print(f"NLP evaluation report: {report}")
