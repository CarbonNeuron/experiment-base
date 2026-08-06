"""Compare multigrid and matched baselines on diagnostic sequence tasks."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from multigrid import (  # noqa: E402
    PrimitiveEvaluationConfig,
    SymbolicModelConfig,
    run_primitive_evaluation,
)


CONFIG = PrimitiveEvaluationConfig(
    tasks=(
        "associative_recall",
        "multi_query_recall",
        "induction",
        "copying",
        "nested_scopes",
        "state_tracking",
        "paraphrased_retrieval",
    ),
    mechanisms=(
        "multigrid",
        "softmax",
        "linear_attention",
        "ssm",
        "gru",
    ),
    capacities=(4, 8, 16, 32, 64),
    train_capacity=16,
    curriculum_capacities=(2, 4, 8, 16),
    train_steps=2_000,
    batch_size=512,
    eval_batches=16,
    learning_rate=1e-3,
    warmup_steps=200,
    save_every=1_000,
    compile=True,
    compile_mode="default",
    compile_backend="auto",
    output_dir=Path("checkpoints/multigrid-evaluation-v2"),
    model=SymbolicModelConfig(
        d_model=128,
        n_layers=4,
        d_ff=256,
        n_heads=8,
        max_seq_len=512,
        n_memory_slots=64,
        d_memory=64,
        d_key=64,
    ),
)


if __name__ == "__main__":
    report = run_primitive_evaluation(CONFIG)
    print(f"Evaluation report: {report}")
