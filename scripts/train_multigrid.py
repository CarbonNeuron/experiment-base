"""Train the causal multigrid-memory language model on WikiText-103."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from config import (  # noqa: E402
    DataConfig,
    ExperimentConfig,
    RuntimeConfig,
    TrainingConfig,
)
from experiment import run_experiment  # noqa: E402
from multigrid import MultigridMemoryConfig  # noqa: E402


CONFIG = ExperimentConfig(
    model=MultigridMemoryConfig(
        d_model=128,
        n_layers=6,
        n_cycles=2,
        d_ff=512,
        max_seq_len=512,
        n_memory_slots=256,
        d_memory=64,
        d_key=64,
    ),
    data=DataConfig(batch_size=8, num_workers=2),
    training=TrainingConfig(epochs=3, grad_accum_steps=1, ce_backend="full"),
    runtime=RuntimeConfig(
        compile=True,
        compile_mode="default",
        compile_backend="auto",
        checkpoint_dir=Path("checkpoints/multigrid"),
    ),
)


if __name__ == "__main__":
    run_experiment(CONFIG)
