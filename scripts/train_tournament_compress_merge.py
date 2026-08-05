"""Train the tournament Hydra transformer with learned compression merges."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from config import (  # noqa: E402
    DataConfig,
    ExperimentConfig,
    RuntimeConfig,
    TournamentHydraConfig,
    TrainingConfig,
)
from experiment import run_experiment  # noqa: E402


CONFIG = ExperimentConfig(
    model=TournamentHydraConfig(
        d_embed=128,
        n_heads=8,
        n_intake_layers=2,
        n_blocks=3,
        n_experts=8,
        merge_schedule=(4, 2),
        n_expert_layers=2,
        n_merge_layers=1,
        merge_mode="compress",
        d_ff_ratio=4.0,
        max_seq_len=512,
    ),
    data=DataConfig(batch_size=128, num_workers=2),
    training=TrainingConfig(
        epochs=3,
        grad_accum_steps=1,
        ce_backend="sampled",
        ce_negative_samples=4096,
    ),
    runtime=RuntimeConfig(
        compile=True,
        compile_mode="default",
        compile_backend="auto",
        checkpoint_dir=Path("checkpoints/tournament_compress_merge"),
    ),
)


if __name__ == "__main__":
    run_experiment(CONFIG)
