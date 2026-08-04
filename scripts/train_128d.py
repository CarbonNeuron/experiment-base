"""Train the compact 128d baseline."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from config import (  # noqa: E402
    DataConfig,
    ExperimentConfig,
    RuntimeConfig,
    TrainingConfig,
    TransformerConfig,
)
from experiment import run_experiment  # noqa: E402


CONFIG = ExperimentConfig(
    model=TransformerConfig(
        d_model=128,
        n_heads=8,
        n_layers=6,
        d_ff=512,
        max_seq_len=512,
    ),
    data=DataConfig(batch_size=8, num_workers=2),
    training=TrainingConfig(epochs=3, grad_accum_steps=1),
    runtime=RuntimeConfig(
        compile=True,
        compile_mode="default",
        compile_backend="auto",
        checkpoint_dir=Path("checkpoints/128d"),
    ),
)


if __name__ == "__main__":
    run_experiment(CONFIG)
