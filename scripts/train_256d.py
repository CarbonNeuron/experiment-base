"""Train a mid-sized 256d transformer."""

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
        d_model=256,
        n_heads=8,
        n_layers=8,
        d_ff=1024,
        max_seq_len=512,
    ),
    data=DataConfig(batch_size=4, num_workers=2),
    training=TrainingConfig(epochs=3, grad_accum_steps=2),
    runtime=RuntimeConfig(checkpoint_dir=Path("checkpoints/256d")),
)


if __name__ == "__main__":
    run_experiment(CONFIG)
