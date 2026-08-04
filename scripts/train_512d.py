"""Train the larger 512d baseline."""

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
        d_model=512,
        n_heads=8,
        n_layers=12,
        d_ff=2048,
        max_seq_len=512,
    ),
    data=DataConfig(batch_size=2, num_workers=2),
    training=TrainingConfig(epochs=3, grad_accum_steps=4),
    runtime=RuntimeConfig(checkpoint_dir=Path("checkpoints/512d")),
)


if __name__ == "__main__":
    run_experiment(CONFIG)
