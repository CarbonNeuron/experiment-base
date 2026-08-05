"""Train the growing-then-frozen-width transformer."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from config import (  # noqa: E402
    DataConfig,
    ExperimentConfig,
    GrowingWidthConfig,
    HardNegativeIndexConfig,
    HardNegativeRetrievalConfig,
    RuntimeConfig,
    TrainingConfig,
)
from experiment import run_experiment  # noqa: E402


CONFIG = ExperimentConfig(
    model=GrowingWidthConfig(
        d_embed=128,
        n_heads=8,
        n_layers=12,
        d_ff_ratio=4.0,
        max_seq_len=512,
    ),
    data=DataConfig(batch_size=128, num_workers=2),
    training=TrainingConfig(
        epochs=3,
        grad_accum_steps=1,
        ce_backend="sampled",
        ce_negative_samples=4096,
        hard_negative_retrieval=HardNegativeRetrievalConfig(
            enabled=True,
            hard_k=32,
            query_chunk_size=1024,
            loss_weight=0.25,
            warmup_steps=1000,
            max_positions_per_batch=4096,
            index=HardNegativeIndexConfig(
                num_clusters=512,
                nprobe=8,
                max_candidates_per_query=2048,
            ),
        ),
    ),
    runtime=RuntimeConfig(
        compile=True,
        compile_mode="default",
        compile_backend="auto",
        checkpoint_dir=Path("checkpoints/growing_width"),
    ),
)


if __name__ == "__main__":
    run_experiment(CONFIG)
