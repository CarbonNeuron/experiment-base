"""Train the growing-then-frozen-width transformer."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch  # noqa: E402

from config import (  # noqa: E402
    DataConfig,
    GrowingWidthConfig,
    RuntimeConfig,
    TrainingConfig,
)
from data import build_dataloaders  # noqa: E402
from model import GrowingWidthTransformer  # noqa: E402
from trainer import Trainer, resolve_device  # noqa: E402


MODEL_CONFIG = GrowingWidthConfig(
    d_embed=128,
    n_heads=8,
    n_layers=12,
    d_ff_ratio=4.0,
    max_seq_len=512,
)
DATA_CONFIG = DataConfig(batch_size=128, num_workers=2)
TRAINING_CONFIG = TrainingConfig(
    epochs=3,
    grad_accum_steps=1,
    ce_backend="sampled",
    ce_negative_samples=4096,
)
RUNTIME_CONFIG = RuntimeConfig(
    compile=True,
    compile_mode="default",
    compile_backend="auto",
    checkpoint_dir=Path("checkpoints/growing_width"),
)


def main() -> None:
    """Build the growing-width model and train it with the shared Trainer."""
    torch.manual_seed(TRAINING_CONFIG.seed)
    device = resolve_device(RUNTIME_CONFIG.device)
    model = GrowingWidthTransformer(MODEL_CONFIG).to(device)
    train_loader, val_loader = build_dataloaders(
        DATA_CONFIG,
        seq_len=MODEL_CONFIG.max_seq_len,
        device=device,
    )
    print(f"device={device} dtype={RUNTIME_CONFIG.dtype}")
    print(
        f"widths={model.layer_widths} vocab={model.vocab_size:,} "
        f"parameters={model.num_parameters():,} total, "
        f"{model.num_parameters(trainable_only=True):,} trainable"
    )
    Trainer(
        model,
        train_loader,
        val_loader,
        MODEL_CONFIG,
        TRAINING_CONFIG,
        RUNTIME_CONFIG,
        device,
    ).fit()


if __name__ == "__main__":
    main()
