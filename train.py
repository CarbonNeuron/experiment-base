"""CLI entry point for the generic transformer experiment."""

from __future__ import annotations

import argparse
from pathlib import Path

from config import (
    COMPILE_BACKENDS,
    COMPILE_MODES,
    DTYPES,
    DataConfig,
    ExperimentConfig,
    RuntimeConfig,
    TrainingConfig,
    TransformerConfig,
)
from experiment import run_experiment


def parse_args() -> argparse.Namespace:
    """Parse CLI values without constructing models, data, or optimizers."""
    parser = argparse.ArgumentParser(description=__doc__)

    model = parser.add_argument_group("model")
    model.add_argument(
        "--embed-path",
        type=Path,
        default=None,
        help="Optional local OpenAI SVD table (default: download/cache from HF)",
    )
    model.add_argument("--d-model", type=int, default=128)
    model.add_argument("--n-heads", type=int, default=8)
    model.add_argument("--n-layers", type=int, default=6)
    model.add_argument("--d-ff", type=int, default=512)
    model.add_argument("--seq-len", type=int, default=512)
    model.add_argument("--dropout", type=float, default=0.1)

    data = parser.add_argument_group("data")
    data.add_argument("--batch-size", type=int, default=8)
    data.add_argument("--num-workers", type=int, default=2)
    data.add_argument("--cache-dir", type=Path, default=Path("data_cache"))

    training = parser.add_argument_group("training")
    training.add_argument("--epochs", type=int, default=3)
    training.add_argument("--lr", type=float, default=3e-4)
    training.add_argument("--weight-decay", type=float, default=0.1)
    training.add_argument("--warmup-steps", type=int, default=500)
    training.add_argument("--grad-accum-steps", type=int, default=1)
    training.add_argument("--max-grad-norm", type=float, default=1.0)
    training.add_argument(
        "--ce-chunk-size",
        type=int,
        default=1024,
        help="Flattened token positions per tied-output loss chunk (0=full logits)",
    )
    training.add_argument("--max-steps", type=int, default=0)
    training.add_argument("--eval-every", type=int, default=500)
    training.add_argument("--eval-batches", type=int, default=50)
    training.add_argument("--save-every", type=int, default=1000)
    training.add_argument("--seed", type=int, default=42)

    runtime = parser.add_argument_group("runtime")
    runtime.add_argument("--device", default="auto")
    runtime.add_argument("--dtype", choices=DTYPES, default="bf16")
    runtime.add_argument("--compile", action="store_true")
    runtime.add_argument(
        "--compile-mode",
        choices=COMPILE_MODES,
        default="default",
        help="torch.compile mode for the token/transformer encoder path",
    )
    runtime.add_argument(
        "--compile-backend",
        choices=COMPILE_BACKENDS,
        default="auto",
        help="Compiler backend (auto avoids CUDA Inductor when Triton is absent)",
    )
    runtime.add_argument(
        "--checkpoint-dir", type=Path, default=Path("checkpoints")
    )
    runtime.add_argument("--resume", type=Path)
    return parser.parse_args()


def config_from_args(args: argparse.Namespace) -> ExperimentConfig:
    """Translate the CLI boundary into the same config used by Python scripts."""
    return ExperimentConfig(
        model=TransformerConfig(
            d_model=args.d_model,
            n_heads=args.n_heads,
            n_layers=args.n_layers,
            d_ff=args.d_ff,
            max_seq_len=args.seq_len,
            dropout=args.dropout,
        ),
        data=DataConfig(
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            cache_dir=args.cache_dir,
        ),
        training=TrainingConfig(
            epochs=args.epochs,
            lr=args.lr,
            weight_decay=args.weight_decay,
            warmup_steps=args.warmup_steps,
            grad_accum_steps=args.grad_accum_steps,
            max_grad_norm=args.max_grad_norm,
            ce_chunk_size=args.ce_chunk_size,
            max_steps=args.max_steps,
            eval_every=args.eval_every,
            eval_batches=args.eval_batches,
            save_every=args.save_every,
            seed=args.seed,
        ),
        runtime=RuntimeConfig(
            device=args.device,
            dtype=args.dtype,
            compile=args.compile,
            compile_mode=args.compile_mode,
            compile_backend=args.compile_backend,
            checkpoint_dir=args.checkpoint_dir,
            resume=args.resume,
        ),
        embed_path=args.embed_path,
    )


def main() -> None:
    run_experiment(config_from_args(parse_args()))


if __name__ == "__main__":
    main()
