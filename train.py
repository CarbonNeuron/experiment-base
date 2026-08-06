"""CLI entry point for the generic transformer experiment."""

from __future__ import annotations

import argparse
from pathlib import Path

from config import (
    COMPILE_BACKENDS,
    COMPILE_MODES,
    DTYPES,
    FFN_TYPES,
    HARD_NEGATIVE_BACKENDS,
    HARD_NEGATIVE_LOSSES,
    LOSS_BACKENDS,
    DataConfig,
    ExperimentConfig,
    HardNegativeDiagnosticsConfig,
    HardNegativeIndexConfig,
    HardNegativeRetrievalConfig,
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
    model.add_argument(
        "--ffn-type",
        choices=FFN_TYPES,
        default="gelu",
        help="Feed-forward nonlinearity (gelu or activation-free quatspin)",
    )
    model.add_argument(
        "--n-quats",
        type=int,
        default=None,
        help="QuatSpin channels per FFN (default: d-model)",
    )

    data = parser.add_argument_group("data")
    data.add_argument("--batch-size", type=int, default=8)
    data.add_argument("--num-workers", type=int, default=2)
    data.add_argument(
        "--val-num-workers",
        type=int,
        default=0,
        help="Validation workers (0 avoids Windows process-spawn stalls)",
    )
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
    training.add_argument(
        "--ce-backend",
        choices=LOSS_BACKENDS,
        default="tiled",
        help="Training loss backend; validation always uses exact tiled loss",
    )
    training.add_argument(
        "--ce-negative-samples",
        type=int,
        default=4096,
        help="Shared negatives per batch when --ce-backend=sampled",
    )
    training.add_argument("--max-steps", type=int, default=0)
    training.add_argument("--eval-every", type=int, default=500)
    training.add_argument("--eval-batches", type=int, default=50)
    training.add_argument("--save-every", type=int, default=1000)
    training.add_argument("--seed", type=int, default=42)

    hard = parser.add_argument_group("hard-negative retrieval")
    hard.add_argument("--hard-negatives", action="store_true")
    hard.add_argument(
        "--hard-negative-backend",
        choices=HARD_NEGATIVE_BACKENDS,
        default="ivf",
    )
    hard.add_argument("--hard-k", type=int, default=32)
    hard.add_argument("--hard-retrieve-extra", type=int, default=8)
    hard.add_argument("--hard-query-chunk-size", type=int, default=1024)
    hard.add_argument("--hard-loss-weight", type=float, default=0.25)
    hard.add_argument("--hard-warmup-steps", type=int, default=1000)
    hard.add_argument(
        "--hard-loss-type", choices=HARD_NEGATIVE_LOSSES, default="candidate_ce"
    )
    hard.add_argument("--hard-pairwise-margin", type=float, default=0.0)
    hard.add_argument(
        "--hard-normalize-directions",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    hard.add_argument(
        "--hard-normalize-queries",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    hard.add_argument("--hard-position-fraction", type=float, default=1.0)
    hard.add_argument("--hard-max-positions", type=int)
    hard.add_argument(
        "--hard-invalid-token-ids",
        default="",
        help="Comma-separated vocabulary IDs excluded from retrieved negatives",
    )
    hard.add_argument("--hard-index-path", type=Path)
    hard.add_argument("--hard-index-rebuild", action="store_true")
    hard.add_argument("--hard-index-clusters", type=int, default=512)
    hard.add_argument("--hard-index-nprobe", type=int, default=8)
    hard.add_argument("--hard-index-max-candidates", type=int, default=2048)
    hard.add_argument("--hard-index-build-batch-size", type=int, default=8192)
    hard.add_argument("--hard-index-kmeans-iterations", type=int, default=8)
    hard.add_argument("--hard-index-vocab-chunk-size", type=int, default=8192)
    hard.add_argument("--hard-index-seed", type=int, default=0)
    hard.add_argument("--hard-log-interval", type=int, default=100)
    hard.add_argument("--hard-exact-recall-interval", type=int, default=0)
    hard.add_argument("--hard-exact-recall-query-count", type=int, default=64)

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
            ffn_type=args.ffn_type,
            n_quats=args.n_quats,
        ),
        data=DataConfig(
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            val_num_workers=args.val_num_workers,
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
            ce_backend=args.ce_backend,
            ce_negative_samples=args.ce_negative_samples,
            max_steps=args.max_steps,
            eval_every=args.eval_every,
            eval_batches=args.eval_batches,
            save_every=args.save_every,
            seed=args.seed,
            hard_negative_retrieval=HardNegativeRetrievalConfig(
                enabled=args.hard_negatives,
                backend=args.hard_negative_backend,
                hard_k=args.hard_k,
                retrieve_extra=args.hard_retrieve_extra,
                query_chunk_size=args.hard_query_chunk_size,
                loss_weight=args.hard_loss_weight,
                warmup_steps=args.hard_warmup_steps,
                loss_type=args.hard_loss_type,
                pairwise_margin=args.hard_pairwise_margin,
                normalize_directions=args.hard_normalize_directions,
                normalize_queries=args.hard_normalize_queries,
                position_fraction=args.hard_position_fraction,
                max_positions_per_batch=args.hard_max_positions,
                invalid_token_ids=tuple(
                    int(value)
                    for value in args.hard_invalid_token_ids.split(",")
                    if value.strip()
                ),
                index=HardNegativeIndexConfig(
                    path=args.hard_index_path,
                    rebuild=args.hard_index_rebuild,
                    num_clusters=args.hard_index_clusters,
                    nprobe=args.hard_index_nprobe,
                    max_candidates_per_query=args.hard_index_max_candidates,
                    build_batch_size=args.hard_index_build_batch_size,
                    kmeans_iterations=args.hard_index_kmeans_iterations,
                    vocab_chunk_size=args.hard_index_vocab_chunk_size,
                    seed=args.hard_index_seed,
                ),
                diagnostics=HardNegativeDiagnosticsConfig(
                    log_interval=args.hard_log_interval,
                    exact_recall_interval=args.hard_exact_recall_interval,
                    exact_recall_query_count=args.hard_exact_recall_query_count,
                ),
            ),
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
