"""Benchmark sampled-only and hybrid output losses on the repository model.

Example (uses the local released artifact when available)::

    python benchmarks/benchmark_hard_negative.py \
        --embed-path embeddings/openai_svd_embeddings_128d.pt --steps 5

Comma-separated sweep arguments may contain multiple values. Exact retrieval
is opt-in because it is a diagnostic backend for the 100k-token vocabulary.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from itertools import product
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from config import (  # noqa: E402
    HardNegativeIndexConfig,
    HardNegativeRetrievalConfig,
    TransformerConfig,
)
from models import GenericTransformer  # noqa: E402
from output_retrieval import (  # noqa: E402
    ExactStaticOutputIndex,
    HardNegativeTrainer,
    IVFStaticOutputIndex,
)
from output_retrieval.metrics import recall_at_k  # noqa: E402


def csv_values(value: str, converter):
    return [converter(item) for item in value.split(",")]


def synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--embed-path", type=Path)
    parser.add_argument(
        "--device", default="cuda" if torch.cuda.is_available() else "cpu"
    )
    parser.add_argument("--dimension", type=int, default=128)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--sequence-length", type=int, default=64)
    parser.add_argument("--layers", type=int, default=2)
    parser.add_argument("--steps", type=int, default=3)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--random-negatives", default="4096")
    parser.add_argument("--hard-k", default="32")
    parser.add_argument("--nprobe", default="8")
    parser.add_argument("--query-chunk-size", default="1024")
    parser.add_argument("--hard-loss-weight", default="0.25")
    parser.add_argument("--num-clusters", type=int, default=512)
    parser.add_argument("--max-candidates", type=int, default=2048)
    parser.add_argument("--kmeans-iterations", type=int, default=5)
    parser.add_argument("--include-exact", action="store_true")
    parser.add_argument("--recall-queries", type=int, default=64)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)
    torch.manual_seed(11)
    config = TransformerConfig(
        d_model=args.dimension,
        n_heads=max(1, args.dimension // 16),
        n_layers=args.layers,
        d_ff=4 * args.dimension,
        max_seq_len=args.sequence_length,
        dropout=0.0,
    )
    model = GenericTransformer(config, args.embed_path).to(device)
    initial_state = {
        key: value.detach().clone() for key, value in model.state_dict().items()
    }
    inputs = torch.randint(
        model.vocab_size,
        (args.batch_size, args.sequence_length),
        device=device,
    )
    targets = torch.randint(
        model.vocab_size,
        (args.batch_size, args.sequence_length),
        device=device,
    )
    directions = model.embeddings.directions
    normalized = torch.nn.functional.normalize(directions.float(), dim=-1)

    def run_case(
        name: str,
        negative_count: int,
        hard_trainer: HardNegativeTrainer | None,
        hard_weight: float,
    ) -> dict[str, float | int | str]:
        model.load_state_dict(initial_state)
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
        durations: list[float] = []
        final_loss = 0.0
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)
        for step in range(args.warmup + args.steps):
            optimizer.zero_grad(set_to_none=True)
            synchronize(device)
            start = time.perf_counter()
            _, loss = model(
                inputs,
                targets,
                loss_chunk_size=1024,
                loss_backend="sampled",
                loss_negative_samples=negative_count,
                hard_negative_trainer=hard_trainer,
                hard_loss_weight=hard_weight,
            )
            assert loss is not None
            loss.backward()
            optimizer.step()
            synchronize(device)
            elapsed = time.perf_counter() - start
            if step >= args.warmup:
                durations.append(elapsed)
            final_loss = loss.item()
        mean_seconds = sum(durations) / len(durations)
        return {
            "case": name,
            "random_negatives": negative_count,
            "hard_k": hard_trainer.config.hard_k if hard_trainer else 0,
            "hard_weight": hard_weight,
            "step_seconds": mean_seconds,
            "tokens_per_second": inputs.numel() / mean_seconds,
            "peak_bytes": (
                torch.cuda.max_memory_allocated(device) if device.type == "cuda" else 0
            ),
            "final_training_loss": final_loss,
        }

    negative_counts = csv_values(args.random_negatives, int)
    results = [run_case("sampled", count, None, 0.0) for count in negative_counts]
    index_cache: dict[int, IVFStaticOutputIndex] = {}
    sweep = product(
        negative_counts,
        csv_values(args.hard_k, int),
        csv_values(args.nprobe, int),
        csv_values(args.query_chunk_size, int),
        csv_values(args.hard_loss_weight, float),
    )
    for negative_count, hard_k, nprobe, query_chunk_size, hard_weight in sweep:
        if nprobe not in index_cache:
            index_cache[nprobe] = IVFStaticOutputIndex.build(
                normalized,
                num_clusters=args.num_clusters,
                nprobe=nprobe,
                max_candidates_per_query=args.max_candidates,
                kmeans_iterations=args.kmeans_iterations,
            )
        retrieval_config = HardNegativeRetrievalConfig(
            enabled=True,
            hard_k=hard_k,
            query_chunk_size=query_chunk_size,
            loss_weight=hard_weight,
            warmup_steps=0,
            index=HardNegativeIndexConfig(
                num_clusters=args.num_clusters,
                nprobe=nprobe,
                max_candidates_per_query=args.max_candidates,
            ),
        )
        hard_trainer = HardNegativeTrainer(
            index_cache[nprobe], directions, retrieval_config
        )
        result = run_case("hybrid_ivf", negative_count, hard_trainer, hard_weight)
        result.update(nprobe=nprobe, query_chunk_size=query_chunk_size)
        metrics = model.last_hard_negative_metrics
        if metrics is not None:
            result.update(
                retrieval_seconds=metrics["retrieval_seconds"].item(),
                candidate_score_seconds=metrics["candidate_score_seconds"].item(),
                hard_loss_seconds=metrics["hard_loss_seconds"].item(),
            )
        results.append(result)

    recall_queries = torch.nn.functional.normalize(
        torch.randn(args.recall_queries, args.dimension, device=device), dim=-1
    )
    exact = ExactStaticOutputIndex(normalized)
    _, exact_ids = exact.search(recall_queries, max(csv_values(args.hard_k, int)))
    recalls = {}
    for nprobe, index in index_cache.items():
        _, approximate_ids = index.search(recall_queries, exact_ids.size(1))
        recalls[str(nprobe)] = recall_at_k(approximate_ids, exact_ids).item()

    if args.include_exact:
        exact_config = HardNegativeRetrievalConfig(
            enabled=True,
            backend="exact",
            hard_k=max(csv_values(args.hard_k, int)),
            warmup_steps=0,
        )
        exact_trainer = HardNegativeTrainer(exact, directions, exact_config)
        results.append(
            run_case(
                "hybrid_exact",
                negative_counts[0],
                exact_trainer,
                csv_values(args.hard_loss_weight, float)[0],
            )
        )

    print(json.dumps({"results": results, "ivf_recall_at_k": recalls}, indent=2))


if __name__ == "__main__":
    main()
