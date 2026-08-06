# Multigrid Evaluation Results

Score shown: **Token accuracy**. The best result in each test row is bold.

## Evaluation setup

| Setting | Value |
| --- | --- |
| Tests | 7 |
| Mechanisms | 5 |
| Capacities | 4, 8, 16, 32, 64 |
| Training capacity | 16 |
| Training steps | 2000 |
| Batch size | 512 |
| Evaluation batches | 16 |
| Seed | 42 |

## Mean token accuracy across capacities

| Test | Multigrid | Softmax | Linear Attention | SSM | GRU |
| --- | --- | --- | --- | --- | --- |
| Associative Recall | 11.91% | 12.38% | 11.84% | 9.13% | **12.57%** |
| Multi Query Recall | **99.23%** | 12.17% | 11.96% | 6.97% | 12.29% |
| Induction | **98.36%** | 98.36% | 62.93% | 60.47% | 80.68% |
| Copying | **99.96%** | 46.62% | 35.89% | 1.04% | 91.62% |
| Nested Scopes | **65.81%** | 49.89% | 30.12% | 17.98% | 23.34% |
| State Tracking | **65.52%** | 12.59% | 14.67% | 2.00% | 15.46% |
| Paraphrased Retrieval | **73.54%** | 12.20% | 11.95% | 5.11% | 12.25% |

## Scores by capacity

### Capacity 4 — Interpolation

| Test | Multigrid | Softmax | Linear Attention | SSM | GRU |
| --- | --- | --- | --- | --- | --- |
| Associative Recall | 25.78% | 26.12% | 26.34% | 22.95% | **26.79%** |
| Multi Query Recall | **100.00%** | 26.50% | 26.70% | 18.19% | 26.46% |
| Induction | **100.00%** | **100.00%** | **100.00%** | **100.00%** | **100.00%** |
| Copying | **99.82%** | 13.07% | 15.54% | 1.03% | 83.13% |
| Nested Scopes | **98.79%** | 98.23% | 27.92% | 35.30% | 30.10% |
| State Tracking | **100.00%** | 26.18% | 26.31% | 2.70% | 25.69% |
| Paraphrased Retrieval | **99.66%** | 27.01% | 26.12% | 11.83% | 26.76% |

### Capacity 8 — Interpolation

| Test | Multigrid | Softmax | Linear Attention | SSM | GRU |
| --- | --- | --- | --- | --- | --- |
| Associative Recall | 15.61% | **16.21%** | 15.37% | 10.12% | 15.58% |
| Multi Query Recall | **100.00%** | 15.56% | 15.65% | 7.59% | 15.23% |
| Induction | **100.00%** | **100.00%** | **100.00%** | **100.00%** | **100.00%** |
| Copying | **100.00%** | 68.42% | 28.66% | 1.09% | 99.56% |
| Nested Scopes | **88.23%** | 54.94% | 49.93% | 30.26% | 19.29% |
| State Tracking | **100.00%** | 15.56% | 15.68% | 1.90% | 14.83% |
| Paraphrased Retrieval | **99.54%** | 15.15% | 15.91% | 5.71% | 15.47% |

### Capacity 16 — Interpolation

| Test | Multigrid | Softmax | Linear Attention | SSM | GRU |
| --- | --- | --- | --- | --- | --- |
| Associative Recall | 9.86% | 9.86% | 10.03% | 6.81% | **10.47%** |
| Multi Query Recall | **100.00%** | 10.32% | 9.48% | 5.02% | 10.49% |
| Induction | **100.00%** | **100.00%** | **100.00%** | **100.00%** | **100.00%** |
| Copying | **100.00%** | **100.00%** | **100.00%** | 1.07% | **100.00%** |
| Nested Scopes | **100.00%** | 92.96% | 65.51% | 19.63% | 59.68% |
| State Tracking | **99.99%** | 12.26% | 11.77% | 2.02% | 12.83% |
| Paraphrased Retrieval | **86.45%** | 9.91% | 9.64% | 3.97% | 10.40% |

### Capacity 32 — Extrapolation

| Test | Multigrid | Softmax | Linear Attention | SSM | GRU |
| --- | --- | --- | --- | --- | --- |
| Associative Recall | 5.21% | 6.37% | 4.70% | 3.44% | **6.56%** |
| Multi Query Recall | **99.71%** | 5.52% | 5.08% | 2.44% | 5.99% |
| Induction | 97.00% | **100.00%** | 12.95% | 1.22% | 99.05% |
| Copying | **100.00%** | 25.99% | 28.37% | 1.00% | 92.55% |
| Nested Scopes | **27.85%** | 1.83% | 4.62% | 3.22% | 5.08% |
| State Tracking | **26.22%** | 5.78% | 10.21% | 1.67% | 12.18% |
| Paraphrased Retrieval | **54.06%** | 5.73% | 5.37% | 2.32% | 5.37% |

### Capacity 64 — Extrapolation

| Test | Multigrid | Softmax | Linear Attention | SSM | GRU |
| --- | --- | --- | --- | --- | --- |
| Associative Recall | 3.08% | 3.33% | 2.73% | 2.34% | **3.44%** |
| Multi Query Recall | **96.42%** | 2.96% | 2.87% | 1.60% | 3.30% |
| Induction | **94.80%** | 91.78% | 1.72% | 1.11% | 4.36% |
| Copying | **100.00%** | 25.63% | 6.88% | 1.04% | 82.88% |
| Nested Scopes | **14.21%** | 1.51% | 2.63% | 1.47% | 2.54% |
| State Tracking | 1.39% | 3.18% | 9.36% | 1.72% | **11.74%** |
| Paraphrased Retrieval | **27.99%** | 3.19% | 2.72% | 1.71% | 3.27% |

## Mechanisms

| Mechanism | Parameters | Δ vs. multigrid | FFN width |
| --- | --- | --- | --- |
| Multigrid | 5,007,364 | +0 | 256 |
| Softmax | 5,007,640 | +276 | 4,486 |
| Linear Attention | 5,006,884 | -480 | 4,417 |
| SSM | 5,006,860 | -504 | 4,547 |
| GRU | 5,007,116 | -248 | 4,355 |

## Runtime

| Mechanism | Sequence length | Prefill (ms) | Prefill (tokens/s) | Decode (ms/token) | Peak memory (MiB) |
| --- | --- | --- | --- | --- | --- |
| Multigrid | 64 | 12.82 | 4,991.0 | 231.37 | 100.2 |
| Multigrid | 128 | 15.39 | 8,316.2 | 282.03 | 101.0 |
| Multigrid | 256 | 17.58 | 14,565.2 | 271.32 | 102.4 |
| Multigrid | 512 | 19.23 | 26,621.7 | 262.41 | 104.7 |
| Softmax | 64 | 1.91 | 33,498.3 | 1.60 | 106.0 |
| Softmax | 128 | 1.72 | 74,429.2 | 1.52 | 107.2 |
| Softmax | 256 | 1.78 | 144,008.6 | 1.61 | 109.9 |
| Softmax | 512 | 1.90 | 269,817.2 | 1.48 | 114.7 |
| Linear Attention | 64 | 3.37 | 18,972.5 | 15.48 | 205.3 |
| Linear Attention | 128 | 2.95 | 43,453.7 | 15.78 | 206.5 |
| Linear Attention | 256 | 3.01 | 85,121.6 | 16.09 | 209.1 |
| Linear Attention | 512 | 3.12 | 163,898.3 | 15.84 | 214.2 |
| SSM | 64 | 2.27 | 28,195.0 | 10.55 | 203.7 |
| SSM | 128 | 2.06 | 62,053.4 | 10.46 | 205.7 |
| SSM | 256 | 2.14 | 119,483.0 | 10.69 | 207.3 |
| SSM | 512 | 2.30 | 222,941.1 | 10.57 | 212.1 |
| GRU | 64 | 12.59 | 5,081.9 | 11.92 | 203.0 |
| GRU | 128 | 20.90 | 6,124.8 | 20.98 | 204.3 |
| GRU | 256 | 42.06 | 6,086.5 | 39.75 | 206.8 |
| GRU | 512 | 78.16 | 6,551.0 | 78.04 | 211.4 |

## Notes

- WikiText remains a separate secondary evaluation via scripts/train_multigrid.py; these results isolate mechanism-level computation first.
- All mechanisms use full-prefix replay because the public model APIs do not expose inference caches.
