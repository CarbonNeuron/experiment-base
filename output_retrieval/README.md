# Static hard-negative retrieval

The model's effective output rows are `s * D * Q^T`, where `D` contains the
frozen unit directions, `Q` is the learned orthogonal matrix, and
`s = exp(log_magnitude) > 0`. For row-major hidden states, the fixed-space
query is therefore `q = hidden @ Q`. The static index contains `D`, never the
rotated effective table, so optimizer updates do not invalidate it.

Retrieval runs on normalized, detached float32 queries. Candidate IDs are
discrete and have no gradient. The selected rows of the canonical (unnormalized
unless that is the model definition) direction table are then gathered and
rescored as `s * dot(q, direction)` with the original differentiable query.
Thus gradients reach hidden states, the rotation generator, and global scale,
but not the fixed directions or index traversal.

The default hybrid objective keeps the existing uniformly sampled and
importance-corrected loss intact and adds a candidate cross-entropy:

```text
sampled_loss + hard_weight * logsumexp(0, hard_logits - positive_logit)
```

Deterministic, context-dependent IVF candidates are deliberately not inserted
into the sampled-softmax correction. A pairwise softplus variant is available,
but candidate cross-entropy is the default.

## Backends and persistence

`ExactStaticOutputIndex` streams vocabulary chunks and is for tests, recall
measurement, debugging, and explicitly selected small runs.
`IVFStaticOutputIndex` is the dependency-free production backend. It clusters
directions once, probes the best centroids, caps candidates per query, and
scores only those candidates. IVF tensors are derived data, saved separately
from model checkpoints, and validated with a SHA-256 fingerprint of the fixed
directions and relevant index settings. Under initialized `torch.distributed`,
rank zero builds/saves and other ranks load after a barrier.

The repository currently has no FAISS dependency or built-in DDP launcher.
It has no padding token in the WikiText stream; `-1` remains the loss ignore
index, and other reserved IDs can be excluded with
`--hard-invalid-token-ids`. The tokenizer's EOT token is training data and is
not excluded by default.

## Starting configuration

Use sampled training and enable retrieval explicitly:

```bash
python train.py --ce-backend sampled --hard-negatives \
  --hard-k 32 --hard-loss-weight 0.25 --hard-warmup-steps 1000 \
  --hard-index-clusters 512 --hard-index-nprobe 8 \
  --hard-index-max-candidates 2048
```

These are conservative starting values, not universal optima. Tune hard-k,
the random-negative count, nprobe, candidate cap, and hard-loss weight jointly.
Increasing nprobe generally improves recall and costs more irregular candidate
work. Position subsampling bounds overhead while retaining a per-selected-
position mean loss.

Run the benchmark with a compatible local embedding artifact:

```bash
python benchmarks/benchmark_hard_negative.py \
  --embed-path embeddings/openai_svd_embeddings_128d.pt \
  --random-negatives 2048,4096 --hard-k 16,32 --nprobe 4,8
```
