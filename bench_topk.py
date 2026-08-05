"""Benchmark brute-force topk vs flat fast_topk on the GrowingWidthTransformer."""

import torch
import time
import sys
sys.path.insert(0, ".")

from config import GrowingWidthConfig
from models import GrowingWidthTransformer

DEVICE = "cuda:0"
DTYPE = torch.bfloat16
BATCH = 8
SEQ_LEN = 512
K = 32
WARMUP = 3
ITERS = 20


def bench(name, fn, k):
    # warmup
    for _ in range(WARMUP):
        fn(k)
    torch.cuda.synchronize(DEVICE)
    t0 = time.perf_counter()
    for _ in range(ITERS):
        fn(k)
        torch.cuda.synchronize(DEVICE)
    elapsed = (time.perf_counter() - t0) / ITERS * 1000
    return elapsed


def main():
    print(f"Device: {DEVICE}, batch: {BATCH}, seq_len: {SEQ_LEN}, k: {K}")
    cfg = GrowingWidthConfig()
    model = GrowingWidthTransformer(cfg, embed_path=None).to(DEVICE, DTYPE)
    model.eval()

    hidden = torch.randn(BATCH * SEQ_LEN, cfg.d_embed, device=DEVICE, dtype=DTYPE)

    # 1. Brute-force (embeddings.topk - chunked matmul)
    bf_ms = bench("brute-force", lambda k: model.embeddings.topk(hidden, k), K)

    # 2. fast_topk (flat exact - single matmul)
    ft_ms = bench("fast_topk", lambda k: model.fast_topk(hidden, k), K)

    # 3. Raw flat matmul (no rescoring, just queries @ directions.T)
    from torch.nn import functional as F
    dirs_norm = F.normalize(model.embeddings.directions.float(), dim=-1)
    R = model.embeddings.rotation.matrix
    def raw_flat(k):
        q = F.normalize((hidden @ R).float(), dim=-1)
        return (q @ dirs_norm.T).topk(k, dim=-1)
    raw_ms = bench("raw flat matmul", lambda k: raw_flat(k), K)

    print(f"  brute-force topk:  {bf_ms:8.2f} ms")
    print(f"  fast_topk (flat):  {ft_ms:8.2f} ms  ({bf_ms/ft_ms:.1f}x vs brute)")
    print(f"  raw flat matmul:   {raw_ms:8.2f} ms  ({bf_ms/raw_ms:.1f}x vs brute)")

    # Recall check
    with torch.no_grad():
        bf_s, bf_i = model.embeddings.topk(hidden[:256], K)
        ft_s, ft_i = model.fast_topk(hidden[:256], K)
    matches = sum(
        len(set(bf_i[r].tolist()) & set(ft_i[r].tolist()))
        for r in range(bf_i.size(0))
    )
    total = bf_i.size(0) * K
    print(f"  recall@{K}: {matches/total*100:.1f}%")


if __name__ == "__main__":
    main()
