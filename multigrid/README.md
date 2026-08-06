# Multigrid evaluation

WikiText perplexity does not reveal which long-context computations a model
can implement. The diagnostic runner therefore trains every mechanism from
scratch on each primitive before evaluating language modeling:

- exact and multi-query associative recall;
- induction and copying across arbitrary distractor delays;
- nested scope/variable binding and state tracking under distractors;
- needle retrieval through three equivalent symbolic query templates.

Run the complete comparison from the repository root:

```bash
python scripts/evaluate_multigrid.py
```

The preset compares multigrid memory with matched-parameter softmax attention,
gated causal linear attention, a diagonal state-space model, and a GRU. The
multigrid model fixes the target parameter count; each baseline's FFN width is
chosen to reach the nearest count while embedding width, depth, vocabulary,
and output head remain fixed.

Compilation is enabled by default with the repository's automatic backend
selection. Embeddings, multigrid V-cycles, block FFNs, and output heads are
compiled. The episodic-memory Python boundary stays outside Dynamo to avoid
guards on changing `MemoryState` histories, but its complete softmax-addressed
read/write recurrence uses a fused Triton forward and reverse-time backward on
CUDA/ROCm. The linear-attention and diagonal-SSM baselines likewise use fused
Triton forward/reverse scans instead of launching one PyTorch graph per token.
CPU, hash addressing, missing Triton, and oversized state dimensions fall back
automatically. Set `use_triton_memory=False` to compare multigrid memory against
its reference recurrence or `compile=False` for broader debugging. On ROCm,
softmax runs eagerly because SDPA already selects causal Flash Attention; GRU
uses the fused eager MIOpen/cuDNN path because Dynamo does not support RNN
modules. The short capacity sweep runs eagerly so it does not compile five
one-off validation shapes.

Each preset task follows a 2 → 4 → 8 → 16 capacity curriculum for 2,000 updates
at batch 512, then is tested at capacities 4, 8, 16, 32, and 64. Shorter
curriculum examples are
right-padded after their supervised answers to the capacity-16 tensor length,
so the curriculum does not cause compiler churn. The answer loss is restricted
to the 96 valid value symbols; this removes the irrelevant preliminary task of
suppressing 416 tokens that can never be answers. Supervised positions never
contain their answer token, avoiding the common copy-from-input benchmark leak.

Results are written beneath `checkpoints/multigrid-evaluation-v2/`:

- `report.json` contains configuration, parameter counts, accuracy curves,
  exact-match rates, and runtime measurements;
- `accuracy.csv` is convenient for plotting capacity curves;
- `runtime.csv` reports prefill speed, peak CUDA allocation, and uncached
  decoding latency;
- `<mechanism>/<task>/latest.pt` allows deterministic resume.

Decoding is explicitly labeled `full-prefix replay`: none of the public model
APIs currently exposes an inference cache, so claiming cached per-token latency
would be misleading. WikiText remains the secondary language-model evaluation:

```bash
python scripts/train_multigrid.py
```

Edit the configuration-only preset to select a smaller task/mechanism subset
for quick experiments. Keep the same seed and capacities for comparable runs.
