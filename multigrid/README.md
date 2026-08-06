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

Each V-cycle shares its refinement, restriction, and prolongation operators
across every pyramid level. Training at a short context therefore updates all
parameters used at longer contexts instead of reserving inactive parameters
for unseen levels. It also makes the learned computation naturally reusable
when the pyramid grows during length extrapolation.

Each preset task follows a 2 → 4 → 8 → 16 capacity curriculum for 2,000 updates
at batch 512, then is tested at capacities 4, 8, 16, 32, and 64. Shorter
curriculum examples are
right-padded after their supervised answers to the capacity-16 tensor length,
so the curriculum does not cause compiler churn. The answer loss is restricted
to the 96 valid value symbols; this removes the irrelevant preliminary task of
suppressing 416 tokens that can never be answers. Supervised positions never
contain their answer token, avoiding the common copy-from-input benchmark leak.

Corrected scale-shared results are written beneath
`checkpoints/multigrid-evaluation-v3/`:

- `report.json` contains configuration, parameter counts, accuracy curves,
  exact-match rates, and runtime measurements;
- `accuracy.csv` is convenient for plotting capacity curves;
- `runtime.csv` reports prefill speed, peak CUDA allocation, and uncached
  decoding latency;
- `<mechanism>/<task>/latest.pt` allows deterministic resume.

The existing `multigrid-evaluation-v2` report remains a historical result from
the level-specific architecture. Its checkpoints are intentionally not reused
by `v3`, whose shared operators have a different optimizer layout.

Decoding is explicitly labeled `full-prefix replay`: none of the public model
APIs currently exposes an inference cache, so claiming cached per-token latency
would be misleading.

## WikiText transfer evaluation

After the primitive sweep, run the matched-mechanism NLP comparison:

```bash
python scripts/evaluate_multigrid_nlp.py
```

This trains the same five parameter-matched mechanisms on identical
deterministic WikiText-103 token windows. Every model shares the same frozen
SVD token directions and tied vocabulary projection. Fixed sinusoidal
positions make evaluation beyond the 512-token training context meaningful;
the preset reports exact validation perplexity at 128, 256, 512, 1,024, and
2,048 tokens. Its primary length metric scores the same aligned final 128
target tokens at every length, so only the amount of available prefix context
changes. Full-sequence and context-quartile perplexities remain secondary
diagnostics.

The current R9700 preset uses 10,000 updates at batch 8, or 40,960,000 training
tokens per mechanism. Training and validation both use exact full-vocabulary
cross-entropy. Runs checkpoint independently and resume automatically beneath
`checkpoints/multigrid-nlp-evaluation-v2/`. The earlier `v1` checkpoints are
left intact because they use level-specific V-cycle parameters and are not
compatible with the corrected scale-shared model.

The mechanism comparison uses zero dropout. A shared V-cycle refinement is
invoked once per pyramid level, so applying the same nominal dropout rate as a
single-call baseline would impose substantially stronger effective
regularization on Multigrid.

The default seed-42 run is a screening experiment. A final winner claim should
rerun the leading mechanisms with at least seeds 43 and 44. If their
perplexities are close, give each finalist a small learning-rate sweep; the
shared optimizer recipe tests plug-compatible learning, not each architecture's
individually tuned ceiling.

Outputs include:

- `report.md` and `report.json` for the complete comparison;
- `perplexity.csv` for context and position curves;
- `learning_curves.csv` for token-matched optimization curves;
- `training.csv` for parameter counts, throughput, memory, and elapsed time.

`scripts/train_multigrid.py` remains the standalone Multigrid-only WikiText
training preset. It is useful for architecture development, but it is not a
matched comparison.

Edit the configuration-only preset to select a smaller task/mechanism subset
for quick experiments. Keep the same seed and capacities for comparable runs.
