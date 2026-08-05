# Transformer Experiment Base

A deliberately conventional decoder-only transformer for controlled language
model experiments. The baseline uses the `cl100k_base` vocabulary and the
OpenAI 128-dimensional SVD embedding table through the reusable
[`svd-embeds`](https://github.com/CarbonNeuron/svd-embeds) package.

## Baseline architecture

- 6 pre-norm transformer blocks, 8 attention heads, and a 512-wide GELU FFN
- PyTorch scaled dot-product causal attention
- frozen unit SVD token directions
- one positive learned global magnitude
- an identity-initialized learned orthogonal rotation
- tied input and output embeddings
- learned absolute position embeddings whose initial mean L2 norm is derived
  from (and matches) the source SVD token embeddings

Only the SVD directions are frozen. The global magnitude, orthogonal rotation,
position embeddings, transformer blocks, and final layer norm are trainable.
Vocabulary size and tokenizer come directly from `svd-embeds`; they are not
duplicated in the transformer configuration.

## Setup

Clone the repository and create an environment:

```bash
git clone https://github.com/CarbonNeuron/experiment-base.git
cd experiment-base
python -m venv .venv
```

Activate it on Linux/macOS:

```bash
source .venv/bin/activate
pip install -r requirements.txt
hf auth login
python -m unittest discover -s tests -v
python train.py --device auto
```

On Windows, activate with `.venv\Scripts\activate` instead. Authentication is
currently needed because the `Carbun1/FixingEmbeds` artifact repository is
private. The first default model run downloads the approximately 49 MiB OpenAI
128d table into the normal Hugging Face cache; later runs reuse it. You can
instead provide a compatible local table with `--embed-path`.

The training script downloads and caches WikiText-103, uses bf16 autocast on
GPU by default, displays training and validation progress with `tqdm`, and
writes checkpoints under `checkpoints/`. By default, exact tied-output
cross-entropy uses the `svd-embeds` tiled autograd path, avoiding a retained or
materialized `[batch, sequence, 100277]` logits tensor. `--ce-chunk-size`
bounds the token side of each tile; use `0` for conventional full logits.

For faster small-model experiments, select `--ce-backend sampled` and control
the shared uniform sample with `--ce-negative-samples` (default 4096). This
keeps the complete tokenizer and vocabulary but scores only each true target
and the sampled negatives, with importance correction. Validation always uses
the exact tiled 100k-way loss. The ready-made `scripts/train_128d.py` preset
enables sampled training and encoder compilation by default.

Sampled training can optionally add static hard negatives retrieved from the
frozen SVD directions. Enable it with `--hard-negatives`; it is disabled by
default, and validation remains exact tiled cross-entropy. The production
backend is a dependency-free, fingerprinted IVF index cached separately from
model checkpoints. Exact chunked retrieval is available for tests and recall
diagnostics. Candidate selection is detached, selected logits are recomputed
with gradients, and the hard candidate-CE is added separately from the
existing sampled-softmax correction. See
[`output_retrieval/README.md`](output_retrieval/README.md) for the math,
configuration, tradeoffs, and benchmark command.

Training DataLoader workers are persistent across epochs. Validation defaults
to `--val-num-workers 0` because spawning short-lived worker processes causes a
long `0/50` pause on Windows; its contiguous cached-tensor slices do not benefit
from multiprocessing. This can be overridden independently if needed.

`--compile` applies `torch.compile` to the token/position/transformer encoder
while leaving the vocabulary-loss loop eager. This keeps compile
graphs focused and preserves the loss's bounded-memory behavior. Select a mode
with `--compile-mode` and a backend with `--compile-backend`. The default
`auto` backend uses Inductor when supported, but selects `aot_eager` on CUDA
machines without Triton instead of failing during the first matrix multiply.
Validation uses the compiled encoder too. Training and evaluation modes may
create separate graphs on first use, after which each graph is reused. Any
remaining backend failure automatically falls back to eager execution.

Native CUDA Inductor requires Triton. Standard Windows PyTorch installations
often do not include it; install a compatible `triton-windows` build if you
want Inductor acceleration there, then use `--compile-backend inductor`.

Architecture and training settings are exposed as command-line arguments; run
`python train.py --help` for the full list. `--d-model` selects the matching
OpenAI SVD width and must be one of the released artifact dimensions.

## Code layout

Each module owns one concern:

| Module | Responsibility |
|---|---|
| `config.py` | Structured model, data, training, and runtime settings |
| `data.py` | WikiText token cache and DataLoader construction |
| `models/` | Baseline, Hydra, and growing-width architecture components |
| `model.py` | Backward-compatible exports for `models/` |
| `multigrid/` | Multigrid-memory architecture and benchmarks |
| `training/` | Trainer, objectives, runtime policy, and hard-negative lifecycle |
| `trainer.py` | Backward-compatible exports for `training/` |
| `experiments/` | Model registry and training-objective selection |
| `experiment.py` | Compose any registered model, data, and the shared trainer |
| `train.py` | Translate command-line arguments into configuration |
| `output_retrieval/` | Static exact/IVF indexes, filtering, hard loss, and index I/O |
| `benchmarks/` | Sampled versus hybrid output-training benchmark |

Experiments call `run_experiment()` directly and only define an
`ExperimentConfig`. Models that return their own optimized loss and models that
return logits (such as multigrid) are adapted through `training/objectives.py`,
so presets do not implement private optimizer or checkpoint loops.

To add a new architecture, bind its config type, constructor, and objective
with the public `experiments.register_model()` API. No edits to the runner or
trainer are required. See [experiments/README.md](experiments/README.md) for a
complete template and the supported extension contracts.

## Model-size scripts

Ready-to-edit Python presets live under `scripts/`:

```bash
python scripts/train_128d.py
python scripts/train_256d.py
python scripts/train_512d.py
python scripts/train_multigrid.py
python scripts/train_growing_width.py
```

They cover compact 128d, mid-sized 256d, and larger 512d transformers. Each
file contains its complete `ExperimentConfig` and writes to a separate
checkpoint directory. See `scripts/README.md` for the preset table.

For a quick GPU check without committing to a full run:

```bash
python train.py --max-steps 2 --eval-every 0 --save-every 0
```

Enable encoder compilation with:

```bash
python train.py --compile --compile-mode default
```

## Embedding invariant

If `D` is the frozen unit-direction table, `log_s` one learned scalar, and `Q`
the learned orthogonal rotation initialized to identity, the tied table is:

```text
E = exp(log_s) * D * Q^T
Q = exp(0.5 * (A - A^T))
```

The global scale and orthogonal transform preserve the source table's pairwise
cosine similarities throughout training. The scale begins at the source mean
token magnitude; the rotation begins at identity.
