# Transformer Experiment Base

A deliberately conventional decoder-only transformer for controlled language
model experiments. The baseline uses the `cl100k_base` vocabulary and the
OpenAI 128-dimensional SVD embedding table through the reusable
[`svd-embeds`](https://github.com/CarbonNeuron/svd-embeds) package.

## Baseline architecture

- 6 pre-norm transformer blocks, 8 attention heads, and a 512-wide GELU FFN
- PyTorch scaled dot-product causal attention
- frozen unit SVD token directions
- learned per-token magnitudes and an identity-initialized shared rotation
- tied input and output embeddings
- learned absolute position embeddings whose initial mean L2 norm is derived
  from (and matches) the source SVD token embeddings

Only the SVD directions are frozen. The token norms, rotation, position
embeddings, transformer blocks, and final layer norm are trainable. Vocabulary
size and tokenizer come directly from `svd-embeds`; they are not duplicated in
the transformer configuration.

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
writes checkpoints under `checkpoints/`. Exact tied-output cross-entropy is
computed in activation-checkpointed token chunks, avoiding a retained
`[batch, sequence, 100277]` logits tensor. Adjust memory/throughput with
`--ce-chunk-size`; use `0` for the conventional full-logit loss.

`--compile` applies `torch.compile` to the token/position/transformer encoder
while leaving the checkpointed vocabulary-loss loop eager. This keeps compile
graphs focused and preserves the loss's bounded-memory behavior. Select a mode
with `--compile-mode` and a backend with `--compile-backend`. The default
`auto` backend uses Inductor when supported, but selects `aot_eager` on CUDA
machines without Triton instead of failing during the first matrix multiply.
Any remaining backend failure automatically falls back to eager execution.

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
| `model.py` | Transformer architecture and encoder compilation |
| `trainer.py` | Optimization, evaluation, progress, and checkpoints |
| `experiment.py` | Compose configured model, data, and trainer |
| `train.py` | Translate command-line arguments into configuration |

Experiments can call `run_experiment()` directly and do not need to imitate the
CLI. This keeps architecture changes independent from dataset and training-loop
changes.

## Model-size scripts

Ready-to-edit Python presets live under `scripts/`:

```bash
python scripts/train_128d.py
python scripts/train_256d.py
python scripts/train_512d.py
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

If `D` is the frozen unit-direction table, `n` the learned per-token norms, and
`R` the learned rotation initialized to identity, the tied table is:

```text
E = R(n[:, None] * D)
```

At initialization this reconstructs the source SVD table exactly.
