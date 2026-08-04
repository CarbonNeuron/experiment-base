# Transformer Experiment Base

A deliberately conventional decoder-only transformer for controlled language
model experiments. The baseline uses the `cl100k_base` vocabulary and the
OpenAI 128-dimensional SVD embedding table.

## Baseline architecture

- 6 pre-norm transformer blocks, 8 attention heads, and a 512-wide GELU FFN
- PyTorch scaled dot-product causal attention
- frozen unit SVD token directions
- learned per-token magnitudes and an identity-initialized shared rotation
- tied input and output embeddings
- learned absolute position embeddings whose initial mean L2 norm is derived
  from (and matches) the source SVD token embeddings

Only the SVD directions are frozen. The token norms, rotation, position
embeddings, transformer blocks, and final layer norm are trainable.

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
python -m unittest discover -s tests -v
python train.py --device auto
```

On Windows, activate with `.venv\Scripts\activate` instead. The first default
model run automatically downloads the approximately 49 MiB OpenAI 128d SVD
table from `Carbun1/FixingEmbeds` on Hugging Face and stores it in the normal
Hugging Face cache. You can instead provide any compatible local table with
`--embed-path`.

The training script downloads and caches WikiText-103, uses bf16 autocast on
GPU by default, displays training and validation progress with `tqdm`, and
writes checkpoints under `checkpoints/`. Architecture and training settings
are exposed as command-line arguments; run
`python train.py --help` for the full list.

For a quick GPU check without committing to a full run:

```bash
python train.py --max-steps 2 --eval-every 0 --save-every 0
```

## Embedding invariant

If `D` is the frozen unit-direction table, `n` the learned per-token norms, and
`R` the learned rotation initialized to identity, the tied table is:

```text
E = R(n[:, None] * D)
```

At initialization this reconstructs the source SVD table exactly.
