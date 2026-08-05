# Model-size scripts

These are normal Python configurations, not wrappers around long CLI commands.
Run one from the repository root:

```bash
python scripts/train_128d.py
python scripts/train_256d.py
python scripts/train_512d.py
python scripts/train_compound_q.py
python scripts/train_hydra.py
python scripts/train_chained_hydra.py
python scripts/train_tournament_hydra.py
python scripts/train_growing_width.py
python scripts/train_multigrid.py
```

Each file owns one model preset and writes to a separate checkpoint directory.
Edit its `CONFIG` object to change that experiment without touching the model,
data pipeline, trainer, or CLI.

| Script | Width | Layers | FFN width | Batch | Accumulation | Train loss |
|---|---:|---:|---:|---:|---:|---|
| `train_128d.py` | 128 | 6 | 512 | 8 | 1 | 4096 sampled negatives |
| `train_256d.py` | 256 | 8 | 1024 | 4 | 2 | Exact tiled |
| `train_512d.py` | 512 | 12 | 2048 | 2 | 4 | Exact tiled |

All validation uses exact tiled cross-entropy, including the 128d sampled-loss
preset.

These entry points intentionally contain configuration only. New presets for a
registered architecture should follow the same pattern and call
`run_experiment(CONFIG)`; they should not construct a `Trainer` or optimizer.
For an entirely new architecture, follow the registration template in
[`experiments/README.md`](../experiments/README.md).
