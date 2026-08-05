# Adding an experiment

An experiment has three independent pieces:

1. an architecture config with a `max_seq_len` field;
2. an `nn.Module` constructor accepting `(model_config, embed_path)`;
3. a training objective describing the model's forward API.

The dataset, optimizer, scheduler, evaluation, progress reporting, resume, and
checkpoint behavior all come from the shared runner and trainer.

## Add a preset for an existing architecture

Create a file under `scripts/`, construct an `ExperimentConfig`, and pass it to
`run_experiment`. No registry changes are needed. The existing scripts are
complete examples.

## Add a new architecture

Keep the config and model in their architecture package. Register the pair at
the application boundary:

```python
from dataclasses import dataclass, asdict

from torch import Tensor, nn

from config import DataConfig, ExperimentConfig, RuntimeConfig, TrainingConfig
from experiment import run_experiment
from experiments import register_model
from training import LogitsCrossEntropy


@dataclass
class MyModelConfig:
    d_model: int = 128
    max_seq_len: int = 512

    def to_dict(self):
        return asdict(self)


class MyModel(nn.Module):
    def __init__(self, config: MyModelConfig, embed_path=None):
        super().__init__()
        self.config = config
        # Build the model here.

    @property
    def vocab_size(self) -> int:
        ...

    def num_parameters(self, trainable_only: bool = False) -> int:
        ...

    def forward(self, input_ids: Tensor) -> Tensor:
        # Return [batch, sequence, vocabulary] logits.
        ...


register_model(
    MyModelConfig,
    MyModel,
    objective=LogitsCrossEntropy,
    name="my-model",
)

CONFIG = ExperimentConfig(
    model=MyModelConfig(),
    data=DataConfig(),
    training=TrainingConfig(),
    runtime=RuntimeConfig(checkpoint_dir="checkpoints/my-model"),
)

if __name__ == "__main__":
    run_experiment(CONFIG)
```

Registration is intentionally explicit: one entry binds construction and loss
semantics, so the runner never guesses based on a model class or forward return
shape. `ModelProvidedLoss` is the default for repository models implementing
`forward(input_ids, targets, loss_...) -> (logits, loss)`. Use
`LogitsCrossEntropy` for models returning logits only, or implement the small
`TrainingObjective` protocol for a different task.

Library code and tests can create an isolated `ModelRegistry` and pass it to
`run_experiment(..., registry=registry)`. This avoids process-global
registration and makes experimental packages independently testable.

## Optional capabilities

- `compile_encoder(mode=..., backend=...)` enables the shared compile switch.
- An `embeddings` module with frozen `directions` is required only when
  hard-negative retrieval is enabled.
- The trainer always handles next-token shifting. A custom objective receives
  the original `[batch, sequence + 1]` token batch.

New architectures should not add optimizer loops or branches to
`experiment.py` or `trainer.py`.
