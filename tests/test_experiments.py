"""Tests for the public experiment extension API."""

from dataclasses import asdict, dataclass
from pathlib import Path
import unittest
from unittest.mock import patch

from torch import Tensor, nn

from config import ExperimentConfig, RuntimeConfig
from experiment import run_experiment
from experiments import DEFAULT_REGISTRY, ModelRegistry
from multigrid import MultigridMemoryConfig
from training import LogitsCrossEntropy


@dataclass
class ToyConfig:
    max_seq_len: int = 8
    vocab_size: int = 11

    def to_dict(self):
        return asdict(self)


class ToyModel(nn.Module):
    def __init__(self, config: ToyConfig, embed_path=None) -> None:
        super().__init__()
        self.config = config
        self.seen_embed_path = embed_path
        self.projection = nn.Linear(1, config.vocab_size)

    @property
    def vocab_size(self) -> int:
        return self.config.vocab_size

    def num_parameters(self, trainable_only: bool = False) -> int:
        return sum(
            parameter.numel()
            for parameter in self.parameters()
            if not trainable_only or parameter.requires_grad
        )

    def forward(self, input_ids: Tensor) -> Tensor:
        return self.projection(input_ids.float().unsqueeze(-1))


class ExperimentRegistryTests(unittest.TestCase):
    def test_custom_architecture_registers_without_central_changes(self) -> None:
        registry = ModelRegistry()
        registration = registry.register(
            ToyConfig,
            ToyModel,
            objective=LogitsCrossEntropy,
            name="toy",
        )
        config = ToyConfig()
        model = registry.build_model(config, "embeddings.pt")

        self.assertEqual(registration.display_name, "toy")
        self.assertIsInstance(model, ToyModel)
        self.assertEqual(model.seen_embed_path, "embeddings.pt")
        self.assertIsInstance(
            registry.objective_for(config), LogitsCrossEntropy
        )

    def test_registry_rejects_duplicates_and_explains_missing_types(self) -> None:
        registry = ModelRegistry()
        registry.register(ToyConfig, ToyModel)

        with self.assertRaisesRegex(ValueError, "already registered"):
            registry.register(ToyConfig, ToyModel)

        @dataclass
        class MissingConfig:
            max_seq_len: int = 4

        with self.assertRaisesRegex(TypeError, "register_model"):
            registry.resolve(MissingConfig())

    def test_subclass_uses_the_most_specific_registration(self) -> None:
        @dataclass
        class SpecializedToyConfig(ToyConfig):
            pass

        class SpecializedToyModel(ToyModel):
            pass

        registry = ModelRegistry()
        registry.register(ToyConfig, ToyModel)
        self.assertIsInstance(
            registry.build_model(SpecializedToyConfig()), ToyModel
        )

        registry.register(SpecializedToyConfig, SpecializedToyModel)
        self.assertIsInstance(
            registry.build_model(SpecializedToyConfig()),
            SpecializedToyModel,
        )

    def test_runner_accepts_an_isolated_registry(self) -> None:
        registry = ModelRegistry()
        registry.register(ToyConfig, ToyModel, objective=LogitsCrossEntropy)
        config = ExperimentConfig(
            model=ToyConfig(),
            runtime=RuntimeConfig(device="cpu", dtype="fp32"),
        )
        expected_path = Path("custom-checkpoint.pt")

        class FakeTrainer:
            def __init__(self, *args, objective, **kwargs) -> None:
                del args, kwargs
                self.objective = objective

            def fit(self) -> Path:
                if not isinstance(self.objective, LogitsCrossEntropy):
                    raise AssertionError("runner selected the wrong objective")
                return expected_path

        with (
            patch("experiment.build_dataloaders", return_value=([], [])),
            patch("experiment.Trainer", FakeTrainer),
        ):
            self.assertEqual(
                run_experiment(config, registry=registry), expected_path
            )

    def test_default_catalog_keeps_multigrid_loss_contract(self) -> None:
        registration = DEFAULT_REGISTRY.resolve(MultigridMemoryConfig())
        self.assertEqual(registration.display_name, "multigrid-memory")
        self.assertIsInstance(
            registration.make_objective(), LogitsCrossEntropy
        )


if __name__ == "__main__":
    unittest.main()
