from pathlib import Path
import tempfile
import unittest

import torch

from config import ChainedHydraConfig
from model import ChainedHydraTransformer, RecursiveHydraBlock


class ChainedHydraTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls._temporary_directory = tempfile.TemporaryDirectory()
        temporary_path = Path(cls._temporary_directory.name)
        cls.embed_path = temporary_path / "embeddings_32.pt"
        cls.small_embed_path = temporary_path / "embeddings_16.pt"
        torch.manual_seed(23)
        torch.save(torch.randn(100_277, 32) * 0.2, cls.embed_path)
        torch.save(torch.randn(100_277, 16) * 0.2, cls.small_embed_path)

    @classmethod
    def tearDownClass(cls) -> None:
        cls._temporary_directory.cleanup()

    @staticmethod
    def chained_config(**overrides: object) -> ChainedHydraConfig:
        values = {
            "d_embed": 32,
            "n_heads": 4,
            "n_intake_layers": 1,
            "n_blocks": 2,
            "n_streams": 2,
            "depth": 1,
            "n_stream_layers": 2,
            "n_merge_layers": 1,
            "d_ff_ratio": 2.0,
            "max_seq_len": 64,
            "dropout": 0.0,
        }
        values.update(overrides)
        return ChainedHydraConfig(**values)

    def make_model(self, **overrides: object) -> ChainedHydraTransformer:
        return ChainedHydraTransformer(
            self.chained_config(**overrides), self.embed_path
        )

    def test_forward_shapes(self) -> None:
        model = self.make_model()
        input_ids = torch.randint(0, model.vocab_size, (2, 16))

        with torch.no_grad():
            logits, loss = model(input_ids)

        self.assertIsNone(loss)
        assert logits is not None
        self.assertEqual(logits.shape, (2, 16, model.vocab_size))

    def test_gradient_flow(self) -> None:
        model = self.make_model()
        input_ids = torch.randint(0, model.vocab_size, (1, 4))
        targets = torch.randint(0, model.vocab_size, (1, 4))

        _, loss = model(input_ids, targets)
        assert loss is not None
        loss.backward()

        missing_gradients = [
            name
            for name, parameter in model.named_parameters()
            if parameter.requires_grad and parameter.grad is None
        ]
        self.assertEqual(missing_gradients, [])

    def test_nested_forward(self) -> None:
        model = self.make_model(depth=2)
        input_ids = torch.randint(0, model.vocab_size, (2, 8))

        with torch.no_grad():
            hidden = model.encode(input_ids)

        self.assertEqual(hidden.shape, (2, 8, model.config.d_embed))

    def test_chainable(self) -> None:
        model = self.make_model(n_blocks=3)
        input_ids = torch.randint(0, model.vocab_size, (2, 8))

        with torch.no_grad():
            hidden = model.encode(input_ids)

        self.assertEqual(hidden.shape, (2, 8, model.config.d_embed))

    def test_loss_backward(self) -> None:
        model = self.make_model()
        input_ids = torch.randint(0, model.vocab_size, (2, 8))
        targets = torch.randint(0, model.vocab_size, (2, 8))

        logits, loss = model(
            input_ids,
            targets,
            loss_chunk_size=4,
            loss_backend="sampled",
            loss_negative_samples=31,
        )

        self.assertIsNone(logits)
        assert loss is not None
        self.assertEqual(loss.ndim, 0)
        loss.backward()

    def test_recursive_block_isolation(self) -> None:
        inputs = torch.randn(2, 8, 32)
        for depth in (1, 2):
            block = RecursiveHydraBlock(
                d_embed=32,
                n_streams=2,
                depth=depth,
                n_stream_layers=2,
                n_merge_layers=1,
                n_heads=4,
                d_ff_ratio=2.0,
                dropout=0.0,
                layer_norm_eps=1e-5,
            )
            with self.subTest(depth=depth):
                self.assertEqual(block(inputs).shape, inputs.shape)

    def test_depth_3(self) -> None:
        config = self.chained_config(
            d_embed=16,
            n_heads=2,
            n_blocks=1,
            n_streams=2,
            depth=3,
            n_stream_layers=1,
            n_merge_layers=1,
        )
        model = ChainedHydraTransformer(config, self.small_embed_path)
        input_ids = torch.randint(0, model.vocab_size, (1, 4))

        with torch.no_grad():
            hidden = model.encode(input_ids)

        self.assertEqual(hidden.shape, (1, 4, config.d_embed))


if __name__ == "__main__":
    unittest.main()
