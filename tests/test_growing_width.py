"""Tests for the progressively widening decoder-only transformer."""

from pathlib import Path
import tempfile
import unittest

import torch
from torch import nn

from config import GrowingWidthConfig
from model import GrowingWidthTransformer


def make_model(
    tmp_path: Path,
    *,
    d_embed: int = 16,
    d_model: int = 32,
    n_layers: int = 3,
) -> GrowingWidthTransformer:
    """Create a small model backed by a valid test embedding artifact."""
    embed_path = tmp_path / "embeddings.pt"
    torch.save(torch.randn(100_277, d_embed) * 0.2, embed_path)
    config = GrowingWidthConfig(
        d_embed=d_embed,
        d_model=d_model,
        n_heads=4,
        n_layers=n_layers,
        d_ff_ratio=2.0,
        max_seq_len=12,
        dropout=0.0,
    )
    return GrowingWidthTransformer(config, embed_path)


class GrowingWidthTransformerTests(unittest.TestCase):
    def test_width_schedule_is_linear_and_head_divisible(self) -> None:
        config = GrowingWidthConfig(
            d_embed=128,
            d_model=1024,
            n_heads=16,
            n_layers=12,
        )
        self.assertEqual(
            config.layer_widths,
            [128, 208, 288, 368, 448, 528, 624, 704, 784, 864, 944, 1024],
        )
        self.assertTrue(all(width % 16 == 0 for width in config.layer_widths))

    def test_forward_shapes(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            model = make_model(Path(directory))
            input_ids = torch.randint(0, model.vocab_size, (2, 7))
            hidden = model.encode(input_ids)
            self.assertEqual(hidden.shape, (2, 7, 16))
            with torch.no_grad():
                logits, loss = model(input_ids)
            assert logits is not None
            self.assertEqual(logits.shape, (2, 7, model.vocab_size))
            self.assertIsNone(loss)

    def test_gradients_reach_all_trainable_parameters(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            model = make_model(Path(directory))
            input_ids = torch.randint(0, model.vocab_size, (2, 7))
            targets = torch.randint(0, model.vocab_size, (2, 7))
            _, loss = model(
                input_ids,
                targets,
                loss_chunk_size=4,
                loss_backend="sampled",
                loss_negative_samples=31,
            )
            assert loss is not None
            loss.backward()

            self.assertIsNone(model.embeddings.directions.grad)
            missing = [
                name
                for name, parameter in model.named_parameters()
                if parameter.requires_grad and parameter.grad is None
            ]
            self.assertEqual(missing, [])

    def test_equal_width_uses_only_identity_transitions(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            model = make_model(
                Path(directory), d_embed=16, d_model=16, n_layers=3
            )
            self.assertEqual(model.layer_widths, [16, 16, 16])
            self.assertTrue(
                all(
                    isinstance(projection, nn.Identity)
                    for projection in model.projections
                )
            )
            self.assertIsInstance(model.output_projection, nn.Linear)


if __name__ == "__main__":
    unittest.main()
