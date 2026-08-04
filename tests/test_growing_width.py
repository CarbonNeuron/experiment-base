"""Tests for the growing-then-frozen-width decoder."""

from pathlib import Path
import tempfile
import unittest

import torch

from config import GrowingWidthConfig
from model import GrowingWidthTransformer


def make_model(
    tmp_path: Path,
    *,
    d_embed: int = 16,
    n_layers: int = 4,
    embed_cutoff_ratio: float = 0.5,
) -> GrowingWidthTransformer:
    """Create a small model backed by a valid test embedding artifact."""
    embed_path = tmp_path / "embeddings.pt"
    torch.save(torch.randn(100_277, d_embed) * 0.2, embed_path)
    config = GrowingWidthConfig(
        d_embed=d_embed,
        n_heads=4,
        n_layers=n_layers,
        d_ff_ratio=2.0,
        max_seq_len=12,
        dropout=0.0,
        embed_cutoff_ratio=embed_cutoff_ratio,
    )
    return GrowingWidthTransformer(config, embed_path)


class GrowingWidthTransformerTests(unittest.TestCase):
    def test_stream_stops_growing_at_embed_cutoff(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            model = make_model(Path(directory))
            input_widths: list[int] = []
            handles = [
                block.register_forward_pre_hook(
                    lambda _module, args: input_widths.append(args[0].size(-1))
                )
                for block in model.blocks
            ]
            try:
                model.encode(torch.randint(0, model.vocab_size, (2, 7)))
            finally:
                for handle in handles:
                    handle.remove()

            self.assertEqual(input_widths, [16, 32, 48, 48])
            cutoff = model.config.embed_cutoff_layer
            self.assertEqual(input_widths[cutoff:], [48, 48])
            self.assertEqual(model.layer_widths, input_widths)

    def test_embed_channels_decay_linearly(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            model = make_model(Path(directory))
            model.eval()
            input_ids = torch.randint(0, model.vocab_size, (2, 7))
            embedded = model.embeddings(input_ids).detach()
            observed: list[torch.Tensor] = []

            # Remove reconstruction writes so this test observes source decay
            # independently of the post-cutoff learned deltas.
            with torch.no_grad():
                for block in model.blocks:
                    if block.embed_write is not None:
                        block.embed_write.weight.zero_()

            handles = [
                block.register_forward_pre_hook(
                    lambda _module, args: observed.append(
                        args[0][..., :16].detach().clone()
                    )
                )
                for block in model.blocks
            ]
            try:
                model.encode(input_ids)
            finally:
                for handle in handles:
                    handle.remove()

            self.assertEqual(
                [model.embed_decay(index) for index in range(4)],
                [1.0, 0.5, 0.0, 0.0],
            )
            for actual, decay in zip(observed, [1.0, 0.5, 0.0, 0.0]):
                torch.testing.assert_close(actual, embedded * decay)

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

    def test_post_cutoff_blocks_write_to_embed_channels(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            model = make_model(Path(directory))
            self.assertEqual(model.config.embed_cutoff_layer, 2)
            self.assertIsNone(model.blocks[0].embed_write)
            self.assertIsNone(model.blocks[1].embed_write)
            self.assertIsNotNone(model.blocks[2].embed_write)
            self.assertIsNotNone(model.blocks[3].embed_write)

            input_ids = torch.randint(0, model.vocab_size, (2, 7))
            last_block = model.blocks[-1]
            assert last_block.embed_write is not None
            fixed_scratch = torch.arange(16, dtype=torch.float32)
            handle = last_block.register_forward_hook(
                lambda _module, _args, output: fixed_scratch.expand_as(output)
            )
            try:
                with torch.no_grad():
                    for block in model.blocks:
                        if block.embed_write is not None:
                            block.embed_write.weight.zero_()
                    without_write = model.encode(input_ids)
                    last_block.embed_write.weight.copy_(torch.eye(16))
                    with_write = model.encode(input_ids)
            finally:
                handle.remove()

            torch.testing.assert_close(without_write, torch.zeros_like(without_write))
            self.assertGreater(with_write.abs().max().item(), 0.0)


if __name__ == "__main__":
    unittest.main()
