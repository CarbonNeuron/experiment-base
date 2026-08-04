from pathlib import Path
import tempfile
import unittest

import torch

from model import GenericTransformer, TransformerConfig


def make_model(tmp_path: Path) -> tuple[GenericTransformer, torch.Tensor]:
    torch.manual_seed(7)
    source = torch.randn(100_277, 16) * 0.3
    embed_path = tmp_path / "embeddings.pt"
    torch.save(source, embed_path)
    config = TransformerConfig(
        d_model=16,
        n_heads=4,
        n_layers=2,
        d_ff=32,
        max_seq_len=12,
        dropout=0.0,
    )
    return GenericTransformer(config, embed_path), source


class GenericTransformerTests(unittest.TestCase):
    def test_svd_initialization_and_freezing(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            model, source = make_model(Path(directory))
            effective = model.effective_embeddings().detach()

            torch.testing.assert_close(effective, source)
            self.assertNotIn(
                "embeddings.directions", dict(model.named_parameters())
            )
            self.assertIn(
                "embeddings.directions", dict(model.named_buffers())
            )
            torch.testing.assert_close(
                model.embeddings.position_embedding.weight.norm(dim=-1).mean(),
                source.norm(dim=-1).mean(),
            )
            torch.testing.assert_close(
                model.embeddings.rotation.weight,
                torch.eye(model.config.d_model),
            )
            self.assertNotIn("embeddings.directions", model.state_dict())

    def test_forward_and_embedding_gradients(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            model, _ = make_model(Path(directory))
            input_ids = torch.randint(0, model.vocab_size, (2, 8))
            targets = torch.randint(0, model.vocab_size, (2, 8))

            with torch.no_grad():
                logits, _ = model(input_ids)
            assert logits is not None
            self.assertEqual(logits.shape, (2, 8, model.vocab_size))

            no_logits, loss = model(
                input_ids,
                targets,
                loss_chunk_size=3,
            )
            self.assertIsNone(no_logits)
            self.assertIsNotNone(loss)
            assert loss is not None
            self.assertTrue(torch.isfinite(loss))
            loss.backward()

            self.assertIsNone(model.embeddings.directions.grad)
            self.assertIsNotNone(model.embeddings.norms.grad)
            self.assertIsNotNone(model.embeddings.rotation.weight.grad)

    def test_vocab_and_compiled_encoder_come_from_embedding_module(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            model, _ = make_model(Path(directory))
            model.eval()
            input_ids = torch.randint(0, model.vocab_size, (2, 8))

            with torch.no_grad():
                eager = model.encode(input_ids)
                model.compile_encoder(backend="eager")
                compiled = model.encode(input_ids)

            self.assertEqual(model.vocab_size, model.embeddings.num_embeddings)
            self.assertFalse(hasattr(model.config, "vocab_size"))
            self.assertFalse(
                any("_orig_mod" in key for key in model.state_dict())
            )
            torch.testing.assert_close(compiled, eager)

    def test_rejects_wrong_embedding_shape(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            embed_path = Path(directory) / "wrong.pt"
            torch.save(torch.randn(10, 8), embed_path)
            config = TransformerConfig(
                d_model=16,
                n_heads=4,
                n_layers=1,
                d_ff=32,
            )
            with self.assertRaisesRegex(ValueError, "embedding shape"):
                GenericTransformer(config, embed_path)


if __name__ == "__main__":
    unittest.main()
