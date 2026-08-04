from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import torch
from torch.nn import functional as F

from config import (
    HardNegativeIndexConfig,
    HardNegativeRetrievalConfig,
    TransformerConfig,
)
from model import GenericTransformer, resolve_compile_backend
from output_retrieval import ExactStaticOutputIndex, HardNegativeTrainer


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

            source_subset = F.normalize(source[:128], dim=-1)
            effective_subset = F.normalize(effective[:128], dim=-1)
            torch.testing.assert_close(
                effective_subset @ effective_subset.T,
                source_subset @ source_subset.T,
            )
            self.assertEqual(model.embeddings.log_magnitude.numel(), 1)
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
                model.embeddings.rotation.matrix,
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
            self.assertIsNotNone(model.embeddings.log_magnitude.grad)
            self.assertIsNotNone(model.embeddings.rotation.generator.grad)

    def test_fixed_space_rotation_orientation_matches_direct_logits(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            model, _ = make_model(Path(directory))
            with torch.no_grad():
                model.embeddings.rotation.generator.normal_(std=0.2)
            hidden = torch.randn(9, model.config.d_model)
            token_ids = torch.randint(0, model.vocab_size, (9,))
            directions = model.embeddings.directions[token_ids]
            effective = model.embeddings.token_embeddings(token_ids)
            direct = (hidden * effective).sum(dim=-1)
            transformed = model.embeddings.magnitude * (
                model.transform_hidden_to_fixed_space(hidden) * directions
            ).sum(dim=-1)
            torch.testing.assert_close(direct, transformed, rtol=1e-5, atol=1e-5)

    def test_sampled_loss_uses_the_same_full_vocabulary(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            model, _ = make_model(Path(directory))
            input_ids = torch.randint(0, model.vocab_size, (2, 8))
            targets = torch.randint(0, model.vocab_size, (2, 8))
            logits, loss = model(
                input_ids,
                targets,
                loss_chunk_size=3,
                loss_backend="sampled",
                loss_negative_samples=31,
            )
            self.assertIsNone(logits)
            assert loss is not None
            self.assertTrue(torch.isfinite(loss))
            loss.backward()
            self.assertEqual(model.vocab_size, 100_277)
            self.assertIsNotNone(model.embeddings.log_magnitude.grad)

    def test_disabled_hard_path_is_sampled_only_regression(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            model, _ = make_model(Path(directory))
            input_ids = torch.randint(0, model.vocab_size, (2, 8))
            targets = torch.randint(0, model.vocab_size, (2, 8))
            torch.manual_seed(101)
            _, baseline = model(
                input_ids,
                targets,
                loss_chunk_size=3,
                loss_backend="sampled",
                loss_negative_samples=31,
            )
            torch.manual_seed(101)
            _, disabled = model(
                input_ids,
                targets,
                loss_chunk_size=3,
                loss_backend="sampled",
                loss_negative_samples=31,
                hard_negative_trainer=None,
                hard_loss_weight=0.25,
            )
            torch.testing.assert_close(disabled, baseline, rtol=0, atol=0)
            self.assertIsNone(model.last_hard_negative_metrics)

    def test_hybrid_forward_avoids_full_projection_and_backpropagates(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            model, _ = make_model(Path(directory))
            config = HardNegativeRetrievalConfig(
                enabled=True,
                backend="exact",
                hard_k=3,
                retrieve_extra=2,
                warmup_steps=0,
                index=HardNegativeIndexConfig(vocab_chunk_size=8192),
            )
            directions = F.normalize(model.embeddings.directions.float(), dim=-1)
            index = ExactStaticOutputIndex(directions, vocab_chunk_size=8192)
            trainer = HardNegativeTrainer(index, model.embeddings.directions, config)
            input_ids = torch.randint(0, model.vocab_size, (2, 4))
            targets = torch.randint(0, model.vocab_size, (2, 4))
            with patch.object(
                model.embeddings,
                "project",
                side_effect=AssertionError("full projection must not run"),
            ):
                logits, loss = model(
                    input_ids,
                    targets,
                    loss_chunk_size=4,
                    loss_backend="sampled",
                    loss_negative_samples=31,
                    hard_negative_trainer=trainer,
                    hard_loss_weight=0.25,
                )
            self.assertIsNone(logits)
            assert loss is not None
            loss.backward()
            self.assertIsNotNone(model.embeddings.log_magnitude.grad)
            self.assertGreater(
                model.embeddings.rotation.generator.grad.abs().sum().item(), 0
            )
            self.assertIsNotNone(model.last_hard_negative_metrics)

    def test_vocab_and_compiled_encoder_come_from_embedding_module(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            model, _ = make_model(Path(directory))
            model.eval()
            input_ids = torch.randint(0, model.vocab_size, (2, 8))

            with torch.no_grad():
                eager = model.encode(input_ids)
                selected = model.compile_encoder(backend="eager")
                compiled = model.encode(input_ids)

            self.assertEqual(selected, "eager")
            self.assertEqual(model.vocab_size, model.embeddings.num_embeddings)
            self.assertFalse(hasattr(model.config, "vocab_size"))
            self.assertFalse(
                any("_orig_mod" in key for key in model.state_dict())
            )
            torch.testing.assert_close(compiled, eager)

    def test_auto_compile_backend_avoids_inductor_without_triton(self) -> None:
        with patch("model.importlib.util.find_spec", return_value=None):
            self.assertEqual(
                resolve_compile_backend("auto", "cuda"), "aot_eager"
            )
            self.assertEqual(resolve_compile_backend("auto", "cpu"), "inductor")
        with patch("model.importlib.util.find_spec", return_value=object()):
            self.assertEqual(resolve_compile_backend("auto", "cuda"), "inductor")
        self.assertEqual(resolve_compile_backend("eager", "cuda"), "eager")

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
