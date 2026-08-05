from pathlib import Path
import tempfile
import unittest

import torch

from config import HydraConfig
from model import HydraTransformer


class HydraTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls._temporary_directory = tempfile.TemporaryDirectory()
        cls.embed_path = Path(cls._temporary_directory.name) / "embeddings.pt"
        torch.manual_seed(17)
        torch.save(torch.randn(100_277, 32) * 0.2, cls.embed_path)

    @classmethod
    def tearDownClass(cls) -> None:
        cls._temporary_directory.cleanup()

    @staticmethod
    def hydra_config() -> HydraConfig:
        return HydraConfig(
            d_embed=32,
            n_streams=4,
            n_heads_per_stream=4,
            n_intake_layers=1,
            n_stream_layers=2,
            n_merge_layers=1,
            d_ff_ratio=2.0,
            max_seq_len=64,
            dropout=0.0,
        )

    def make_model(self) -> HydraTransformer:
        return HydraTransformer(self.hydra_config(), self.embed_path)

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

    def test_streams_are_independent(self) -> None:
        model = self.make_model()
        model.eval()
        input_ids = torch.randint(0, model.vocab_size, (1, 8))

        with torch.no_grad():
            hidden = model.embedding_dropout(model.embeddings(input_ids))
            for block in model.intake_blocks:
                hidden = block(hidden)
            hidden = model.intake_norm(hidden)

            outputs = []
            for stream_blocks in model.streams:
                stream_hidden = hidden
                for block in stream_blocks:
                    stream_hidden = block(stream_hidden)
                outputs.append(stream_hidden)

        for output in outputs[1:]:
            self.assertFalse(torch.equal(outputs[0], output))

    def test_loss_backward_works(self) -> None:
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

    def test_slice_preserves_embed_dim(self) -> None:
        model = self.make_model()
        input_ids = torch.randint(0, model.vocab_size, (2, 16))

        with torch.no_grad():
            hidden = model.encode(input_ids)

        self.assertEqual(hidden.shape, (2, 16, model.config.d_embed))


if __name__ == "__main__":
    unittest.main()
