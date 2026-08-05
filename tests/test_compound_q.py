from pathlib import Path
import tempfile
import unittest

import torch

from config import CompoundQConfig, TransformerConfig
from model import CompoundQAttention, CompoundQTransformer, GenericTransformer


class CompoundQTests(unittest.TestCase):
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
    def compound_config() -> CompoundQConfig:
        return CompoundQConfig(
            d_model=32,
            n_heads=4,
            n_layers=2,
            d_ff=64,
            max_seq_len=64,
            dropout=0.0,
        )

    def make_model(self) -> CompoundQTransformer:
        return CompoundQTransformer(self.compound_config(), self.embed_path)

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

    def test_compound_q_produces_different_output_than_standard(self) -> None:
        standard_config = TransformerConfig(
            d_model=32,
            n_heads=4,
            n_layers=2,
            d_ff=64,
            max_seq_len=64,
            dropout=0.0,
        )
        standard = GenericTransformer(standard_config, self.embed_path)
        compound = self.make_model()
        compound.embeddings.load_state_dict(standard.embeddings.state_dict())
        compound.final_norm.load_state_dict(standard.final_norm.state_dict())

        with torch.no_grad():
            for standard_block, compound_block in zip(
                standard.blocks, compound.blocks, strict=True
            ):
                compound_block.attn_norm.load_state_dict(
                    standard_block.attn_norm.state_dict()
                )
                compound_block.ffn_norm.load_state_dict(
                    standard_block.ffn_norm.state_dict()
                )
                compound_block.ffn.load_state_dict(
                    standard_block.ffn.state_dict()
                )
                query, key, value = standard_block.attn.qkv.weight.chunk(3)
                compound_block.attn.q1_proj.weight.copy_(query)
                compound_block.attn.q2_proj.weight.copy_(query)
                compound_block.attn.q3_proj.weight.copy_(query)
                compound_block.attn.k_proj.weight.copy_(key)
                compound_block.attn.v_proj.weight.copy_(value)
                compound_block.attn.out_proj.weight.copy_(
                    standard_block.attn.out_proj.weight
                )

        standard.eval()
        compound.eval()
        input_ids = torch.randint(0, standard.vocab_size, (1, 8))
        with torch.no_grad():
            standard_logits, _ = standard(input_ids)
            compound_logits, _ = compound(input_ids)

        assert standard_logits is not None
        assert compound_logits is not None
        self.assertGreater(
            (standard_logits - compound_logits).abs().max().item(), 0.0
        )
        correlation = torch.corrcoef(
            torch.stack((standard_logits.flatten(), compound_logits.flatten()))
        )[0, 1]
        self.assertGreater(correlation.item(), 0.99)

    def test_three_q_projections_exist(self) -> None:
        attention = CompoundQAttention(self.compound_config())
        self.assertTrue(hasattr(attention, "q1_proj"))
        self.assertTrue(hasattr(attention, "q2_proj"))
        self.assertTrue(hasattr(attention, "q3_proj"))

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


if __name__ == "__main__":
    unittest.main()
