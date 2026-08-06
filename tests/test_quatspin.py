import sys
import unittest
from unittest.mock import patch

import torch

from config import CompoundQConfig, TransformerConfig
from model import QuatRMSNorm, QuatSpinFFN, TransformerBlock, quat_mul
from models.baseline import CompoundQBlock
from train import config_from_args, parse_args


class QuatSpinTests(unittest.TestCase):
    def test_hamilton_product_matches_known_result(self) -> None:
        left = torch.tensor([[1.0, 2.0, 3.0, 4.0]])
        right = torch.tensor([[5.0, 6.0, 7.0, 8.0]])
        expected = torch.tensor([[-60.0, 12.0, 30.0, 24.0]])
        torch.testing.assert_close(quat_mul(left, right), expected)

    def test_quaternion_norm_has_learned_channel_scale(self) -> None:
        norm = QuatRMSNorm(3)
        values = torch.randn(2, 5, 12)
        output = norm(values).reshape(2, 5, 3, 4)
        torch.testing.assert_close(
            torch.linalg.vector_norm(output, dim=-1),
            torch.ones(2, 5, 3),
        )

    def test_ffn_preserves_shape_and_backpropagates(self) -> None:
        ffn = QuatSpinFFN(d_model=8, n_quats=3, dropout=0.0)
        inputs = torch.randn(2, 5, 8, requires_grad=True)
        output = ffn(inputs)
        self.assertEqual(output.shape, inputs.shape)

        output.square().mean().backward()
        self.assertIsNotNone(inputs.grad)
        self.assertTrue(
            all(parameter.grad is not None for parameter in ffn.parameters())
        )

    def test_transformer_families_select_quatspin_from_config(self) -> None:
        transformer = TransformerBlock(
            TransformerConfig(
                d_model=8,
                n_heads=2,
                n_layers=1,
                d_ff=16,
                ffn_type="quatspin",
                n_quats=3,
                dropout=0.0,
            )
        )
        compound = CompoundQBlock(
            CompoundQConfig(
                d_model=8,
                n_heads=2,
                n_layers=1,
                d_ff=16,
                ffn_type="quatspin",
                n_quats=3,
                dropout=0.0,
            )
        )
        self.assertIsInstance(transformer.ffn, QuatSpinFFN)
        self.assertIsInstance(compound.ffn, QuatSpinFFN)

    def test_cli_exposes_quatspin_configuration(self) -> None:
        with patch.object(
            sys,
            "argv",
            ["train.py", "--ffn-type", "quatspin", "--n-quats", "17"],
        ):
            config = config_from_args(parse_args())
        self.assertEqual(config.model.ffn_type, "quatspin")
        self.assertEqual(config.model.n_quats, 17)

    def test_rejects_invalid_quatspin_settings(self) -> None:
        with self.assertRaisesRegex(ValueError, "ffn_type"):
            TransformerConfig(ffn_type="unknown")
        with self.assertRaisesRegex(ValueError, "n_quats"):
            TransformerConfig(ffn_type="quatspin", n_quats=0)


if __name__ == "__main__":
    unittest.main()
