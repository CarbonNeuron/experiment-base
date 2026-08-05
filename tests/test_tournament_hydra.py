from pathlib import Path
import tempfile
import unittest

import torch

from config import TournamentHydraConfig
from model import (
    CompressMergeBlock,
    FFNMergeBlock,
    HydraAttention,
    TournamentBlock,
    TournamentHydraTransformer,
    TournamentRound,
)


class TournamentHydraTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls._temporary_directory = tempfile.TemporaryDirectory()
        cls.embed_path = (
            Path(cls._temporary_directory.name) / "embeddings_32.pt"
        )
        torch.manual_seed(29)
        torch.save(torch.randn(100_277, 32) * 0.2, cls.embed_path)

    @classmethod
    def tearDownClass(cls) -> None:
        cls._temporary_directory.cleanup()

    @staticmethod
    def tournament_config(**overrides: object) -> TournamentHydraConfig:
        values = {
            "d_embed": 32,
            "n_heads": 4,
            "n_intake_layers": 1,
            "n_blocks": 1,
            "n_experts": 4,
            "merge_schedule": (2, 2),
            "n_expert_layers": 2,
            "n_merge_layers": 1,
            "d_ff_ratio": 2.0,
            "max_seq_len": 64,
            "dropout": 0.0,
        }
        values.update(overrides)
        return TournamentHydraConfig(**values)

    def make_model(self, **overrides: object) -> TournamentHydraTransformer:
        return TournamentHydraTransformer(
            self.tournament_config(**overrides), self.embed_path
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

    def test_8_experts(self) -> None:
        model = self.make_model(
            n_experts=8,
            merge_schedule=(4, 2),
            n_expert_layers=1,
        )
        input_ids = torch.randint(0, model.vocab_size, (1, 4))

        with torch.no_grad():
            hidden = model.encode(input_ids)

        self.assertEqual(model.config.n_rounds, 2)
        self.assertEqual(hidden.shape, (1, 4, model.config.d_embed))

    def test_funnel_4_2(self) -> None:
        model = self.make_model(
            n_experts=8,
            merge_schedule=(4, 2),
            n_expert_layers=1,
        )
        input_ids = torch.randint(0, model.vocab_size, (1, 4))

        with torch.no_grad():
            hidden = model.encode(input_ids)

        self.assertEqual(len(model.tournament_blocks[0].rounds), 2)
        self.assertEqual(hidden.shape, (1, 4, model.config.d_embed))

    def test_single_merge(self) -> None:
        model = self.make_model(
            n_experts=8,
            merge_schedule=(8,),
            n_expert_layers=1,
        )
        input_ids = torch.randint(0, model.vocab_size, (1, 4))

        with torch.no_grad():
            hidden = model.encode(input_ids)

        self.assertEqual(hidden.shape, (1, 4, model.config.d_embed))

    def test_schedule_2_2_2(self) -> None:
        model = self.make_model(
            n_experts=8,
            merge_schedule=(2, 2, 2),
            n_expert_layers=1,
        )
        input_ids = torch.randint(0, model.vocab_size, (1, 4))

        with torch.no_grad():
            hidden = model.encode(input_ids)

        self.assertEqual(hidden.shape, (1, 4, model.config.d_embed))

    def test_chained(self) -> None:
        model = self.make_model(n_blocks=3, n_expert_layers=1)
        input_ids = torch.randint(0, model.vocab_size, (1, 4))

        with torch.no_grad():
            hidden = model.encode(input_ids)

        self.assertEqual(hidden.shape, (1, 4, model.config.d_embed))

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

    def test_tournament_round_isolation(self) -> None:
        eight_streams = [torch.randn(2, 8, 32) for _ in range(8)]
        four_stream_round = TournamentRound(
            n_groups=2,
            group_size=4,
            d_embed=32,
            n_merge_layers=1,
            d_ff_ratio=2.0,
            dropout=0.0,
            layer_norm_eps=1e-5,
        )
        four_stream_outputs = four_stream_round(eight_streams)

        self.assertEqual(len(four_stream_outputs), 2)
        self.assertTrue(
            all(
                output.shape == eight_streams[0].shape
                for output in four_stream_outputs
            )
        )

        four_streams = [torch.randn(2, 8, 32) for _ in range(4)]
        two_stream_round = TournamentRound(
            n_groups=2,
            group_size=2,
            d_embed=32,
            n_merge_layers=1,
            d_ff_ratio=2.0,
            dropout=0.0,
            layer_norm_eps=1e-5,
        )
        two_stream_outputs = two_stream_round(four_streams)

        self.assertEqual(len(two_stream_outputs), 2)
        self.assertTrue(
            all(
                output.shape == four_streams[0].shape
                for output in two_stream_outputs
            )
        )

    def test_tournament_block_isolation(self) -> None:
        inputs = torch.randn(2, 8, 32)
        block = TournamentBlock(
            d_embed=32,
            n_experts=4,
            merge_schedule=(2, 2),
            n_expert_layers=2,
            n_merge_layers=1,
            n_heads=4,
            d_ff_ratio=2.0,
            dropout=0.0,
            layer_norm_eps=1e-5,
        )

        self.assertEqual(block(inputs).shape, inputs.shape)

    def test_experts_are_independent(self) -> None:
        block = TournamentBlock(
            d_embed=32,
            n_experts=4,
            merge_schedule=(2, 2),
            n_expert_layers=2,
            n_merge_layers=1,
            n_heads=4,
            d_ff_ratio=2.0,
            dropout=0.0,
            layer_norm_eps=1e-5,
        )
        weight_pointers = [
            expert[0].attn.qkv.weight.data_ptr() for expert in block.experts
        ]

        self.assertEqual(len(set(weight_pointers)), 4)

    def test_2_experts(self) -> None:
        model = self.make_model(n_experts=2, merge_schedule=(2,))
        input_ids = torch.randint(0, model.vocab_size, (1, 4))

        with torch.no_grad():
            hidden = model.encode(input_ids)

        self.assertEqual(model.config.n_rounds, 1)
        self.assertEqual(hidden.shape, (1, 4, model.config.d_embed))

    def test_ffn_merge_forward(self) -> None:
        model = self.make_model(merge_mode="ffn")
        input_ids = torch.randint(0, model.vocab_size, (2, 16))

        with torch.no_grad():
            logits, loss = model(input_ids)

        self.assertIsNone(loss)
        assert logits is not None
        self.assertEqual(logits.shape, (2, 16, model.vocab_size))

    def test_ffn_merge_gradient_flow(self) -> None:
        model = self.make_model(merge_mode="ffn")
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

    def test_compress_merge_forward(self) -> None:
        model = self.make_model(merge_mode="compress")
        input_ids = torch.randint(0, model.vocab_size, (2, 16))

        with torch.no_grad():
            logits, loss = model(input_ids)

        self.assertIsNone(loss)
        assert logits is not None
        self.assertEqual(logits.shape, (2, 16, model.vocab_size))

    def test_compress_merge_gradient_flow(self) -> None:
        model = self.make_model(merge_mode="compress")
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

    def test_compress_merge_has_attention(self) -> None:
        block = TournamentBlock(
            d_embed=32,
            n_experts=4,
            merge_schedule=(2, 2),
            n_expert_layers=2,
            n_merge_layers=1,
            n_heads=4,
            d_ff_ratio=2.0,
            dropout=0.0,
            layer_norm_eps=1e-5,
            merge_mode="compress",
        )

        compress_blocks = [
            merge_block
            for round_module in block.rounds
            for merger in round_module.group_mergers
            for merge_block in merger
            if isinstance(merge_block, CompressMergeBlock)
        ]
        self.assertTrue(compress_blocks)
        self.assertTrue(
            all(
                any(
                    isinstance(module, HydraAttention)
                    for module in merge_block.modules()
                )
                for merge_block in compress_blocks
            )
        )

    def test_compress_merge_learned_projection(self) -> None:
        block = TournamentBlock(
            d_embed=32,
            n_experts=4,
            merge_schedule=(2, 2),
            n_expert_layers=2,
            n_merge_layers=1,
            n_heads=4,
            d_ff_ratio=2.0,
            dropout=0.0,
            layer_norm_eps=1e-5,
            merge_mode="compress",
        )

        self.assertTrue(
            any(
                isinstance(module, CompressMergeBlock)
                for module in block.modules()
            )
        )

    def test_compress_merge_output_not_slice(self) -> None:
        torch.manual_seed(41)
        compress_block = TournamentBlock(
            d_embed=32,
            n_experts=4,
            merge_schedule=(2, 2),
            n_expert_layers=2,
            n_merge_layers=1,
            n_heads=4,
            d_ff_ratio=2.0,
            dropout=0.0,
            layer_norm_eps=1e-5,
            merge_mode="compress",
        )
        torch.manual_seed(41)
        full_block = TournamentBlock(
            d_embed=32,
            n_experts=4,
            merge_schedule=(2, 2),
            n_expert_layers=2,
            n_merge_layers=1,
            n_heads=4,
            d_ff_ratio=2.0,
            dropout=0.0,
            layer_norm_eps=1e-5,
            merge_mode="full",
        )
        inputs = torch.randn(2, 8, 32)

        with torch.no_grad():
            compress_output = compress_block(inputs)
            full_output = full_block(inputs)

        self.assertFalse(torch.allclose(compress_output, full_output))

    def test_ffn_merge_no_attention(self) -> None:
        block = TournamentBlock(
            d_embed=32,
            n_experts=4,
            merge_schedule=(2, 2),
            n_expert_layers=2,
            n_merge_layers=1,
            n_heads=4,
            d_ff_ratio=2.0,
            dropout=0.0,
            layer_norm_eps=1e-5,
            merge_mode="ffn",
        )

        merge_blocks = [
            merge_block
            for round_module in block.rounds
            for merger in round_module.group_mergers
            for merge_block in merger
        ]
        self.assertTrue(merge_blocks)
        self.assertTrue(
            all(isinstance(module, FFNMergeBlock) for module in merge_blocks)
        )
        self.assertFalse(
            any(
                isinstance(module, HydraAttention)
                for merge_block in merge_blocks
                for module in merge_block.modules()
            )
        )

    def test_ffn_merge_fewer_params(self) -> None:
        full_model = self.make_model(merge_mode="full")
        ffn_model = self.make_model(merge_mode="ffn")

        full_params = sum(
            parameter.numel()
            for parameter in full_model.parameters()
            if parameter.requires_grad
        )
        ffn_params = sum(
            parameter.numel()
            for parameter in ffn_model.parameters()
            if parameter.requires_grad
        )

        self.assertLess(ffn_params, full_params)

    def test_full_merge_default(self) -> None:
        block = TournamentBlock(
            d_embed=32,
            n_experts=4,
            merge_schedule=(2, 2),
            n_expert_layers=2,
            n_merge_layers=1,
            n_heads=4,
            d_ff_ratio=2.0,
            dropout=0.0,
            layer_norm_eps=1e-5,
        )

        merge_blocks = [
            merge_block
            for round_module in block.rounds
            for merger in round_module.group_mergers
            for merge_block in merger
        ]
        self.assertTrue(
            any(
                isinstance(module, HydraAttention)
                for merge_block in merge_blocks
                for module in merge_block.modules()
            )
        )


if __name__ == "__main__":
    unittest.main()
