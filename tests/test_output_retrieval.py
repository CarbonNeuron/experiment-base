from pathlib import Path
import tempfile
import unittest

import torch
from torch.nn import functional as F

from config import (
    HardNegativeIndexConfig,
    HardNegativeRetrievalConfig,
)
from output_retrieval import (
    ExactStaticOutputIndex,
    HardNegativeTrainer,
    IVFStaticOutputIndex,
    build_or_load_index,
    filter_hard_negatives,
    hard_negative_loss_from_logits,
)
from output_retrieval.metrics import recall_at_k


class StaticOutputRetrievalTests(unittest.TestCase):
    def test_exact_backend_matches_brute_force_without_full_allocation(self) -> None:
        torch.manual_seed(1)
        vectors = F.normalize(torch.randn(37, 6), dim=-1)
        queries = F.normalize(torch.randn(5, 6), dim=-1)
        index = ExactStaticOutputIndex(vectors, vocab_chunk_size=7)
        scores, ids = index.search(queries, 5)
        expected_scores, expected_ids = (queries @ vectors.T).topk(5, dim=-1)
        torch.testing.assert_close(scores, expected_scores)
        torch.testing.assert_close(ids, expected_ids)
        self.assertLessEqual(index.max_score_columns, 7)
        self.assertLess(index.max_score_columns, vectors.size(0))

    def test_positive_invalid_and_duplicate_filtering_with_padding(self) -> None:
        ids = torch.tensor([[3, 3, 7, -1, 8, 2], [1, 5, 6, 7, 8, 9]])
        scores = torch.arange(12, dtype=torch.float32).reshape(2, 6)
        batch = filter_hard_negatives(
            ids,
            scores,
            torch.tensor([3, 1]),
            hard_k=4,
            vocab_size=8,
            invalid_token_ids=(7,),
        )
        torch.testing.assert_close(batch.token_ids[0], torch.tensor([2, 0, 0, 0]))
        torch.testing.assert_close(
            batch.valid_mask[0], torch.tensor([True, False, False, False])
        )
        torch.testing.assert_close(batch.token_ids[1], torch.tensor([5, 6, 0, 0]))

    def test_candidate_loss_equivalence_empty_mask_and_stability(self) -> None:
        positive = torch.tensor([2.0, 1.0], requires_grad=True)
        hard = torch.tensor([[3.0, -1.0], [1.0e20, -1.0e20]], requires_grad=True)
        mask = torch.tensor([[True, True], [False, False]])
        losses = hard_negative_loss_from_logits(positive, hard, mask)
        manual = -torch.log(
            torch.exp(positive[0])
            / (torch.exp(positive[0]) + torch.exp(hard[0]).sum())
        )
        torch.testing.assert_close(losses[0], manual)
        torch.testing.assert_close(losses[1], torch.tensor(0.0))
        self.assertTrue(torch.isfinite(losses).all())
        losses.sum().backward()
        self.assertTrue(torch.isfinite(positive.grad).all())
        self.assertTrue(torch.isfinite(hard.grad).all())

        pairwise = hard_negative_loss_from_logits(
            positive.detach(),
            hard.detach(),
            mask,
            loss_type="pairwise",
            pairwise_margin=0.5,
        )
        expected_pairwise = F.softplus(
            0.5 + hard.detach()[0] - positive.detach()[0]
        ).mean()
        torch.testing.assert_close(pairwise[0], expected_pairwise)
        torch.testing.assert_close(pairwise[1], torch.tensor(0.0))

    def test_detached_retrieval_and_exact_logit_gradient_flow(self) -> None:
        torch.manual_seed(2)
        directions = F.normalize(torch.randn(23, 5), dim=-1).requires_grad_()
        config = HardNegativeRetrievalConfig(
            enabled=True,
            backend="exact",
            hard_k=4,
            retrieve_extra=2,
            warmup_steps=0,
            index=HardNegativeIndexConfig(vocab_chunk_size=6),
        )
        index = ExactStaticOutputIndex(directions.detach(), vocab_chunk_size=6)
        trainer = HardNegativeTrainer(index, directions, config)
        hidden = torch.randn(4, 5, requires_grad=True)
        generator = torch.randn(5, 5, requires_grad=True)
        skew = 0.5 * (generator - generator.T)
        rotation = torch.matrix_exp(skew)
        log_scale = torch.tensor(0.3, requires_grad=True)
        loss, _ = trainer.compute(
            hidden,
            torch.tensor([0, 1, 2, 3]),
            rotation,
            log_scale.exp(),
        )
        loss.backward()
        self.assertGreater(hidden.grad.abs().sum().item(), 0)
        self.assertGreater(generator.grad.abs().sum().item(), 0)
        self.assertGreater(log_scale.grad.abs().sum().item(), 0)
        self.assertIsNone(directions.grad)
        self.assertIsNone(trainer.last_retrieval_queries.grad_fn)

        empty_loss, _ = trainer.compute(
            hidden,
            torch.full((4,), -1),
            rotation,
            log_scale.exp(),
        )
        self.assertEqual(empty_loss.item(), 0.0)

    def test_positive_scale_does_not_change_ranking(self) -> None:
        torch.manual_seed(3)
        logits = torch.randn(6, 29)
        expected = logits.topk(7, dim=-1).indices
        for scale in (0.01, 1.0, 100.0):
            torch.testing.assert_close(
                (scale * logits).topk(7, dim=-1).indices, expected
            )

    def test_ivf_recall_and_candidate_cap(self) -> None:
        torch.manual_seed(4)
        vectors = F.normalize(torch.randn(400, 12), dim=-1)
        queries = F.normalize(torch.randn(24, 12), dim=-1)
        exact = ExactStaticOutputIndex(vectors, vocab_chunk_size=73)
        ivf = IVFStaticOutputIndex.build(
            vectors,
            num_clusters=16,
            nprobe=6,
            max_candidates_per_query=180,
            build_batch_size=80,
            kmeans_iterations=5,
            seed=5,
        )
        _, exact_ids = exact.search(queries, 10)
        _, approximate_ids = ivf.search(queries, 10)
        recall = recall_at_k(approximate_ids, exact_ids)
        self.assertGreaterEqual(recall.item(), 0.60, f"recall={recall.item():.3f}")
        self.assertLessEqual(ivf.max_scored_candidates, 180)

    def test_index_save_load_and_fingerprint_validation(self) -> None:
        torch.manual_seed(6)
        vectors = F.normalize(torch.randn(64, 7), dim=-1)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "index.pt"
            config = HardNegativeRetrievalConfig(
                enabled=True,
                index=HardNegativeIndexConfig(
                    path=path,
                    num_clusters=8,
                    nprobe=2,
                    max_candidates_per_query=32,
                    build_batch_size=16,
                    kmeans_iterations=2,
                ),
            )
            first, fingerprint, saved_path = build_or_load_index(vectors, config)
            self.assertEqual(saved_path, path)
            self.assertTrue(path.is_file())
            second, loaded_fingerprint, _ = build_or_load_index(vectors, config)
            self.assertEqual(fingerprint, loaded_fingerprint)
            query = F.normalize(torch.randn(3, 7), dim=-1)
            torch.testing.assert_close(
                first.search(query, 4)[1], second.search(query, 4)[1]
            )
            with self.assertRaisesRegex(ValueError, "fingerprint mismatch"):
                build_or_load_index(vectors.roll(1, 0), config)


if __name__ == "__main__":
    unittest.main()
