from pathlib import Path
import tempfile
import unittest

import torch
from torch.utils.data import DataLoader

from config import RuntimeConfig, TrainingConfig, TransformerConfig
from data import TokenDataset
from model import GenericTransformer
from trainer import Trainer


class TrainerTests(unittest.TestCase):
    def test_final_partial_accumulation_group_updates_and_checkpoints(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            table = torch.randn(100_277, 4) * 0.2
            embed_path = root / "table.pt"
            torch.save(table, embed_path)

            model_config = TransformerConfig(
                d_model=4,
                n_heads=1,
                n_layers=1,
                d_ff=8,
                max_seq_len=4,
                dropout=0.0,
            )
            model = GenericTransformer(model_config, embed_path)
            dataset = TokenDataset(
                torch.randint(0, model.vocab_size, (10,)), seq_len=4
            )
            loader = DataLoader(dataset, batch_size=1, shuffle=False)
            training = TrainingConfig(
                epochs=1,
                warmup_steps=0,
                grad_accum_steps=3,
                ce_chunk_size=2,
                max_steps=1,
                eval_every=0,
                save_every=0,
            )
            runtime = RuntimeConfig(
                device="cpu",
                dtype="fp32",
                checkpoint_dir=root / "checkpoints",
            )
            trainer = Trainer(
                model,
                loader,
                loader,
                model_config,
                training,
                runtime,
                torch.device("cpu"),
            )

            checkpoint_path = trainer.fit()
            self.assertEqual(trainer.step, 1)
            self.assertTrue(checkpoint_path.is_file())
            checkpoint = torch.load(
                checkpoint_path, map_location="cpu", weights_only=False
            )
            self.assertNotIn("embeddings.directions", checkpoint["model"])


if __name__ == "__main__":
    unittest.main()
