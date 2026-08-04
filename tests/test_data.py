from pathlib import Path
import unittest
from unittest.mock import patch

import torch

from config import DataConfig
from data import TokenDataset, build_dataloaders


class DataLoaderTests(unittest.TestCase):
    def test_validation_avoids_worker_spawn_by_default(self) -> None:
        dataset = TokenDataset(torch.arange(65), seq_len=8)
        config = DataConfig(
            batch_size=2,
            num_workers=2,
            val_num_workers=0,
            cache_dir=Path("unused"),
        )
        with patch("data.load_wikitext", return_value=dataset):
            train_loader, val_loader = build_dataloaders(
                config, seq_len=8, device=torch.device("cpu")
            )

        self.assertEqual(train_loader.num_workers, 2)
        self.assertTrue(train_loader.persistent_workers)
        self.assertEqual(val_loader.num_workers, 0)
        self.assertFalse(val_loader.persistent_workers)


if __name__ == "__main__":
    unittest.main()
