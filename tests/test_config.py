import unittest

from config import ExperimentConfig
from scripts.train_128d import CONFIG as CONFIG_128D
from scripts.train_256d import CONFIG as CONFIG_256D
from scripts.train_512d import CONFIG as CONFIG_512D


class ExperimentConfigTests(unittest.TestCase):
    def test_default_config_is_composed_from_independent_sections(self) -> None:
        config = ExperimentConfig()
        self.assertEqual(config.model.d_model, 128)
        self.assertEqual(config.data.batch_size, 8)
        self.assertEqual(config.data.val_num_workers, 0)
        self.assertEqual(config.training.ce_chunk_size, 1024)
        self.assertEqual(config.training.ce_backend, "tiled")
        self.assertEqual(config.runtime.compile_backend, "auto")

    def test_size_scripts_select_matching_artifacts_and_directories(self) -> None:
        configs = (CONFIG_128D, CONFIG_256D, CONFIG_512D)
        self.assertEqual([item.model.d_model for item in configs], [128, 256, 512])
        self.assertEqual(
            [item.model.d_ff for item in configs], [512, 1024, 2048]
        )
        self.assertEqual(
            [item.runtime.checkpoint_dir.name for item in configs],
            ["128d", "256d", "512d"],
        )
        self.assertEqual(CONFIG_128D.training.ce_backend, "sampled")
        self.assertEqual(CONFIG_128D.training.ce_negative_samples, 4096)


if __name__ == "__main__":
    unittest.main()
