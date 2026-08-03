"""Tests for H100 training migration artifacts (configs + launch wrapper).

Run with: python3 -m pytest tests/test_h100_train_config.py
Or standalone: python3 tests/test_h100_train_config.py
"""

import os
import re
import subprocess
import unittest

import yaml

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TRAIN_DIR = os.path.join(REPO_ROOT, "train")
RECIPES_DIR = os.path.join(REPO_ROOT, "recipes", "accelerate_configs")

DATASET_PATH = "/user/dengxianglong/datasets/ReasonLite-Dataset"
OUTPUT_ROOT = "/user/dengxianglong/outputs"


def _load_yaml(rel_path):
    with open(os.path.join(REPO_ROOT, rel_path)) as f:
        return yaml.safe_load(f)


class Stage1ConfigTests(unittest.TestCase):
    def test_dataset_path_filled(self):
        cfg = _load_yaml("train/config_stage1.yaml")
        self.assertEqual(cfg["dataset_name"], DATASET_PATH)

    def test_global_batch_is_256(self):
        # User decision: per_device_train_batch_size=32, grad_accum=1, 8 GPUs.
        cfg = _load_yaml("train/config_stage1.yaml")
        per_dev = cfg["per_device_train_batch_size"]
        ga = cfg["gradient_accumulation_steps"]
        # single-node 8-GPU: global batch = per_dev * ga * 8
        self.assertEqual(per_dev, 32)
        self.assertEqual(per_dev * ga * 8, 256)

    def test_uses_flash_attention_3(self):
        cfg = _load_yaml("train/config_stage1.yaml")
        self.assertEqual(cfg["attn_implementation"], "flash_attention_3")

    def test_output_dir_on_shared_storage(self):
        cfg = _load_yaml("train/config_stage1.yaml")
        self.assertTrue(cfg["output_dir"].startswith(OUTPUT_ROOT))


class Stage2ConfigTests(unittest.TestCase):
    def test_dataset_path_filled_and_matches_stage1(self):
        cfg = _load_yaml("train/config_stage2.yaml")
        self.assertEqual(cfg["dataset_name"], DATASET_PATH)

    def test_global_batch_is_256_strict(self):
        # User decision: per_device_train_batch_size=32 for both stages,
        # global batch held at 256 => grad_accum=1 on 8 GPUs.
        cfg = _load_yaml("train/config_stage2.yaml")
        per_dev = cfg["per_device_train_batch_size"]
        ga = cfg["gradient_accumulation_steps"]
        self.assertEqual(per_dev, 32)
        self.assertEqual(per_dev * ga * 8, 256)

    def test_uses_flash_attention_3(self):
        cfg = _load_yaml("train/config_stage2.yaml")
        self.assertEqual(cfg["attn_implementation"], "flash_attention_3")

    def test_output_dir_distinct_from_stage1(self):
        s1 = _load_yaml("train/config_stage1.yaml")["output_dir"]
        s2 = _load_yaml("train/config_stage2.yaml")["output_dir"]
        self.assertNotEqual(s1, s2)


class AccelerateConfigTests(unittest.TestCase):
    def test_zero1_yaml_exists_and_is_zero1(self):
        path = os.path.join(RECIPES_DIR, "zero1.yaml")
        self.assertTrue(os.path.isfile(path), f"missing {path}")
        with open(path) as f:
            cfg = yaml.safe_load(f)
        self.assertEqual(cfg["distributed_type"], "DEEPSPEED")
        self.assertEqual(cfg["deepspeed_config"]["zero_stage"], 1)
        self.assertEqual(cfg["mixed_precision"], "bf16")

    def test_zero1_single_node_eight_gpu(self):
        with open(os.path.join(RECIPES_DIR, "zero1.yaml")) as f:
            cfg = yaml.safe_load(f)
        self.assertEqual(cfg["num_processes"], 8)
        self.assertEqual(cfg["num_machines"], 1)


class LaunchScriptTests(unittest.TestCase):
    def test_stage_scripts_single_node_eight_gpu(self):
        for name in ("stage1.sh", "stage2.sh"):
            with open(os.path.join(TRAIN_DIR, name)) as f:
                content = f.read()
            self.assertIn("--num_processes 8", content)
            self.assertIn("--num_machines 1", content)
            self.assertIn("--machine_rank 0", content)
            self.assertIn("--main_process_ip 127.0.0.1", content)
            self.assertIn("recipes/accelerate_configs/zero1.yaml", content)
            # stage scripts source the SSOT file for coordinates/paths
            self.assertIn("setup_env.sh", content)
            # TrlParser uses --config (not --config_file) to load the YAML
            self.assertIn("--config ", content)

    def test_setup_env_script_sourced_by_launch(self):
        # setup_env.sh is the SSOT for cluster coordinates; launch_h100.sh
        # must source it rather than re-declaring the values.
        with open(os.path.join(TRAIN_DIR, "launch_h100.sh")) as f:
            launch = f.read()
        self.assertIn("source", launch)
        self.assertRegex(launch, r"setup_env\.sh")

    def test_setup_env_singlesource_coordinates(self):
        # The cctl coordinates live ONLY in setup_env.sh; launch_h100.sh
        # references them via variables, never as literal duplicates.
        with open(os.path.join(TRAIN_DIR, "setup_env.sh")) as f:
            setup = f.read()
        for key in ("CCTL_CLUSTER", "CCTL_RESOURCE_POOL", "CCTL_PROJECT",
                    "CCTL_BILLING", "CCTL_IMAGE", "DATASET_PATH",
                    "OUTPUT_ROOT", "REASONLITE_WORKSPACE_ROOT"):
            self.assertIn(key, setup, f"setup_env.sh missing {key}")
        # launch script must not hardcode the literal cluster name
        with open(os.path.join(TRAIN_DIR, "launch_h100.sh")) as f:
            launch = f.read()
        self.assertNotIn("paratera_train", launch,
                         "launch_h100.sh must source coordinates, not hardcode")

    def test_pip_install_in_entry(self):
        # User decision B: venv is built at runtime via pip install in the
        # pytorch-job entry script (no custom Docker image).
        with open(os.path.join(TRAIN_DIR, "launch_h100.sh")) as f:
            launch = f.read()
        self.assertIn("pip install", launch)
        self.assertIn("requirements_train.txt", launch)

    def test_smoke_and_full_modes(self):
        with open(os.path.join(TRAIN_DIR, "launch_h100.sh")) as f:
            launch = f.read()
        self.assertIn("smoke", launch)
        self.assertIn("full", launch)
        self.assertIn("--max_steps 3", launch)


if __name__ == "__main__":
    unittest.main()
