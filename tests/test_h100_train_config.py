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


def _parse_shell_exports(rel_path):
    """Parse `export KEY="..."` lines from a sourced shell file.

    Handles two forms:
      export KEY="${KEY:-value}"   -> value
      export KEY="${OTHER}/path"   -> ${OTHER}/path (raw, kept for presence check)
    """
    env = {}
    with open(os.path.join(REPO_ROOT, rel_path)) as f:
        for line in f:
            m = re.match(r'\s*export\s+([A-Z_]+)="\$\{[A-Z_]+:-([^}]*)\}"', line)
            if m:
                env[m.group(1)] = m.group(2)
                continue
            m = re.match(r'\s*export\s+([A-Z_]+)="(.+)"', line)
            if m:
                env[m.group(1)] = m.group(2)
    return env


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

    def test_train_split_is_medium(self):
        # open-r1 sft.py reads `dataset_train_split` (TRL ScriptArguments,
        # default "train"), NOT `dataset_split`. The ReasonLite dataset has
        # splits high/medium/cot; stage1 trains on `medium`.
        cfg = _load_yaml("train/config_stage1.yaml")
        self.assertEqual(cfg["dataset_train_split"], "medium")
        self.assertNotIn("dataset_split", cfg)

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

    def test_train_split_is_high(self):
        # open-r1 sft.py reads `dataset_train_split` (TRL ScriptArguments,
        # default "train"), NOT `dataset_split`. stage2 trains on `high`.
        cfg = _load_yaml("train/config_stage2.yaml")
        self.assertEqual(cfg["dataset_train_split"], "high")
        self.assertNotIn("dataset_split", cfg)

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
    def test_stage_scripts_single_node_config(self):
        for name in ("stage1.sh", "stage2.sh"):
            with open(os.path.join(TRAIN_DIR, name)) as f:
                content = f.read()
            # num_processes is parameterized via NPROC (default 8), not hardcoded
            self.assertIn("--num_processes \"${NPROC}\"", content)
            self.assertIn("NPROC=\"${NPROC:-8}\"", content)
            self.assertIn("--num_machines 1", content)
            self.assertIn("--machine_rank 0", content)
            self.assertIn("--main_process_ip 127.0.0.1", content)
            self.assertIn("recipes/accelerate_configs/zero1.yaml", content)
            # stage scripts source the SSOT file for coordinates/paths
            self.assertIn("setup_env.sh", content)
            # TrlParser uses --config (not --config_file) to load the YAML
            self.assertIn("--config ", content)
            # Entry is the ReasonLite adapter (wraps open-r1 get_dataset to
            # normalize parquet -> messages), NOT open-r1's raw sft.py.
            self.assertIn("sft_reasonlite.py", content)
            self.assertNotIn("src/open_r1/sft.py", content)

    def test_launch_sets_nproc_per_mode(self):
        # smoke defaults NPROC=1 (1-GPU flow validation); full defaults NPROC=8.
        with open(os.path.join(TRAIN_DIR, "launch_h100.sh")) as f:
            launch = f.read()
        self.assertIn('NPROC="${NPROC:-1}"', launch)
        self.assertIn('NPROC="${NPROC:-8}"', launch)


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
        setup_env = _parse_shell_exports("train/setup_env.sh")
        for key in ("CCTL_CLUSTER", "CCTL_RESOURCE_POOL", "CCTL_PROJECT",
                    "CCTL_BILLING", "CCTL_IMAGE", "DATASET_PATH",
                    "OUTPUT_ROOT", "REASONLITE_WORKSPACE_ROOT",
                    "REASONLITE_GIT_REPO", "REASONLITE_GIT_REF",
                    "HF_ENDPOINT", "HF_DATASETS_CACHE", "TMPDIR"):
            self.assertIn(key, setup_env, f"setup_env.sh missing {key}")
        # HF endpoint must be the domestic mirror (nodes cannot reach huggingface.co)
        self.assertEqual(setup_env["HF_ENDPOINT"], "https://hf-mirror.com")
        # The ReasonLite repo URL must be the Codeup intranet fork (reachable
        # from training nodes), and the git ref must be this worktree's branch.
        self.assertIn("codeup.aliyun.com", setup_env["REASONLITE_GIT_REPO"])
        self.assertEqual(setup_env["REASONLITE_GIT_REF"], "worktree-train-on-h100")
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

    def test_launch_calls_openr1_patch(self):
        # launch_h100.sh must invoke patch_openr1.sh after editable install.
        with open(os.path.join(TRAIN_DIR, "launch_h100.sh")) as f:
            launch = f.read()
        self.assertIn("patch_openr1.sh", launch)


class PatchOpenR1Tests(unittest.TestCase):
    """patch_openr1.sh inserts a ParallelismConfig import after `import trl`,
    idempotently. Tested against a synthetic configs.py copy."""

    def _make_fake_configs(self, open_r1_dir):
        cfg = os.path.join(open_r1_dir, "configs.py")
        with open(cfg, "w") as f:
            f.write("from __future__ import annotations\n"
                    "import trl\n"
                    "\n"
                    "@dataclass\n"
                    "class SFTConfig(trl.SFTConfig):\n"
                    "    pass\n")
        return cfg

    def test_patch_inserts_import(self):
        import tempfile
        with tempfile.TemporaryDirectory() as td:
            open_r1_src = os.path.join(td, "src", "open_r1")
            os.makedirs(open_r1_src)
            cfg = self._make_fake_configs(open_r1_src)
            env = dict(os.environ, OPENR1_ROOT=td)
            rc = subprocess.run(
                ["bash", os.path.join(TRAIN_DIR, "patch_openr1.sh")],
                env=env, capture_output=True, text=True)
            self.assertEqual(rc.returncode, 0, rc.stderr)
            with open(cfg) as f:
                content = f.read()
            self.assertIn("from accelerate.parallelism_config import ParallelismConfig",
                          content)
            # import sits right after `import trl`
            self.assertIn("import trl\n"
                          "from accelerate.parallelism_config import ParallelismConfig",
                          content)

    def test_patch_is_idempotent(self):
        import tempfile
        with tempfile.TemporaryDirectory() as td:
            open_r1_src = os.path.join(td, "src", "open_r1")
            os.makedirs(open_r1_src)
            cfg = self._make_fake_configs(open_r1_src)
            env = dict(os.environ, OPENR1_ROOT=td)
            first = subprocess.run(["bash", os.path.join(TRAIN_DIR, "patch_openr1.sh")],
                                   env=env, capture_output=True, text=True)
            self.assertEqual(first.returncode, 0, first.stderr)
            with open(cfg) as f:
                once = f.read()
            second = subprocess.run(["bash", os.path.join(TRAIN_DIR, "patch_openr1.sh")],
                                    env=env, capture_output=True, text=True)
            self.assertEqual(second.returncode, 0, second.stderr)
            with open(cfg) as f:
                twice = f.read()
            self.assertEqual(once, twice)


class SftReasonliteAdapterTests(unittest.TestCase):
    """train/sft_reasonlite.py is the training entry: it wraps open-r1's
    get_dataset to normalize the ReasonLite parquet ({prompt, answer, ...})
    into TRL's conversational `messages` format. The transform is a pure
    top-level function so it imports and tests without open-r1/torch."""

    def _load_transform(self):
        import importlib.util
        path = os.path.join(TRAIN_DIR, "sft_reasonlite.py")
        spec = importlib.util.spec_from_file_location("sft_reasonlite", path)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        return mod

    def test_to_messages_maps_prompt_and_answer(self):
        mod = self._load_transform()
        row = {"prompt": "2+2?", "answer": "<think>add</think>\n4",
               "expected_answer": "4", "vote": {}, "problem_source": "x"}
        out = mod.to_messages(row)
        self.assertEqual(out, {"messages": [
            {"role": "user", "content": "2+2?"},
            {"role": "assistant", "content": "<think>add</think>\n4"},
        ]})

    def test_to_messages_requires_prompt_and_answer(self):
        # Fail fast: a row missing prompt/answer must raise, not silently skip.
        mod = self._load_transform()
        with self.assertRaises(KeyError):
            mod.to_messages({"answer": "4"})
        with self.assertRaises(KeyError):
            mod.to_messages({"prompt": "2+2?"})

    def test_entry_wraps_openr1_get_dataset(self):
        # The adapter reuses open-r1's main/get_dataset (no training-loop copy)
        # and drops the metadata columns via remove_columns.
        with open(os.path.join(TRAIN_DIR, "sft_reasonlite.py")) as f:
            src = f.read()
        self.assertIn("get_dataset", src)
        self.assertIn("from open_r1.sft import main", src)
        self.assertIn("remove_columns", src)


if __name__ == "__main__":
    unittest.main()



