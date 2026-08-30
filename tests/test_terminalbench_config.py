from __future__ import annotations

import importlib
import io
import json
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import patch

import yaml

from scripts.materialize_terminalbench_split import materialize_terminalbench_split
from skillopt.config import flatten_config, load_config
from skillopt.envs.terminalbench.adapter import TerminalBenchAdapter


class TerminalBenchConfigTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.repository_root = Path(__file__).resolve().parents[1]
        cls.default_config = cls.repository_root / "configs/terminalbench/default.yaml"
        cls.smoke_config = cls.repository_root / "configs/terminalbench/smoke.yaml"
        cls.initial_skill = (
            cls.repository_root
            / "skillopt/envs/terminalbench/skills/initial.md"
        )

    def setUp(self) -> None:
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary_directory.name)
        task_source = self.root / "task_ids.json"
        task_source.write_text(
            json.dumps([f"task-{index}" for index in range(10)]),
            encoding="utf-8",
        )
        self.split_dir = self.root / "split"
        materialize_terminalbench_split(task_source, self.split_dir, seed=7)

        self.task_source = self.root / "terminal-bench-v2.1"
        self.task_source.mkdir()
        self.harbor_base_config = self.root / "harbor-base.yaml"
        self.harbor_base_config.write_text(
            yaml.safe_dump(
                {
                    "job_name": "validated-baseline",
                    "jobs_dir": str(self.root / "owner-jobs"),
                    "n_attempts": 1,
                    "n_concurrent_trials": 1,
                    "environment": {"type": "docker"},
                    "agents": [
                        {
                            "name": "terminus-2",
                            "model_name": "openai/DeepSeek-V4-Flash-0731",
                            "skills": [],
                        }
                    ],
                    "datasets": [{"path": str(self.task_source)}],
                    "tasks": [],
                },
                sort_keys=False,
            ),
            encoding="utf-8",
        )
        self.harbor_log = self.root / "harbor-invocations.jsonl"
        self.harbor_executable = self.root / "harbor"
        self.harbor_executable.write_text(
            "#!/usr/bin/env python3\n"
            "import json\n"
            "import sys\n"
            f"log_path = {str(self.harbor_log)!r}\n"
            "with open(log_path, 'a', encoding='utf-8') as handle:\n"
            "    handle.write(json.dumps(sys.argv[1:]) + '\\n')\n"
            "if sys.argv[1:] == ['--version']:\n"
            "    print('0.20.0')\n"
            "    raise SystemExit(0)\n"
            "if sys.argv[1:2] == ['run'] and '--print-config' in sys.argv[1:]:\n"
            "    raise SystemExit(0)\n"
            "raise SystemExit(97)\n",
            encoding="utf-8",
        )
        self.harbor_executable.chmod(0o755)

    def tearDown(self) -> None:
        self.temporary_directory.cleanup()

    def _overrides(self) -> list[str]:
        return [
            f"env.split_dir={self.split_dir}",
            f"env.harbor_base_config={self.harbor_base_config}",
            f"env.harbor_executable={self.harbor_executable}",
            f"env.out_root={self.root / 'output'}",
        ]

    def _load_registry_modules(self):
        loaded_before = set(sys.modules)
        openai_stub = types.ModuleType("openai")
        openai_stub.AzureOpenAI = type("AzureOpenAI", (), {})
        openai_stub.OpenAI = type("OpenAI", (), {})
        try:
            with patch.dict(sys.modules, {"openai": openai_stub}):
                train_module = importlib.import_module("scripts.train")
                eval_module = importlib.import_module("scripts.eval_only")
        finally:
            for module_name in set(sys.modules) - loaded_before:
                if module_name.startswith("skillopt.model") or module_name in {
                    "scripts.train",
                    "scripts.eval_only",
                }:
                    sys.modules.pop(module_name, None)
        return train_module, eval_module

    def test_checked_in_configs_load_with_string_inheritance(self) -> None:
        default_raw = yaml.safe_load(self.default_config.read_text(encoding="utf-8"))
        smoke_raw = yaml.safe_load(self.smoke_config.read_text(encoding="utf-8"))
        default = load_config(str(self.default_config))
        smoke = load_config(str(self.smoke_config))

        self.assertEqual(default_raw["_base_"], "../_base_/default.yaml")
        self.assertIsInstance(default_raw["_base_"], str)
        self.assertEqual(smoke_raw["_base_"], "default.yaml")
        self.assertIsInstance(smoke_raw["_base_"], str)
        self.assertEqual(default["env"]["name"], "terminalbench")
        self.assertEqual(smoke["env"]["name"], "terminalbench")
        self.assertEqual(smoke["model"]["optimizer"], "deepseek-v4-flash")

    def test_default_config_flattens_for_terminalbench_adapter(self) -> None:
        flat = flatten_config(load_config(str(self.default_config)))

        self.assertEqual(flat["env"], "terminalbench")
        self.assertEqual(flat["optimizer_backend"], "openai_compatible")
        self.assertEqual(flat["optimizer_model"], "deepseek-v4-flash")
        self.assertEqual(flat["reasoning_effort"], "max")
        self.assertNotEqual(flat["optimizer_model"], "DeepSeek-V4-Flash-0731")
        self.assertEqual(
            flat["split_dir"],
            "REPLACE_WITH_MATERIALIZED_TERMINALBENCH_SPLIT",
        )
        self.assertEqual(
            flat["harbor_base_config"],
            "REPLACE_WITH_VALIDATED_HARBOR_BASE_CONFIG.yaml",
        )

    def test_initial_skill_exists_and_is_semantically_blank(self) -> None:
        flat = flatten_config(load_config(str(self.default_config)))
        configured_path = self.repository_root / flat["skill_init"]

        self.assertEqual(configured_path, self.initial_skill)
        self.assertTrue(configured_path.is_file())
        self.assertEqual(configured_path.read_text(encoding="utf-8").strip(), "")

    def test_smoke_config_is_one_item_minimal_eval_workload(self) -> None:
        flat = flatten_config(load_config(str(self.smoke_config)))

        self.assertEqual(flat["limit"], 1)
        self.assertEqual(flat["n_concurrent_trials"], 1)
        self.assertEqual(flat["num_epochs"], 1)
        self.assertEqual(flat["batch_size"], 1)
        self.assertEqual(flat["accumulation"], 1)
        self.assertEqual(flat["minibatch_size"], 1)
        self.assertEqual(flat["merge_batch_size"], 2)
        self.assertEqual(flat["sel_env_num"], 1)
        self.assertEqual(flat["test_env_num"], 1)
        self.assertFalse(flat["use_slow_update"])
        self.assertFalse(flat["use_meta_skill"])
        self.assertFalse(flat["eval_test"])

    def test_smoke_config_builds_one_item_eval_without_rollout(self) -> None:
        train_module, _ = self._load_registry_modules()
        flat = flatten_config(
            load_config(str(self.smoke_config), overrides=self._overrides())
        )
        adapter = train_module.get_adapter(flat)
        adapter.setup(flat)

        items = adapter.build_eval_env(0, "valid_seen", flat["seed"])

        self.assertEqual(len(items), 1)
        self.assertEqual(items, adapter.dataloader.val_items)

    def test_repository_configs_have_no_machine_paths_or_credentials(self) -> None:
        for config_path in (self.default_config, self.smoke_config):
            with self.subTest(config=config_path.name):
                raw = yaml.safe_load(config_path.read_text(encoding="utf-8"))
                text = config_path.read_text(encoding="utf-8")
                self.assertNotIn("/home/", text)
                self.assertNotIn("/mnt/", text)
                self.assertNotIn("${", text)
                self._assert_no_credentials(raw)

        flat = flatten_config(load_config(str(self.default_config)))
        self.assertFalse(Path(flat["split_dir"]).is_absolute())
        self.assertFalse(Path(flat["harbor_base_config"]).is_absolute())

    def _assert_no_credentials(self, value, path: str = "config") -> None:
        if isinstance(value, dict):
            for key, child in value.items():
                leaf = str(key).casefold()
                if any(token in leaf for token in ("api_key", "token", "secret", "password")):
                    self.assertIn(child, (None, ""), f"credential at {path}.{key}")
                self._assert_no_credentials(child, f"{path}.{key}")
        elif isinstance(value, list):
            for index, child in enumerate(value):
                self._assert_no_credentials(child, f"{path}[{index}]")

    def test_supported_overrides_replace_external_paths(self) -> None:
        flat = flatten_config(
            load_config(str(self.default_config), overrides=self._overrides())
        )

        self.assertEqual(flat["split_dir"], str(self.split_dir))
        self.assertEqual(flat["harbor_base_config"], str(self.harbor_base_config))
        self.assertEqual(flat["harbor_executable"], str(self.harbor_executable))
        self.assertEqual(flat["out_root"], str(self.root / "output"))

    def test_train_cli_loader_applies_cfg_options(self) -> None:
        train_module, _ = self._load_registry_modules()
        argv = [
            "train.py",
            "--config",
            str(self.default_config),
            "--cfg-options",
            *self._overrides(),
        ]

        with patch.object(sys, "argv", argv):
            cfg = train_module.load_config(train_module.parse_args())

        self.assertEqual(cfg["env"], "terminalbench")
        self.assertEqual(cfg["split_dir"], str(self.split_dir))
        self.assertEqual(cfg["harbor_base_config"], str(self.harbor_base_config))
        self.assertEqual(cfg["optimizer_backend"], "openai_compatible")
        self.assertEqual(cfg["optimizer_model"], "deepseek-v4-flash")
        self.assertEqual(cfg["reasoning_effort"], "max")

    def test_train_and_eval_registries_construct_and_setup_adapter(self) -> None:
        train_module, eval_module = self._load_registry_modules()
        flat = flatten_config(
            load_config(str(self.default_config), overrides=self._overrides())
        )

        train_adapter = train_module.get_adapter(flat)
        eval_adapter = eval_module.get_adapter(flat)
        train_adapter.setup(flat)
        eval_adapter.setup(flat)

        self.assertIsInstance(train_adapter, TerminalBenchAdapter)
        self.assertIsInstance(eval_adapter, TerminalBenchAdapter)
        self.assertEqual(len(train_adapter.dataloader.train_items), 1)
        self.assertEqual(len(eval_adapter.dataloader.test_items), 8)

        invocations = [
            json.loads(line)
            for line in self.harbor_log.read_text(encoding="utf-8").splitlines()
        ]
        self.assertTrue(invocations)
        for invocation in invocations:
            self.assertTrue(
                invocation == ["--version"]
                or (
                    invocation[:1] == ["run"]
                    and "--print-config" in invocation
                )
            )

    def test_eval_cli_still_requires_skill(self) -> None:
        _, eval_module = self._load_registry_modules()
        argv = ["eval_only.py", "--config", str(self.default_config)]

        with patch.object(sys, "argv", argv), patch("sys.stderr", io.StringIO()):
            with self.assertRaises(SystemExit) as raised:
                eval_module.parse_args()

        self.assertEqual(raised.exception.code, 2)

    def test_eval_cli_accepts_documented_skill_and_overrides(self) -> None:
        _, eval_module = self._load_registry_modules()
        argv = [
            "eval_only.py",
            "--config",
            str(self.smoke_config),
            "--skill",
            str(self.initial_skill),
            "--split",
            "valid_seen",
            "--cfg-options",
            *self._overrides(),
        ]

        with patch.object(sys, "argv", argv):
            args = eval_module.parse_args()
        flat = flatten_config(load_config(args.config, overrides=args.cfg_options))

        self.assertEqual(args.skill, str(self.initial_skill))
        self.assertEqual(args.split, "valid_seen")
        self.assertEqual(flat["split_dir"], str(self.split_dir))
        self.assertEqual(flat["harbor_base_config"], str(self.harbor_base_config))


if __name__ == "__main__":
    unittest.main()
