from __future__ import annotations

import importlib
import json
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import patch

from scripts.materialize_terminalbench_split import materialize_terminalbench_split
from skillopt.config import flatten_config
from skillopt.datasets.base import BatchSpec
from skillopt.envs.base import EnvAdapter
from skillopt.envs.terminalbench.adapter import TerminalBenchAdapter
from skillopt.envs.terminalbench.dataloader import TerminalBenchDataLoader


class TerminalBenchAdapterTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary_directory.name)
        source_path = self.root / "task_ids.json"
        source_path.write_text(
            json.dumps([f"task-{index}" for index in range(10)]),
            encoding="utf-8",
        )
        self.split_dir = self.root / "split"
        materialize_terminalbench_split(source_path, self.split_dir, seed=7)
        self.base_config = self.root / "harbor.yaml"
        self.base_config.write_text("job_name: fixture\n", encoding="utf-8")

    def tearDown(self) -> None:
        self.temporary_directory.cleanup()

    def _adapter(self, **kwargs) -> TerminalBenchAdapter:
        return TerminalBenchAdapter(
            split_dir=str(self.split_dir),
            harbor_base_config=str(self.base_config),
            **kwargs,
        )

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

    def test_adapter_instantiates_with_explicit_static_config(self) -> None:
        adapter = self._adapter(n_concurrent_trials=3, analyst_workers=2)

        self.assertIsInstance(adapter, TerminalBenchAdapter)
        self.assertEqual(adapter.n_concurrent_trials, 3)
        self.assertEqual(adapter.analyst_workers, 2)

    def test_setup_initializes_loader_and_runner(self) -> None:
        adapter = self._adapter(harbor_executable="fixture-harbor")
        runner = object()

        with patch(
            "skillopt.envs.terminalbench.adapter.HarborRunner",
            return_value=runner,
        ) as runner_class:
            adapter.setup({"env": "terminalbench"})

        self.assertEqual(len(adapter.dataloader.train_items), 1)
        self.assertEqual(len(adapter.dataloader.val_items), 1)
        self.assertEqual(len(adapter.dataloader.test_items), 8)
        self.assertIs(adapter.runner, runner)
        runner_class.assert_called_once_with(
            adapter.harbor_base_config,
            "fixture-harbor",
        )

    def test_get_dataloader_returns_terminalbench_loader(self) -> None:
        adapter = self._adapter()

        self.assertIs(adapter.get_dataloader(), adapter.dataloader)
        self.assertIsInstance(adapter.get_dataloader(), TerminalBenchDataLoader)

    def test_build_env_from_batch_preserves_order_and_duplicates(self) -> None:
        adapter = self._adapter()
        payload = [{"id": "task-b"}, {"id": "task-a"}, {"id": "task-b"}]
        batch = BatchSpec(
            phase="train",
            split="train",
            seed=42,
            batch_size=3,
            payload=payload,
        )

        items = adapter.build_env_from_batch(batch)

        self.assertEqual(items, payload)
        self.assertIsNot(items, payload)

    def test_build_train_and_eval_env_use_loaded_split_batches(self) -> None:
        adapter = self._adapter()
        with patch("skillopt.envs.terminalbench.adapter.HarborRunner"):
            adapter.setup({"env": "terminalbench"})

        train_items = adapter.build_train_env(batch_size=1, seed=11)
        val_items = adapter.build_eval_env(
            env_num=1,
            split="valid_seen",
            seed=11,
        )
        test_items = adapter.build_eval_env(
            env_num=1,
            split="valid_unseen",
            seed=11,
        )

        self.assertEqual(train_items, adapter.dataloader.train_items)
        self.assertEqual(val_items, adapter.dataloader.val_items)
        self.assertEqual(test_items, adapter.dataloader.test_items[:1])

    def test_rollout_is_thin_and_returns_phase6_result_unchanged(self) -> None:
        adapter = self._adapter(n_concurrent_trials=4)
        runner = object()
        adapter.runner = runner
        items = [{"id": "task-a"}]
        results = [{"id": "task-a", "hard": 1.0, "soft": 1.0}]
        rollout_dir = self.root / "runs" / "step-1" / "rollout"
        skill_content = "  keep this exact\n\n"

        with patch(
            "skillopt.envs.terminalbench.adapter.run_terminalbench_rollout",
            return_value=results,
        ) as rollout:
            returned = adapter.rollout(items, skill_content, str(rollout_dir))

        self.assertIs(returned, results)
        rollout.assert_called_once()
        call = rollout.call_args
        self.assertIs(call.args[0], items)
        self.assertEqual(call.kwargs["skill_content"], skill_content)
        self.assertEqual(call.kwargs["rollout_dir"], str(rollout_dir))
        self.assertIs(call.kwargs["runner"], runner)
        self.assertEqual(call.kwargs["n_concurrent_trials"], 4)

    def test_blank_initial_skill_uses_same_rollout_path_unchanged(self) -> None:
        adapter = self._adapter()
        adapter.runner = object()
        initial_path = (
            Path(__file__).parents[1]
            / "skillopt"
            / "envs"
            / "terminalbench"
            / "skills"
            / "initial.md"
        )
        skill_content = initial_path.read_text(encoding="utf-8")

        with patch(
            "skillopt.envs.terminalbench.adapter.run_terminalbench_rollout",
            return_value=[],
        ) as rollout:
            adapter.rollout(
                [{"id": "task-a"}],
                skill_content,
                str(self.root / "baseline"),
            )

        self.assertEqual(skill_content.strip(), "")
        self.assertEqual(rollout.call_args.kwargs["skill_content"], skill_content)

    def test_result_name_uses_rollout_directory_and_exact_task_ids(self) -> None:
        adapter = self._adapter()
        adapter.runner = object()
        result_names: list[str] = []

        def capture(*args, **kwargs):
            result_names.append(kwargs["result_name"])
            return []

        first = self.root / "train" / "rollout"
        second = self.root / "selection_eval"
        with patch(
            "skillopt.envs.terminalbench.adapter.run_terminalbench_rollout",
            side_effect=capture,
        ):
            adapter.rollout([{"id": "task-a"}], "skill", str(first))
            adapter.rollout([{"id": "task-a"}], "skill", str(first))
            adapter.rollout([{"id": "task-b"}], "skill", str(first))
            adapter.rollout([{"id": "task-a"}], "skill", str(second))

        self.assertEqual(result_names[0], result_names[1])
        self.assertNotEqual(result_names[0], result_names[2])
        self.assertNotEqual(result_names[0], result_names[3])
        self.assertRegex(result_names[0], r"^terminalbench-[0-9a-f]{16}$")

    def test_lossy_trainer_topup_directory_names_do_not_collide(self) -> None:
        adapter = self._adapter()
        adapter.runner = object()
        result_names: list[str] = []
        shared_rollout_dir = self.root / "rollout_prev" / "topup" / "task_a"

        def capture(*args, **kwargs):
            result_names.append(kwargs["result_name"])
            return []

        with patch(
            "skillopt.envs.terminalbench.adapter.run_terminalbench_rollout",
            side_effect=capture,
        ):
            adapter.rollout([{"id": "task+a"}], "skill", str(shared_rollout_dir))
            adapter.rollout([{"id": "task_a"}], "skill", str(shared_rollout_dir))

        self.assertNotEqual(result_names[0], result_names[1])

    def test_train_and_eval_registries_lazy_load_adapter(self) -> None:
        train_module, eval_module = self._load_registry_modules()
        cfg = {
            "env": "terminalbench",
            "split_dir": str(self.split_dir),
            "harbor_base_config": str(self.base_config),
        }

        train_adapter = train_module.get_adapter(cfg)
        eval_adapter = eval_module.get_adapter(cfg)

        self.assertIsInstance(train_adapter, TerminalBenchAdapter)
        self.assertIsInstance(eval_adapter, TerminalBenchAdapter)

    def test_structured_config_flattens_into_explicit_constructor_parameters(self) -> None:
        train_module, _ = self._load_registry_modules()
        flat = flatten_config(
            {
                "env": {
                    "name": "terminalbench",
                    "split_dir": str(self.split_dir),
                    "harbor_base_config": str(self.base_config),
                    "harbor_executable": "fixture-harbor",
                    "n_concurrent_trials": 5,
                    "analyst_workers": 3,
                    "limit": 1,
                }
            }
        )

        adapter = train_module.get_adapter(flat)

        self.assertEqual(flat["env"], "terminalbench")
        self.assertEqual(adapter.harbor_executable, "fixture-harbor")
        self.assertEqual(adapter.n_concurrent_trials, 5)
        self.assertEqual(adapter.analyst_workers, 3)
        self.assertEqual(adapter.dataloader.limit, 1)

    def test_adapter_uses_inherited_reflection_without_model_calls(self) -> None:
        self.assertIs(TerminalBenchAdapter.reflect, EnvAdapter.reflect)

    def test_task_types_use_trainers_existing_uncategorized_bucket(self) -> None:
        self.assertEqual(self._adapter().get_task_types(), ["other"])

    def test_invalid_static_config_fails_clearly(self) -> None:
        with self.assertRaisesRegex(FileNotFoundError, "split_dir"):
            TerminalBenchAdapter(
                split_dir=str(self.root / "missing-split"),
                harbor_base_config=str(self.base_config),
            )
        with self.assertRaisesRegex(FileNotFoundError, "harbor_base_config"):
            TerminalBenchAdapter(
                split_dir=str(self.split_dir),
                harbor_base_config=str(self.root / "missing-config.yaml"),
            )
        with self.assertRaisesRegex(ValueError, "positive integer"):
            self._adapter(n_concurrent_trials=0)


if __name__ == "__main__":
    unittest.main()
