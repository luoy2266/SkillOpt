from __future__ import annotations

import copy
import json
import os
import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import yaml

from skillopt.envs.terminalbench.harbor_runner import (
    EXPECTED_HARBOR_VERSION,
    HarborArtifactConflictError,
    HarborConfigError,
    HarborExecutionDisabledError,
    HarborParityError,
    HarborRunner,
    HarborVersionError,
    assert_harbor_config_parity,
    build_harbor_config,
    load_harbor_base_config,
)
from skillopt.envs.terminalbench.skill_pack import package_skill_content


class TerminalBenchHarborConfigTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary_directory.name)
        self.task_source = self.root / "terminal-bench-v2.1"
        self.task_source.mkdir()

    def tearDown(self) -> None:
        self.temporary_directory.cleanup()

    def _base_config(self) -> dict:
        return {
            "job_name": "validated-baseline",
            "jobs_dir": str(self.root / "owner-jobs"),
            "n_attempts": 2,
            "timeout_multiplier": 1.25,
            "n_concurrent_trials": 3,
            "retry": {"max_retries": 1},
            "environment": {
                "type": "docker",
                "override_cpus": 4,
                "override_memory_mb": 8192,
                "override_storage_mb": 20480,
            },
            "verifier": {"override_timeout_sec": 90},
            "agents": [
                {
                    "name": "terminus-2",
                    "model_name": "openai/DeepSeek-V4-Flash-0731",
                    "skills": [],
                    "override_timeout_sec": 600,
                    "kwargs": {"max_turns": 40, "reasoning": "enabled"},
                }
            ],
            "datasets": [
                {
                    "path": str(self.task_source),
                    "task_names": ["old-selection"],
                    "exclude_task_names": ["old-exclusion"],
                    "n_tasks": 1,
                }
            ],
            "tasks": [],
        }

    def _write_base_config(self, config: dict | None = None) -> Path:
        path = self.root / "base.yaml"
        path.write_text(
            yaml.safe_dump(config or self._base_config(), sort_keys=False),
            encoding="utf-8",
        )
        return path

    def _write_fake_harbor(
        self,
        version: str = EXPECTED_HARBOR_VERSION,
        *,
        accept_config: bool = True,
    ) -> tuple[Path, Path]:
        executable = self.root / f"fake-harbor-{version.replace('.', '-')}"
        log_path = self.root / "fake-harbor-invocations.jsonl"
        executable.write_text(
            "#!/usr/bin/env python3\n"
            "import json\n"
            "import sys\n"
            f"log_path = {str(log_path)!r}\n"
            "with open(log_path, 'a', encoding='utf-8') as handle:\n"
            "    handle.write(json.dumps(sys.argv[1:]) + '\\n')\n"
            "if sys.argv[1:] == ['--version']:\n"
            f"    print({version!r})\n"
            "    raise SystemExit(0)\n"
            "if sys.argv[1:2] == ['run'] and '--print-config' in sys.argv[1:]:\n"
            f"    raise SystemExit({0 if accept_config else 2})\n"
            "raise SystemExit(97)\n",
            encoding="utf-8",
        )
        executable.chmod(0o755)
        return executable, log_path

    def _skill_dir(self, content: str = "# Reusable guidance\n") -> Path:
        packaged = package_skill_content(content, self.root / "skill-artifacts")
        self.assertIsNotNone(packaged.skill_dir)
        return packaged.skill_dir

    def _build(self, *, skills=(), result_name="comparison") -> dict:
        return build_harbor_config(
            base_config=self._base_config(),
            task_ids=["task-b", "task-a"],
            harbor_skills=skills,
            result_name=result_name,
            jobs_dir=self.root / "resolved-jobs",
        )

    def test_loads_valid_yaml_base_config(self) -> None:
        loaded = load_harbor_base_config(self._write_base_config())

        self.assertEqual(loaded["agents"][0]["name"], "terminus-2")
        self.assertEqual(loaded["environment"]["type"], "docker")
        self.assertEqual(loaded["datasets"][0]["path"], str(self.task_source))

    def test_baseline_and_skill_overlay_use_native_directory_paths(self) -> None:
        baseline = self._build(skills=(), result_name="baseline")
        skill_dir = self._skill_dir()
        candidate = self._build(skills=[skill_dir], result_name="skillopt")

        self.assertEqual(baseline["agents"][0]["skills"], [])
        self.assertEqual(candidate["agents"][0]["skills"], [str(skill_dir)])
        self.assertTrue(Path(candidate["agents"][0]["skills"][0]).is_dir())
        self.assertNotEqual(
            candidate["agents"][0]["skills"],
            [str(skill_dir / "SKILL.md")],
        )

    def test_task_ids_map_to_exact_dataset_filter(self) -> None:
        resolved = self._build()

        self.assertEqual(resolved["datasets"][0]["task_names"], ["task-a", "task-b"])
        self.assertIsNone(resolved["datasets"][0]["exclude_task_names"])
        self.assertIsNone(resolved["datasets"][0]["n_tasks"])
        self.assertEqual(resolved["tasks"], [])

    def test_duplicate_task_ids_fail_without_silent_deduplication(self) -> None:
        with self.assertRaisesRegex(HarborConfigError, "duplicate"):
            build_harbor_config(
                base_config=self._base_config(),
                task_ids=["task-a", "task-a"],
                harbor_skills=[],
                result_name="baseline",
                jobs_dir=self.root / "jobs",
            )

    def test_exact_task_ids_reject_all_fnmatch_metacharacters(self) -> None:
        for metacharacter in "*?[]":
            with self.subTest(metacharacter=metacharacter):
                with self.assertRaisesRegex(HarborConfigError, "fnmatch metacharacters"):
                    build_harbor_config(
                        base_config=self._base_config(),
                        task_ids=[f"task-{metacharacter}-id"],
                        harbor_skills=[],
                        result_name="baseline",
                        jobs_dir=self.root / "jobs",
                    )

    def test_parity_allows_only_skills_name_and_output(self) -> None:
        baseline = self._build(result_name="baseline")
        candidate = build_harbor_config(
            base_config=self._base_config(),
            task_ids=["task-a", "task-b"],
            harbor_skills=[self._skill_dir()],
            result_name="skillopt",
            jobs_dir=self.root / "different-output-root",
        )

        assert_harbor_config_parity(baseline, candidate)
        self.assertEqual(
            set(baseline["datasets"][0]["task_names"]),
            set(candidate["datasets"][0]["task_names"]),
        )

    def test_parity_rejects_model_difference(self) -> None:
        baseline = self._build(result_name="baseline")
        candidate = self._build(skills=[self._skill_dir()], result_name="skillopt")
        candidate["agents"][0]["model_name"] = "different-model"

        with self.assertRaisesRegex(HarborParityError, "outside the parity allowlist"):
            assert_harbor_config_parity(baseline, candidate)

    def test_parity_rejects_timeout_resource_and_environment_differences(self) -> None:
        mutations = (
            ("timeout", lambda config: config.__setitem__("timeout_multiplier", 9.0)),
            (
                "resource",
                lambda config: config["environment"].__setitem__("override_cpus", 99),
            ),
            (
                "environment",
                lambda config: config["environment"].__setitem__("force_build", True),
            ),
            ("attempts", lambda config: config.__setitem__("n_attempts", 9)),
            (
                "concurrency",
                lambda config: config.__setitem__("n_concurrent_trials", 9),
            ),
        )
        for label, mutate in mutations:
            with self.subTest(label=label):
                baseline = self._build(result_name=f"baseline-{label}")
                candidate = self._build(result_name=f"skillopt-{label}")
                mutate(candidate)
                with self.assertRaises(HarborParityError):
                    assert_harbor_config_parity(baseline, candidate)

    def test_builds_do_not_mutate_or_leak_skills(self) -> None:
        base = self._base_config()
        original = copy.deepcopy(base)
        skill_dir = self._skill_dir()

        candidate = build_harbor_config(
            base_config=base,
            task_ids=["task-a"],
            harbor_skills=[skill_dir],
            result_name="skillopt",
            jobs_dir=self.root / "resolved-jobs",
        )
        baseline = build_harbor_config(
            base_config=base,
            task_ids=["task-a"],
            harbor_skills=[],
            result_name="baseline",
            jobs_dir=self.root / "resolved-jobs",
        )

        self.assertEqual(base, original)
        self.assertEqual(candidate["agents"][0]["skills"], [str(skill_dir)])
        self.assertEqual(baseline["agents"][0]["skills"], [])

    def test_invalid_version_and_missing_base_config_fail(self) -> None:
        wrong_harbor, _ = self._write_fake_harbor("0.21.0")
        with self.assertRaises(HarborVersionError):
            HarborRunner(self._write_base_config(), wrong_harbor)

        valid_harbor, _ = self._write_fake_harbor()
        with self.assertRaises(FileNotFoundError):
            HarborRunner(self.root / "missing.yaml", valid_harbor)

    def test_invalid_base_contract_fails_loudly(self) -> None:
        invalid = self._base_config()
        invalid["agents"][0]["skills"] = ["preexisting-skill"]

        with self.assertRaisesRegex(HarborConfigError, "baseline"):
            load_harbor_base_config(self._write_base_config(invalid))

    def test_harbor_schema_rejection_fails_loudly(self) -> None:
        rejecting_harbor, _ = self._write_fake_harbor(accept_config=False)

        with self.assertRaisesRegex(HarborConfigError, "rejected config"):
            HarborRunner(self._write_base_config(), rejecting_harbor)

    def test_dry_run_serializes_config_without_starting_job(self) -> None:
        fake_harbor, log_path = self._write_fake_harbor()
        runner = HarborRunner(self._write_base_config(), fake_harbor)

        prepared = runner.prepare(
            task_ids=["task-b", "task-a"],
            harbor_skills=[self._skill_dir()],
            result_name="skillopt-dry-run",
            output_root=self.root / "output",
        )
        manifest = runner.dry_run(prepared)

        self.assertTrue(prepared.resolved_config_path.is_file())
        self.assertTrue(prepared.dry_run_path.is_file())
        self.assertFalse(prepared.expected_job_dir.exists())
        self.assertFalse(manifest["execution_started"])
        self.assertEqual(manifest["harbor_version"], EXPECTED_HARBOR_VERSION)
        self.assertEqual(manifest["task_ids"], ["task-a", "task-b"])
        self.assertEqual(manifest["skills"], list(prepared.harbor_skills))

        invocations = [
            json.loads(line)
            for line in log_path.read_text(encoding="utf-8").splitlines()
        ]
        self.assertIn(["--version"], invocations)
        self.assertTrue(
            all(
                invocation == ["--version"] or "--print-config" in invocation
                for invocation in invocations
            )
        )
        with self.assertRaises(HarborExecutionDisabledError):
            runner.run(prepared)

    def test_environment_secret_reference_is_preserved_without_plaintext_artifact(self) -> None:
        test_secret = "fixture-secret-that-must-not-be-written"
        base = self._base_config()
        base["agents"][0]["env"] = {"TEST_API_KEY": "${TEST_API_KEY}"}
        base["agents"][0]["kwargs"]["headers"] = {
            "Authorization": "Bearer ${TEST_API_KEY}"
        }
        fake_harbor, _ = self._write_fake_harbor()

        with patch.dict(os.environ, {"TEST_API_KEY": test_secret}):
            runner = HarborRunner(self._write_base_config(base), fake_harbor)
            prepared = runner.prepare(
                task_ids=["task-a"],
                harbor_skills=[],
                result_name="secret-reference",
                output_root=self.root / "output",
            )
            runner.dry_run(prepared)

        resolved_text = prepared.resolved_config_path.read_text(encoding="utf-8")
        dry_run_text = prepared.dry_run_path.read_text(encoding="utf-8")
        self.assertIn("${TEST_API_KEY}", resolved_text)
        self.assertNotIn(test_secret, resolved_text)
        self.assertNotIn(test_secret, dry_run_text)

    def test_plaintext_secret_in_base_config_fails_before_artifact_write(self) -> None:
        fixtures = []

        env_secret = self._base_config()
        env_secret["agents"][0]["env"] = {
            "TEST_API_KEY": "plaintext-fixture-secret"
        }
        fixtures.append(env_secret)

        header_secret = self._base_config()
        header_secret["agents"][0]["kwargs"]["headers"] = {
            "Authorization": "Bearer plaintext-fixture-token"
        }
        fixtures.append(header_secret)

        legacy_env_secret = self._base_config()
        legacy_env_secret["environment"]["env"] = [
            "TEST_API_KEY=plaintext-fixture-secret"
        ]
        fixtures.append(legacy_env_secret)

        for index, base in enumerate(fixtures):
            with self.subTest(index=index):
                with self.assertRaisesRegex(HarborConfigError, "plaintext value"):
                    load_harbor_base_config(self._write_base_config(base))

    def test_deterministic_artifact_conflict_is_not_overwritten(self) -> None:
        fake_harbor, _ = self._write_fake_harbor()
        runner = HarborRunner(self._write_base_config(), fake_harbor)
        prepared = runner.prepare(
            task_ids=["task-a"],
            harbor_skills=[],
            result_name="same-name",
            output_root=self.root / "output",
        )
        prepared.resolved_config_path.write_text("tampered", encoding="utf-8")

        with self.assertRaises(HarborArtifactConflictError):
            runner.prepare(
                task_ids=["task-a"],
                harbor_skills=[],
                result_name="same-name",
                output_root=self.root / "output",
            )

    def test_path_traversal_fails(self) -> None:
        with self.assertRaisesRegex(HarborConfigError, "traversal"):
            build_harbor_config(
                base_config=self._base_config(),
                task_ids=["task-a"],
                harbor_skills=[],
                result_name="baseline",
                jobs_dir="../unsafe",
            )

    def test_resolved_config_is_accepted_by_real_harbor_parser(self) -> None:
        harbor = shutil.which("harbor")
        if harbor is None:
            self.skipTest("Harbor executable is not installed")
        version = subprocess.run(
            [harbor, "--version"],
            check=True,
            capture_output=True,
            text=True,
            timeout=30,
        ).stdout.strip()
        if version != EXPECTED_HARBOR_VERSION:
            self.skipTest(f"Harbor {EXPECTED_HARBOR_VERSION} is not installed")

        resolved = self._build(skills=[self._skill_dir()])
        resolved_path = self.root / "resolved.json"
        resolved_path.write_text(
            json.dumps(resolved, ensure_ascii=False, indent=2, sort_keys=True),
            encoding="utf-8",
        )
        completed = subprocess.run(
            [harbor, "run", "--config", str(resolved_path), "--print-config"],
            cwd=self.root,
            check=False,
            capture_output=True,
            text=True,
            timeout=30,
        )

        self.assertEqual(completed.returncode, 0, completed.stdout + completed.stderr)
        parsed = json.loads(completed.stdout)
        self.assertEqual(parsed["agents"][0]["skills"], resolved["agents"][0]["skills"])
        self.assertEqual(
            parsed["datasets"][0]["task_names"],
            resolved["datasets"][0]["task_names"],
        )


if __name__ == "__main__":
    unittest.main()
