from __future__ import annotations

import json
import fcntl
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from scripts.terminalbench_formal_identity import experiment_lock_name


class TerminalBenchFormalWrapperTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary_directory.name)
        self.repository_root = Path(__file__).resolve().parents[1]
        self.project_root = self.root / "SkillOpt"
        self.formal_root = self.root / "formal" / "experiment-001"
        self.call_log = self.root / "calls.jsonl"
        self.fake_bin = self.root / "bin"
        self.fake_bin.mkdir()
        self._write_fake_project()
        self.wrapper = self._write_test_wrapper()

    def tearDown(self) -> None:
        self.temporary_directory.cleanup()

    def _write_fake_project(self) -> None:
        python_path = self.project_root / ".venv" / "bin" / "python"
        python_path.parent.mkdir(parents=True)
        python_path.symlink_to(sys.executable)
        scripts_dir = self.project_root / "scripts"
        scripts_dir.mkdir()
        configs_dir = self.project_root / "configs" / "terminalbench"
        configs_dir.mkdir(parents=True)
        (configs_dir / "formal.yaml").write_text("env: {}\n", encoding="utf-8")
        skill_path = (
            self.project_root
            / "skillopt"
            / "envs"
            / "terminalbench"
            / "skills"
            / "initial.md"
        )
        skill_path.parent.mkdir(parents=True)
        skill_path.write_text("\n", encoding="utf-8")

        logger = (
            "from pathlib import Path\n"
            "import json, os, sys, time\n"
            "with Path(os.environ['FORMAL_WRAPPER_CALL_LOG']).open('a', encoding='utf-8') as handle:\n"
            "    handle.write(json.dumps({'command': Path(sys.argv[0]).name, 'argv': sys.argv[1:]}) + '\\n')\n"
            "time.sleep(float(os.environ.get('FAKE_COMMAND_SLEEP', '0')))\n"
        )
        (scripts_dir / "preflight_terminalbench.py").write_text(
            logger
            + "raise SystemExit(int(os.environ.get('FAKE_PREFLIGHT_EXIT', '0')))\n",
            encoding="utf-8",
        )
        for name in (
            "train.py",
            "eval_only.py",
            "freeze_terminalbench_skill.py",
            "aggregate_terminalbench_results.py",
            "probe_terminalbench_formal_service.py",
        ):
            (scripts_dir / name).write_text(logger, encoding="utf-8")

        docker = self.fake_bin / "docker"
        docker.write_text(
            "#!/usr/bin/env bash\n"
            "set -euo pipefail\n"
            "if [[ \"$*\" == \"network ls -q\" ]]; then\n"
            "  echo network-id\n"
            "elif [[ \"${1:-} ${2:-}\" == \"network inspect\" ]]; then\n"
            "  echo '[{\"IPAM\":{\"Config\":[{\"Subnet\":\"172.18.0.0/16\",\"Gateway\":\"172.18.0.1\"}]}}]'\n"
            "else\n"
            "  exit 97\n"
            "fi\n",
            encoding="utf-8",
        )
        docker.chmod(0o755)
        harbor = self.fake_bin / "harbor"
        harbor.write_text(
            "#!/usr/bin/env bash\n"
            "echo harbor-must-not-run >>\"$FORMAL_WRAPPER_CALL_LOG\"\n"
            "exit 98\n",
            encoding="utf-8",
        )
        harbor.chmod(0o755)

    def _write_test_wrapper(self) -> Path:
        scripts_dir = self.project_root / "scripts"
        wrapper = scripts_dir / "run_terminalbench_formal_stage.sh"
        wrapper.write_text(
            (self.repository_root / "scripts" / wrapper.name).read_text(encoding="utf-8"),
            encoding="utf-8",
        )
        wrapper.chmod(0o755)
        identity = scripts_dir / "terminalbench_formal_identity.py"
        identity.write_text(
            (
                self.repository_root
                / "scripts"
                / "terminalbench_formal_identity.py"
            ).read_text(encoding="utf-8"),
            encoding="utf-8",
        )
        return wrapper

    def _environment(
        self,
        *,
        preflight_exit: int = 0,
        overrides: dict[str, str | None] | None = None,
    ) -> dict[str, str]:
        environment = dict(os.environ)
        environment.update(
            {
                "PATH": f"{self.fake_bin}:{environment['PATH']}",
                "SKILLOPT_FORMAL_HEAD": "a" * 40,
                "SKILLOPT_FORMAL_DOCKER_MODE": "sg",
                "SKILLOPT_FORMAL_EXPERIMENT_ID": "experiment-001",
                "SKILLOPT_RUNTIME_ROOT": str(self.root / "runtime"),
                "SKILLOPT_FORMAL_ROOT": str(self.formal_root),
                "SKILLOPT_TBENCH_CONCURRENCY": "1",
                "DEEPSEEK_API_KEY": "test-secret-not-logged",
                "HTTP_PROXY": "http://proxy.example",
                "HTTPS_PROXY": "http://proxy.example",
                "http_proxy": "http://proxy.example",
                "https_proxy": "http://proxy.example",
                "NO_PROXY": "localhost,127.0.0.1,::1",
                "no_proxy": "localhost,127.0.0.1,::1",
                "TERMINALBENCH_ROOT": str(self.root / "terminal-bench"),
                "TERMINALBENCH_SPLIT_DIR": str(self.root / "split"),
                "TERMINALBENCH_HARBOR_BASE_CONFIG": str(self.root / "harbor.yaml"),
                "TERMINALBENCH_FORMAL_CACHE_ROOT": str(self.root / "cache"),
                "FORMAL_WRAPPER_CALL_LOG": str(self.call_log),
                "FAKE_PREFLIGHT_EXIT": str(preflight_exit),
            }
        )
        for name, value in (overrides or {}).items():
            if value is None:
                environment.pop(name, None)
            else:
                environment[name] = value
        return environment

    def _run(
        self,
        stage: str,
        *,
        preflight_exit: int = 0,
        overrides: dict[str, str | None] | None = None,
    ) -> subprocess.CompletedProcess[str]:
        if self.call_log.exists():
            self.call_log.unlink()
        return subprocess.run(
            [str(self.wrapper), stage],
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
            timeout=10,
            env=self._environment(
                preflight_exit=preflight_exit,
                overrides=overrides,
            ),
        )

    def _calls(self) -> list[dict[str, object]]:
        if not self.call_log.exists():
            return []
        return [
            json.loads(line)
            for line in self.call_log.read_text(encoding="utf-8").splitlines()
        ]

    @staticmethod
    def _argument(arguments: list[str], name: str) -> str:
        return arguments[arguments.index(name) + 1]

    def test_preflight_stage_succeeds_without_fallthrough(self) -> None:
        completed = self._run("preflight")

        self.assertEqual(completed.returncode, 0, completed.stderr)
        calls = self._calls()
        self.assertEqual([call["command"] for call in calls], ["preflight_terminalbench.py"])
        arguments = calls[0]["argv"]
        self.assertIsInstance(arguments, list)
        self.assertEqual(self._argument(arguments, "--condition"), "training")
        self.assertEqual(
            self._argument(arguments, "--output-root"),
            str(self.formal_root / "preflight"),
        )
        self.assertEqual(
            self._argument(arguments, "--manifest-out"),
            str(self.formal_root / "manifests" / "preflight.experiment_manifest.json"),
        )
        self.assertNotIn("harbor-must-not-run", self.call_log.read_text(encoding="utf-8"))

    def test_preflight_stage_propagates_failure_without_fallthrough(self) -> None:
        completed = self._run("preflight", preflight_exit=17)

        self.assertEqual(completed.returncode, 17)
        self.assertEqual(
            [call["command"] for call in self._calls()],
            ["preflight_terminalbench.py"],
        )

    def test_unknown_stage_still_fails(self) -> None:
        completed = self._run("unknown-stage")

        self.assertEqual(completed.returncode, 2)
        self.assertIn("unknown formal stage", completed.stderr)
        self.assertEqual(self._calls(), [])

    def test_all_formal_stages_require_explicit_experiment_id(self) -> None:
        for stage in (
            "probe",
            "preflight",
            "training",
            "freeze-skill",
            "baseline-test",
            "skill-test",
            "aggregate",
        ):
            with self.subTest(stage=stage):
                completed = self._run(
                    stage,
                    overrides={"SKILLOPT_FORMAL_EXPERIMENT_ID": None},
                )
                self.assertEqual(completed.returncode, 2)
                self.assertIn(
                    "OPERATOR INPUT REQUIRED: SKILLOPT_FORMAL_EXPERIMENT_ID",
                    completed.stderr,
                )
                self.assertEqual(self._calls(), [])

    def test_formal_stage_requires_explicit_runtime_root(self) -> None:
        completed = self._run(
            "preflight",
            overrides={"SKILLOPT_RUNTIME_ROOT": None},
        )

        self.assertEqual(completed.returncode, 2)
        self.assertIn(
            "OPERATOR INPUT REQUIRED: SKILLOPT_RUNTIME_ROOT",
            completed.stderr,
        )
        self.assertEqual(self._calls(), [])

    def test_formal_stage_requires_explicit_concurrency(self) -> None:
        completed = self._run(
            "preflight",
            overrides={"SKILLOPT_TBENCH_CONCURRENCY": None},
        )

        self.assertEqual(completed.returncode, 2)
        self.assertIn(
            "OPERATOR INPUT REQUIRED: SKILLOPT_TBENCH_CONCURRENCY",
            completed.stderr,
        )
        self.assertEqual(self._calls(), [])

    def test_training_still_runs_preflight_before_trainer(self) -> None:
        completed = self._run("training")

        self.assertEqual(completed.returncode, 0, completed.stderr)
        self.assertEqual(
            [call["command"] for call in self._calls()],
            ["preflight_terminalbench.py", "train.py"],
        )
        preflight_args = self._calls()[0]["argv"]
        train_args = self._calls()[1]["argv"]
        self.assertEqual(self._argument(preflight_args, "--concurrency"), "1")
        self.assertIn("env.n_concurrent_trials=1", train_args)

    def test_training_stops_when_preflight_fails(self) -> None:
        completed = self._run("training", preflight_exit=19)

        self.assertEqual(completed.returncode, 19)
        self.assertEqual(
            [call["command"] for call in self._calls()],
            ["preflight_terminalbench.py"],
        )

    def test_evaluation_stages_keep_the_preflight_gate(self) -> None:
        for stage in ("baseline-test", "skill-test"):
            with self.subTest(stage=stage):
                completed = self._run(stage)
                self.assertEqual(completed.returncode, 0, completed.stderr)
                self.assertEqual(
                    [call["command"] for call in self._calls()],
                    ["preflight_terminalbench.py", "eval_only.py"],
                )

    def test_skill_test_uses_frozen_skill_and_provenance(self) -> None:
        completed = self._run("skill-test")

        self.assertEqual(completed.returncode, 0, completed.stderr)
        calls = self._calls()
        preflight_args = calls[0]["argv"]
        eval_args = calls[1]["argv"]
        frozen_root = self.root / "runtime" / "skills" / "experiment-001"
        self.assertEqual(
            self._argument(preflight_args, "--skill"),
            str(frozen_root / "best_skill.md"),
        )
        self.assertEqual(
            self._argument(preflight_args, "--skill-provenance"),
            str(frozen_root / "skill_provenance.json"),
        )
        self.assertEqual(
            self._argument(eval_args, "--skill"),
            str(frozen_root / "best_skill.md"),
        )

    def test_freeze_stage_has_no_preflight_trainer_or_eval_fallthrough(self) -> None:
        completed = self._run("freeze-skill")

        self.assertEqual(completed.returncode, 0, completed.stderr)
        calls = self._calls()
        self.assertEqual(
            [call["command"] for call in calls],
            ["freeze_terminalbench_skill.py"],
        )
        arguments = calls[0]["argv"]
        self.assertEqual(
            self._argument(arguments, "--training-output"),
            str(self.formal_root / "training"),
        )
        self.assertEqual(
            self._argument(arguments, "--output-root"),
            str(self.root / "runtime" / "skills" / "experiment-001"),
        )

    def test_aggregate_stage_is_read_only_over_evaluation_artifacts(self) -> None:
        completed = self._run("aggregate")

        self.assertEqual(completed.returncode, 0, completed.stderr)
        calls = self._calls()
        self.assertEqual(
            [call["command"] for call in calls],
            ["aggregate_terminalbench_results.py"],
        )
        arguments = calls[0]["argv"]
        self.assertEqual(
            self._argument(arguments, "--baseline-output"),
            str(self.formal_root / "baseline-test"),
        )
        self.assertEqual(
            self._argument(arguments, "--skill-output"),
            str(self.formal_root / "skill-test"),
        )

    def test_evaluation_stages_stop_when_preflight_fails(self) -> None:
        for stage in ("baseline-test", "skill-test"):
            with self.subTest(stage=stage):
                completed = self._run(stage, preflight_exit=23)
                self.assertEqual(completed.returncode, 23)
                self.assertEqual(
                    [call["command"] for call in self._calls()],
                    ["preflight_terminalbench.py"],
                )

    def test_project_and_runtime_roots_are_portable(self) -> None:
        runtime_root = self.root / "runtime-portable"
        completed = self._run(
            "preflight",
            overrides={
                "SKILLOPT_RUNTIME_ROOT": str(runtime_root),
                "SKILLOPT_FORMAL_ROOT": None,
                "TERMINALBENCH_ROOT": None,
                "TERMINALBENCH_SPLIT_DIR": None,
                "TERMINALBENCH_HARBOR_BASE_CONFIG": None,
                "TERMINALBENCH_FORMAL_CACHE_ROOT": None,
            },
        )

        self.assertEqual(completed.returncode, 0, completed.stderr)
        arguments = self._calls()[0]["argv"]
        self.assertEqual(
            self._argument(arguments, "--config"),
            str(self.project_root / "configs" / "terminalbench" / "formal.yaml"),
        )
        self.assertEqual(
            self._argument(arguments, "--split-dir"),
            str(runtime_root / "splits" / "tbench-v2.1-s42"),
        )
        self.assertEqual(
            self._argument(arguments, "--cache-root"),
            str(runtime_root / "cache" / "terminal-bench-v2.1"),
        )

    def test_granular_runtime_overrides_take_precedence(self) -> None:
        split_dir = self.root / "explicit-split"
        completed = self._run(
            "preflight",
            overrides={
                "SKILLOPT_RUNTIME_ROOT": str(self.root / "runtime-other"),
                "TERMINALBENCH_SPLIT_DIR": str(split_dir),
            },
        )

        self.assertEqual(completed.returncode, 0, completed.stderr)
        self.assertEqual(
            self._argument(self._calls()[0]["argv"], "--split-dir"),
            str(split_dir),
        )

    def test_direct_mode_without_proxy_is_allowed(self) -> None:
        completed = self._run(
            "preflight",
            overrides={name: None for name in (
                "HTTP_PROXY",
                "HTTPS_PROXY",
                "http_proxy",
                "https_proxy",
                "NO_PROXY",
                "no_proxy",
            )},
        )

        self.assertEqual(completed.returncode, 0, completed.stderr)

    def test_partial_proxy_is_rejected(self) -> None:
        completed = self._run(
            "preflight",
            overrides={
                "HTTP_PROXY": "http://proxy.example",
                "HTTPS_PROXY": None,
                "http_proxy": None,
                "https_proxy": None,
            },
        )

        self.assertEqual(completed.returncode, 2)
        self.assertIn("must both be set", completed.stderr)

    def test_server_concurrency_is_passed_to_preflight_and_training(self) -> None:
        completed = self._run(
            "training",
            overrides={"SKILLOPT_TBENCH_CONCURRENCY": "16"},
        )

        self.assertEqual(completed.returncode, 0, completed.stderr)
        calls = self._calls()
        self.assertEqual(self._argument(calls[0]["argv"], "--concurrency"), "16")
        self.assertIn("env.n_concurrent_trials=16", calls[1]["argv"])

    def test_invalid_concurrency_is_rejected(self) -> None:
        for value in ("0", "-1", "not-an-integer"):
            with self.subTest(value=value):
                completed = self._run(
                    "preflight",
                    overrides={"SKILLOPT_TBENCH_CONCURRENCY": value},
                )
                self.assertEqual(completed.returncode, 2)
                self.assertIn("positive integer", completed.stderr)

    def test_same_experiment_execution_lock_is_nonblocking(self) -> None:
        lock_root = self.root / "runtime" / "locks"
        lock_root.mkdir(parents=True)
        lock_path = lock_root / experiment_lock_name("experiment-001")
        with lock_path.open("w", encoding="utf-8") as handle:
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
            completed = self._run("training")

        self.assertEqual(completed.returncode, 2)
        self.assertIn("already running", completed.stderr)
        self.assertEqual(self._calls(), [])

    def test_different_experiments_use_independent_locks(self) -> None:
        lock_root = self.root / "runtime" / "locks"
        lock_root.mkdir(parents=True)
        lock_path = lock_root / experiment_lock_name("experiment-001")
        with lock_path.open("w", encoding="utf-8") as handle:
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
            completed = self._run(
                "training",
                overrides={
                    "SKILLOPT_FORMAL_EXPERIMENT_ID": "experiment-002",
                    "SKILLOPT_FORMAL_ROOT": str(self.root / "formal" / "experiment-002"),
                },
            )

        self.assertEqual(completed.returncode, 0, completed.stderr)


if __name__ == "__main__":
    unittest.main()
