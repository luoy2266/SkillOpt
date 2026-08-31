from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


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
            "import json, os, sys\n"
            "with Path(os.environ['FORMAL_WRAPPER_CALL_LOG']).open('a', encoding='utf-8') as handle:\n"
            "    handle.write(json.dumps({'command': Path(sys.argv[0]).name, 'argv': sys.argv[1:]}) + '\\n')\n"
        )
        (scripts_dir / "preflight_terminalbench.py").write_text(
            logger
            + "raise SystemExit(int(os.environ.get('FAKE_PREFLIGHT_EXIT', '0')))\n",
            encoding="utf-8",
        )
        for name in ("train.py", "eval_only.py", "probe_terminalbench_formal_service.py"):
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
        source = (
            self.repository_root / "scripts" / "run_terminalbench_formal_stage.sh"
        ).read_text(encoding="utf-8")
        source = source.replace(
            'PROJECT_ROOT="/home/yunl/projects/SkillOpt"',
            f'PROJECT_ROOT="{self.project_root}"',
        )
        wrapper = self.root / "run-terminalbench-formal-stage.sh"
        wrapper.write_text(source, encoding="utf-8")
        wrapper.chmod(0o755)
        return wrapper

    def _environment(self, *, preflight_exit: int = 0) -> dict[str, str]:
        environment = dict(os.environ)
        environment.update(
            {
                "PATH": f"{self.fake_bin}:{environment['PATH']}",
                "SKILLOPT_FORMAL_HEAD": "a" * 40,
                "SKILLOPT_FORMAL_DOCKER_MODE": "sg",
                "SKILLOPT_FORMAL_EXPERIMENT_ID": "experiment-001",
                "SKILLOPT_FORMAL_ROOT": str(self.formal_root),
                "DEEPSEEK_API_KEY": "test-secret-not-logged",
                "HTTP_PROXY": "http://proxy.example",
                "HTTPS_PROXY": "http://proxy.example",
                "NO_PROXY": "localhost,127.0.0.1,::1",
                "TERMINALBENCH_ROOT": str(self.root / "terminal-bench"),
                "TERMINALBENCH_SPLIT_DIR": str(self.root / "split"),
                "TERMINALBENCH_HARBOR_BASE_CONFIG": str(self.root / "harbor.yaml"),
                "TERMINALBENCH_FORMAL_CACHE_ROOT": str(self.root / "cache"),
                "FORMAL_WRAPPER_CALL_LOG": str(self.call_log),
                "FAKE_PREFLIGHT_EXIT": str(preflight_exit),
            }
        )
        return environment

    def _run(self, stage: str, *, preflight_exit: int = 0) -> subprocess.CompletedProcess[str]:
        if self.call_log.exists():
            self.call_log.unlink()
        return subprocess.run(
            [str(self.wrapper), stage],
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
            timeout=10,
            env=self._environment(preflight_exit=preflight_exit),
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

    def test_training_still_runs_preflight_before_trainer(self) -> None:
        completed = self._run("training")

        self.assertEqual(completed.returncode, 0, completed.stderr)
        self.assertEqual(
            [call["command"] for call in self._calls()],
            ["preflight_terminalbench.py", "train.py"],
        )

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

    def test_evaluation_stages_stop_when_preflight_fails(self) -> None:
        for stage in ("baseline-test", "skill-test"):
            with self.subTest(stage=stage):
                completed = self._run(stage, preflight_exit=23)
                self.assertEqual(completed.returncode, 23)
                self.assertEqual(
                    [call["command"] for call in self._calls()],
                    ["preflight_terminalbench.py"],
                )


if __name__ == "__main__":
    unittest.main()
