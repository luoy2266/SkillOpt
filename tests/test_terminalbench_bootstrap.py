from __future__ import annotations

import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import yaml


class TerminalBenchBootstrapTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary_directory.name)
        self.repository_root = Path(__file__).resolve().parents[1]
        self.script = self.repository_root / "scripts" / "bootstrap_terminalbench_server.sh"

    def tearDown(self) -> None:
        self.temporary_directory.cleanup()

    def _run_init(
        self,
        runtime_root: Path,
        *,
        proxy: bool,
        experiment_id: str | None = "server-exp-001",
    ) -> subprocess.CompletedProcess[str]:
        environment = dict(os.environ)
        environment.update(
            {
                "SKILLOPT_PYTHON": sys.executable,
                "SKILLOPT_RUNTIME_ROOT": str(runtime_root),
                "SKILLOPT_TBENCH_CONCURRENCY": "8",
                "TERMINALBENCH_ROOT": str(runtime_root / "datasets" / "terminal-bench-2-1"),
            }
        )
        if experiment_id is None:
            environment.pop("SKILLOPT_FORMAL_EXPERIMENT_ID", None)
        else:
            environment["SKILLOPT_FORMAL_EXPERIMENT_ID"] = experiment_id
        for name in (
            "HTTP_PROXY",
            "HTTPS_PROXY",
            "http_proxy",
            "https_proxy",
            "NO_PROXY",
            "no_proxy",
        ):
            environment.pop(name, None)
        if proxy:
            environment.update(
                {
                    "HTTP_PROXY": "http://proxy.example:8080",
                    "HTTPS_PROXY": "http://proxy.example:8080",
                    "http_proxy": "http://proxy.example:8080",
                    "https_proxy": "http://proxy.example:8080",
                }
            )
        return subprocess.run(
            [str(self.script), "init"],
            cwd=self.repository_root,
            env=environment,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
            timeout=15,
        )

    def test_init_creates_portable_runtime_skeleton_without_credentials(self) -> None:
        runtime_root = self.root / "runtime-direct"

        completed = self._run_init(runtime_root, proxy=False)

        self.assertEqual(completed.returncode, 0, completed.stderr)
        for name in ("datasets", "splits", "harbor-configs", "outputs", "skills", "cache", "logs", "locks"):
            self.assertTrue((runtime_root / name).is_dir(), name)
        manifest = yaml.safe_load(
            (runtime_root / "splits" / "tbench-v2.1-s42" / "split_manifest.json").read_text(
                encoding="utf-8"
            )
        )
        self.assertEqual(
            manifest["semantic_sha256"],
            "bd36fe2f37a67cd2b46149263522d833166d3a4d036c8e9af082e742ad017500",
        )
        env_template = (runtime_root / "terminalbench-formal.env.example").read_text(
            encoding="utf-8"
        )
        self.assertIn("DEEPSEEK_API_KEY=<REQUIRED_SECRET>", env_template)
        self.assertIn(
            "SKILLOPT_TBENCH_CONCURRENCY=<REQUIRED_POSITIVE_INTEGER>",
            env_template,
        )
        self.assertNotIn("test-secret", env_template)

    def test_inspect_reports_missing_operator_inputs_without_defaults(self) -> None:
        environment = dict(os.environ)
        environment["SKILLOPT_PYTHON"] = sys.executable
        for name in (
            "SKILLOPT_RUNTIME_ROOT",
            "SKILLOPT_FORMAL_EXPERIMENT_ID",
            "SKILLOPT_TBENCH_CONCURRENCY",
            "TERMINALBENCH_ROOT",
            "TERMINALBENCH_FORMAL_CACHE_ROOT",
            "TERMINALBENCH_SPLIT_DIR",
            "TERMINALBENCH_HARBOR_BASE_CONFIG",
            "SKILLOPT_FORMAL_ENV_TEMPLATE_OUT",
        ):
            environment.pop(name, None)

        completed = subprocess.run(
            [str(self.script), "inspect"],
            cwd=self.repository_root,
            env=environment,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
            timeout=15,
        )

        self.assertEqual(completed.returncode, 2)
        for name in (
            "SKILLOPT_RUNTIME_ROOT",
            "SKILLOPT_FORMAL_EXPERIMENT_ID",
            "SKILLOPT_TBENCH_CONCURRENCY",
        ):
            self.assertIn(f"{name}=OPERATOR INPUT REQUIRED", completed.stdout)
        self.assertIn("operator_inputs_required=3", completed.stdout)
        self.assertNotIn("skillopt-runtime", completed.stdout)

    def test_init_requires_runtime_root_and_concurrency(self) -> None:
        cases = (
            (
                "SKILLOPT_RUNTIME_ROOT",
                {
                    "SKILLOPT_TBENCH_CONCURRENCY": "8",
                },
            ),
            (
                "SKILLOPT_TBENCH_CONCURRENCY",
                {
                    "SKILLOPT_RUNTIME_ROOT": str(self.root / "runtime"),
                },
            ),
        )
        for missing_name, values in cases:
            with self.subTest(missing_name=missing_name):
                environment = dict(os.environ)
                environment["SKILLOPT_PYTHON"] = sys.executable
                environment.update(values)
                environment.pop(missing_name, None)
                completed = subprocess.run(
                    [str(self.script), "init"],
                    cwd=self.repository_root,
                    env=environment,
                    text=True,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    check=False,
                    timeout=15,
                )
                self.assertEqual(completed.returncode, 2)
                self.assertIn(
                    f"OPERATOR INPUT REQUIRED: {missing_name}",
                    completed.stderr,
                )

    def test_init_does_not_invent_experiment_id(self) -> None:
        runtime_root = self.root / "runtime-no-experiment"

        completed = self._run_init(
            runtime_root,
            proxy=False,
            experiment_id=None,
        )

        self.assertEqual(completed.returncode, 0, completed.stderr)
        self.assertIn(
            "SKILLOPT_FORMAL_EXPERIMENT_ID=OPERATOR INPUT REQUIRED before probe/preflight",
            completed.stdout,
        )
        template = (runtime_root / "terminalbench-formal.env.example").read_text(
            encoding="utf-8"
        )
        self.assertIn(
            "SKILLOPT_FORMAL_EXPERIMENT_ID=<REQUIRED_OPERATOR_VALUE>",
            template,
        )

    def test_direct_and_proxy_modes_render_expected_harbor_environment(self) -> None:
        direct_root = self.root / "direct"
        proxy_root = self.root / "proxy"
        self.assertEqual(self._run_init(direct_root, proxy=False).returncode, 0)
        self.assertEqual(self._run_init(proxy_root, proxy=True).returncode, 0)

        direct = yaml.safe_load(
            (direct_root / "harbor-configs" / "tbench-v2.1-formal.yaml").read_text(
                encoding="utf-8"
            )
        )
        proxy = yaml.safe_load(
            (proxy_root / "harbor-configs" / "tbench-v2.1-formal.yaml").read_text(
                encoding="utf-8"
            )
        )
        self.assertEqual(direct["n_concurrent_trials"], 8)
        self.assertNotIn("HTTP_PROXY", direct["environment"]["env"])
        for name in ("HTTP_PROXY", "HTTPS_PROXY", "http_proxy", "https_proxy", "NO_PROXY", "no_proxy"):
            self.assertEqual(proxy["environment"]["env"][name], f"${{{name}}}")

    def test_partial_proxy_blocks_init(self) -> None:
        environment = dict(os.environ)
        environment.update(
            {
                "SKILLOPT_PYTHON": sys.executable,
                "SKILLOPT_RUNTIME_ROOT": str(self.root / "partial"),
                "HTTP_PROXY": "http://proxy.example",
            }
        )
        for name in ("HTTPS_PROXY", "http_proxy", "https_proxy"):
            environment.pop(name, None)

        completed = subprocess.run(
            [str(self.script), "init"],
            cwd=self.repository_root,
            env=environment,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
            timeout=15,
        )

        self.assertEqual(completed.returncode, 2)
        self.assertIn("partial", completed.stderr)

    def test_bootstrap_contains_no_host_mutation_or_benchmark_commands(self) -> None:
        text = self.script.read_text(encoding="utf-8")

        for forbidden in (
            "sudo ",
            "usermod",
            "daemon.json >",
            "docker system",
            "harbor run",
            "scripts/train.py",
            "scripts/eval_only.py",
            "api.deepseek.com",
        ):
            self.assertNotIn(forbidden, text)


if __name__ == "__main__":
    unittest.main()
