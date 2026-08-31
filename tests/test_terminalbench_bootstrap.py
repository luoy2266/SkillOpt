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
    ) -> subprocess.CompletedProcess[str]:
        environment = dict(os.environ)
        environment.update(
            {
                "SKILLOPT_PYTHON": sys.executable,
                "SKILLOPT_RUNTIME_ROOT": str(runtime_root),
                "SKILLOPT_FORMAL_EXPERIMENT_ID": "server-exp-001",
                "SKILLOPT_TBENCH_CONCURRENCY": "8",
                "TERMINALBENCH_ROOT": str(runtime_root / "datasets" / "terminal-bench-2-1"),
            }
        )
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
        self.assertIn("DEEPSEEK_API_KEY=<REQUIRED>", env_template)
        self.assertNotIn("test-secret", env_template)

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
