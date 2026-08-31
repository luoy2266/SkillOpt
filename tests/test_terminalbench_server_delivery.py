from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import yaml

from scripts.preflight_terminalbench import _validate_harbor_config
from scripts.render_terminalbench_harbor_config import (
    render_harbor_config,
    write_harbor_config,
)
from scripts.terminalbench_formal_identity import (
    experiment_lock_name,
    read_environment_value,
    systemd_unit_name,
)


class TerminalBenchServerDeliveryTests(unittest.TestCase):
    def setUp(self) -> None:
        self.repository_root = Path(__file__).resolve().parents[1]
        self.template = (
            self.repository_root
            / "configs"
            / "terminalbench"
            / "harbor-formal.template.yaml"
        )
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary_directory.name)
        self.runtime_root = self.root / "runtime"
        self.tasks_path = self.runtime_root / "datasets" / "terminal-bench-2-1" / "tasks"
        self.cache_root = self.runtime_root / "cache" / "terminal-bench-v2.1"
        self.tasks_path.mkdir(parents=True)
        self.cache_root.mkdir(parents=True)

    def tearDown(self) -> None:
        self.temporary_directory.cleanup()

    def _render(self, *, concurrency: int, proxy_mode: str) -> dict:
        return render_harbor_config(
            template_path=self.template,
            runtime_root=self.runtime_root,
            tasks_path=self.tasks_path,
            cache_root=self.cache_root,
            concurrency=concurrency,
            proxy_mode=proxy_mode,
        )

    def test_reviewed_template_is_secret_free_and_machine_independent(self) -> None:
        text = self.template.read_text(encoding="utf-8")

        self.assertNotIn("/home/yunl", text)
        self.assertNotIn("DEEPSEEK_API_KEY", text)
        self.assertIn("deepseek/deepseek-v4-flash", text)
        self.assertIn("reasoning_effort: max", text)
        self.assertIn("create_host_path: false", text)

    def test_generator_is_deterministic_and_validates_direct_mode(self) -> None:
        first = self._render(concurrency=16, proxy_mode="direct")
        second = self._render(concurrency=16, proxy_mode="direct")
        output = self.root / "harbor-direct.yaml"
        write_harbor_config(first, output)

        self.assertEqual(first, second)
        self.assertEqual(first["n_concurrent_trials"], 16)
        self.assertEqual(first["n_attempts"], 1)
        self.assertEqual(first["retry"]["max_retries"], 0)
        self.assertNotIn("HTTP_PROXY", first["environment"]["env"])
        _validate_harbor_config(
            output,
            tasks_path=self.tasks_path,
            cache_root=self.cache_root,
            concurrency=16,
            proxy_configured=False,
        )

    def test_generator_emits_complete_proxy_reference_contract(self) -> None:
        config = self._render(concurrency=8, proxy_mode="environment")
        output = self.root / "harbor-proxy.yaml"
        write_harbor_config(config, output)

        for name in (
            "HTTP_PROXY",
            "HTTPS_PROXY",
            "http_proxy",
            "https_proxy",
            "NO_PROXY",
            "no_proxy",
        ):
            self.assertEqual(config["environment"]["env"][name], f"${{{name}}}")
        _validate_harbor_config(
            output,
            tasks_path=self.tasks_path,
            cache_root=self.cache_root,
            concurrency=8,
            proxy_configured=True,
        )

    def test_environment_template_contains_only_placeholders(self) -> None:
        template = (
            self.repository_root
            / "configs"
            / "terminalbench"
            / "terminalbench-formal.env.example"
        )
        text = template.read_text(encoding="utf-8")

        self.assertIn("SKILLOPT_RUNTIME_ROOT=<REQUIRED_OPERATOR_VALUE>", text)
        self.assertIn(
            "SKILLOPT_FORMAL_EXPERIMENT_ID=<REQUIRED_OPERATOR_VALUE>",
            text,
        )
        self.assertIn(
            "SKILLOPT_TBENCH_CONCURRENCY=<REQUIRED_POSITIVE_INTEGER>",
            text,
        )
        self.assertIn("DEEPSEEK_API_KEY=<REQUIRED_SECRET>", text)
        self.assertNotIn("SKILLOPT_TBENCH_CONCURRENCY=1", text)
        self.assertNotIn("/home/yunl", text)

    def test_formal_entrypoints_have_no_operator_input_fallbacks(self) -> None:
        wrapper = (
            self.repository_root / "scripts" / "run_terminalbench_formal_stage.sh"
        ).read_text(encoding="utf-8")
        bootstrap = (
            self.repository_root / "scripts" / "bootstrap_terminalbench_server.sh"
        ).read_text(encoding="utf-8")

        self.assertNotIn("tbench-v2.1-dsv4flash-s42-formal-001", wrapper)
        self.assertNotIn("tbench-v2.1-server-formal-001", bootstrap)
        self.assertNotIn("SKILLOPT_TBENCH_CONCURRENCY:-1", wrapper)
        self.assertNotIn("SKILLOPT_TBENCH_CONCURRENCY:-1", bootstrap)
        self.assertNotIn("PROJECT_ROOT}/../skillopt-runtime", wrapper)
        self.assertNotIn("PROJECT_ROOT}/../skillopt-runtime", bootstrap)

    def test_systemd_launcher_requires_operator_owned_environment_values(self) -> None:
        launcher = self.repository_root / "scripts" / "run_terminalbench_formal_systemd.sh"
        required = {
            "SKILLOPT_RUNTIME_ROOT": str(self.runtime_root),
            "SKILLOPT_FORMAL_EXPERIMENT_ID": "server-test-001",
            "SKILLOPT_TBENCH_CONCURRENCY": "8",
        }
        for missing_name in required:
            with self.subTest(missing_name=missing_name):
                env_file = self.root / f"missing-{missing_name}.env"
                env_file.write_text(
                    "\n".join(
                        f"{name}={value}"
                        for name, value in required.items()
                        if name != missing_name
                    )
                    + "\n",
                    encoding="utf-8",
                )
                environment = dict(os.environ)
                environment.update(
                    {
                        "SKILLOPT_PYTHON": sys.executable,
                        "SKILLOPT_FORMAL_ENV_FILE": str(env_file),
                    }
                )
                completed = subprocess.run(
                    [str(launcher), "probe"],
                    cwd=self.repository_root,
                    env=environment,
                    text=True,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    check=False,
                    timeout=10,
                )
                self.assertEqual(completed.returncode, 2)
                self.assertIn(
                    f"OPERATOR INPUT REQUIRED: {missing_name}",
                    completed.stderr,
                )

    def test_formal_entrypoints_have_no_developer_machine_paths(self) -> None:
        for relative_path in (
            "scripts/run_terminalbench_formal_stage.sh",
            "scripts/run_terminalbench_formal_systemd.sh",
            "scripts/probe_terminalbench_formal_systemd.sh",
        ):
            with self.subTest(path=relative_path):
                text = (self.repository_root / relative_path).read_text(encoding="utf-8")
                self.assertNotIn("/home/yunl", text)

    def test_systemd_launcher_exposes_complete_delivery_lifecycle(self) -> None:
        launcher = (
            self.repository_root / "scripts" / "run_terminalbench_formal_systemd.sh"
        ).read_text(encoding="utf-8")

        for stage in (
            "probe",
            "preflight",
            "training",
            "freeze-skill",
            "baseline-test",
            "skill-test",
            "aggregate",
        ):
            self.assertIn(stage, launcher)

    def test_manifest_concurrency_contract_accepts_positive_values_only(self) -> None:
        schema = json.loads(
            (
                self.repository_root
                / "configs"
                / "terminalbench"
                / "experiment_manifest.schema.json"
            ).read_text(encoding="utf-8")
        )
        contract = schema["properties"]["execution"]["properties"][
            "n_concurrent_trials"
        ]

        self.assertEqual(contract["type"], "integer")
        self.assertEqual(contract["minimum"], 1)
        self.assertGreaterEqual(1, contract["minimum"])
        self.assertGreaterEqual(16, contract["minimum"])
        self.assertLess(0, contract["minimum"])

    def test_systemd_and_lock_identities_are_experiment_scoped(self) -> None:
        first = systemd_unit_name("server experiment/a", "training")
        second = systemd_unit_name("server experiment/b", "training")

        self.assertNotEqual(first, second)
        self.assertTrue(first.startswith("skillopt-tbench-server-experiment-a-"))
        self.assertLessEqual(len(first), 200)
        self.assertNotEqual(
            experiment_lock_name("server experiment/a"),
            experiment_lock_name("server experiment/b"),
        )

    def test_environment_file_reader_returns_only_requested_identity(self) -> None:
        env_file = self.root / "formal.env"
        env_file.write_text(
            "DEEPSEEK_API_KEY=secret-not-returned\n"
            "SKILLOPT_FORMAL_EXPERIMENT_ID='server-exp-001'\n",
            encoding="utf-8",
        )

        self.assertEqual(
            read_environment_value(env_file, "SKILLOPT_FORMAL_EXPERIMENT_ID"),
            "server-exp-001",
        )

    def test_generated_yaml_contains_no_template_sentinels(self) -> None:
        config = self._render(concurrency=4, proxy_mode="direct")
        text = yaml.safe_dump(config, sort_keys=False)

        self.assertNotIn("__CONCURRENCY__", text)
        self.assertNotIn("__TASKS_PATH__", text)
        self.assertNotIn("__CACHE_ROOT__", text)
        self.assertNotIn("__JOBS_DIR__", text)


if __name__ == "__main__":
    unittest.main()
