from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import patch

import yaml

from scripts.preflight_terminalbench import (
    CACHE_MANIFEST_FIELDS,
    EXPECTED_CACHE_CONTAINER_ROOT,
    EXPECTED_CACHE_ENV,
    EXPECTED_HIGH_RISK_ASSETS,
    EXPECTED_TERMINUS_CLASS_MODULE,
    EXPECTED_TERMINUS_CLASS_NAME,
    PreflightFailure,
    _sha256_tree,
    _terminus_version,
    _validate_cache_contract,
    _validate_harbor_config,
    _validate_terminus_version,
)
from scripts.probe_terminalbench_formal_service import collect_probe_status


class TerminalBenchFormalPreflightTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary_directory.name)

    def tearDown(self) -> None:
        self.temporary_directory.cleanup()

    def _write_fake_harbor_tool(
        self,
        *,
        version: str = "2.0.0",
        registry_error: bool = False,
        version_error: bool = False,
        wrong_class: bool = False,
    ) -> tuple[Path, Path]:
        tool_root = Path(tempfile.mkdtemp(prefix="harbor-tool-", dir=self.root))
        bin_dir = tool_root / "bin"
        bin_dir.mkdir()
        base_python = Path("/usr/bin/python3")
        python_details = subprocess.run(
            [
                str(base_python),
                "-c",
                "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}'); "
                "print(sys.base_prefix); print(sys.executable)",
            ],
            check=True,
            text=True,
            stdout=subprocess.PIPE,
        ).stdout.splitlines()
        python_version, base_prefix, base_executable = python_details
        interpreter = bin_dir / "python"
        interpreter.symlink_to(base_python)
        (tool_root / "pyvenv.cfg").write_text(
            f"home = {Path(base_executable).parent}\n"
            "include-system-site-packages = false\n"
            f"version = {python_version}\n"
            f"executable = {base_executable}\n",
            encoding="utf-8",
        )
        site_packages = (
            tool_root / "lib" / f"python{python_version}" / "site-packages"
        )

        def write_package(relative_path: str, content: str = "") -> None:
            path = site_packages / relative_path
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(content, encoding="utf-8")

        write_package("harbor/__init__.py")
        write_package("harbor/agents/__init__.py")
        write_package("harbor/agents/terminus_2/__init__.py")
        version_body = (
            "raise RuntimeError('version failure')"
            if version_error
            else f"return {version!r}"
        )
        write_package(
            "harbor/agents/terminus_2/terminus_2.py",
            "class Terminus2:\n"
            "    @staticmethod\n"
            "    def name():\n"
            "        return 'terminus-2'\n"
            "    def version(self):\n"
            f"        {version_body}\n",
        )
        if registry_error:
            factory_body = (
                "class AgentFactory:\n"
                "    @classmethod\n"
                "    def get_agent_class(cls, name):\n"
                "        raise RuntimeError('registry failure')\n"
            )
        elif wrong_class:
            factory_body = (
                "class Decoy:\n"
                "    @staticmethod\n"
                "    def name():\n"
                "        return 'terminus-2'\n"
                "    def version(self):\n"
                f"        return {version!r}\n"
                "class AgentFactory:\n"
                "    @classmethod\n"
                "    def get_agent_class(cls, name):\n"
                "        return Decoy\n"
            )
        else:
            factory_body = (
                "from harbor.agents.terminus_2.terminus_2 import Terminus2\n"
                "class AgentFactory:\n"
                "    @classmethod\n"
                "    def get_agent_class(cls, name):\n"
                "        return Terminus2\n"
            )
        write_package("harbor/agents/factory.py", factory_body)
        write_package("harbor/models/__init__.py")
        write_package("harbor/models/agent/__init__.py")
        write_package(
            "harbor/models/agent/name.py",
            "from enum import Enum\n"
            "class AgentName(str, Enum):\n"
            "    TERMINUS_2 = 'terminus-2'\n",
        )
        launcher = bin_dir / "harbor"
        launcher.write_text(
            f"#!{interpreter}\nraise SystemExit(97)\n",
            encoding="utf-8",
        )
        launcher.chmod(0o755)
        self.assertEqual(Path(base_prefix), Path(base_executable).parent.parent)
        return launcher, interpreter

    @staticmethod
    def _runtime_identity(version: str = "2.0.0") -> str:
        return json.dumps(
            {
                "registry_name": "terminus-2",
                "agent_name": "terminus-2",
                "class_module": EXPECTED_TERMINUS_CLASS_MODULE,
                "class_name": EXPECTED_TERMINUS_CLASS_NAME,
                "version": version,
            }
        )

    def test_terminus_discovery_uses_uv_shebang_interpreter_and_registry(self) -> None:
        launcher, interpreter = self._write_fake_harbor_tool()

        version = _terminus_version(str(launcher))

        self.assertEqual(version, "2.0.0")
        self.assertTrue(interpreter.is_symlink())
        self.assertEqual(interpreter.readlink(), Path("/usr/bin/python3"))

    def test_terminus_discovery_subprocess_preserves_interpreter_symlink(self) -> None:
        launcher, interpreter = self._write_fake_harbor_tool()
        completed = subprocess.CompletedProcess(
            args=[],
            returncode=0,
            stdout=self._runtime_identity(),
            stderr="",
        )

        with patch(
            "scripts.preflight_terminalbench.subprocess.run", return_value=completed
        ) as run:
            self.assertEqual(_terminus_version(str(launcher)), "2.0.0")

        command = run.call_args.args[0]
        self.assertEqual(command[0], str(interpreter))
        self.assertNotEqual(command[0], str(interpreter.resolve()))
        self.assertFalse(run.call_args.kwargs["shell"])
        self.assertNotIn("env", run.call_args.kwargs)

    def test_terminus_discovery_isolated_from_current_python_modules(self) -> None:
        launcher, _ = self._write_fake_harbor_tool()
        decoy = types.ModuleType("harbor")
        decoy.__file__ = "/decoy/current-python/harbor.py"

        with patch.dict(sys.modules, {"harbor": decoy}):
            version = _terminus_version(str(launcher))

        self.assertEqual(version, "2.0.0")

    def test_terminus_discovery_does_not_require_standalone_distribution(self) -> None:
        launcher, interpreter = self._write_fake_harbor_tool()
        metadata_probe = subprocess.run(
            [
                str(interpreter),
                "-c",
                "from importlib import metadata; "
                "\ntry:\n metadata.version('terminus-2')"
                "\nexcept metadata.PackageNotFoundError:\n print('NOT_INSTALLED')",
            ],
            check=True,
            text=True,
            stdout=subprocess.PIPE,
        )

        self.assertEqual(metadata_probe.stdout.strip(), "NOT_INSTALLED")
        self.assertEqual(_terminus_version(str(launcher)), "2.0.0")

    def test_terminus_discovery_rejects_missing_harbor(self) -> None:
        with self.assertRaisesRegex(PreflightFailure, "Harbor executable not found"):
            _terminus_version(str(self.root / "missing-harbor"))

    def test_terminus_discovery_rejects_invalid_launcher(self) -> None:
        launcher = self.root / "harbor"
        launcher.write_text("not a Python launcher\n", encoding="utf-8")
        launcher.chmod(0o755)

        with self.assertRaisesRegex(PreflightFailure, "no Python shebang"):
            _terminus_version(str(launcher))

    def test_terminus_discovery_rejects_non_python_shebang(self) -> None:
        launcher = self.root / "harbor"
        launcher.write_text("#!/bin/sh\n", encoding="utf-8")
        launcher.chmod(0o755)

        with self.assertRaisesRegex(PreflightFailure, "non-Python shebang"):
            _terminus_version(str(launcher))

    def test_terminus_discovery_rejects_missing_shebang_interpreter(self) -> None:
        launcher = self.root / "harbor"
        launcher.write_text(
            f"#!{self.root / 'missing' / 'python'}\n",
            encoding="utf-8",
        )
        launcher.chmod(0o755)

        with self.assertRaisesRegex(PreflightFailure, "interpreter is unavailable"):
            _terminus_version(str(launcher))

    def test_terminus_discovery_rejects_subprocess_failure_without_source_fallback(
        self,
    ) -> None:
        tool_root = self.root / "tool"
        interpreter = tool_root / "bin" / "python"
        interpreter.parent.mkdir(parents=True)
        interpreter.write_text("#!/bin/sh\nexit 9\n", encoding="utf-8")
        interpreter.chmod(0o755)
        launcher = tool_root / "bin" / "harbor"
        launcher.write_text(f"#!{interpreter}\n", encoding="utf-8")
        launcher.chmod(0o755)
        source = (
            tool_root
            / "lib/python3.12/site-packages/harbor/agents/terminus_2/terminus_2.py"
        )
        source.parent.mkdir(parents=True)
        source.write_text(
            "class Terminus2:\n"
            "    def version(self):\n"
            "        return '2.0.0'\n",
            encoding="utf-8",
        )

        with self.assertRaisesRegex(PreflightFailure, "exit status 9"):
            _terminus_version(str(launcher))

    def test_terminus_discovery_rejects_timeout(self) -> None:
        launcher, _ = self._write_fake_harbor_tool()

        with patch(
            "scripts.preflight_terminalbench.subprocess.run",
            side_effect=subprocess.TimeoutExpired(["python"], 30),
        ):
            with self.assertRaisesRegex(PreflightFailure, "timed out"):
                _terminus_version(str(launcher))

    def test_terminus_discovery_rejects_malformed_json(self) -> None:
        launcher, _ = self._write_fake_harbor_tool()
        completed = subprocess.CompletedProcess(
            args=[], returncode=0, stdout="not-json", stderr=""
        )

        with patch(
            "scripts.preflight_terminalbench.subprocess.run", return_value=completed
        ):
            with self.assertRaisesRegex(PreflightFailure, "malformed JSON"):
                _terminus_version(str(launcher))

    def test_terminus_discovery_rejects_empty_version(self) -> None:
        launcher, _ = self._write_fake_harbor_tool(version="")

        with self.assertRaisesRegex(PreflightFailure, "version is empty"):
            _terminus_version(str(launcher))

    def test_terminus_discovery_rejects_registry_failure(self) -> None:
        launcher, _ = self._write_fake_harbor_tool(registry_error=True)

        with self.assertRaisesRegex(PreflightFailure, "subprocess failed"):
            _terminus_version(str(launcher))

    def test_terminus_discovery_rejects_version_method_failure(self) -> None:
        launcher, _ = self._write_fake_harbor_tool(version_error=True)

        with self.assertRaisesRegex(PreflightFailure, "subprocess failed"):
            _terminus_version(str(launcher))

    def test_terminus_discovery_rejects_wrong_registry_class_identity(self) -> None:
        launcher, _ = self._write_fake_harbor_tool(wrong_class=True)

        with self.assertRaisesRegex(PreflightFailure, "class_module"):
            _terminus_version(str(launcher))

    def test_terminus_expected_version_mismatch_blocks(self) -> None:
        with self.assertRaisesRegex(PreflightFailure, "expected 2.0.0, got 2.0.1"):
            _validate_terminus_version("2.0.1")

    def _write_cache(self, *, omit: str | None = None, bad_hash: str | None = None) -> Path:
        cache_root = self.root / "cache"
        cache_root.mkdir()
        rows = []
        for asset_id, classification in EXPECTED_HIGH_RISK_ASSETS.items():
            if asset_id == omit:
                continue
            if classification == "required-cache":
                relative_path = f"huggingface/{asset_id}"
                asset_path = cache_root / relative_path
                asset_path.mkdir(parents=True)
                (asset_path / "artifact.bin").write_bytes(asset_id.encode("utf-8"))
                sha256 = _sha256_tree(asset_path)
                if asset_id == bad_hash:
                    sha256 = "0" * 64
            else:
                relative_path = "-"
                sha256 = "-"
            rows.append(
                {
                    "asset_id": asset_id,
                    "classification": classification,
                    "relative_path": relative_path,
                    "sha256": sha256,
                    "tasks": f"task-{asset_id}",
                    "source": f"source:{asset_id}",
                }
            )
        manifest = cache_root / "MANIFEST.tsv"
        manifest.write_text(
            "\t".join(CACHE_MANIFEST_FIELDS)
            + "\n"
            + "".join(
                "\t".join(row[field] for field in CACHE_MANIFEST_FIELDS) + "\n"
                for row in rows
            ),
            encoding="utf-8",
        )
        return cache_root

    def _write_harbor_config(self, cache_root: Path, *, read_only: bool = True) -> Path:
        tasks_path = self.root / "tasks"
        tasks_path.mkdir(exist_ok=True)
        config = {
            "job_name": "formal-base",
            "jobs_dir": str(self.root / "jobs"),
            "n_attempts": 1,
            "n_concurrent_trials": 1,
            "retry": {"max_retries": 0},
            "environment": {
                "type": "docker",
                "env": {
                    **{name: f"${{{name}}}" for name in (
                        "HTTP_PROXY",
                        "HTTPS_PROXY",
                        "http_proxy",
                        "https_proxy",
                        "NO_PROXY",
                        "no_proxy",
                    )},
                    **EXPECTED_CACHE_ENV,
                },
                "mounts": [
                    {
                        "type": "bind",
                        "source": str(cache_root.resolve()),
                        "target": EXPECTED_CACHE_CONTAINER_ROOT,
                        "read_only": read_only,
                        "bind": {"create_host_path": False},
                    }
                ],
            },
            "agents": [
                {
                    "name": "terminus-2",
                    "model_name": "deepseek/deepseek-v4-flash",
                    "skills": [],
                    "kwargs": {"reasoning_effort": "max", "max_turns": 250},
                }
            ],
            "datasets": [{"path": str(tasks_path)}],
        }
        path = self.root / "harbor.yaml"
        path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
        return path

    def test_cache_contract_validates_required_assets_and_hashes(self) -> None:
        cache_root = self._write_cache()

        state = _validate_cache_contract(cache_root)

        self.assertEqual(state["root"], str(cache_root.resolve()))
        self.assertEqual(state["container_root"], EXPECTED_CACHE_CONTAINER_ROOT)
        self.assertEqual(len(state["assets"]), len(EXPECTED_HIGH_RISK_ASSETS))

    def test_cache_contract_rejects_missing_required_asset(self) -> None:
        cache_root = self._write_cache(omit="hf-distilbert-sst2")

        with self.assertRaisesRegex(PreflightFailure, "missing high-risk asset"):
            _validate_cache_contract(cache_root)

    def test_cache_contract_rejects_hash_mismatch(self) -> None:
        cache_root = self._write_cache(bad_hash="hf-qwen2.5-1.5b-instruct")

        with self.assertRaisesRegex(PreflightFailure, "SHA-256 mismatch"):
            _validate_cache_contract(cache_root)

    def test_cache_tree_hash_supports_internal_huggingface_symlinks(self) -> None:
        asset = self.root / "models--example--repo"
        blob = asset / "blobs" / "abc123"
        snapshot_file = asset / "snapshots" / "revision" / "config.json"
        blob.parent.mkdir(parents=True)
        snapshot_file.parent.mkdir(parents=True)
        blob.write_bytes(b"first")
        snapshot_file.symlink_to("../../blobs/abc123")

        initial = _sha256_tree(asset)
        blob.write_bytes(b"second")

        self.assertNotEqual(_sha256_tree(asset), initial)

    def test_cache_tree_hash_rejects_symlinks_outside_asset(self) -> None:
        asset = self.root / "asset"
        asset.mkdir()
        outside = self.root / "outside.bin"
        outside.write_bytes(b"external")
        (asset / "unsafe.bin").symlink_to(outside)

        with self.assertRaisesRegex(PreflightFailure, "unsafe symlink"):
            _sha256_tree(asset)

    def test_harbor_cache_mount_must_be_exact_and_read_only(self) -> None:
        cache_root = self._write_cache()
        config_path = self._write_harbor_config(cache_root)
        tasks_path = self.root / "tasks"

        config = _validate_harbor_config(
            config_path,
            tasks_path=tasks_path,
            cache_root=cache_root,
        )

        self.assertTrue(config["environment"]["mounts"][0]["read_only"])

        invalid_path = self._write_harbor_config(cache_root, read_only=False)
        with self.assertRaisesRegex(PreflightFailure, "read-only bind mount"):
            _validate_harbor_config(
                invalid_path,
                tasks_path=tasks_path,
                cache_root=cache_root,
            )

    def test_service_probe_is_secret_safe_and_checks_credential_bridge(self) -> None:
        cache_root = self._write_cache()
        secret = "sk-THIS-MUST-NOT-APPEAR"
        head = "a" * 40
        network_payload = json.dumps(
            [{"IPAM": {"Config": [{"Subnet": "172.17.0.0/16", "Gateway": "172.17.0.1"}]}}]
        )

        def fake_run(command: list[str]) -> str:
            key = tuple(command)
            if key[-2:] == ("rev-parse", "HEAD"):
                return head
            if key[-2:] == ("status", "--short"):
                return ""
            if key == ("id", "-nG"):
                return "yunl docker"
            if key == ("docker", "network", "ls", "-q"):
                return "network-id"
            if key == ("docker", "network", "inspect", "network-id"):
                return network_payload
            if key[:2] == ("docker", "info"):
                return "29.1.3"
            raise AssertionError(command)

        no_proxy = "localhost,127.0.0.1,127.0.0.11,::1,172.17.0.0/16,172.17.0.1"
        statuses = collect_probe_status(
            {
                "INVOCATION_ID": "formal-probe-invocation",
                "SKILLOPT_FORMAL_HEAD": head,
                "SKILLOPT_FORMAL_DOCKER_MODE": "sg",
                "DEEPSEEK_API_KEY": secret,
                "OPTIMIZER_OPENAI_COMPATIBLE_API_KEY": secret,
                "OPTIMIZER_OPENAI_COMPATIBLE_BASE_URL": "https://api.deepseek.com",
                "HTTP_PROXY": "http://proxy.example",
                "http_proxy": "http://proxy.example",
                "HTTPS_PROXY": "http://proxy.example",
                "https_proxy": "http://proxy.example",
                "NO_PROXY": no_proxy,
                "no_proxy": no_proxy,
                "TERMINALBENCH_FORMAL_CACHE_ROOT": str(cache_root),
            },
            run=fake_run,
        )

        self.assertTrue(statuses)
        self.assertNotIn("FAIL", statuses.values())
        self.assertNotIn("MISSING", statuses.values())
        rendered = "\n".join(f"{key}={value}" for key, value in statuses.items())
        self.assertNotIn(secret, rendered)


if __name__ == "__main__":
    unittest.main()
