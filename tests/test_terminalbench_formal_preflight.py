from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import yaml

from scripts.preflight_terminalbench import (
    CACHE_MANIFEST_FIELDS,
    EXPECTED_CACHE_CONTAINER_ROOT,
    EXPECTED_CACHE_ENV,
    EXPECTED_HIGH_RISK_ASSETS,
    PreflightFailure,
    _sha256_tree,
    _validate_cache_contract,
    _validate_harbor_config,
)
from scripts.probe_terminalbench_formal_service import collect_probe_status


class TerminalBenchFormalPreflightTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary_directory.name)

    def tearDown(self) -> None:
        self.temporary_directory.cleanup()

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
