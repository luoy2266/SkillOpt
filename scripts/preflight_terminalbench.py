#!/usr/bin/env python3
"""Fail-closed preflight and manifest writer for formal Terminal-Bench runs."""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.metadata
import json
import os
import re
import shutil
import subprocess
import sys
import tomllib
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from skillopt.config import flatten_config, load_config
from skillopt.envs.terminalbench.dataloader import TerminalBenchDataLoader
from skillopt.envs.terminalbench.harbor_runner import load_harbor_base_config
from skillopt.envs.terminalbench.skill_pack import (
    is_semantically_blank,
    render_skill_artifact,
)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
EXPECTED_BRANCH = "terminalbench-v2.1"
EXPECTED_TBENCH_HEAD = "7131e4375048a0e408a8fb404b5f499d726b695b"
EXPECTED_SPLIT_SHA256 = "8fa19aa350b90a7c39c3cde56f87a93bbfcb450586b416dc700c4c0b35827894"
EXPECTED_COUNTS = {"train": 9, "val": 9, "test": 71}
EXPECTED_HARBOR_VERSION = "0.20.0"
EXPECTED_TERMINUS_VERSION = "2.0.0"
EXPECTED_TARGET_MODEL = "deepseek/deepseek-v4-flash"
EXPECTED_OPTIMIZER_MODEL = "deepseek-v4-flash"
EXPECTED_UNDERLYING_MODEL = "DeepSeek-V4-Flash-0731"
EXPECTED_REASONING_EFFORT = "max"
EXPECTED_OPTIMIZER_CAP = 16384
EXPECTED_OPTIMIZER_ENDPOINT = "https://api.deepseek.com"
EXPECTED_AGENT = "terminus-2"
EXPECTED_TARGET_MAX_TURNS = 250
EXPECTED_CACHE_CONTAINER_ROOT = "/opt/skillopt-cache/terminal-bench-v2.1"
CACHE_MANIFEST_NAME = "MANIFEST.tsv"
EXPECTED_CACHE_ENV = {
    "SKILLOPT_TERMINALBENCH_CACHE_ROOT": EXPECTED_CACHE_CONTAINER_ROOT,
    "HF_HOME": f"{EXPECTED_CACHE_CONTAINER_ROOT}/huggingface",
    "HF_HUB_CACHE": f"{EXPECTED_CACHE_CONTAINER_ROOT}/huggingface/hub",
    "HF_DATASETS_CACHE": f"{EXPECTED_CACHE_CONTAINER_ROOT}/huggingface/datasets",
}
EXPECTED_HIGH_RISK_ASSETS = {
    "hf-distilbert-sst2": "required-cache",
    "hf-qwen2.5-1.5b-instruct": "required-cache",
    "hf-openthoughts-1k-sample-hub": "required-cache",
    "hf-openthoughts-1k-sample-prepared": "required-cache",
    "hf-bge-small-zh-v1.5-rev-7999e1d3": "required-cache",
    "caffe-cifar10": "runtime-network-only",
    "povray-2.2-archives": "runtime-network-only",
    "qemu-5.2.0-source": "runtime-network-only",
    "mteb-results-repository": "runtime-network-only",
    "alpine-3.19-extended-iso": "image-build-only",
    "oewn-sqlite": "image-build-only",
    "allenai-c4-shard": "image-build-only",
    "yelp-review-full-parquet": "image-build-only",
}
CACHE_MANIFEST_FIELDS = (
    "asset_id",
    "classification",
    "relative_path",
    "sha256",
    "tasks",
    "source",
)
REQUIRED_PROXY_NAMES = (
    "HTTP_PROXY",
    "HTTPS_PROXY",
    "http_proxy",
    "https_proxy",
    "NO_PROXY",
    "no_proxy",
)


class PreflightFailure(RuntimeError):
    """Raised after one or more refusal conditions fail."""


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _sha256_tree(path: Path) -> str:
    if path.is_symlink():
        raise PreflightFailure(f"cache asset must not be a symlink: {path}")
    if path.is_file():
        return _sha256(path)
    if not path.is_dir():
        raise PreflightFailure(f"cache asset is not a file or directory: {path}")
    root = path.resolve()
    digest = hashlib.sha256()
    entries = sorted(path.rglob("*"), key=lambda item: item.relative_to(path).as_posix())
    file_count = 0
    for item in entries:
        relative = item.relative_to(path).as_posix().encode("utf-8")
        if item.is_symlink():
            try:
                resolved = item.resolve(strict=True)
                resolved.relative_to(root)
            except (FileNotFoundError, RuntimeError, ValueError) as exc:
                raise PreflightFailure(
                    f"cache asset contains unsafe symlink: {item}"
                ) from exc
            if not resolved.is_file():
                raise PreflightFailure(
                    f"cache asset symlink must resolve to a file: {item}"
                )
            digest.update(b"L\0")
            digest.update(relative)
            digest.update(b"\0")
            digest.update(os.readlink(item).encode("utf-8"))
            digest.update(b"\0")
            digest.update(bytes.fromhex(_sha256(resolved)))
            digest.update(b"\n")
            file_count += 1
        elif item.is_file():
            digest.update(b"F\0")
            digest.update(relative)
            digest.update(b"\0")
            digest.update(bytes.fromhex(_sha256(item)))
            digest.update(b"\n")
            file_count += 1
        elif not item.is_dir():
            raise PreflightFailure(f"cache asset contains unsupported entry: {item}")
    if not file_count:
        raise PreflightFailure(f"cache asset directory is empty: {path}")
    return digest.hexdigest()


def _run(command: list[str], *, cwd: Path | None = None) -> str:
    completed = subprocess.run(
        command,
        cwd=cwd,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    if completed.returncode != 0:
        detail = completed.stderr.strip() or completed.stdout.strip() or "no output"
        raise PreflightFailure(f"command failed ({' '.join(command)}): {detail}")
    return completed.stdout.strip()


def _git(path: Path, *args: str) -> str:
    return _run(["git", "-C", str(path), *args])


def _require_clean_git(path: Path, *, label: str) -> tuple[str, str]:
    branch = _git(path, "branch", "--show-current")
    head = _git(path, "rev-parse", "HEAD")
    status = _git(path, "status", "--short")
    if status:
        raise PreflightFailure(f"{label} worktree is dirty")
    return branch, head


def _harbor_version(executable: str) -> str:
    return _run([executable, "--version"]).splitlines()[-1].strip()


def _terminus_version(executable: str) -> str:
    launcher = Path(shutil.which(executable) or executable).resolve()
    first_line = launcher.read_text(encoding="utf-8").splitlines()[0]
    if not first_line.startswith("#!"):
        raise PreflightFailure(f"Harbor launcher has no Python shebang: {launcher}")
    interpreter = Path(first_line[2:]).resolve()
    roots = sorted((interpreter.parent.parent / "lib").glob("python*/site-packages"))
    for root in roots:
        source = root / "harbor/agents/terminus_2/terminus_2.py"
        if not source.is_file():
            continue
        match = re.search(
            r"def version\(self\).*?return\s+[\"']([^\"']+)[\"']",
            source.read_text(encoding="utf-8"),
            flags=re.DOTALL,
        )
        if match:
            return match.group(1)
    raise PreflightFailure("Unable to locate the installed Terminus-2 version")


def _split_state(split_dir: Path) -> tuple[dict[str, list[str]], str]:
    loader = TerminalBenchDataLoader(
        split_dir=str(split_dir),
        split_mode="split_dir",
        limit=0,
    )
    loader.setup({})
    task_ids = {
        "train": [item["id"] for item in loader.train_items],
        "val": [item["id"] for item in loader.val_items],
        "test": [item["id"] for item in loader.test_items],
    }
    counts = {name: len(ids) for name, ids in task_ids.items()}
    if counts != EXPECTED_COUNTS:
        raise PreflightFailure(f"split counts mismatch: {counts!r}")
    sets = {name: set(ids) for name, ids in task_ids.items()}
    if sets["train"] & sets["val"] or sets["train"] & sets["test"] or sets["val"] & sets["test"]:
        raise PreflightFailure("split contains cross-split overlap")
    if len(set().union(*sets.values())) != 89:
        raise PreflightFailure("split does not contain exactly 89 unique tasks")
    manifest = split_dir / "split_manifest.json"
    digest = _sha256(manifest)
    if digest != EXPECTED_SPLIT_SHA256:
        raise PreflightFailure(
            f"split manifest SHA-256 mismatch: expected {EXPECTED_SPLIT_SHA256}, got {digest}"
        )
    return task_ids, digest


def _validate_formal_config(config_path: Path) -> dict[str, Any]:
    flat = flatten_config(load_config(str(config_path)))
    expected = {
        "env": "terminalbench",
        "optimizer_backend": "openai_compatible",
        "optimizer_model": EXPECTED_OPTIMIZER_MODEL,
        "reasoning_effort": EXPECTED_REASONING_EFFORT,
        "optimizer_openai_compatible_completion_cap": EXPECTED_OPTIMIZER_CAP,
        "num_epochs": 4,
        "train_size": 0,
        "batch_size": 40,
        "accumulation": 1,
        "seed": 42,
        "minibatch_size": 8,
        "merge_batch_size": 8,
        "analyst_workers": 16,
        "failure_only": False,
        "edit_budget": 4,
        "use_slow_update": True,
        "slow_update_samples": 20,
        "use_meta_skill": True,
        "sel_env_num": 0,
        "test_env_num": 0,
        "eval_test": False,
        "limit": 0,
        "n_concurrent_trials": 1,
    }
    mismatches = {
        key: {"expected": value, "actual": flat.get(key)}
        for key, value in expected.items()
        if flat.get(key) != value
    }
    if mismatches:
        raise PreflightFailure(f"formal config mismatch: {mismatches!r}")
    plaintext_credentials = sorted(
        key for key, value in flat.items() if _is_sensitive_key(key) and value
    )
    if plaintext_credentials:
        raise PreflightFailure(
            "formal config must not contain credential values: "
            f"{plaintext_credentials}"
        )
    return flat


def _validate_harbor_config(
    path: Path,
    *,
    tasks_path: Path,
    cache_root: Path,
) -> dict[str, Any]:
    config = load_harbor_base_config(path)
    if len(config.get("agents") or []) != 1:
        raise PreflightFailure("Harbor config must contain exactly one agent")
    if len(config.get("datasets") or []) != 1:
        raise PreflightFailure("Harbor config must contain exactly one dataset")
    agent = config["agents"][0]
    if config.get("n_attempts", 1) != 1:
        raise PreflightFailure("Harbor n_attempts must equal 1")
    if config.get("n_concurrent_trials", 1) != 1:
        raise PreflightFailure("Harbor n_concurrent_trials must equal 1")
    if (config.get("retry") or {}).get("max_retries", 0) != 0:
        raise PreflightFailure("Harbor retry.max_retries must equal 0")
    if agent.get("name") != EXPECTED_AGENT:
        raise PreflightFailure("Harbor agent must be Terminus-2")
    if agent.get("model_name") != EXPECTED_TARGET_MODEL:
        raise PreflightFailure("Harbor target request model mismatch")
    if (agent.get("kwargs") or {}).get("reasoning_effort") != EXPECTED_REASONING_EFFORT:
        raise PreflightFailure("Harbor target reasoning_effort mismatch")
    if (agent.get("kwargs") or {}).get("max_turns") != EXPECTED_TARGET_MAX_TURNS:
        raise PreflightFailure("Harbor Terminus-2 max_turns mismatch")
    if agent.get("skills") not in (None, []):
        raise PreflightFailure("Harbor base config must retain skills=[]")
    dataset_path = Path(config["datasets"][0]["path"]).expanduser().resolve()
    if dataset_path != tasks_path.resolve():
        raise PreflightFailure(
            f"Harbor dataset path mismatch: expected {tasks_path}, got {dataset_path}"
        )
    environment = config.get("environment") or {}
    if environment.get("type") != "docker":
        raise PreflightFailure("Harbor environment must be Docker")
    proxy_env = environment.get("env") or {}
    for name in REQUIRED_PROXY_NAMES:
        if proxy_env.get(name) != f"${{{name}}}":
            raise PreflightFailure(f"Harbor environment.env must reference ${{{name}}}")
    for name, expected in EXPECTED_CACHE_ENV.items():
        if proxy_env.get(name) != expected:
            raise PreflightFailure(
                f"Harbor environment.env cache path mismatch for {name}"
            )
    expected_source = str(cache_root.resolve())
    matching_mounts = [
        mount
        for mount in environment.get("mounts") or []
        if mount.get("target") == EXPECTED_CACHE_CONTAINER_ROOT
    ]
    if len(matching_mounts) != 1:
        raise PreflightFailure(
            "Harbor config must contain exactly one formal cache mount"
        )
    mount = matching_mounts[0]
    if (
        mount.get("type") != "bind"
        or Path(str(mount.get("source", ""))).expanduser().resolve() != cache_root.resolve()
        or mount.get("read_only") is not True
    ):
        raise PreflightFailure(
            "Harbor formal cache mount must be an exact read-only bind mount"
        )
    bind_options = mount.get("bind") or {}
    if bind_options.get("create_host_path") is not False:
        raise PreflightFailure(
            "Harbor formal cache mount must set bind.create_host_path=false"
        )
    if str(mount.get("source")) != expected_source:
        raise PreflightFailure("Harbor formal cache mount source must be canonical")
    return config


def _validate_cache_contract(cache_root: Path) -> dict[str, Any]:
    if cache_root.is_symlink() or not cache_root.is_dir():
        raise PreflightFailure(f"formal cache root is missing or unsafe: {cache_root}")
    if not os.access(cache_root, os.R_OK | os.X_OK):
        raise PreflightFailure(f"formal cache root is not readable: {cache_root}")
    manifest_path = cache_root / CACHE_MANIFEST_NAME
    if manifest_path.is_symlink() or not manifest_path.is_file():
        raise PreflightFailure(f"formal cache manifest is missing: {manifest_path}")
    if not os.access(manifest_path, os.R_OK):
        raise PreflightFailure(f"formal cache manifest is not readable: {manifest_path}")

    with manifest_path.open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        if tuple(reader.fieldnames or ()) != CACHE_MANIFEST_FIELDS:
            raise PreflightFailure(
                f"formal cache manifest header must be {CACHE_MANIFEST_FIELDS}"
            )
        rows = list(reader)
    by_id: dict[str, dict[str, str]] = {}
    for line_number, row in enumerate(rows, start=2):
        asset_id = str(row.get("asset_id") or "").strip()
        if not asset_id or asset_id in by_id:
            raise PreflightFailure(
                f"invalid or duplicate cache asset_id at line {line_number}"
            )
        classification = str(row.get("classification") or "").strip()
        if classification not in {
            "required-cache",
            "runtime-network-only",
            "image-build-only",
        }:
            raise PreflightFailure(
                f"invalid cache classification for {asset_id}: {classification!r}"
            )
        relative_path = str(row.get("relative_path") or "").strip()
        expected_sha256 = str(row.get("sha256") or "").strip().casefold()
        tasks = str(row.get("tasks") or "").strip()
        source = str(row.get("source") or "").strip()
        if not tasks or not source:
            raise PreflightFailure(
                f"cache asset {asset_id} must declare tasks and source"
            )
        if classification == "required-cache":
            lexical = Path(relative_path)
            if (
                not relative_path
                or lexical.is_absolute()
                or ".." in lexical.parts
                or not relative_path.startswith("huggingface/")
            ):
                raise PreflightFailure(
                    f"required cache asset {asset_id} has unsafe relative_path"
                )
            asset_path = cache_root / lexical
            if not asset_path.exists() or not os.access(asset_path, os.R_OK):
                raise PreflightFailure(
                    f"required cache asset is missing or unreadable: {asset_id}"
                )
            if re.fullmatch(r"[0-9a-f]{64}", expected_sha256) is None:
                raise PreflightFailure(
                    f"required cache asset {asset_id} must declare SHA-256"
                )
            actual_sha256 = _sha256_tree(asset_path)
            if actual_sha256 != expected_sha256:
                raise PreflightFailure(
                    f"required cache asset SHA-256 mismatch: {asset_id}"
                )
        elif relative_path != "-" or expected_sha256 != "-":
            raise PreflightFailure(
                f"{classification} asset {asset_id} must use '-' path and SHA-256"
            )
        by_id[asset_id] = {
            field: str(row.get(field) or "").strip()
            for field in CACHE_MANIFEST_FIELDS
        }

    for asset_id, classification in EXPECTED_HIGH_RISK_ASSETS.items():
        row = by_id.get(asset_id)
        if row is None:
            raise PreflightFailure(f"cache manifest is missing high-risk asset {asset_id}")
        if row["classification"] != classification:
            raise PreflightFailure(
                f"cache asset classification mismatch for {asset_id}"
            )
    return {
        "root": str(cache_root.resolve()),
        "container_root": EXPECTED_CACHE_CONTAINER_ROOT,
        "manifest_path": str(manifest_path.resolve()),
        "manifest_sha256": _sha256(manifest_path),
        "assets": [by_id[asset_id] for asset_id in sorted(by_id)],
    }


def _task_timeout_contract(
    tasks_path: Path,
    task_ids: dict[str, list[str]],
) -> dict[str, Any]:
    selected_ids = sorted({task_id for ids in task_ids.values() for task_id in ids})
    agent_seconds: dict[str, float] = {}
    verifier_seconds: dict[str, float] = {}
    build_seconds: dict[str, float] = {}
    for task_id in selected_ids:
        task_path = tasks_path / task_id / "task.toml"
        if not task_path.is_file():
            raise PreflightFailure(f"missing task.toml for {task_id}: {task_path}")
        task = tomllib.loads(task_path.read_text(encoding="utf-8"))
        if (task.get("task") or {}).get("name") != f"terminal-bench/{task_id}":
            raise PreflightFailure(f"task identity mismatch in {task_path}")
        try:
            agent_seconds[task_id] = float(task["agent"]["timeout_sec"])
            verifier_seconds[task_id] = float(task["verifier"]["timeout_sec"])
            build_seconds[task_id] = float(task["environment"]["build_timeout_sec"])
        except (KeyError, TypeError, ValueError) as exc:
            raise PreflightFailure(f"invalid timeout contract in {task_path}") from exc
    return {
        "source": "Terminal-Bench task.toml",
        "target_max_turns": EXPECTED_TARGET_MAX_TURNS,
        "agent_seconds_by_task": agent_seconds,
        "verifier_seconds_by_task": verifier_seconds,
        "build_seconds_by_task": build_seconds,
    }


def _proxy_entries(value: str) -> set[str]:
    return {entry.strip().casefold() for entry in value.split(",") if entry.strip()}


def _validate_proxy_environment(docker_local_addresses: list[str]) -> None:
    missing = [name for name in REQUIRED_PROXY_NAMES if not os.environ.get(name)]
    if missing:
        raise PreflightFailure(f"missing proxy environment variable names: {missing}")
    if os.environ["HTTP_PROXY"] != os.environ["http_proxy"]:
        raise PreflightFailure("HTTP_PROXY and http_proxy differ")
    if os.environ["HTTPS_PROXY"] != os.environ["https_proxy"]:
        raise PreflightFailure("HTTPS_PROXY and https_proxy differ")
    if os.environ["NO_PROXY"] != os.environ["no_proxy"]:
        raise PreflightFailure("NO_PROXY and no_proxy differ")
    entries = _proxy_entries(os.environ["NO_PROXY"])
    required = {
        "localhost",
        "127.0.0.1",
        "127.0.0.11",
        "::1",
        *(address.casefold() for address in docker_local_addresses if address),
    }
    missing_entries = sorted(required - entries)
    if missing_entries:
        raise PreflightFailure(f"NO_PROXY/no_proxy missing required local entries: {missing_entries}")


def _docker_state(tasks_path: Path, *, concurrency: int) -> dict[str, Any]:
    engine = _run(["docker", "version", "--format", "{{.Server.Version}}"])
    _run(["docker", "info", "--format", "{{.DockerRootDir}}"])
    network_ids = _run(["docker", "network", "ls", "-q"]).splitlines()
    networks = (
        json.loads(_run(["docker", "network", "inspect", *network_ids]))
        if network_ids
        else []
    )
    network_subnets: list[str] = []
    network_gateways: list[str] = []
    for network in networks:
        for entry in ((network.get("IPAM") or {}).get("Config") or []):
            subnet = str(entry.get("Subnet") or "").strip()
            gateway = str(entry.get("Gateway") or "").strip()
            if subnet:
                network_subnets.append(subnet)
            if gateway:
                network_gateways.append(gateway)
    max_storage_mb = 0
    for task_dir in tasks_path.iterdir():
        task_toml = task_dir / "task.toml"
        if not task_toml.is_file():
            continue
        task = tomllib.loads(task_toml.read_text(encoding="utf-8"))
        max_storage_mb = max(max_storage_mb, int(task["environment"]["storage_mb"]))
    free_bytes = shutil.disk_usage(tasks_path).free
    required_bytes = max_storage_mb * 1024 * 1024 * concurrency
    if free_bytes < required_bytes:
        raise PreflightFailure(
            f"insufficient storage: free={free_bytes}, required={required_bytes}"
        )
    daemon_config = Path("/etc/docker/daemon.json")
    address_pools_configured = False
    if daemon_config.is_file():
        try:
            address_pools_configured = bool(
                json.loads(daemon_config.read_text(encoding="utf-8")).get("default-address-pools")
            )
        except (OSError, json.JSONDecodeError):
            address_pools_configured = False
    if concurrency > 1 and not address_pools_configured:
        raise PreflightFailure(
            "Docker default-address-pools must be configured when concurrency > 1"
        )
    return {
        "engine": engine,
        "network_subnets": sorted(set(network_subnets)),
        "network_gateways": sorted(set(network_gateways)),
        "free_bytes": free_bytes,
        "minimum_declared_bytes": required_bytes,
        "default_address_pools_configured": address_pools_configured,
        "docker_default_address_pools_configured": address_pools_configured,
    }


def _validate_persistent_runtime() -> None:
    if not os.environ.get("INVOCATION_ID"):
        raise PreflightFailure("formal stage must run inside a systemd invocation")
    if os.environ.get("SKILLOPT_FORMAL_DOCKER_MODE") != "sg":
        raise PreflightFailure("formal stage must run through the frozen sg docker path")
    groups = set(_run(["id", "-nG"]).split())
    if "docker" not in groups:
        raise PreflightFailure("current user is not a member of the docker group")
    _run(["systemctl", "--user", "show-environment"])


def _skill_state(
    skill_path: Path,
    *,
    condition: str,
    training_output: Path | None,
) -> dict[str, Any]:
    if not skill_path.is_file():
        raise PreflightFailure(f"skill file is missing: {skill_path}")
    content = skill_path.read_text(encoding="utf-8")
    blank = is_semantically_blank(content)
    training_step: int | None = None
    selection_score: float | None = None
    origin: str | None = None
    if condition in {"training", "baseline-test"}:
        if not blank:
            raise PreflightFailure(f"{condition} requires the blank initial skill")
    else:
        if blank:
            raise PreflightFailure("skill-test requires a nonblank learned skill")
        if training_output is None:
            raise PreflightFailure("skill-test requires --training-output")
        expected = (training_output / "best_skill.md").resolve()
        if skill_path.resolve() != expected:
            raise PreflightFailure("skill-test must use training_output/best_skill.md")
        state_path = training_output / "runtime_state.json"
        if not state_path.is_file():
            raise PreflightFailure(f"missing training runtime state: {state_path}")
        state = json.loads(state_path.read_text(encoding="utf-8"))
        training_step = int(state["best_step"])
        selection_score = float(state["best_score"])
        origin = str(state["best_origin"])
    native_sha256 = None
    if not blank:
        native_sha256 = hashlib.sha256(render_skill_artifact(content)).hexdigest()
    return {
        "path": str(skill_path.resolve()),
        "bytes": len(content.encode("utf-8")),
        "sha256": hashlib.sha256(content.encode("utf-8")).hexdigest(),
        "native_skill_sha256": native_sha256,
        "is_blank": blank,
        "training_step": training_step,
        "selection_score": selection_score,
        "origin": origin,
    }


def _is_sensitive_key(key: str) -> bool:
    normalized = re.sub(r"[^a-z0-9]+", "_", key.casefold()).strip("_")
    return (
        normalized
        in {
            "api_key",
            "apikey",
            "authorization",
            "credential",
            "credentials",
            "password",
            "passwd",
            "secret",
            "token",
        }
        or normalized.endswith(
            (
                "_api_key",
                "_authorization",
                "_credential",
                "_credentials",
                "_password",
                "_secret",
                "_token",
            )
        )
    )


def _redact_config(config: dict[str, Any]) -> dict[str, Any]:
    redacted: dict[str, Any] = {}
    for key, value in config.items():
        if _is_sensitive_key(key):
            redacted[key] = "<redacted>" if value else value
        else:
            redacted[key] = value
    return redacted


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(PROJECT_ROOT / "configs/terminalbench/formal.yaml"))
    parser.add_argument("--tbench-root", required=True)
    parser.add_argument("--split-dir", required=True)
    parser.add_argument("--harbor-base-config", required=True)
    parser.add_argument("--cache-root", required=True)
    parser.add_argument("--harbor-executable", default="harbor")
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--manifest-out", required=True)
    parser.add_argument("--log-path", required=True)
    parser.add_argument("--experiment-id", required=True)
    parser.add_argument("--condition", choices=("training", "baseline-test", "skill-test"), required=True)
    parser.add_argument("--skill", required=True)
    parser.add_argument("--training-output")
    parser.add_argument("--expected-skillopt-head", required=True)
    parser.add_argument("--require-persistent-runtime", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config_path = Path(args.config).expanduser().resolve()
    tbench_root = Path(args.tbench_root).expanduser().resolve()
    tasks_path = tbench_root / "tasks"
    split_dir = Path(args.split_dir).expanduser().resolve()
    harbor_config_path = Path(args.harbor_base_config).expanduser().resolve()
    cache_root = Path(args.cache_root).expanduser().resolve()
    output_root = Path(args.output_root).expanduser().resolve()
    manifest_path = Path(args.manifest_out).expanduser().resolve()
    log_path = Path(args.log_path).expanduser().resolve()
    skill_path = Path(args.skill).expanduser().resolve()
    training_output = (
        Path(args.training_output).expanduser().resolve()
        if args.training_output
        else None
    )

    if output_root.exists():
        raise PreflightFailure(f"output root already exists: {output_root}")
    branch, skillopt_head = _require_clean_git(PROJECT_ROOT, label="SkillOpt")
    if branch != EXPECTED_BRANCH:
        raise PreflightFailure(f"SkillOpt branch mismatch: {branch!r}")
    if skillopt_head != args.expected_skillopt_head:
        raise PreflightFailure(
            f"SkillOpt HEAD mismatch: expected {args.expected_skillopt_head}, got {skillopt_head}"
        )
    _, tbench_head = _require_clean_git(tbench_root, label="Terminal-Bench")
    if tbench_head != EXPECTED_TBENCH_HEAD:
        raise PreflightFailure(
            f"Terminal-Bench HEAD mismatch: expected {EXPECTED_TBENCH_HEAD}, got {tbench_head}"
        )
    if not tasks_path.is_dir():
        raise PreflightFailure(f"Terminal-Bench tasks directory is missing: {tasks_path}")

    flat_config = _validate_formal_config(config_path)
    task_ids, split_manifest_sha256 = _split_state(split_dir)
    cache_state = _validate_cache_contract(cache_root)
    harbor_config = _validate_harbor_config(
        harbor_config_path,
        tasks_path=tasks_path,
        cache_root=cache_root,
    )
    timeout_contract = _task_timeout_contract(tasks_path, task_ids)
    harbor_version = _harbor_version(args.harbor_executable)
    if harbor_version != EXPECTED_HARBOR_VERSION:
        raise PreflightFailure(
            f"Harbor version mismatch: expected {EXPECTED_HARBOR_VERSION}, got {harbor_version}"
        )
    terminus_version = _terminus_version(args.harbor_executable)
    if terminus_version != EXPECTED_TERMINUS_VERSION:
        raise PreflightFailure(
            f"Terminus-2 version mismatch: expected {EXPECTED_TERMINUS_VERSION}, got {terminus_version}"
        )

    if not os.environ.get("DEEPSEEK_API_KEY"):
        raise PreflightFailure("DEEPSEEK_API_KEY is missing")
    if not os.environ.get("OPTIMIZER_OPENAI_COMPATIBLE_API_KEY"):
        raise PreflightFailure("OPTIMIZER_OPENAI_COMPATIBLE_API_KEY is missing")
    if os.environ.get("OPTIMIZER_OPENAI_COMPATIBLE_BASE_URL") != EXPECTED_OPTIMIZER_ENDPOINT:
        raise PreflightFailure("optimizer OpenAI-compatible endpoint mismatch")

    docker_state = _docker_state(tasks_path, concurrency=1)
    _validate_proxy_environment(
        docker_state["network_subnets"] + docker_state["network_gateways"]
    )
    if args.require_persistent_runtime:
        _validate_persistent_runtime()

    skill = _skill_state(
        skill_path,
        condition=args.condition,
        training_output=training_output,
    )
    cli_overrides = [
        f"env.split_dir={split_dir}",
        f"env.harbor_base_config={harbor_config_path}",
        f"env.out_root={output_root}",
    ]
    manifest = {
        "schema_version": "skillopt-terminalbench-formal-v1",
        "experiment_id": args.experiment_id,
        "condition": args.condition,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "versions": {
            "skillopt_branch": branch,
            "skillopt_head": skillopt_head,
            "terminalbench_head": tbench_head,
            "harbor": harbor_version,
            "terminus_2": terminus_version,
            "docker_engine": docker_state["engine"],
            "python": sys.version.split()[0],
            "openai_sdk": importlib.metadata.version("openai"),
        },
        "dataset": {
            "checkout": str(tbench_root),
            "tasks_path": str(tasks_path),
            "split_dir": str(split_dir),
            "split_manifest_sha256": split_manifest_sha256,
            "split_manifest": json.loads(
                (split_dir / "split_manifest.json").read_text(encoding="utf-8")
            ),
            "counts": {**EXPECTED_COUNTS, "unique": 89},
            "task_ids": task_ids,
        },
        "models": {
            "underlying_identity": EXPECTED_UNDERLYING_MODEL,
            "target": {
                "request_model": EXPECTED_TARGET_MODEL,
                "reasoning_effort": EXPECTED_REASONING_EFFORT,
                "transport": "Harbor -> Terminus-2 -> LiteLLM",
            },
            "optimizer": {
                "backend": "openai_compatible",
                "request_model": EXPECTED_OPTIMIZER_MODEL,
                "reasoning_effort": EXPECTED_REASONING_EFFORT,
                "completion_cap": EXPECTED_OPTIMIZER_CAP,
                "endpoint": EXPECTED_OPTIMIZER_ENDPOINT,
            },
        },
        "cache": cache_state,
        "execution": {
            "config_path": str(config_path),
            "config_sha256": _sha256(config_path),
            "source_config_text": config_path.read_text(encoding="utf-8"),
            "resolved_config": _redact_config(flat_config),
            "cli_overrides": cli_overrides,
            "harbor_base_config_path": str(harbor_config_path),
            "harbor_base_config_sha256": _sha256(harbor_config_path),
            "harbor_base_config_text": harbor_config_path.read_text(encoding="utf-8"),
            "n_attempts": int(harbor_config.get("n_attempts", 1)),
            "n_concurrent_trials": int(harbor_config.get("n_concurrent_trials", 1)),
            "harbor_max_retries": int((harbor_config.get("retry") or {}).get("max_retries", 0)),
            "timeouts": timeout_contract,
            "docker": docker_state,
        },
        "skill": skill,
        "artifacts": {
            "output_root": str(output_root),
            "log_path": str(log_path),
            "manifest_path": str(manifest_path),
        },
        "environment_variable_names": [
            "SKILLOPT_FORMAL_HEAD",
            "SKILLOPT_FORMAL_DOCKER_MODE",
            "DEEPSEEK_API_KEY",
            "OPTIMIZER_OPENAI_COMPATIBLE_API_KEY",
            "OPTIMIZER_OPENAI_COMPATIBLE_BASE_URL",
            "TERMINALBENCH_FORMAL_CACHE_ROOT",
            *REQUIRED_PROXY_NAMES,
        ],
    }
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    if manifest_path.exists():
        raise PreflightFailure(f"manifest already exists: {manifest_path}")
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({
        "status": "READY",
        "condition": args.condition,
        "experiment_id": args.experiment_id,
        "manifest": str(manifest_path),
        "output_root": str(output_root),
        "default_address_pools_configured": docker_state["default_address_pools_configured"],
    }, ensure_ascii=False))


if __name__ == "__main__":
    try:
        main()
    except PreflightFailure as exc:
        print(json.dumps({"status": "BLOCKED", "error": str(exc)}), file=sys.stderr)
        raise SystemExit(2) from exc
