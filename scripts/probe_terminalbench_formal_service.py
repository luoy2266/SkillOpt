#!/usr/bin/env python3
"""Secret-safe service-context probe for formal Terminal-Bench stages."""

from __future__ import annotations

import json
import os
import subprocess
from collections.abc import Callable, Mapping
from pathlib import Path

from scripts.preflight_terminalbench import (
    EXPECTED_OPTIMIZER_ENDPOINT,
    _proxy_entries,
    _validate_cache_contract,
)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
CommandRunner = Callable[[list[str]], str]


def _run(command: list[str]) -> str:
    completed = subprocess.run(
        command,
        text=True,
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
        timeout=30,
    )
    if completed.returncode != 0:
        raise RuntimeError(command[0])
    return completed.stdout.strip()


def _docker_local_addresses(run: CommandRunner) -> set[str]:
    network_ids = run(["docker", "network", "ls", "-q"]).splitlines()
    if not network_ids:
        return set()
    networks = json.loads(run(["docker", "network", "inspect", *network_ids]))
    addresses: set[str] = set()
    for network in networks:
        for entry in ((network.get("IPAM") or {}).get("Config") or []):
            for key in ("Subnet", "Gateway"):
                value = str(entry.get(key) or "").strip().casefold()
                if value:
                    addresses.add(value)
    return addresses


def collect_probe_status(
    environ: Mapping[str, str] | None = None,
    *,
    run: CommandRunner | None = None,
) -> dict[str, str]:
    env = dict(os.environ if environ is None else environ)
    command_runner = run or _run
    statuses: dict[str, str] = {}

    statuses["SYSTEMD_INVOCATION"] = "PASS" if env.get("INVOCATION_ID") else "FAIL"
    expected_head = env.get("SKILLOPT_FORMAL_HEAD", "")
    try:
        actual_head = command_runner(
            ["git", "-C", str(PROJECT_ROOT), "rev-parse", "HEAD"]
        )
        clean = not command_runner(
            ["git", "-C", str(PROJECT_ROOT), "status", "--short"]
        )
        statuses["SKILLOPT_FORMAL_HEAD"] = (
            "PASS" if bool(expected_head) and clean and actual_head == expected_head else "FAIL"
        )
    except Exception:
        statuses["SKILLOPT_FORMAL_HEAD"] = "FAIL"

    deepseek_key = env.get("DEEPSEEK_API_KEY", "")
    optimizer_key = env.get("OPTIMIZER_OPENAI_COMPATIBLE_API_KEY", "")
    statuses["DEEPSEEK_API_KEY"] = "SET" if deepseek_key else "MISSING"
    statuses["OPTIMIZER_OPENAI_COMPATIBLE_API_KEY"] = (
        "PASS" if deepseek_key and optimizer_key == deepseek_key else "FAIL"
    )
    statuses["OPTIMIZER_OPENAI_COMPATIBLE_BASE_URL"] = (
        "PASS"
        if env.get("OPTIMIZER_OPENAI_COMPATIBLE_BASE_URL")
        == EXPECTED_OPTIMIZER_ENDPOINT
        else "FAIL"
    )

    for upper, lower in (("HTTP_PROXY", "http_proxy"), ("HTTPS_PROXY", "https_proxy")):
        statuses[upper] = "SET" if env.get(upper) else "MISSING"
        statuses[lower] = (
            "PASS" if env.get(upper) and env.get(lower) == env.get(upper) else "FAIL"
        )

    no_proxy = env.get("NO_PROXY", "")
    lower_no_proxy = env.get("no_proxy", "")
    statuses["NO_PROXY"] = "SET" if no_proxy else "MISSING"
    try:
        required_no_proxy = {
            "localhost",
            "127.0.0.1",
            "127.0.0.11",
            "::1",
            *_docker_local_addresses(command_runner),
        }
        effective_entries = _proxy_entries(no_proxy)
        statuses["no_proxy"] = (
            "PASS"
            if no_proxy
            and lower_no_proxy == no_proxy
            and required_no_proxy.issubset(effective_entries)
            else "FAIL"
        )
    except Exception:
        statuses["no_proxy"] = "FAIL"

    try:
        groups = set(command_runner(["id", "-nG"]).split())
        statuses["SG_DOCKER"] = (
            "PASS"
            if env.get("SKILLOPT_FORMAL_DOCKER_MODE") == "sg" and "docker" in groups
            else "FAIL"
        )
        command_runner(["docker", "info", "--format", "{{.ServerVersion}}"])
        statuses["DOCKER_ACCESS"] = "PASS"
    except Exception:
        statuses["DOCKER_ACCESS"] = "FAIL"
        statuses.setdefault("SG_DOCKER", "FAIL")

    try:
        cache_root = Path(env.get("TERMINALBENCH_FORMAL_CACHE_ROOT", ""))
        _validate_cache_contract(cache_root)
        statuses["FORMAL_CACHE"] = "PASS"
    except Exception:
        statuses["FORMAL_CACHE"] = "FAIL"
    return statuses


def main() -> None:
    statuses = collect_probe_status()
    for name, status in statuses.items():
        print(f"{name}={status}")
    if any(status in {"FAIL", "MISSING"} for status in statuses.values()):
        raise SystemExit(2)


if __name__ == "__main__":
    main()
