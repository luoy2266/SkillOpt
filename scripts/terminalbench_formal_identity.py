#!/usr/bin/env python3
"""Stable experiment-scoped names for formal Terminal-Bench orchestration."""

from __future__ import annotations

import argparse
import hashlib
import re
import shlex
from pathlib import Path

_UNSAFE_COMPONENT = re.compile(r"[^A-Za-z0-9_.-]+")


def sanitize_component(value: str, *, max_length: int = 96) -> str:
    raw = str(value or "").strip()
    sanitized = _UNSAFE_COMPONENT.sub("-", raw).strip("-.")
    if not sanitized:
        raise ValueError("experiment identity must contain a systemd-safe character")
    if len(sanitized) <= max_length:
        return sanitized
    digest = hashlib.sha256(raw.encode("utf-8")).hexdigest()[:12]
    prefix_length = max_length - len(digest) - 1
    return f"{sanitized[:prefix_length].rstrip('-.')}-{digest}"


def systemd_unit_name(experiment_id: str, stage: str) -> str:
    safe_stage = sanitize_component(stage, max_length=32)
    prefix = "skillopt-tbench-"
    suffix = f"-{safe_stage}"
    component_limit = 200 - len(prefix) - len(suffix)
    safe_experiment = sanitize_component(experiment_id, max_length=component_limit)
    return f"{prefix}{safe_experiment}{suffix}"


def experiment_lock_name(experiment_id: str) -> str:
    return f"{sanitize_component(experiment_id, max_length=128)}.lock"


def read_environment_value(path: Path, name: str) -> str:
    if not path.is_file():
        raise ValueError(f"environment file is missing: {path}")
    found: str | None = None
    for line_number, raw_line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[7:].lstrip()
        if "=" not in line:
            continue
        key, raw_value = line.split("=", 1)
        if key.strip() != name:
            continue
        try:
            values = shlex.split(raw_value, comments=True, posix=True)
        except ValueError as exc:
            raise ValueError(
                f"invalid {name} value in {path}:{line_number}: {exc}"
            ) from exc
        if len(values) != 1 or not values[0]:
            raise ValueError(f"{name} must be one non-empty value in {path}:{line_number}")
        if found is not None:
            raise ValueError(f"duplicate {name} entry in {path}")
        found = values[0]
    if found is None:
        raise ValueError(f"{name} is missing from {path}")
    return found


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    unit_parser = subparsers.add_parser("unit-name")
    unit_parser.add_argument("--experiment-id", required=True)
    unit_parser.add_argument("--stage", required=True)

    lock_parser = subparsers.add_parser("lock-name")
    lock_parser.add_argument("--experiment-id", required=True)

    env_parser = subparsers.add_parser("env-value")
    env_parser.add_argument("--file", type=Path, required=True)
    env_parser.add_argument("--name", required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.command == "unit-name":
        print(systemd_unit_name(args.experiment_id, args.stage))
    elif args.command == "lock-name":
        print(experiment_lock_name(args.experiment_id))
    else:
        print(read_environment_value(args.file, args.name))


if __name__ == "__main__":
    main()
