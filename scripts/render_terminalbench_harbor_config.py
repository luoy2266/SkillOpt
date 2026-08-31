#!/usr/bin/env python3
"""Render the reviewed secret-free Harbor 0.20.0 formal config."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import yaml

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_TEMPLATE = PROJECT_ROOT / "configs/terminalbench/harbor-formal.template.yaml"
PROXY_NAMES = (
    "HTTP_PROXY",
    "HTTPS_PROXY",
    "http_proxy",
    "https_proxy",
    "NO_PROXY",
    "no_proxy",
)


def _positive_int(value: int | str) -> int:
    try:
        parsed = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError("concurrency must be a positive integer") from exc
    if parsed <= 0:
        raise ValueError("concurrency must be a positive integer")
    return parsed


def render_harbor_config(
    *,
    template_path: Path,
    runtime_root: Path,
    tasks_path: Path,
    cache_root: Path,
    concurrency: int,
    proxy_mode: str,
) -> dict[str, Any]:
    config = yaml.safe_load(template_path.read_text(encoding="utf-8"))
    if not isinstance(config, dict):
        raise ValueError(f"Harbor template must be a YAML object: {template_path}")
    concurrency = _positive_int(concurrency)
    runtime_root = runtime_root.expanduser().resolve()
    tasks_path = tasks_path.expanduser().resolve()
    cache_root = cache_root.expanduser().resolve()

    config["jobs_dir"] = str(runtime_root / "outputs" / "harbor-jobs")
    config["n_concurrent_trials"] = concurrency
    config["datasets"][0]["path"] = str(tasks_path)
    environment = config["environment"]
    environment["mounts"][0]["source"] = str(cache_root)
    environment_env = environment.setdefault("env", {})
    for name in PROXY_NAMES:
        environment_env.pop(name, None)
    if proxy_mode == "environment":
        environment_env.update({name: f"${{{name}}}" for name in PROXY_NAMES})
    elif proxy_mode != "direct":
        raise ValueError("proxy_mode must be 'direct' or 'environment'")
    return config


def write_harbor_config(config: dict[str, Any], output_path: Path) -> None:
    output_path = output_path.expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        yaml.safe_dump(config, sort_keys=False, allow_unicode=True),
        encoding="utf-8",
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runtime-root", type=Path, required=True)
    parser.add_argument("--tasks-path", type=Path)
    parser.add_argument("--cache-root", type=Path)
    parser.add_argument("--concurrency", required=True)
    parser.add_argument(
        "--proxy-mode",
        choices=("direct", "environment"),
        required=True,
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--template", type=Path, default=DEFAULT_TEMPLATE)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    runtime_root = args.runtime_root.expanduser().resolve()
    config = render_harbor_config(
        template_path=args.template,
        runtime_root=runtime_root,
        tasks_path=args.tasks_path or runtime_root / "datasets" / "terminal-bench-2-1" / "tasks",
        cache_root=args.cache_root or runtime_root / "cache" / "terminal-bench-v2.1",
        concurrency=_positive_int(args.concurrency),
        proxy_mode=args.proxy_mode,
    )
    write_harbor_config(config, args.output)
    print(args.output.expanduser().resolve())


if __name__ == "__main__":
    main()
