"""SkillOpt adapter for Terminal-Bench v2.1 Harbor rollouts."""

from __future__ import annotations

import hashlib
import json
import os
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from skillopt.datasets.base import BatchSpec
from skillopt.envs.base import EnvAdapter
from skillopt.envs.terminalbench.dataloader import TerminalBenchDataLoader
from skillopt.envs.terminalbench.harbor_runner import HarborRunner
from skillopt.envs.terminalbench.rollout import run_terminalbench_rollout


class TerminalBenchAdapter(EnvAdapter):
    """Connect Terminal-Bench split batches to the Phase 6 rollout path."""

    def __init__(
        self,
        split_dir: str,
        harbor_base_config: str,
        harbor_executable: str = "harbor",
        n_concurrent_trials: int | None = None,
        analyst_workers: int = 16,
        failure_only: bool = False,
        minibatch_size: int = 8,
        edit_budget: int = 4,
        seed: int = 42,
        limit: int = 0,
    ) -> None:
        split_path = _require_path(split_dir, label="split_dir", directory=True)
        base_config_path = _require_path(
            harbor_base_config,
            label="harbor_base_config",
            directory=False,
        )
        if n_concurrent_trials is not None and (
            isinstance(n_concurrent_trials, bool)
            or not isinstance(n_concurrent_trials, int)
            or n_concurrent_trials <= 0
        ):
            raise ValueError("n_concurrent_trials must be a positive integer")

        self.harbor_base_config = str(base_config_path)
        self.harbor_executable = harbor_executable
        self.n_concurrent_trials = n_concurrent_trials
        self.analyst_workers = analyst_workers
        self.failure_only = failure_only
        self.minibatch_size = minibatch_size
        self.edit_budget = edit_budget
        self.dataloader = TerminalBenchDataLoader(
            split_dir=str(split_path),
            split_mode="split_dir",
            seed=seed,
            limit=limit,
        )
        self.runner: HarborRunner | None = None

    def setup(self, cfg: dict) -> None:
        super().setup(cfg)
        self.dataloader.setup(cfg)
        self.runner = HarborRunner(
            self.harbor_base_config,
            self.harbor_executable,
        )

    def get_dataloader(self) -> TerminalBenchDataLoader:
        return self.dataloader

    def build_env_from_batch(self, batch: BatchSpec, **kwargs):
        return list(batch.payload or [])

    def build_train_env(self, batch_size: int, seed: int, **kwargs):
        batch = self.dataloader.build_train_batch(
            batch_size=batch_size,
            seed=seed,
            **kwargs,
        )
        return self.build_env_from_batch(batch, **kwargs)

    def build_eval_env(self, env_num: int, split: str, seed: int, **kwargs):
        batch = self.dataloader.build_eval_batch(
            env_num=env_num,
            split=split,
            seed=seed,
            **kwargs,
        )
        return self.build_env_from_batch(batch, **kwargs)

    def rollout(
        self,
        env_manager,
        skill_content: str,
        out_dir: str,
        **kwargs,
    ) -> list[dict]:
        if self.runner is None:
            raise RuntimeError("TerminalBenchAdapter.setup() must run before rollout()")
        return run_terminalbench_rollout(
            env_manager,
            skill_content=skill_content,
            rollout_dir=out_dir,
            runner=self.runner,
            result_name=_result_name(out_dir, env_manager),
            n_concurrent_trials=self.n_concurrent_trials,
        )

    def get_task_types(self) -> list[str]:
        return ["other"]


def _result_name(
    rollout_dir: str | os.PathLike[str],
    items: Sequence[Mapping[str, Any]],
) -> str:
    raw = os.fspath(rollout_dir)
    if not isinstance(raw, str) or not raw.strip():
        raise ValueError("rollout directory must not be empty")
    task_ids = [
        item.get("id")
        if isinstance(item, Mapping) and isinstance(item.get("id"), str)
        else None
        for item in items
    ]
    identity = json.dumps(
        {
            "rollout_dir": str(Path(raw).expanduser().resolve()),
            "task_ids": task_ids,
        },
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    )
    digest = hashlib.sha256(identity.encode("utf-8")).hexdigest()[:16]
    return f"terminalbench-{digest}"


def _require_path(
    value: str | os.PathLike[str],
    *,
    label: str,
    directory: bool,
) -> Path:
    raw = os.fspath(value)
    if not isinstance(raw, str) or not raw.strip():
        raise ValueError(f"{label} must not be empty")
    path = Path(raw).expanduser().absolute()
    if directory and not path.is_dir():
        raise FileNotFoundError(f"{label} is not a directory: {path}")
    if not directory and not path.is_file():
        raise FileNotFoundError(f"{label} is not a file: {path}")
    return path
