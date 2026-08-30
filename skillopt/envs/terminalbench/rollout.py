"""Orchestrate one Terminal-Bench rollout through Harbor 0.20.0 artifacts."""

from __future__ import annotations

import json
import os
from collections.abc import Mapping, Sequence
from datetime import datetime
from pathlib import Path
from typing import Any

from skillopt.envs.terminalbench.dataloader import normalize_manifest_item
from skillopt.envs.terminalbench.harbor_runner import HarborRunner, PreparedHarborRun
from skillopt.envs.terminalbench.result_parser import parse_trial_result
from skillopt.envs.terminalbench.skill_pack import package_skill_content
from skillopt.envs.terminalbench.trajectory import (
    conversation_output_path,
    convert_atif_trajectory,
)


class TerminalBenchRolloutError(RuntimeError):
    """Raised when a Harbor rollout artifact violates the Phase 6 contract."""


def run_terminalbench_rollout(
    items: Sequence[Mapping[str, Any]],
    *,
    skill_content: str,
    rollout_dir: str | os.PathLike[str],
    runner: HarborRunner,
    result_name: str,
    n_concurrent_trials: int | None = None,
) -> list[dict[str, Any]]:
    """Run one exact task batch and return SkillOpt rollout dictionaries."""
    task_ids = _input_task_ids(items)
    packaged_skill = package_skill_content(skill_content, rollout_dir)
    prepared = runner.prepare(
        task_ids=task_ids,
        harbor_skills=packaged_skill.harbor_skills,
        result_name=result_name,
        output_root=rollout_dir,
        n_concurrent_trials=n_concurrent_trials,
    )
    _require_single_attempt(prepared)

    runner.run(prepared)
    trials_by_task = _load_completed_job(prepared, task_ids)

    results_by_task: dict[str, dict[str, Any]] = {}
    for task_id in task_ids:
        trial_dir = trials_by_task[task_id]
        result = parse_trial_result(
            trial_dir / "result.json",
            expected_task_id=task_id,
        )
        convert_atif_trajectory(
            trial_dir / "agent" / "trajectory.json",
            expected_task_id=task_id,
            output_path=conversation_output_path(rollout_dir, task_id),
        )
        result.update(
            harbor_config_path=str(prepared.resolved_config_path),
            harbor_job_dir=str(prepared.expected_job_dir),
            skill_sha256=packaged_skill.sha256,
        )
        results_by_task[task_id] = result

    results = [results_by_task[task_id] for task_id in task_ids]
    if len(results) != len(task_ids) or any(
        result.get("id") != task_id for result, task_id in zip(results, task_ids)
    ):
        raise TerminalBenchRolloutError("Terminal-Bench rollout cardinality changed")
    return results


def _input_task_ids(items: Sequence[Mapping[str, Any]]) -> tuple[str, ...]:
    if isinstance(items, (str, bytes)):
        raise TypeError("items must be a sequence of task mappings")
    task_ids = tuple(
        normalize_manifest_item(item, context=f"items[{index}]")["id"]
        for index, item in enumerate(items)
    )
    if not task_ids:
        raise TerminalBenchRolloutError("Terminal-Bench rollout items must not be empty")
    if len(set(task_ids)) != len(task_ids):
        raise TerminalBenchRolloutError("Terminal-Bench rollout items contain duplicate IDs")
    return task_ids


def _require_single_attempt(prepared: PreparedHarborRun) -> None:
    n_attempts = prepared.resolved_config.get("n_attempts", 1)
    if isinstance(n_attempts, bool) or not isinstance(n_attempts, int):
        raise TerminalBenchRolloutError("Harbor n_attempts must be the integer 1")
    if n_attempts != 1:
        raise TerminalBenchRolloutError(
            f"Terminal-Bench Phase 6 requires n_attempts=1, found {n_attempts}"
        )


def _load_completed_job(
    prepared: PreparedHarborRun,
    expected_task_ids: Sequence[str],
) -> dict[str, Path]:
    job_dir = prepared.expected_job_dir
    if job_dir.is_symlink() or not job_dir.is_dir():
        raise TerminalBenchRolloutError(f"Harbor job directory is missing: {job_dir}")
    result = _load_json_object(job_dir / "result.json", "Harbor job result")
    _require_timestamp(result.get("started_at"), "job started_at", job_dir)
    _require_timestamp(result.get("finished_at"), "job finished_at", job_dir)

    expected_count = len(expected_task_ids)
    if _required_count(result, "n_total_trials", job_dir) != expected_count:
        raise TerminalBenchRolloutError(
            f"Harbor job trial count does not match input batch: {job_dir}"
        )
    stats = result.get("stats")
    if not isinstance(stats, Mapping):
        raise TerminalBenchRolloutError(f"Harbor job result is missing stats: {job_dir}")
    if _required_count(stats, "n_completed_trials", job_dir) != expected_count:
        raise TerminalBenchRolloutError(f"Harbor job has incomplete trials: {job_dir}")
    for field in ("n_running_trials", "n_pending_trials"):
        if _required_count(stats, field, job_dir) != 0:
            raise TerminalBenchRolloutError(
                f"Harbor job has nonzero {field}: {job_dir}"
            )

    return _discover_trials(job_dir, expected_task_ids)


def _discover_trials(
    job_dir: Path,
    expected_task_ids: Sequence[str],
) -> dict[str, Path]:
    trials_by_task: dict[str, Path] = {}
    for child in job_dir.iterdir():
        if child.is_symlink():
            raise TerminalBenchRolloutError(
                f"Harbor job artifact must not be a symlink: {child}"
            )
        if not child.is_dir():
            continue
        trial_result = _load_json_object(
            child / "result.json",
            "Harbor trial result",
        )
        task_name = trial_result.get("task_name")
        if not isinstance(task_name, str) or not task_name:
            raise TerminalBenchRolloutError(
                f"Harbor trial result is missing task_name: {child / 'result.json'}"
            )
        if task_name in trials_by_task:
            raise TerminalBenchRolloutError(
                f"Duplicate Harbor trials for task {task_name!r}: {job_dir}"
            )
        trials_by_task[task_name] = child

    expected = set(expected_task_ids)
    discovered = set(trials_by_task)
    if discovered != expected:
        missing = sorted(expected - discovered)
        unexpected = sorted(discovered - expected)
        raise TerminalBenchRolloutError(
            "Harbor trial task set does not match input batch: "
            f"missing={missing}, unexpected={unexpected}, job={job_dir}"
        )
    return trials_by_task


def _load_json_object(path: Path, label: str) -> dict[str, Any]:
    if path.is_symlink() or not path.is_file():
        raise TerminalBenchRolloutError(f"{label} is missing: {path}")
    try:
        with path.open(encoding="utf-8") as handle:
            value = json.load(handle, parse_constant=_reject_json_constant)
    except (OSError, UnicodeError, json.JSONDecodeError, ValueError) as exc:
        raise TerminalBenchRolloutError(f"{label} is malformed: {path}") from exc
    if not isinstance(value, dict):
        raise TerminalBenchRolloutError(f"{label} must be a JSON object: {path}")
    return value


def _reject_json_constant(value: str) -> Any:
    raise ValueError(f"non-standard JSON constant {value!r}")


def _required_count(value: Mapping[str, Any], field: str, job_dir: Path) -> int:
    count = value.get(field)
    if isinstance(count, bool) or not isinstance(count, int) or count < 0:
        raise TerminalBenchRolloutError(
            f"Harbor job result has invalid {field}: {job_dir}"
        )
    return count


def _require_timestamp(value: Any, field: str, job_dir: Path) -> None:
    if not isinstance(value, str) or not value.strip():
        raise TerminalBenchRolloutError(
            f"Harbor job is incomplete; missing {field}: {job_dir}"
        )
    try:
        datetime.fromisoformat(value)
    except ValueError as exc:
        raise TerminalBenchRolloutError(
            f"Harbor job has malformed {field}: {job_dir}"
        ) from exc
