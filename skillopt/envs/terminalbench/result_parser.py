"""Parse one Harbor 0.20.0 Terminal-Bench trial result."""
from __future__ import annotations

import json
import math
import os
from collections.abc import Mapping
from datetime import datetime
from pathlib import Path
from typing import Any

from skillopt.envs.terminalbench.dataloader import validate_task_id

_SCORED_AGENT_EXCEPTIONS = {
    "AgentTimeoutError",
    "NonZeroAgentExitCodeError",
}


class InfrastructureInvalidTrialError(RuntimeError):
    """Raised when a trial cannot safely enter SkillOpt scoring."""


def parse_trial_result(
    result_path: str | os.PathLike[str],
    *,
    expected_task_id: str,
) -> dict[str, str | float]:
    """Load and map one explicit Harbor trial ``result.json`` artifact."""
    expected_task_id = validate_task_id(
        expected_task_id,
        context="expected_task_id",
    )
    path = _result_path(result_path)
    result = _load_result(path)

    _require_timestamp(result.get("finished_at"), "finished_at", path)
    _validate_task_identity(result, expected_task_id, path)
    _validate_verifier_timing(result.get("verifier"), path)
    raw_reward = _extract_reward(result.get("verifier_result"), path)
    trial_status = _classify_exception(result, path)

    return {
        "id": expected_task_id,
        "hard": 1.0 if raw_reward == 1.0 else 0.0,
        "soft": raw_reward,
        "raw_reward": raw_reward,
        "trial_status": trial_status,
        "harbor_result_path": str(path.resolve()),
    }


def _result_path(value: str | os.PathLike[str]) -> Path:
    try:
        raw = os.fspath(value)
    except TypeError as exc:
        raise TypeError("result_path must be a string or path-like value") from exc
    if not isinstance(raw, str) or not raw.strip():
        raise ValueError("result_path must not be empty")
    return Path(raw).expanduser()


def _load_result(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise InfrastructureInvalidTrialError(
            f"Harbor trial result artifact is missing: {path}"
        )
    try:
        result = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise InfrastructureInvalidTrialError(
            f"Harbor trial result artifact is unreadable or malformed: {path}"
        ) from exc
    if not isinstance(result, dict):
        raise InfrastructureInvalidTrialError(
            f"Harbor trial result must be a JSON object: {path}"
        )
    return result


def _validate_task_identity(
    result: Mapping[str, Any],
    expected_task_id: str,
    path: Path,
) -> None:
    task_name = result.get("task_name")
    if task_name != expected_task_id:
        raise InfrastructureInvalidTrialError(
            "Harbor trial task_name does not match the expected task ID: "
            f"expected {expected_task_id!r}, got {task_name!r}, path={path}"
        )

    task_id = result.get("task_id")
    if not isinstance(task_id, Mapping):
        raise InfrastructureInvalidTrialError(
            f"Harbor trial result is missing task_id mapping: {path}"
        )
    task_source_path = task_id.get("path")
    if not isinstance(task_source_path, str) or not task_source_path.strip():
        raise InfrastructureInvalidTrialError(
            f"Harbor trial task_id is missing its local path: {path}"
        )
    actual_task_id = Path(task_source_path).name
    if actual_task_id != expected_task_id:
        raise InfrastructureInvalidTrialError(
            "Harbor trial task_id path does not match the expected task ID: "
            f"expected {expected_task_id!r}, got {actual_task_id!r}, path={path}"
        )


def _validate_verifier_timing(value: Any, path: Path) -> None:
    if not isinstance(value, Mapping):
        raise InfrastructureInvalidTrialError(
            f"Harbor trial result is missing verifier timing: {path}"
        )
    _require_timestamp(value.get("started_at"), "verifier.started_at", path)
    _require_timestamp(value.get("finished_at"), "verifier.finished_at", path)


def _require_timestamp(value: Any, field: str, path: Path) -> None:
    if not isinstance(value, str) or not value.strip():
        raise InfrastructureInvalidTrialError(
            f"Harbor trial result is incomplete; missing {field}: {path}"
        )
    try:
        datetime.fromisoformat(value)
    except ValueError as exc:
        raise InfrastructureInvalidTrialError(
            f"Harbor trial result has malformed {field}: {path}"
        ) from exc


def _extract_reward(value: Any, path: Path) -> float:
    if not isinstance(value, Mapping):
        raise InfrastructureInvalidTrialError(
            f"Harbor trial result is missing verifier_result: {path}"
        )
    rewards = value.get("rewards")
    if not isinstance(rewards, Mapping):
        raise InfrastructureInvalidTrialError(
            f"Harbor trial verifier_result is missing rewards: {path}"
        )
    if "reward" not in rewards:
        raise InfrastructureInvalidTrialError(
            f"Harbor trial verifier rewards are missing 'reward': {path}"
        )

    reward = rewards["reward"]
    if isinstance(reward, bool) or not isinstance(reward, (int, float)):
        raise InfrastructureInvalidTrialError(
            f"Harbor trial verifier reward must be numeric: {reward!r}, path={path}"
        )
    raw_reward = float(reward)
    if not math.isfinite(raw_reward) or not 0.0 <= raw_reward <= 1.0:
        raise InfrastructureInvalidTrialError(
            "Harbor trial verifier reward must be finite and within [0, 1]: "
            f"{reward!r}, path={path}"
        )
    return raw_reward


def _classify_exception(result: Mapping[str, Any], path: Path) -> str:
    if "exception_info" not in result:
        raise InfrastructureInvalidTrialError(
            f"Harbor trial result is missing exception_info: {path}"
        )
    exception_info = result["exception_info"]
    if exception_info is None:
        return "completed"
    if not isinstance(exception_info, Mapping):
        raise InfrastructureInvalidTrialError(
            f"Harbor trial exception_info is malformed: {path}"
        )
    exception_type = exception_info.get("exception_type")
    if not isinstance(exception_type, str) or not exception_type:
        raise InfrastructureInvalidTrialError(
            f"Harbor trial exception_info is missing exception_type: {path}"
        )
    if exception_type not in _SCORED_AGENT_EXCEPTIONS:
        raise InfrastructureInvalidTrialError(
            "Harbor trial contains an infrastructure-invalid or unclassified "
            f"exception {exception_type!r}: {path}"
        )
    return exception_type
