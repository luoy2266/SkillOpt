"""Convert pinned Terminus-2 ATIF artifacts for SkillOpt reflection."""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from typing import Any


EXPECTED_ATIF_VERSION = "ATIF-v1.7"
EXPECTED_AGENT_NAME = "terminus-2"
EXPECTED_AGENT_VERSION = "2.0.0"


class TrajectoryConversionError(ValueError):
    """Raised when an ATIF or conversation artifact violates the contract."""


def conversation_output_path(
    rollout_dir: str | os.PathLike[str],
    task_id: str,
) -> Path:
    """Return SkillOpt's required conversation path for one rollout task."""
    safe_task_id = _validate_task_id(task_id)
    return (
        Path(rollout_dir).expanduser().resolve()
        / "predictions"
        / safe_task_id
        / "conversation.json"
    )


def convert_atif_trajectory(
    atif_path: str | os.PathLike[str],
    *,
    expected_task_id: str,
    output_path: str | os.PathLike[str],
) -> list[dict[str, Any]]:
    """Convert one explicit Terminus-2 trajectory and write conversation JSON."""
    task_id = _validate_task_id(expected_task_id)
    source_path = _trajectory_path(atif_path)
    destination = _output_path(output_path, task_id)
    conversation: list[dict[str, Any]] = []
    has_user_context = False
    has_agent_execution = False

    for segment_index, trajectory in enumerate(_load_trajectory_chain(source_path)):
        for expected_step_id, step in enumerate(trajectory["steps"], 1):
            records, user_context, agent_execution = _convert_step(
                step,
                expected_step_id,
            )
            if segment_index > 0 and step.get("is_copied_context") is True:
                continue
            conversation.extend(records)
            has_user_context = has_user_context or user_context
            has_agent_execution = has_agent_execution or agent_execution

    if not conversation:
        raise TrajectoryConversionError("ATIF trajectory has no usable conversation records")
    if not has_user_context:
        raise TrajectoryConversionError("ATIF trajectory has no usable user/task context")
    if not has_agent_execution:
        raise TrajectoryConversionError("ATIF trajectory has no usable agent execution")

    _atomic_write(destination, conversation)
    return conversation


def _load_trajectory_chain(initial_path: Path) -> list[dict[str, Any]]:
    trajectories = []
    seen: set[Path] = set()
    current_path = initial_path
    while True:
        resolved_path = current_path.resolve()
        if resolved_path in seen:
            raise TrajectoryConversionError(
                f"ATIF continuation chain contains a cycle: {current_path}"
            )
        seen.add(resolved_path)
        trajectory = _load_trajectory(current_path)
        trajectories.append(trajectory)

        continuation = trajectory.get("continued_trajectory_ref")
        if continuation is None:
            return trajectories
        if not isinstance(continuation, str) or not continuation:
            raise TrajectoryConversionError(
                f"ATIF continuation must be a non-empty filename: {current_path}"
            )
        continuation_path = Path(continuation)
        if continuation_path.is_absolute() or continuation_path.name != continuation:
            raise TrajectoryConversionError(
                f"ATIF continuation must stay in the agent directory: {continuation!r}"
            )
        current_path = current_path.parent / continuation_path


def _load_trajectory(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise TrajectoryConversionError(f"ATIF trajectory artifact is missing: {path}")
    try:
        with path.open(encoding="utf-8") as handle:
            trajectory = json.load(handle, parse_constant=_reject_json_constant)
    except (OSError, UnicodeError, json.JSONDecodeError, ValueError) as exc:
        raise TrajectoryConversionError(
            f"ATIF trajectory artifact is unreadable or malformed: {path}"
        ) from exc
    if not isinstance(trajectory, dict):
        raise TrajectoryConversionError(f"ATIF trajectory must be a top-level object: {path}")
    if trajectory.get("schema_version") != EXPECTED_ATIF_VERSION:
        raise TrajectoryConversionError(
            f"Expected {EXPECTED_ATIF_VERSION}, found "
            f"{trajectory.get('schema_version')!r}: {path}"
        )

    agent = trajectory.get("agent")
    if not isinstance(agent, dict):
        raise TrajectoryConversionError("ATIF agent must be an object")
    if agent.get("name") != EXPECTED_AGENT_NAME:
        raise TrajectoryConversionError(
            f"Expected ATIF agent {EXPECTED_AGENT_NAME!r}, found {agent.get('name')!r}"
        )
    if agent.get("version") != EXPECTED_AGENT_VERSION:
        raise TrajectoryConversionError(
            f"Expected Terminus-2 {EXPECTED_AGENT_VERSION}, "
            f"found {agent.get('version')!r}"
        )

    steps = trajectory.get("steps")
    if not isinstance(steps, list) or not steps:
        raise TrajectoryConversionError("ATIF steps must be a non-empty list")
    if trajectory.get("subagent_trajectories") not in (None, []):
        raise TrajectoryConversionError(
            "Embedded subagent trajectories are not emitted by pinned Terminus-2"
        )
    return trajectory


def _convert_step(
    step: Any,
    expected_step_id: int,
) -> tuple[list[dict[str, Any]], bool, bool]:
    if not isinstance(step, dict):
        raise TrajectoryConversionError(f"ATIF step {expected_step_id} must be an object")
    step_id = step.get("step_id")
    if isinstance(step_id, bool) or step_id != expected_step_id:
        raise TrajectoryConversionError(
            f"ATIF step_id must be sequential: expected {expected_step_id}, "
            f"found {step_id!r}"
        )
    source = step.get("source")
    if source not in {"system", "user", "agent"}:
        raise TrajectoryConversionError(f"ATIF step {expected_step_id} has invalid source")
    if "message" not in step:
        raise TrajectoryConversionError(f"ATIF step {expected_step_id} is missing message")

    message = _content_text(step["message"], f"ATIF step {expected_step_id} message")
    reasoning = step.get("reasoning_content")
    if reasoning is None:
        reasoning = ""
    elif not isinstance(reasoning, str):
        raise TrajectoryConversionError("ATIF reasoning_content must be text or null")
    action, tool_call_ids = _tool_action(step.get("tool_calls"), expected_step_id)
    feedback = _observation_text(
        step.get("observation"),
        tool_call_ids,
        expected_step_id,
    )

    if source != "agent" and any(
        step.get(field) is not None
        for field in (
            "model_name",
            "reasoning_effort",
            "reasoning_content",
            "tool_calls",
            "metrics",
        )
    ):
        raise TrajectoryConversionError("ATIF agent-only fields require source='agent'")

    records: list[dict[str, Any]] = []
    if source == "user":
        if message.strip():
            records.append({"role": "user", "content": message})
        if feedback:
            records.append(_step_record(step_id, "", "", feedback))
        return records, bool(message.strip()), False

    if source == "system":
        if message.strip():
            records.append({"role": "system", "content": message})
        if feedback:
            records.append(_step_record(step_id, "", "", feedback))
        return records, False, False

    if message.strip():
        records.append({"role": "assistant", "content": message})
    if reasoning or action or feedback:
        records.append(_step_record(step_id, reasoning, action, feedback))
    return records, False, bool(records)


def _tool_action(value: Any, step_id: int) -> tuple[str, set[str]]:
    if value is None:
        return "", set()
    if not isinstance(value, list):
        raise TrajectoryConversionError(f"ATIF step {step_id} tool_calls must be a list")

    actions = []
    identifiers: set[str] = set()
    for index, tool_call in enumerate(value, 1):
        if not isinstance(tool_call, dict):
            raise TrajectoryConversionError(
                f"ATIF step {step_id} tool call {index} must be an object"
            )
        tool_call_id = tool_call.get("tool_call_id")
        function_name = tool_call.get("function_name")
        arguments = tool_call.get("arguments")
        if not isinstance(tool_call_id, str) or not tool_call_id:
            raise TrajectoryConversionError("ATIF tool_call_id must be non-empty text")
        if tool_call_id in identifiers:
            raise TrajectoryConversionError(f"Duplicate ATIF tool_call_id: {tool_call_id}")
        if not isinstance(function_name, str) or not function_name:
            raise TrajectoryConversionError("ATIF function_name must be non-empty text")
        if not isinstance(arguments, dict):
            raise TrajectoryConversionError("ATIF tool call arguments must be an object")
        identifiers.add(tool_call_id)
        serialized = json.dumps(
            arguments,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        actions.append(f"{function_name}({serialized})")
    return "\n".join(actions), identifiers


def _observation_text(value: Any, tool_call_ids: set[str], step_id: int) -> str:
    if value is None:
        return ""
    if not isinstance(value, dict) or not isinstance(value.get("results"), list):
        raise TrajectoryConversionError(
            f"ATIF step {step_id} observation must contain a results list"
        )

    rendered: list[tuple[str | None, str]] = []
    for index, result in enumerate(value["results"], 1):
        if not isinstance(result, dict):
            raise TrajectoryConversionError(
                f"ATIF step {step_id} observation result {index} must be an object"
            )
        source_call_id = result.get("source_call_id")
        if source_call_id is not None and (
            not isinstance(source_call_id, str) or source_call_id not in tool_call_ids
        ):
            raise TrajectoryConversionError(
                f"ATIF observation references unknown tool call {source_call_id!r}"
            )
        content = result.get("content")
        if content is not None:
            text = _content_text(content, "ATIF observation content")
            if text:
                rendered.append((source_call_id, text))

    if len(rendered) == 1:
        return rendered[0][1]
    return "\n\n".join(
        f"[observation {source_call_id or index}]\n{text}"
        for index, (source_call_id, text) in enumerate(rendered, 1)
    )


def _content_text(value: Any, label: str) -> str:
    if isinstance(value, str):
        return value
    if not isinstance(value, list):
        raise TrajectoryConversionError(f"{label} must be text or a ContentPart list")

    rendered = []
    for index, part in enumerate(value, 1):
        if not isinstance(part, dict):
            raise TrajectoryConversionError(f"{label} part {index} must be an object")
        if part.get("type") == "text":
            text = part.get("text")
            if not isinstance(text, str) or part.get("source") is not None:
                raise TrajectoryConversionError(f"{label} has malformed text content")
            rendered.append(text)
            continue
        if part.get("type") == "image":
            source = part.get("source")
            if not isinstance(source, dict) or part.get("text") is not None:
                raise TrajectoryConversionError(f"{label} has malformed image content")
            media_type = source.get("media_type")
            image_path = source.get("path")
            if media_type not in {
                "image/jpeg",
                "image/png",
                "image/gif",
                "image/webp",
            } or not isinstance(image_path, str):
                raise TrajectoryConversionError(f"{label} has malformed image source")
            rendered.append(f"[IMAGE {media_type} {image_path}]")
            continue
        raise TrajectoryConversionError(f"{label} has unsupported content type")
    return "\n".join(rendered)


def _step_record(step: int, reasoning: str, action: str, feedback: str) -> dict[str, Any]:
    return {
        "step": step,
        "reasoning": reasoning,
        "action": action,
        "env_feedback": feedback,
    }


def _atomic_write(path: Path, conversation: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.is_symlink():
        raise TrajectoryConversionError(f"Conversation destination must not be a symlink: {path}")
    if path.exists():
        _reuse_or_raise(path, conversation)
        return

    descriptor, temporary_name = tempfile.mkstemp(
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".tmp",
    )
    temporary_path = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as handle:
            json.dump(conversation, handle, ensure_ascii=False, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        try:
            os.link(temporary_path, path)
        except FileExistsError:
            _reuse_or_raise(path, conversation)
        except OSError as exc:
            raise TrajectoryConversionError(f"Unable to atomically write: {path}") from exc
    finally:
        temporary_path.unlink(missing_ok=True)


def _reuse_or_raise(path: Path, conversation: list[dict[str, Any]]) -> None:
    if path.is_symlink() or not path.is_file():
        raise TrajectoryConversionError(f"Conversation destination is not a file: {path}")
    try:
        with path.open(encoding="utf-8") as handle:
            existing = json.load(handle, parse_constant=_reject_json_constant)
    except (OSError, UnicodeError, json.JSONDecodeError, ValueError) as exc:
        raise TrajectoryConversionError(
            f"Existing conversation artifact is unreadable or malformed: {path}"
        ) from exc
    if existing != conversation:
        raise TrajectoryConversionError(f"Existing conversation artifact conflicts: {path}")


def _trajectory_path(value: str | os.PathLike[str]) -> Path:
    path = Path(value).expanduser().resolve()
    if path.name != "trajectory.json":
        raise TrajectoryConversionError(
            f"Expected an explicit primary agent/trajectory.json path: {path}"
        )
    return path


def _output_path(value: str | os.PathLike[str], task_id: str) -> Path:
    path = Path(os.path.abspath(Path(value).expanduser()))
    if (
        path.name != "conversation.json"
        or path.parent.name != task_id
        or path.parent.parent.name != "predictions"
    ):
        raise TrajectoryConversionError(
            "Conversation output must be predictions/<expected-task-id>/conversation.json"
        )
    return path


def _validate_task_id(value: str) -> str:
    if not isinstance(value, str) or not value or value != value.strip():
        raise TrajectoryConversionError("expected_task_id must be non-empty text")
    if value in {".", ".."} or Path(value).name != value:
        raise TrajectoryConversionError("expected_task_id must not contain path traversal")
    return value


def _reject_json_constant(value: str) -> None:
    raise ValueError(f"Invalid JSON numeric constant: {value}")
