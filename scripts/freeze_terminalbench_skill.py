#!/usr/bin/env python3
"""Freeze one completed formal Terminal-Bench training skill fail closed."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from skillopt.envs.terminalbench.skill_pack import (  # noqa: E402
    SKILL_FILENAME,
    SKILL_NAME,
    is_semantically_blank,
    render_skill_artifact,
)

FROZEN_SKILL_SCHEMA = "skillopt-terminalbench-frozen-skill-v1"


class FreezeFailure(RuntimeError):
    """Raised when a training result cannot be frozen safely."""


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_object(path: Path, *, label: str) -> dict[str, Any]:
    if not path.is_file():
        raise FreezeFailure(f"missing {label}: {path}")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise FreezeFailure(f"invalid {label}: {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise FreezeFailure(f"{label} must be a JSON object: {path}")
    return value


def _load_array(path: Path, *, label: str) -> list[dict[str, Any]]:
    if not path.is_file():
        raise FreezeFailure(f"missing {label}: {path}")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise FreezeFailure(f"invalid {label}: {path}: {exc}") from exc
    if not isinstance(value, list) or any(not isinstance(item, dict) for item in value):
        raise FreezeFailure(f"{label} must be a JSON array of objects: {path}")
    return value


def _positive_int(value: Any, *, label: str) -> int:
    if isinstance(value, bool):
        raise FreezeFailure(f"{label} must be a positive integer")
    try:
        parsed = int(value)
    except (TypeError, ValueError) as exc:
        raise FreezeFailure(f"{label} must be a positive integer") from exc
    if parsed <= 0:
        raise FreezeFailure(f"{label} must be a positive integer")
    return parsed


def _finite_number(value: Any, *, label: str) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError) as exc:
        raise FreezeFailure(f"{label} must be numeric") from exc
    if not math.isfinite(parsed):
        raise FreezeFailure(f"{label} must be finite")
    return parsed


def _resolved_path(value: Any, *, label: str) -> Path:
    if not isinstance(value, str) or not value.strip():
        raise FreezeFailure(f"{label} must be a non-empty path")
    return Path(value).expanduser().resolve()


def _expected_training_steps(
    manifest: dict[str, Any],
    runtime_config: dict[str, Any],
) -> tuple[int, int, int]:
    execution = manifest.get("execution")
    dataset = manifest.get("dataset")
    if not isinstance(execution, dict) or not isinstance(dataset, dict):
        raise FreezeFailure("training manifest is missing execution or dataset state")
    frozen_config = execution.get("resolved_config")
    counts = dataset.get("counts")
    if not isinstance(frozen_config, dict) or not isinstance(counts, dict):
        raise FreezeFailure("training manifest is missing frozen config or split counts")

    num_epochs = _positive_int(frozen_config.get("num_epochs"), label="num_epochs")
    batch_size = _positive_int(frozen_config.get("batch_size"), label="batch_size")
    accumulation = _positive_int(
        frozen_config.get("accumulation"), label="accumulation"
    )
    configured_train_size = int(frozen_config.get("train_size", 0) or 0)
    split_train_size = _positive_int(counts.get("train"), label="training split count")
    train_size = configured_train_size if configured_train_size > 0 else split_train_size
    if configured_train_size > 0 and configured_train_size != split_train_size:
        raise FreezeFailure(
            "frozen train_size does not match the training split count: "
            f"config={configured_train_size}, split={split_train_size}"
        )

    steps_per_epoch = math.ceil(train_size / (batch_size * accumulation))
    expected_steps = num_epochs * steps_per_epoch
    if _positive_int(runtime_config.get("train_size"), label="runtime train_size") != train_size:
        raise FreezeFailure("runtime train_size does not match the frozen training contract")
    if (
        _positive_int(runtime_config.get("steps_per_epoch"), label="runtime steps_per_epoch")
        != steps_per_epoch
    ):
        raise FreezeFailure("runtime steps_per_epoch does not match the derived value")
    for key, expected in (
        ("num_epochs", num_epochs),
        ("batch_size", batch_size),
        ("accumulation", accumulation),
    ):
        if _positive_int(runtime_config.get(key), label=f"runtime {key}") != expected:
            raise FreezeFailure(f"runtime {key} does not match the training manifest")
    if bool(frozen_config.get("eval_test")) or bool(runtime_config.get("eval_test")):
        raise FreezeFailure("formal training must complete with evaluation.eval_test=false")
    return train_size, steps_per_epoch, expected_steps


def validate_completed_training(
    *,
    experiment_id: str,
    training_output: Path,
    training_manifest_path: Path,
    expected_skillopt_head: str | None = None,
) -> dict[str, Any]:
    training_output = training_output.expanduser().resolve()
    training_manifest_path = training_manifest_path.expanduser().resolve()
    manifest = _load_object(training_manifest_path, label="training experiment manifest")
    if manifest.get("condition") != "training":
        raise FreezeFailure("source experiment manifest condition must be training")
    if manifest.get("experiment_id") != experiment_id:
        raise FreezeFailure("source training experiment ID does not match freeze request")
    versions = manifest.get("versions")
    artifacts = manifest.get("artifacts")
    if not isinstance(versions, dict) or not isinstance(artifacts, dict):
        raise FreezeFailure("training manifest is missing versions or artifacts")
    if _resolved_path(artifacts.get("output_root"), label="training output identity") != training_output:
        raise FreezeFailure("training manifest output identity does not match training output")
    if expected_skillopt_head and versions.get("skillopt_head") != expected_skillopt_head:
        raise FreezeFailure(
            "training SkillOpt HEAD mismatch: "
            f"expected {expected_skillopt_head}, got {versions.get('skillopt_head')}"
        )

    runtime_config_path = training_output / "config.json"
    summary_path = training_output / "summary.json"
    runtime_state_path = training_output / "runtime_state.json"
    history_path = training_output / "history.json"
    best_skill_path = training_output / "best_skill.md"
    runtime_config = _load_object(runtime_config_path, label="training runtime config")
    summary = _load_object(summary_path, label="training summary")
    runtime_state = _load_object(runtime_state_path, label="training runtime state")
    history = _load_array(history_path, label="training history")
    if not best_skill_path.is_file():
        raise FreezeFailure(f"missing training best skill: {best_skill_path}")

    _, steps_per_epoch, expected_steps = _expected_training_steps(manifest, runtime_config)
    if summary.get("config") != runtime_config:
        raise FreezeFailure("training summary config does not match runtime config")
    if int(summary.get("total_steps", -1)) != expected_steps:
        raise FreezeFailure(
            f"training summary is incomplete: expected {expected_steps} steps, "
            f"got {summary.get('total_steps')}"
        )
    if len(history) != expected_steps:
        raise FreezeFailure(
            f"training history is incomplete: expected {expected_steps} steps, got {len(history)}"
        )
    expected_numbers = list(range(1, expected_steps + 1))
    actual_numbers = [item.get("step") for item in history]
    if actual_numbers != expected_numbers:
        raise FreezeFailure("training history steps are not complete and sequential")
    if int(runtime_state.get("last_completed_step", -1)) != expected_steps:
        raise FreezeFailure("runtime state does not record the final completed training step")

    for step, history_record in zip(expected_numbers, history, strict=True):
        step_record_path = training_output / "steps" / f"step_{step:04d}" / "step_record.json"
        step_record = _load_object(step_record_path, label=f"step {step} record")
        if step_record != history_record:
            raise FreezeFailure(f"step {step} record does not match training history")

    for field in ("baseline_test_hard", "baseline_test_soft", "test_hard", "test_soft"):
        if summary.get(field) is not None:
            raise FreezeFailure(f"training summary unexpectedly contains test result {field}")
    if summary.get("final_test_hard") is not None or summary.get("final_test_soft") is not None:
        raise FreezeFailure("training summary unexpectedly contains final test results")

    best_step = int(runtime_state.get("best_step", -1))
    if best_step < 0 or best_step > expected_steps:
        raise FreezeFailure("runtime best_step is outside the completed training history")
    best_score = _finite_number(runtime_state.get("best_score"), label="runtime best_score")
    best_origin = runtime_state.get("best_origin")
    current_origin = runtime_state.get("current_origin")
    if not isinstance(best_origin, str) or not best_origin.strip():
        raise FreezeFailure("runtime best_origin is missing")
    if not isinstance(current_origin, str) or not current_origin.strip():
        raise FreezeFailure("runtime current_origin is missing")
    if int(summary.get("best_step", -1)) != best_step:
        raise FreezeFailure("training summary best_step does not match runtime state")
    if _finite_number(summary.get("best_selection_hard"), label="summary best score") != best_score:
        raise FreezeFailure("training summary best score does not match runtime state")
    if summary.get("best_origin") != best_origin:
        raise FreezeFailure("training summary best_origin does not match runtime state")
    if summary.get("current_origin") != current_origin:
        raise FreezeFailure("training summary current_origin does not match runtime state")

    expected_current_path = training_output / "skills" / f"skill_v{expected_steps:04d}.md"
    current_path = _resolved_path(
        runtime_state.get("current_skill_path"), label="runtime current skill path"
    )
    if current_path != expected_current_path.resolve() or not current_path.is_file():
        raise FreezeFailure("runtime current skill does not identify the final retained skill")
    expected_best_path = training_output / "best_skill.md"
    runtime_best_path = _resolved_path(
        runtime_state.get("best_skill_path"), label="runtime best skill path"
    )
    if runtime_best_path != expected_best_path.resolve():
        raise FreezeFailure("runtime best skill path does not identify best_skill.md")
    version_path = training_output / "skills" / f"skill_v{best_step:04d}.md"
    if not version_path.is_file():
        raise FreezeFailure(f"missing best skill version: {version_path}")
    best_bytes = best_skill_path.read_bytes()
    if version_path.read_bytes() != best_bytes:
        raise FreezeFailure("best_skill.md does not match its retained skill version")

    last_record = history[-1]
    if int(last_record.get("best_step", -1)) != best_step:
        raise FreezeFailure("final history record best_step does not match runtime state")
    if _finite_number(last_record.get("best_score"), label="final history best_score") != best_score:
        raise FreezeFailure("final history best_score does not match runtime state")

    split_identity_type = str((manifest.get("dataset") or {}).get("split_identity_type") or "")
    if split_identity_type == "legacy_materialized_manifest_sha256":
        split_identity_schema = "v1"
    elif split_identity_type == "portable_semantic_sha256":
        split_identity_schema = "v2"
    else:
        raise FreezeFailure(f"unsupported source split identity type: {split_identity_type!r}")

    return {
        "manifest": manifest,
        "manifest_path": training_manifest_path,
        "runtime_config": runtime_config,
        "summary": summary,
        "runtime_state": runtime_state,
        "history": history,
        "best_bytes": best_bytes,
        "best_skill_path": best_skill_path,
        "current_skill_path": current_path,
        "expected_steps": expected_steps,
        "steps_per_epoch": steps_per_epoch,
        "best_step": best_step,
        "best_score": best_score,
        "best_origin": best_origin,
        "split_identity_schema": split_identity_schema,
        "split_identity_type": split_identity_type,
    }


def validate_frozen_skill(
    provenance_path: Path,
    *,
    expected_experiment_id: str | None = None,
    require_source: bool = True,
) -> dict[str, Any]:
    provenance_path = provenance_path.expanduser().resolve()
    provenance = _load_object(provenance_path, label="frozen skill provenance")
    if provenance.get("schema_version") != FROZEN_SKILL_SCHEMA:
        raise FreezeFailure("unsupported frozen skill provenance schema")
    if expected_experiment_id and provenance.get("experiment_id") != expected_experiment_id:
        raise FreezeFailure("frozen skill experiment ID mismatch")

    artifact_root = provenance_path.parent
    artifacts = provenance.get("artifacts")
    raw = provenance.get("raw_skill")
    native = provenance.get("native_skill")
    source = provenance.get("source_training")
    if not all(isinstance(value, dict) for value in (artifacts, raw, native, source)):
        raise FreezeFailure("frozen skill provenance is missing artifact state")
    raw_path = artifact_root / str(artifacts.get("best_skill", ""))
    if not raw_path.is_file():
        raise FreezeFailure(f"frozen raw skill is missing: {raw_path}")
    raw_bytes = raw_path.read_bytes()
    raw_sha256 = hashlib.sha256(raw_bytes).hexdigest()
    if raw_sha256 != raw.get("sha256") or len(raw_bytes) != raw.get("bytes"):
        raise FreezeFailure("frozen raw skill bytes do not match provenance")
    try:
        content = raw_bytes.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise FreezeFailure("frozen raw skill is not UTF-8") from exc
    blank = is_semantically_blank(content)
    if blank != bool(raw.get("is_blank")):
        raise FreezeFailure("frozen raw skill blank state does not match provenance")

    native_path: Path | None = None
    native_sha256: str | None = None
    if blank:
        if native.get("sha256") is not None or native.get("package_identity") != "harbor-skills-empty":
            raise FreezeFailure("blank frozen skill must preserve Harbor skills=[] semantics")
        if artifacts.get("native_skill") is not None:
            raise FreezeFailure("blank frozen skill must not declare a native artifact")
    else:
        native_relative = artifacts.get("native_skill")
        if not isinstance(native_relative, str) or not native_relative:
            raise FreezeFailure("nonblank frozen skill is missing its native artifact path")
        native_path = artifact_root / native_relative
        if not native_path.is_file():
            raise FreezeFailure(f"frozen native skill is missing: {native_path}")
        expected_native = render_skill_artifact(content)
        if native_path.read_bytes() != expected_native:
            raise FreezeFailure("frozen native skill bytes do not match deterministic packaging")
        native_sha256 = hashlib.sha256(expected_native).hexdigest()
        if native_sha256 != native.get("sha256"):
            raise FreezeFailure("frozen native skill SHA-256 does not match provenance")
        if native.get("package_identity") != f"{SKILL_NAME}/{SKILL_FILENAME}":
            raise FreezeFailure("frozen native skill package identity is invalid")

    if require_source:
        source_manifest_path = _resolved_path(
            source.get("manifest_path"), label="source training manifest path"
        )
        if not source_manifest_path.is_file() or _sha256(source_manifest_path) != source.get(
            "manifest_sha256"
        ):
            raise FreezeFailure("source training manifest no longer matches frozen provenance")

    return {
        "provenance": provenance,
        "provenance_path": provenance_path,
        "best_skill_path": raw_path.resolve(),
        "content": content,
        "is_blank": blank,
        "raw_sha256": raw_sha256,
        "native_skill_path": native_path.resolve() if native_path else None,
        "native_sha256": native_sha256,
    }


def freeze_training_skill(
    *,
    experiment_id: str,
    training_output: Path,
    training_manifest_path: Path,
    output_root: Path,
    expected_skillopt_head: str | None = None,
) -> dict[str, Any]:
    state = validate_completed_training(
        experiment_id=experiment_id,
        training_output=training_output,
        training_manifest_path=training_manifest_path,
        expected_skillopt_head=expected_skillopt_head,
    )
    output_root = output_root.expanduser().resolve()
    if output_root.exists():
        raise FreezeFailure(f"frozen skill output already exists: {output_root}")

    best_bytes = state["best_bytes"]
    try:
        content = best_bytes.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise FreezeFailure("training best_skill.md is not UTF-8") from exc
    raw_sha256 = hashlib.sha256(best_bytes).hexdigest()
    blank = is_semantically_blank(content)
    native_bytes = None if blank else render_skill_artifact(content)
    native_sha256 = hashlib.sha256(native_bytes).hexdigest() if native_bytes else None

    manifest = state["manifest"]
    dataset = manifest["dataset"]
    execution = manifest["execution"]
    cache = manifest["cache"]
    versions = manifest["versions"]
    provenance = {
        "schema_version": FROZEN_SKILL_SCHEMA,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "experiment_id": experiment_id,
        "source_training": {
            "manifest_path": str(state["manifest_path"]),
            "manifest_sha256": _sha256(state["manifest_path"]),
            "output_root": str(Path(training_output).expanduser().resolve()),
            "output_identity": str((manifest.get("artifacts") or {}).get("output_root") or ""),
            "expected_steps": state["expected_steps"],
            "completed_steps": len(state["history"]),
        },
        "versions": {
            "skillopt_head": versions.get("skillopt_head"),
            "terminalbench_head": versions.get("terminalbench_head"),
        },
        "split": {
            "split_identity_schema": state["split_identity_schema"],
            "split_identity_type": state["split_identity_type"],
            "sha256": dataset.get("split_manifest_sha256"),
            "legacy_materialized_manifest_sha256": (
                dataset.get("split_manifest_sha256")
                if state["split_identity_schema"] == "v1"
                else None
            ),
        },
        "contracts": {
            "formal_config_sha256": execution.get("config_sha256"),
            "harbor_config_sha256": execution.get("harbor_base_config_sha256"),
            "cache_manifest_sha256": cache.get("manifest_sha256"),
            "concurrency": execution.get("n_concurrent_trials"),
        },
        "selection": {
            "best_step": state["best_step"],
            "best_score": state["best_score"],
            "best_origin": state["best_origin"],
            "best_version": f"skill_v{state['best_step']:04d}",
            "final_retained_version": f"skill_v{state['expected_steps']:04d}",
            "final_retained_sha256": _sha256(state["current_skill_path"]),
        },
        "raw_skill": {
            "bytes": len(best_bytes),
            "sha256": raw_sha256,
            "is_blank": blank,
        },
        "native_skill": {
            "sha256": native_sha256,
            "package_identity": (
                "harbor-skills-empty" if blank else f"{SKILL_NAME}/{SKILL_FILENAME}"
            ),
        },
        "artifacts": {
            "best_skill": "best_skill.md",
            "native_skill": None if blank else f"{SKILL_NAME}/{SKILL_FILENAME}",
            "provenance": "skill_provenance.json",
        },
    }

    output_root.mkdir(parents=True)
    shutil.copyfile(state["best_skill_path"], output_root / "best_skill.md")
    if native_bytes is not None:
        native_path = output_root / SKILL_NAME / SKILL_FILENAME
        native_path.parent.mkdir()
        native_path.write_bytes(native_bytes)
    provenance_path = output_root / "skill_provenance.json"
    provenance_path.write_text(
        json.dumps(provenance, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return validate_frozen_skill(
        provenance_path,
        expected_experiment_id=experiment_id,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment-id", required=True)
    parser.add_argument("--training-output", type=Path, required=True)
    parser.add_argument("--training-manifest", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--expected-skillopt-head")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = freeze_training_skill(
        experiment_id=args.experiment_id,
        training_output=args.training_output,
        training_manifest_path=args.training_manifest,
        output_root=args.output_root,
        expected_skillopt_head=args.expected_skillopt_head,
    )
    print(
        json.dumps(
            {
                "status": "FROZEN",
                "experiment_id": args.experiment_id,
                "skill_is_blank": result["is_blank"],
                "best_skill": str(result["best_skill_path"]),
                "raw_sha256": result["raw_sha256"],
                "native_sha256": result["native_sha256"],
                "provenance": str(result["provenance_path"]),
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    try:
        main()
    except FreezeFailure as exc:
        print(json.dumps({"status": "BLOCKED", "error": str(exc)}), file=sys.stderr)
        raise SystemExit(2) from exc
