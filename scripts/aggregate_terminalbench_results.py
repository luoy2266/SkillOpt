#!/usr/bin/env python3
"""Fail-closed paired aggregate for formal Terminal-Bench evaluations."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.freeze_terminalbench_skill import (  # noqa: E402
    FreezeFailure,
    validate_frozen_skill,
)

EVAL_RESULTS_SCHEMA = "skillopt-terminalbench-eval-results-v1"
AGGREGATE_SCHEMA = "skillopt-terminalbench-aggregate-v1"
VALID_TRIAL_STATUSES = {
    "completed",
    "AgentTimeoutError",
    "NonZeroAgentExitCodeError",
}


class AggregateFailure(RuntimeError):
    """Raised when formal evaluation artifacts cannot be paired fairly."""


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_object(path: Path, *, label: str) -> dict[str, Any]:
    if not path.is_file():
        raise AggregateFailure(f"missing {label}: {path}")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise AggregateFailure(f"invalid {label}: {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise AggregateFailure(f"{label} must be a JSON object: {path}")
    return value


def _finite_reward(value: Any, *, label: str) -> float:
    try:
        reward = float(value)
    except (TypeError, ValueError) as exc:
        raise AggregateFailure(f"{label} must be numeric") from exc
    if not math.isfinite(reward) or not 0.0 <= reward <= 1.0:
        raise AggregateFailure(f"{label} must be finite and within [0, 1]")
    return reward


def _manifest_identity(manifest: dict[str, Any]) -> dict[str, Any]:
    versions = manifest.get("versions")
    dataset = manifest.get("dataset")
    execution = manifest.get("execution")
    cache = manifest.get("cache")
    models = manifest.get("models")
    if not all(isinstance(value, dict) for value in (versions, dataset, execution, cache, models)):
        raise AggregateFailure("experiment manifest is missing formal identity sections")
    task_ids = dataset.get("task_ids")
    if not isinstance(task_ids, dict) or not isinstance(task_ids.get("test"), list):
        raise AggregateFailure("experiment manifest is missing frozen test task IDs")
    return {
        "skillopt_head": versions.get("skillopt_head"),
        "terminalbench_head": versions.get("terminalbench_head"),
        "split_identity_type": dataset.get("split_identity_type"),
        "split_sha256": dataset.get("split_manifest_sha256"),
        "test_task_ids": task_ids.get("test"),
        "config_sha256": execution.get("config_sha256"),
        "harbor_config_sha256": execution.get("harbor_base_config_sha256"),
        "cache_manifest_sha256": cache.get("manifest_sha256"),
        "concurrency": execution.get("n_concurrent_trials"),
        "target_model": (models.get("target") or {}).get("request_model"),
        "underlying_model": models.get("underlying_identity"),
        "reasoning_effort": (models.get("target") or {}).get("reasoning_effort"),
        "n_attempts": execution.get("n_attempts"),
        "max_retries": execution.get("harbor_max_retries"),
        "timeouts": execution.get("timeouts"),
    }


def _validate_parity(
    baseline_manifest: dict[str, Any],
    skill_manifest: dict[str, Any],
    *,
    experiment_id: str,
) -> dict[str, Any]:
    for manifest, condition in (
        (baseline_manifest, "baseline-test"),
        (skill_manifest, "skill-test"),
    ):
        if manifest.get("experiment_id") != experiment_id:
            raise AggregateFailure(f"{condition} experiment ID mismatch")
        if manifest.get("condition") != condition:
            raise AggregateFailure(f"expected {condition} experiment manifest")
    baseline_identity = _manifest_identity(baseline_manifest)
    skill_identity = _manifest_identity(skill_manifest)
    for key in baseline_identity:
        if baseline_identity[key] != skill_identity[key]:
            raise AggregateFailure(f"baseline/skill parity mismatch: {key}")
    if baseline_identity["n_attempts"] != 1 or baseline_identity["max_retries"] != 0:
        raise AggregateFailure("formal evaluations must preserve attempts=1 and retries=0")
    if len(set(baseline_identity["test_task_ids"])) != len(
        baseline_identity["test_task_ids"]
    ):
        raise AggregateFailure("frozen test task IDs contain duplicates")
    return baseline_identity


def _load_evaluation(
    output_root: Path,
    *,
    label: str,
    expected_task_ids: list[str],
) -> tuple[dict[str, Any], dict[str, Any], list[dict[str, Any]]]:
    summary = _load_object(output_root / "eval_summary.json", label=f"{label} summary")
    payload = _load_object(output_root / "eval_results.json", label=f"{label} results")
    if payload.get("schema_version") != EVAL_RESULTS_SCHEMA:
        raise AggregateFailure(f"unsupported {label} results schema")
    if payload.get("split") != "valid_unseen":
        raise AggregateFailure(f"{label} did not evaluate the held-out test split")
    results = payload.get("results")
    if not isinstance(results, list) or any(not isinstance(item, dict) for item in results):
        raise AggregateFailure(f"{label} results must be an array of objects")
    expected_count = len(expected_task_ids)
    if summary.get("n_items") != expected_count or payload.get("n_items") != expected_count:
        raise AggregateFailure(
            f"{label} evaluation is incomplete: expected {expected_count} items"
        )
    actual_ids = [item.get("id") for item in results]
    if actual_ids != expected_task_ids:
        raise AggregateFailure(f"{label} task IDs do not match the frozen test split")

    for index, item in enumerate(results):
        status = item.get("trial_status")
        if status not in VALID_TRIAL_STATUSES:
            raise AggregateFailure(
                f"{label} task {item.get('id')!r} has infrastructure-invalid "
                f"or unclassified status {status!r}"
            )
        raw_reward = _finite_reward(item.get("raw_reward"), label=f"{label} raw reward {index}")
        soft = _finite_reward(item.get("soft"), label=f"{label} soft reward {index}")
        hard = _finite_reward(item.get("hard"), label=f"{label} hard reward {index}")
        if soft != raw_reward or hard not in {0.0, 1.0}:
            raise AggregateFailure(f"{label} task {item.get('id')!r} has invalid verifier scores")
    mean_hard = sum(float(item["hard"]) for item in results) / expected_count
    mean_soft = sum(float(item["raw_reward"]) for item in results) / expected_count
    if not math.isclose(float(summary.get("hard")), mean_hard, rel_tol=0.0, abs_tol=1e-12):
        raise AggregateFailure(f"{label} summary hard score does not match task results")
    if not math.isclose(float(summary.get("soft")), mean_soft, rel_tol=0.0, abs_tol=1e-12):
        raise AggregateFailure(f"{label} summary soft score does not match task results")
    return summary, payload, results


def aggregate_results(
    *,
    experiment_id: str,
    baseline_output: Path,
    baseline_manifest_path: Path,
    skill_output: Path,
    skill_manifest_path: Path,
    skill_provenance_path: Path,
    output_root: Path,
) -> dict[str, Any]:
    baseline_output = baseline_output.expanduser().resolve()
    skill_output = skill_output.expanduser().resolve()
    baseline_manifest_path = baseline_manifest_path.expanduser().resolve()
    skill_manifest_path = skill_manifest_path.expanduser().resolve()
    output_root = output_root.expanduser().resolve()
    if output_root.exists():
        raise AggregateFailure(f"aggregate output already exists: {output_root}")

    baseline_manifest = _load_object(
        baseline_manifest_path, label="baseline experiment manifest"
    )
    skill_manifest = _load_object(skill_manifest_path, label="skill experiment manifest")
    if Path(baseline_manifest.get("artifacts", {}).get("output_root", "")).expanduser().resolve() != baseline_output:
        raise AggregateFailure("baseline manifest output identity mismatch")
    if Path(skill_manifest.get("artifacts", {}).get("output_root", "")).expanduser().resolve() != skill_output:
        raise AggregateFailure("skill manifest output identity mismatch")
    identity = _validate_parity(
        baseline_manifest,
        skill_manifest,
        experiment_id=experiment_id,
    )
    expected_task_ids = [str(task_id) for task_id in identity["test_task_ids"]]
    baseline_summary, baseline_payload, baseline_results = _load_evaluation(
        baseline_output,
        label="baseline",
        expected_task_ids=expected_task_ids,
    )
    skill_summary, skill_payload, skill_results = _load_evaluation(
        skill_output,
        label="skill",
        expected_task_ids=expected_task_ids,
    )

    baseline_skill = baseline_manifest.get("skill")
    skill_state = skill_manifest.get("skill")
    if not isinstance(baseline_skill, dict) or not bool(baseline_skill.get("is_blank")):
        raise AggregateFailure("baseline manifest must identify a blank skill")
    if baseline_skill.get("native_skill_sha256") is not None:
        raise AggregateFailure("baseline manifest must preserve Harbor skills=[] semantics")
    try:
        frozen = validate_frozen_skill(
            skill_provenance_path,
            expected_experiment_id=experiment_id,
        )
    except FreezeFailure as exc:
        raise AggregateFailure(str(exc)) from exc
    if not isinstance(skill_state, dict):
        raise AggregateFailure("skill manifest is missing skill provenance")
    if skill_state.get("sha256") != frozen["raw_sha256"]:
        raise AggregateFailure("skill manifest raw SHA-256 does not match frozen skill")
    if skill_state.get("native_skill_sha256") != frozen["native_sha256"]:
        raise AggregateFailure("skill manifest native SHA-256 does not match frozen skill")
    if bool(skill_state.get("is_blank")) != frozen["is_blank"]:
        raise AggregateFailure("skill manifest blank state does not match frozen skill")
    if Path(str(skill_state.get("frozen_provenance_path") or "")).expanduser().resolve() != frozen[
        "provenance_path"
    ]:
        raise AggregateFailure("skill manifest provenance path does not match frozen skill")
    if skill_state.get("frozen_provenance_sha256") != _sha256(frozen["provenance_path"]):
        raise AggregateFailure("skill manifest provenance SHA-256 does not match frozen skill")
    if skill_state.get("source_training_manifest_sha256") != frozen["provenance"][
        "source_training"
    ]["manifest_sha256"]:
        raise AggregateFailure("skill manifest source training provenance mismatch")
    if baseline_payload.get("skill", {}).get("raw_sha256") != baseline_skill.get("sha256"):
        raise AggregateFailure("baseline result skill SHA-256 does not match its manifest")
    if skill_payload.get("skill", {}).get("raw_sha256") != frozen["raw_sha256"]:
        raise AggregateFailure("skill result SHA-256 does not match frozen skill")
    if any(item.get("skill_sha256") is not None for item in baseline_results):
        raise AggregateFailure("baseline results unexpectedly contain a native skill SHA-256")
    if any(item.get("skill_sha256") != frozen["native_sha256"] for item in skill_results):
        raise AggregateFailure("skill result native SHA-256 does not match frozen skill")

    pairs = []
    wins = ties = losses = 0
    status_counts = {
        "baseline": {status: 0 for status in sorted(VALID_TRIAL_STATUSES)},
        "skill": {status: 0 for status in sorted(VALID_TRIAL_STATUSES)},
    }
    for task_id, baseline_item, skill_item in zip(
        expected_task_ids, baseline_results, skill_results, strict=True
    ):
        baseline_reward = float(baseline_item["raw_reward"])
        skill_reward = float(skill_item["raw_reward"])
        delta = skill_reward - baseline_reward
        if delta > 0:
            wins += 1
        elif delta < 0:
            losses += 1
        else:
            ties += 1
        status_counts["baseline"][baseline_item["trial_status"]] += 1
        status_counts["skill"][skill_item["trial_status"]] += 1
        pairs.append(
            {
                "task_id": task_id,
                "baseline_reward": baseline_reward,
                "skill_reward": skill_reward,
                "delta": delta,
                "baseline_status": baseline_item["trial_status"],
                "skill_status": skill_item["trial_status"],
            }
        )

    baseline_score = float(baseline_summary["soft"])
    skill_score = float(skill_summary["soft"])
    absolute_delta = skill_score - baseline_score
    relative_delta = None if baseline_score == 0 else absolute_delta / baseline_score
    summary = {
        "schema_version": AGGREGATE_SCHEMA,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "experiment_id": experiment_id,
        "status": "COMPLETE",
        "n_items": len(expected_task_ids),
        "scores": {
            "baseline_raw": baseline_score,
            "skill_raw": skill_score,
            "absolute_delta": absolute_delta,
            "relative_delta": relative_delta,
            "wins": wins,
            "ties": ties,
            "losses": losses,
        },
        "failure_counts": status_counts,
        "identity": identity,
        "skill": {
            "raw_sha256": frozen["raw_sha256"],
            "native_sha256": frozen["native_sha256"],
            "is_blank": frozen["is_blank"],
            "provenance_path": str(frozen["provenance_path"]),
            "provenance_sha256": _sha256(frozen["provenance_path"]),
        },
        "inputs": {
            "baseline_manifest": str(baseline_manifest_path),
            "baseline_manifest_sha256": _sha256(baseline_manifest_path),
            "skill_manifest": str(skill_manifest_path),
            "skill_manifest_sha256": _sha256(skill_manifest_path),
            "baseline_results": str(baseline_output / "eval_results.json"),
            "skill_results": str(skill_output / "eval_results.json"),
        },
        "paired_results": pairs,
    }

    output_root.mkdir(parents=True)
    summary_path = output_root / "summary.json"
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    with (output_root / "results.tsv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=(
                "task_id",
                "baseline_reward",
                "skill_reward",
                "delta",
                "baseline_status",
                "skill_status",
            ),
            delimiter="\t",
        )
        writer.writeheader()
        writer.writerows(pairs)
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment-id", required=True)
    parser.add_argument("--baseline-output", type=Path, required=True)
    parser.add_argument("--baseline-manifest", type=Path, required=True)
    parser.add_argument("--skill-output", type=Path, required=True)
    parser.add_argument("--skill-manifest", type=Path, required=True)
    parser.add_argument("--skill-provenance", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summary = aggregate_results(
        experiment_id=args.experiment_id,
        baseline_output=args.baseline_output,
        baseline_manifest_path=args.baseline_manifest,
        skill_output=args.skill_output,
        skill_manifest_path=args.skill_manifest,
        skill_provenance_path=args.skill_provenance,
        output_root=args.output_root,
    )
    print(
        json.dumps(
            {
                "status": summary["status"],
                "experiment_id": args.experiment_id,
                "n_items": summary["n_items"],
                "summary": str(args.output_root.expanduser().resolve() / "summary.json"),
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    try:
        main()
    except AggregateFailure as exc:
        print(json.dumps({"status": "BLOCKED", "error": str(exc)}), file=sys.stderr)
        raise SystemExit(2) from exc
