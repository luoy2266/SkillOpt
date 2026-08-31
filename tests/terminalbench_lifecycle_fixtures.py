from __future__ import annotations

import hashlib
import json
import math
from copy import deepcopy
from pathlib import Path
from typing import Any

from skillopt.envs.terminalbench.skill_pack import render_skill_artifact


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def training_manifest(
    *,
    experiment_id: str,
    output_root: Path,
    test_ids: list[str] | None = None,
    concurrency: int = 4,
) -> dict[str, Any]:
    test_ids = test_ids or ["test-a", "test-b", "test-c"]
    return {
        "schema_version": "skillopt-terminalbench-formal-v1",
        "experiment_id": experiment_id,
        "condition": "training",
        "created_at": "2026-08-31T00:00:00+00:00",
        "versions": {
            "skillopt_branch": "terminalbench-v2.1-delivery",
            "skillopt_head": "a" * 40,
            "terminalbench_head": "7" * 40,
            "harbor": "0.20.0",
            "terminus_2": "2.0.0",
            "docker_engine": "28.0.0",
            "python": "3.12.0",
            "openai_sdk": "1.0.0",
        },
        "dataset": {
            "checkout": "/runtime/datasets/terminal-bench-2-1",
            "tasks_path": "/runtime/datasets/terminal-bench-2-1/tasks",
            "split_dir": "/runtime/splits/tbench-v2.1-s42",
            "split_manifest_sha256": "b" * 64,
            "split_identity_type": "portable_semantic_sha256",
            "split_manifest": {"schema_version": 2, "semantic_sha256": "b" * 64},
            "counts": {"train": 3, "val": 2, "test": len(test_ids), "unique": 5 + len(test_ids)},
            "task_ids": {
                "train": ["train-a", "train-b", "train-c"],
                "val": ["val-a", "val-b"],
                "test": test_ids,
            },
        },
        "models": {
            "underlying_identity": "DeepSeek-V4-Flash-0731",
            "target": {
                "request_model": "deepseek/deepseek-v4-flash",
                "reasoning_effort": "max",
                "transport": "Harbor -> Terminus-2 -> LiteLLM",
            },
            "optimizer": {
                "backend": "openai_compatible",
                "request_model": "deepseek-v4-flash",
                "reasoning_effort": "max",
                "completion_cap": 16384,
                "endpoint": "https://api.deepseek.com",
            },
        },
        "cache": {
            "root": "/runtime/cache/terminal-bench-v2.1",
            "container_root": "/opt/skillopt-cache/terminal-bench-v2.1",
            "manifest_path": "/runtime/cache/terminal-bench-v2.1/MANIFEST.tsv",
            "manifest_sha256": "c" * 64,
            "assets": [],
        },
        "execution": {
            "config_path": "/repo/configs/terminalbench/formal.yaml",
            "config_sha256": "d" * 64,
            "source_config_text": "formal\n",
            "resolved_config": {
                "num_epochs": 2,
                "train_size": 0,
                "batch_size": 2,
                "accumulation": 1,
                "eval_test": False,
                "n_concurrent_trials": concurrency,
            },
            "cli_overrides": [],
            "harbor_base_config_path": "/runtime/harbor-configs/formal.yaml",
            "harbor_base_config_sha256": "e" * 64,
            "harbor_base_config_text": "harbor\n",
            "n_attempts": 1,
            "n_concurrent_trials": concurrency,
            "harbor_max_retries": 0,
            "timeouts": {
                "source": "Terminal-Bench task.toml",
                "target_max_turns": 250,
                "agent_seconds_by_task": {task_id: 120 for task_id in test_ids},
                "verifier_seconds_by_task": {task_id: 60 for task_id in test_ids},
                "build_seconds_by_task": {task_id: 600 for task_id in test_ids},
            },
            "docker": {},
            "resources": {},
        },
        "skill": {
            "path": "/repo/initial.md",
            "bytes": 1,
            "sha256": sha256_bytes(b"\n"),
            "native_skill_sha256": None,
            "is_blank": True,
            "training_step": None,
            "selection_score": None,
            "origin": None,
            "frozen_provenance_path": None,
            "frozen_provenance_sha256": None,
            "source_training_manifest_sha256": None,
        },
        "artifacts": {
            "output_root": str(output_root.resolve()),
            "log_path": str(output_root.parent / "training.log"),
            "manifest_path": str(output_root.parent / "training.manifest.json"),
        },
        "environment_variable_names": [],
    }


def write_completed_training_fixture(
    root: Path,
    *,
    experiment_id: str = "experiment-001",
    best_content: str = "Use careful verification.\n",
    best_step: int = 2,
) -> dict[str, Path | dict[str, Any] | int | str]:
    output = root / "training"
    output.mkdir(parents=True)
    manifest_path = root / "manifests" / "training.experiment_manifest.json"
    manifest = training_manifest(experiment_id=experiment_id, output_root=output)
    write_json(manifest_path, manifest)

    config = {
        "num_epochs": 2,
        "train_size": 3,
        "batch_size": 2,
        "accumulation": 1,
        "steps_per_epoch": 2,
        "eval_test": False,
        "n_concurrent_trials": 4,
    }
    expected_steps = config["num_epochs"] * config["steps_per_epoch"]
    blank = not best_content.strip()
    initial = "\n"
    current_content = best_content if not blank else initial
    (output / "skills").mkdir()
    for step in range(expected_steps + 1):
        content = initial if step == 0 or (not blank and step < best_step) else current_content
        (output / "skills" / f"skill_v{step:04d}.md").write_text(content, encoding="utf-8")
    (output / "best_skill.md").write_text(current_content, encoding="utf-8")

    best_score = 0.0 if blank else 0.75
    best_origin = "initial_skill" if best_step == 0 else f"step_{best_step:04d}"
    current_origin = best_origin
    history = []
    for step in range(1, expected_steps + 1):
        record_best_step = best_step if step >= max(best_step, 1) else 0
        record_best_score = best_score if step >= max(best_step, 1) else 0.0
        record = {
            "step": step,
            "epoch": math.ceil(step / config["steps_per_epoch"]),
            "action": "accept" if step == best_step and best_step > 0 else "reject",
            "best_step": record_best_step,
            "best_score": record_best_score,
            "current_score": record_best_score,
            "current_origin": best_origin if step >= max(best_step, 1) else "initial_skill",
            "best_origin": best_origin if step >= max(best_step, 1) else "initial_skill",
        }
        history.append(record)
        write_json(output / "steps" / f"step_{step:04d}" / "step_record.json", record)

    runtime_state = {
        "last_completed_step": expected_steps,
        "current_skill_path": str((output / "skills" / f"skill_v{expected_steps:04d}.md").resolve()),
        "current_score": best_score,
        "current_origin": current_origin,
        "best_skill_path": str((output / "best_skill.md").resolve()),
        "best_score": best_score,
        "best_step": best_step,
        "best_origin": best_origin,
    }
    summary = {
        "config": config,
        "best_selection_hard": best_score,
        "best_step": best_step,
        "best_origin": best_origin,
        "current_origin": current_origin,
        "total_steps": expected_steps,
        "baseline_test_hard": None,
        "baseline_test_soft": None,
        "test_hard": None,
        "test_soft": None,
        "final_test_hard": None,
        "final_test_soft": None,
    }
    write_json(output / "config.json", config)
    write_json(output / "summary.json", summary)
    write_json(output / "runtime_state.json", runtime_state)
    write_json(output / "history.json", history)
    return {
        "output": output,
        "manifest_path": manifest_path,
        "manifest": manifest,
        "expected_steps": expected_steps,
        "best_content": current_content,
    }


def evaluation_manifest(
    training: dict[str, Any],
    *,
    condition: str,
    output_root: Path,
    raw_content: str,
    native_sha256: str | None,
    provenance_path: Path | None = None,
) -> dict[str, Any]:
    manifest = deepcopy(training)
    manifest["condition"] = condition
    manifest["artifacts"]["output_root"] = str(output_root.resolve())
    raw_bytes = raw_content.encode("utf-8")
    provenance = (
        json.loads(provenance_path.read_text(encoding="utf-8"))
        if provenance_path
        else None
    )
    manifest["skill"] = {
        "path": str((provenance_path.parent / "best_skill.md").resolve()) if provenance_path else "/repo/initial.md",
        "bytes": len(raw_bytes),
        "sha256": sha256_bytes(raw_bytes),
        "native_skill_sha256": native_sha256,
        "is_blank": not raw_content.strip(),
        "training_step": 2 if provenance_path else None,
        "selection_score": 0.75 if provenance_path else None,
        "origin": "step_0002" if provenance_path else None,
        "frozen_provenance_path": str(provenance_path.resolve()) if provenance_path else None,
        "frozen_provenance_sha256": (
            sha256_bytes(provenance_path.read_bytes()) if provenance_path else None
        ),
        "source_training_manifest_sha256": (
            provenance["source_training"]["manifest_sha256"] if provenance else None
        ),
    }
    return manifest


def write_evaluation_fixture(
    output_root: Path,
    *,
    task_ids: list[str],
    rewards: list[float],
    raw_skill: str,
    native_sha256: str | None,
    statuses: list[str] | None = None,
) -> None:
    statuses = statuses or ["completed"] * len(task_ids)
    results = []
    for task_id, reward, status in zip(task_ids, rewards, statuses, strict=True):
        results.append(
            {
                "id": task_id,
                "hard": 1.0 if reward == 1.0 else 0.0,
                "soft": reward,
                "raw_reward": reward,
                "trial_status": status,
                "harbor_result_path": f"/harbor/{task_id}/result.json",
                "harbor_config_path": "/harbor/config.yaml",
                "harbor_job_dir": "/harbor/job",
                "skill_sha256": native_sha256,
            }
        )
    write_json(
        output_root / "eval_results.json",
        {
            "schema_version": "skillopt-terminalbench-eval-results-v1",
            "split": "valid_unseen",
            "n_items": len(results),
            "skill": {"path": "/skill.md", "raw_sha256": sha256_bytes(raw_skill.encode("utf-8"))},
            "results": results,
        },
    )
    write_json(
        output_root / "eval_summary.json",
        {
            "skill": "/skill.md",
            "split": "valid_unseen",
            "n_items": len(results),
            "hard": sum(item["hard"] for item in results) / len(results),
            "soft": sum(rewards) / len(rewards),
        },
    )


def native_sha256(content: str) -> str | None:
    if not content.strip():
        return None
    return sha256_bytes(render_skill_artifact(content))
