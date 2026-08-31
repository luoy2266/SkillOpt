#!/usr/bin/env python3
"""Offline clean-clone acceptance for the Terminal-Bench server lifecycle."""

from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

import yaml

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.aggregate_terminalbench_results import aggregate_results  # noqa: E402
from scripts.freeze_terminalbench_skill import freeze_training_skill  # noqa: E402
from scripts.terminalbench_formal_identity import systemd_unit_name  # noqa: E402
from tests.terminalbench_lifecycle_fixtures import (  # noqa: E402
    evaluation_manifest,
    write_completed_training_fixture,
    write_evaluation_fixture,
    write_json,
)

EXPECTED_SPLIT_SHA256 = "bd36fe2f37a67cd2b46149263522d833166d3a4d036c8e9af082e742ad017500"


def main() -> None:
    with tempfile.TemporaryDirectory(prefix="skillopt-tbench-handoff-") as directory:
        root = Path(directory)
        runtime_root = root / "runtime"
        environment = dict(os.environ)
        environment.update(
            {
                "SKILLOPT_PYTHON": sys.executable,
                "SKILLOPT_RUNTIME_ROOT": str(runtime_root),
                "SKILLOPT_FORMAL_EXPERIMENT_ID": "acceptance-exp-001",
                "SKILLOPT_TBENCH_CONCURRENCY": "4",
                "TERMINALBENCH_ROOT": str(runtime_root / "datasets" / "terminal-bench-2-1"),
            }
        )
        for name in (
            "HTTP_PROXY",
            "HTTPS_PROXY",
            "http_proxy",
            "https_proxy",
            "NO_PROXY",
            "no_proxy",
        ):
            environment.pop(name, None)
        subprocess.run(
            [str(PROJECT_ROOT / "scripts" / "bootstrap_terminalbench_server.sh"), "init"],
            cwd=PROJECT_ROOT,
            env=environment,
            check=True,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )

        split_manifest = json.loads(
            (
                runtime_root
                / "splits"
                / "tbench-v2.1-s42"
                / "split_manifest.json"
            ).read_text(encoding="utf-8")
        )
        if split_manifest.get("semantic_sha256") != EXPECTED_SPLIT_SHA256:
            raise RuntimeError("portable split semantic identity mismatch")
        harbor_config = yaml.safe_load(
            (
                runtime_root
                / "harbor-configs"
                / "tbench-v2.1-formal.yaml"
            ).read_text(encoding="utf-8")
        )
        if harbor_config.get("n_concurrent_trials") != 4:
            raise RuntimeError("Harbor concurrency did not propagate")
        if "HTTP_PROXY" in harbor_config.get("environment", {}).get("env", {}):
            raise RuntimeError("direct-network acceptance unexpectedly rendered a proxy")

        schema = json.loads(
            (
                PROJECT_ROOT
                / "configs"
                / "terminalbench"
                / "experiment_manifest.schema.json"
            ).read_text(encoding="utf-8")
        )
        concurrency_contract = schema["properties"]["execution"]["properties"][
            "n_concurrent_trials"
        ]
        if concurrency_contract.get("minimum") != 1:
            raise RuntimeError("manifest concurrency schema is not positive-integer based")
        units = {
            stage: systemd_unit_name("acceptance-exp-001", stage)
            for stage in ("training", "freeze-skill", "baseline-test", "skill-test", "aggregate")
        }
        if len(set(units.values())) != len(units):
            raise RuntimeError("formal lifecycle stages do not have distinct systemd identities")

        fixture = write_completed_training_fixture(
            root / "fixture",
            experiment_id="acceptance-exp-001",
        )
        frozen_root = runtime_root / "skills" / "acceptance-exp-001"
        frozen = freeze_training_skill(
            experiment_id="acceptance-exp-001",
            training_output=fixture["output"],
            training_manifest_path=fixture["manifest_path"],
            output_root=frozen_root,
            expected_skillopt_head="a" * 40,
        )
        blank_fixture = write_completed_training_fixture(
            root / "blank-fixture",
            experiment_id="acceptance-blank-001",
            best_content="\n",
            best_step=0,
        )
        blank = freeze_training_skill(
            experiment_id="acceptance-blank-001",
            training_output=blank_fixture["output"],
            training_manifest_path=blank_fixture["manifest_path"],
            output_root=runtime_root / "skills" / "acceptance-blank-001",
            expected_skillopt_head="a" * 40,
        )
        if not blank["is_blank"] or blank["native_sha256"] is not None:
            raise RuntimeError("blank frozen skill did not preserve Harbor skills=[] semantics")

        formal_root = runtime_root / "outputs" / "formal" / "acceptance-exp-001"
        baseline_output = formal_root / "baseline-test"
        skill_output = formal_root / "skill-test"
        manifests = formal_root / "manifests"
        task_ids = fixture["manifest"]["dataset"]["task_ids"]["test"]
        baseline_manifest = evaluation_manifest(
            fixture["manifest"],
            condition="baseline-test",
            output_root=baseline_output,
            raw_content="\n",
            native_sha256=None,
        )
        skill_manifest = evaluation_manifest(
            fixture["manifest"],
            condition="skill-test",
            output_root=skill_output,
            raw_content=frozen["content"],
            native_sha256=frozen["native_sha256"],
            provenance_path=frozen_root / "skill_provenance.json",
        )
        baseline_manifest_path = manifests / "baseline-test.experiment_manifest.json"
        skill_manifest_path = manifests / "skill-test.experiment_manifest.json"
        write_json(baseline_manifest_path, baseline_manifest)
        write_json(skill_manifest_path, skill_manifest)
        write_evaluation_fixture(
            baseline_output,
            task_ids=task_ids,
            rewards=[0.0, 0.5, 1.0],
            raw_skill="\n",
            native_sha256=None,
        )
        write_evaluation_fixture(
            skill_output,
            task_ids=task_ids,
            rewards=[1.0, 0.5, 1.0],
            raw_skill=frozen["content"],
            native_sha256=frozen["native_sha256"],
        )
        aggregate = aggregate_results(
            experiment_id="acceptance-exp-001",
            baseline_output=baseline_output,
            baseline_manifest_path=baseline_manifest_path,
            skill_output=skill_output,
            skill_manifest_path=skill_manifest_path,
            skill_provenance_path=frozen_root / "skill_provenance.json",
            output_root=formal_root / "aggregate",
        )
        if aggregate.get("status") != "COMPLETE" or aggregate.get("n_items") != len(task_ids):
            raise RuntimeError("aggregate fixture did not complete")

        print(
            json.dumps(
                {
                    "status": "PASS",
                    "runtime_root": str(runtime_root),
                    "split_semantic_sha256": EXPECTED_SPLIT_SHA256,
                    "concurrency": 4,
                    "systemd_units": units,
                    "freeze_nonblank": "PASS",
                    "freeze_blank": "PASS",
                    "aggregate": "PASS",
                    "benchmark_invocations": 0,
                    "model_requests": 0,
                },
                ensure_ascii=False,
            )
        )


if __name__ == "__main__":
    main()
