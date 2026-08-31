from __future__ import annotations

import json
import tempfile
import unittest
from copy import deepcopy
from pathlib import Path

from scripts.aggregate_terminalbench_results import AggregateFailure, aggregate_results
from scripts.freeze_terminalbench_skill import freeze_training_skill
from tests.terminalbench_lifecycle_fixtures import (
    evaluation_manifest,
    native_sha256,
    write_completed_training_fixture,
    write_evaluation_fixture,
    write_json,
)


class TerminalBenchAggregateTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary_directory.name)
        fixture = write_completed_training_fixture(self.root)
        self.frozen_root = self.root / "skills" / "experiment-001"
        self.frozen = freeze_training_skill(
            experiment_id="experiment-001",
            training_output=fixture["output"],
            training_manifest_path=fixture["manifest_path"],
            output_root=self.frozen_root,
            expected_skillopt_head="a" * 40,
        )
        self.training_manifest = fixture["manifest"]
        self.task_ids = self.training_manifest["dataset"]["task_ids"]["test"]
        self.baseline_output = self.root / "formal" / "baseline-test"
        self.skill_output = self.root / "formal" / "skill-test"
        self.baseline_manifest_path = self.root / "formal" / "manifests" / "baseline.json"
        self.skill_manifest_path = self.root / "formal" / "manifests" / "skill.json"
        self.provenance_path = self.frozen_root / "skill_provenance.json"
        self._write_pair([0.0, 0.5, 1.0], [1.0, 0.5, 0.0])

    def tearDown(self) -> None:
        self.temporary_directory.cleanup()

    def _write_pair(
        self,
        baseline_rewards: list[float],
        skill_rewards: list[float],
        *,
        baseline_statuses: list[str] | None = None,
        skill_statuses: list[str] | None = None,
    ) -> None:
        content = self.frozen["content"]
        native = self.frozen["native_sha256"]
        baseline_manifest = evaluation_manifest(
            self.training_manifest,
            condition="baseline-test",
            output_root=self.baseline_output,
            raw_content="\n",
            native_sha256=None,
        )
        skill_manifest = evaluation_manifest(
            self.training_manifest,
            condition="skill-test",
            output_root=self.skill_output,
            raw_content=content,
            native_sha256=native,
            provenance_path=self.provenance_path,
        )
        write_json(self.baseline_manifest_path, baseline_manifest)
        write_json(self.skill_manifest_path, skill_manifest)
        write_evaluation_fixture(
            self.baseline_output,
            task_ids=self.task_ids,
            rewards=baseline_rewards,
            raw_skill="\n",
            native_sha256=None,
            statuses=baseline_statuses,
        )
        write_evaluation_fixture(
            self.skill_output,
            task_ids=self.task_ids,
            rewards=skill_rewards,
            raw_skill=content,
            native_sha256=native,
            statuses=skill_statuses,
        )

    def _aggregate(self, *, output_name: str = "aggregate") -> dict:
        return aggregate_results(
            experiment_id="experiment-001",
            baseline_output=self.baseline_output,
            baseline_manifest_path=self.baseline_manifest_path,
            skill_output=self.skill_output,
            skill_manifest_path=self.skill_manifest_path,
            skill_provenance_path=self.provenance_path,
            output_root=self.root / output_name,
        )

    def test_complete_pair_outputs_paired_scores(self) -> None:
        summary = self._aggregate()

        self.assertEqual(summary["n_items"], 3)
        self.assertEqual(summary["scores"]["wins"], 1)
        self.assertEqual(summary["scores"]["ties"], 1)
        self.assertEqual(summary["scores"]["losses"], 1)
        self.assertEqual(summary["scores"]["absolute_delta"], 0.0)
        self.assertEqual(len(summary["paired_results"]), 3)
        self.assertTrue((self.root / "aggregate" / "results.tsv").is_file())

    def test_positive_negative_and_zero_relative_delta(self) -> None:
        cases = (
            ([0.0, 0.0, 0.0], [1.0, 0.0, 0.0], 1 / 3, None),
            ([1.0, 1.0, 1.0], [0.0, 1.0, 1.0], -1 / 3, -1 / 3),
            ([0.5, 0.5, 0.5], [0.5, 0.5, 0.5], 0.0, 0.0),
        )
        for index, (baseline, skill, delta, relative) in enumerate(cases):
            with self.subTest(index=index):
                self._write_pair(baseline, skill)
                summary = self._aggregate(output_name=f"aggregate-{index}")
                self.assertAlmostEqual(summary["scores"]["absolute_delta"], delta)
                if relative is None:
                    self.assertIsNone(summary["scores"]["relative_delta"])
                else:
                    self.assertAlmostEqual(summary["scores"]["relative_delta"], relative)

    def test_task_id_mismatch_blocks(self) -> None:
        path = self.skill_output / "eval_results.json"
        payload = json.loads(path.read_text(encoding="utf-8"))
        payload["results"][0]["id"] = "other-task"
        write_json(path, payload)

        with self.assertRaisesRegex(AggregateFailure, "task IDs"):
            self._aggregate()

    def test_manifest_parity_mismatches_block(self) -> None:
        cases = {
            "concurrency": ("execution", "n_concurrent_trials", 8),
            "model": ("models", "underlying_identity", "other-model"),
            "split": ("dataset", "split_manifest_sha256", "f" * 64),
        }
        original = json.loads(self.skill_manifest_path.read_text(encoding="utf-8"))
        for name, (section, key, value) in cases.items():
            with self.subTest(name=name):
                manifest = deepcopy(original)
                manifest[section][key] = value
                write_json(self.skill_manifest_path, manifest)
                with self.assertRaisesRegex(AggregateFailure, "parity mismatch"):
                    self._aggregate(output_name=f"aggregate-{name}")
        write_json(self.skill_manifest_path, original)

    def test_skill_sha_mismatch_blocks(self) -> None:
        manifest = json.loads(self.skill_manifest_path.read_text(encoding="utf-8"))
        manifest["skill"]["sha256"] = "0" * 64
        write_json(self.skill_manifest_path, manifest)

        with self.assertRaisesRegex(AggregateFailure, "raw SHA-256"):
            self._aggregate()

    def test_task_level_native_skill_sha_mismatch_blocks(self) -> None:
        path = self.skill_output / "eval_results.json"
        payload = json.loads(path.read_text(encoding="utf-8"))
        payload["results"][0]["skill_sha256"] = "0" * 64
        write_json(path, payload)

        with self.assertRaisesRegex(AggregateFailure, "native SHA-256"):
            self._aggregate()

    def test_missing_trial_blocks(self) -> None:
        path = self.baseline_output / "eval_results.json"
        payload = json.loads(path.read_text(encoding="utf-8"))
        payload["results"].pop()
        payload["n_items"] -= 1
        write_json(path, payload)

        with self.assertRaisesRegex(AggregateFailure, "incomplete"):
            self._aggregate()

    def test_infrastructure_invalid_status_blocks(self) -> None:
        self._write_pair(
            [0.0, 0.5, 1.0],
            [1.0, 0.5, 0.0],
            skill_statuses=["completed", "InfrastructureError", "completed"],
        )

        with self.assertRaisesRegex(AggregateFailure, "infrastructure-invalid"):
            self._aggregate()

    def test_timeout_and_nonzero_agent_exit_are_counted_not_rewritten(self) -> None:
        self._write_pair(
            [0.0, 0.0, 1.0],
            [0.0, 0.0, 1.0],
            baseline_statuses=["AgentTimeoutError", "NonZeroAgentExitCodeError", "completed"],
            skill_statuses=["completed", "AgentTimeoutError", "NonZeroAgentExitCodeError"],
        )

        summary = self._aggregate()

        self.assertEqual(summary["failure_counts"]["baseline"]["AgentTimeoutError"], 1)
        self.assertEqual(
            summary["failure_counts"]["skill"]["NonZeroAgentExitCodeError"], 1
        )


if __name__ == "__main__":
    unittest.main()
