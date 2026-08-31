from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from scripts.eval_only import _write_terminalbench_eval_results


class TerminalBenchEvalResultsTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary_directory.name)

    def tearDown(self) -> None:
        self.temporary_directory.cleanup()

    def _result(self) -> dict:
        return {
            "id": "task-a",
            "hard": 0.0,
            "soft": 0.5,
            "raw_reward": 0.5,
            "trial_status": "completed",
            "harbor_result_path": "/job/task-a/result.json",
            "harbor_config_path": "/job/config.yaml",
            "harbor_job_dir": "/job",
            "skill_sha256": None,
        }

    def test_writes_lightweight_task_level_terminalbench_results(self) -> None:
        _write_terminalbench_eval_results(
            out_root=str(self.root),
            split="valid_unseen",
            skill_path="/skill.md",
            skill_content="\n",
            results=[self._result()],
        )

        payload = json.loads((self.root / "eval_results.json").read_text(encoding="utf-8"))
        self.assertEqual(payload["schema_version"], "skillopt-terminalbench-eval-results-v1")
        self.assertEqual(payload["n_items"], 1)
        self.assertEqual(payload["results"][0]["id"], "task-a")
        self.assertNotIn("trajectory", payload["results"][0])

    def test_missing_aggregate_field_fails_before_writing(self) -> None:
        result = self._result()
        del result["trial_status"]

        with self.assertRaisesRegex(ValueError, "missing aggregate fields"):
            _write_terminalbench_eval_results(
                out_root=str(self.root),
                split="valid_unseen",
                skill_path="/skill.md",
                skill_content="\n",
                results=[result],
            )
        self.assertFalse((self.root / "eval_results.json").exists())


if __name__ == "__main__":
    unittest.main()
