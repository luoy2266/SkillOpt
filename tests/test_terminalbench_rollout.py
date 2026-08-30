from __future__ import annotations

import json
import shutil
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import yaml

from skillopt.envs.terminalbench.harbor_runner import (
    EXPECTED_HARBOR_VERSION,
    HarborRunner,
)
from skillopt.envs.terminalbench.result_parser import (
    InfrastructureInvalidTrialError,
)
from skillopt.envs.terminalbench.rollout import (
    TerminalBenchRolloutError,
    run_terminalbench_rollout,
)
from skillopt.envs.terminalbench.trajectory import TrajectoryConversionError


class TerminalBenchRolloutTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary_directory.name)
        self.task_source = self.root / "terminal-bench-v2.1"
        self.task_source.mkdir()
        self.base_config_path = self._write_base_config()
        self.harbor_executable = self._write_fake_harbor()

    def tearDown(self) -> None:
        self.temporary_directory.cleanup()

    def _base_config(self, *, n_attempts: int = 1) -> dict:
        return {
            "job_name": "baseline",
            "jobs_dir": str(self.root / "owner-jobs"),
            "n_attempts": n_attempts,
            "n_concurrent_trials": 2,
            "environment": {"type": "docker"},
            "agents": [
                {
                    "name": "terminus-2",
                    "model_name": "openai/DeepSeek-V4-Flash-0731",
                    "skills": [],
                }
            ],
            "datasets": [
                {
                    "path": str(self.task_source),
                    "task_names": ["old-selection"],
                }
            ],
            "tasks": [],
        }

    def _write_base_config(self, *, n_attempts: int = 1) -> Path:
        path = self.root / f"base-{n_attempts}.yaml"
        path.write_text(
            yaml.safe_dump(
                self._base_config(n_attempts=n_attempts),
                sort_keys=False,
            ),
            encoding="utf-8",
        )
        return path

    def _write_fake_harbor(self) -> Path:
        path = self.root / "fake-harbor"
        path.write_text(
            "#!/usr/bin/env python3\n"
            "import sys\n"
            "if sys.argv[1:] == ['--version']:\n"
            f"    print({EXPECTED_HARBOR_VERSION!r})\n"
            "    raise SystemExit(0)\n"
            "if sys.argv[1:2] == ['run'] and '--print-config' in sys.argv[1:]:\n"
            "    raise SystemExit(0)\n"
            "raise SystemExit(97)\n",
            encoding="utf-8",
        )
        path.chmod(0o755)
        return path

    def _runner(self, *, n_attempts: int = 1) -> HarborRunner:
        config_path = (
            self.base_config_path
            if n_attempts == 1
            else self._write_base_config(n_attempts=n_attempts)
        )
        return HarborRunner(config_path, self.harbor_executable)

    def _trial_result(
        self,
        task_id: str,
        *,
        reward: float,
        exception_type: str | None = None,
    ) -> dict:
        exception_info = None
        if exception_type is not None:
            exception_info = {
                "exception_type": exception_type,
                "exception_message": "synthetic exception",
                "exception_traceback": "synthetic traceback",
                "occurred_at": "2026-08-29T12:00:01",
            }
        return {
            "id": "00000000-0000-0000-0000-000000000001",
            "task_name": f"terminal-bench/{task_id}",
            "trial_name": f"{task_id}__synthetic",
            "trial_uri": f"file:///synthetic/{task_id}",
            "task_id": {"path": f"/synthetic/tasks/{task_id}"},
            "source": "synthetic-dataset",
            "task_checksum": "synthetic-checksum",
            "config": {},
            "agent_info": {"name": "terminus-2", "version": "2.0.0"},
            "agent_result": {},
            "verifier_result": {"rewards": {"reward": reward}},
            "exception_info": exception_info,
            "started_at": "2026-08-29T12:00:00",
            "finished_at": "2026-08-29T12:00:02",
            "environment_setup": {
                "started_at": "2026-08-29T12:00:00",
                "finished_at": "2026-08-29T12:00:00.100000",
            },
            "agent_setup": {
                "started_at": "2026-08-29T12:00:00.100000",
                "finished_at": "2026-08-29T12:00:00.200000",
            },
            "agent_execution": {
                "started_at": "2026-08-29T12:00:00.200000",
                "finished_at": "2026-08-29T12:00:01",
            },
            "verifier": {
                "started_at": "2026-08-29T12:00:01",
                "finished_at": "2026-08-29T12:00:02",
            },
            "step_results": None,
        }

    def _trajectory(self, task_id: str, *, failed: bool = False) -> dict:
        message = (
            "Analysis: the attempted implementation failed"
            if failed
            else "Analysis: inspect and implement the task"
        )
        return {
            "schema_version": "ATIF-v1.7",
            "session_id": f"session-{task_id}",
            "agent": {
                "name": "terminus-2",
                "version": "2.0.0",
                "model_name": "openai/DeepSeek-V4-Flash-0731",
            },
            "steps": [
                {
                    "step_id": 1,
                    "source": "user",
                    "message": f"Complete task {task_id}.",
                },
                {
                    "step_id": 2,
                    "source": "agent",
                    "model_name": "openai/DeepSeek-V4-Flash-0731",
                    "message": message,
                    "reasoning_content": "Use the terminal evidence.",
                    "tool_calls": [
                        {
                            "tool_call_id": "call_2_1",
                            "function_name": "bash_command",
                            "arguments": {"keystrokes": "ls", "duration": 0.1},
                        }
                    ],
                    "observation": {
                        "results": [
                            {
                                "source_call_id": "call_2_1",
                                "content": "synthetic terminal output",
                            }
                        ]
                    },
                },
                {
                    "step_id": 3,
                    "source": "agent",
                    "model_name": "openai/DeepSeek-V4-Flash-0731",
                    "message": "Final response: task execution finished.",
                    "reasoning_content": None,
                    "tool_calls": None,
                    "observation": None,
                },
            ],
        }

    def _write_job(
        self,
        prepared,
        trials: list[dict],
        *,
        started: bool = True,
        finished: bool = True,
        completed_count: int | None = None,
        total_count: int | None = None,
        running_count: int = 0,
        pending_count: int = 0,
    ) -> None:
        prepared.expected_job_dir.mkdir(parents=True)
        expected_count = len(prepared.task_ids)
        job_result = {
            "id": "00000000-0000-0000-0000-000000000010",
            "started_at": "2026-08-29T12:00:00" if started else None,
            "updated_at": "2026-08-29T12:00:03",
            "finished_at": "2026-08-29T12:00:03" if finished else None,
            "n_total_trials": expected_count if total_count is None else total_count,
            "stats": {
                "n_completed_trials": (
                    expected_count if completed_count is None else completed_count
                ),
                "n_errored_trials": sum(
                    trial.get("exception_type") is not None for trial in trials
                ),
                "n_running_trials": running_count,
                "n_pending_trials": pending_count,
                "n_cancelled_trials": 0,
                "n_retries": 0,
                "evals": {},
            },
        }
        (prepared.expected_job_dir / "result.json").write_text(
            json.dumps(job_result),
            encoding="utf-8",
        )
        for index, trial in enumerate(trials):
            trial_dir = prepared.expected_job_dir / trial.get(
                "directory_name",
                f"trial-{index}",
            )
            trial_dir.mkdir()
            task_id = trial["task_id"]
            result_artifact = trial.get("result_artifact", "valid")
            if result_artifact == "valid":
                trial_result = self._trial_result(
                    task_id,
                    reward=trial.get("reward", 1.0),
                    exception_type=trial.get("exception_type"),
                )
                if "task_name" in trial:
                    trial_result["task_name"] = trial["task_name"]
                if "task_path" in trial:
                    trial_result["task_id"] = {"path": trial["task_path"]}
                (trial_dir / "result.json").write_text(
                    json.dumps(trial_result),
                    encoding="utf-8",
                )
            elif result_artifact == "malformed":
                (trial_dir / "result.json").write_text("{malformed", encoding="utf-8")
            if trial.get("trajectory", True):
                agent_dir = trial_dir / "agent"
                agent_dir.mkdir()
                (agent_dir / "trajectory.json").write_text(
                    json.dumps(
                        self._trajectory(
                            task_id,
                            failed=trial.get("reward", 1.0) == 0,
                        )
                    ),
                    encoding="utf-8",
                )

    def _run(
        self,
        items: list[dict],
        trials: list[dict],
        *,
        skill_content: str = "",
        job_options: dict | None = None,
    ) -> tuple[list[dict], object]:
        runner = self._runner()
        captured = {}

        def synthesize(prepared):
            captured["prepared"] = prepared
            self._write_job(prepared, trials, **(job_options or {}))

        with patch.object(runner, "run", side_effect=synthesize) as execute:
            results = run_terminalbench_rollout(
                items,
                skill_content=skill_content,
                rollout_dir=self.root / "rollout",
                runner=runner,
                result_name="synthetic-job",
            )
        execute.assert_called_once()
        return results, captured["prepared"]

    def test_single_task_success_packages_skill_and_writes_conversation(self) -> None:
        results, prepared = self._run(
            [{"id": "task-a"}],
            [{"task_id": "task-a", "reward": 1}],
            skill_content="# Reusable guidance\n",
        )

        self.assertEqual(len(results), 1)
        self.assertEqual(results[0]["id"], "task-a")
        self.assertEqual(results[0]["hard"], 1.0)
        self.assertEqual(results[0]["soft"], 1.0)
        self.assertIsNotNone(results[0]["skill_sha256"])
        self.assertEqual(results[0]["harbor_job_dir"], str(prepared.expected_job_dir))
        self.assertEqual(
            results[0]["harbor_config_path"],
            str(prepared.resolved_config_path),
        )
        self.assertEqual(len(prepared.harbor_skills), 1)
        self.assertTrue(
            (self.root / "rollout/predictions/task-a/conversation.json").is_file()
        )

    def test_valid_failure_returns_score_and_still_writes_conversation(self) -> None:
        results, prepared = self._run(
            [{"id": "task-a"}],
            [{"task_id": "task-a", "reward": 0}],
        )

        self.assertEqual(results[0]["hard"], 0.0)
        self.assertEqual(results[0]["soft"], 0.0)
        self.assertIsNone(results[0]["skill_sha256"])
        self.assertEqual(prepared.harbor_skills, ())
        conversation = json.loads(
            (self.root / "rollout/predictions/task-a/conversation.json").read_text()
        )
        self.assertIn("failed", conversation[1]["content"])

    def test_out_of_order_trials_return_original_input_order(self) -> None:
        items = [{"id": task_id} for task_id in ("task-a", "task-b", "task-c")]
        trials = [
            {"task_id": "task-c", "directory_name": "trial-c"},
            {"task_id": "task-a", "directory_name": "trial-a"},
            {"task_id": "task-b", "directory_name": "trial-b"},
        ]

        results, _ = self._run(items, trials)

        self.assertEqual([result["id"] for result in results], ["task-a", "task-b", "task-c"])

    def test_duplicate_input_ids_fail_before_execution(self) -> None:
        runner = self._runner()

        with patch.object(runner, "prepare") as prepare, patch.object(
            runner,
            "run",
        ) as execute:
            with self.assertRaisesRegex(TerminalBenchRolloutError, "duplicate IDs"):
                run_terminalbench_rollout(
                    [{"id": "task-a"}, {"id": "task-a"}],
                    skill_content="",
                    rollout_dir=self.root / "rollout",
                    runner=runner,
                    result_name="duplicate-input",
                )
        prepare.assert_not_called()
        execute.assert_not_called()

    def test_agent_timeout_with_reward_zero_is_not_rejected_by_job_error_count(self) -> None:
        results, _ = self._run(
            [{"id": "task-a"}],
            [
                {
                    "task_id": "task-a",
                    "reward": 0,
                    "exception_type": "AgentTimeoutError",
                }
            ],
        )

        self.assertEqual(results[0]["trial_status"], "AgentTimeoutError")
        self.assertEqual(results[0]["soft"], 0.0)
        self.assertTrue(
            (self.root / "rollout/predictions/task-a/conversation.json").is_file()
        )

    def test_missing_trial_fails_closed(self) -> None:
        with self.assertRaisesRegex(TerminalBenchRolloutError, "missing=.*task-b"):
            self._run(
                [{"id": "task-a"}, {"id": "task-b"}],
                [{"task_id": "task-a"}],
            )

    def test_duplicate_trial_fails_closed(self) -> None:
        with self.assertRaisesRegex(TerminalBenchRolloutError, "Duplicate"):
            self._run(
                [{"id": "task-a"}],
                [
                    {"task_id": "task-a", "directory_name": "first"},
                    {"task_id": "task-a", "directory_name": "second"},
                ],
            )

    def test_unexpected_trial_fails_closed(self) -> None:
        with self.assertRaisesRegex(TerminalBenchRolloutError, "unexpected=.*task-b"):
            self._run(
                [{"id": "task-a"}],
                [{"task_id": "task-b"}],
            )

    def test_unprefixed_task_name_fails_discovery(self) -> None:
        with self.assertRaisesRegex(
            TerminalBenchRolloutError,
            "not the canonical Terminal-Bench name",
        ):
            self._run(
                [{"id": "task-a"}],
                [{"task_id": "task-a", "task_name": "task-a"}],
            )

    def test_wrong_task_name_prefix_fails_discovery(self) -> None:
        with self.assertRaisesRegex(
            TerminalBenchRolloutError,
            "not the canonical Terminal-Bench name",
        ):
            self._run(
                [{"id": "task-a"}],
                [{"task_id": "task-a", "task_name": "other-bench/task-a"}],
            )

    def test_missing_task_name_fails_discovery(self) -> None:
        with self.assertRaisesRegex(
            TerminalBenchRolloutError,
            "not the canonical Terminal-Bench name",
        ):
            self._run(
                [{"id": "task-a"}],
                [{"task_id": "task-a", "task_name": None}],
            )

    def test_mismatched_task_path_and_canonical_name_fail_discovery(self) -> None:
        with self.assertRaisesRegex(
            TerminalBenchRolloutError,
            "not the canonical Terminal-Bench name",
        ):
            self._run(
                [{"id": "task-a"}],
                [
                    {
                        "task_id": "task-a",
                        "task_path": "/synthetic/tasks/task-b",
                    }
                ],
            )

    def test_incomplete_job_fails_closed(self) -> None:
        with self.assertRaisesRegex(TerminalBenchRolloutError, "missing job finished_at"):
            self._run(
                [{"id": "task-a"}],
                [{"task_id": "task-a"}],
                job_options={"finished": False},
            )

    def test_job_requires_started_timestamp(self) -> None:
        with self.assertRaisesRegex(TerminalBenchRolloutError, "missing job started_at"):
            self._run(
                [{"id": "task-a"}],
                [{"task_id": "task-a"}],
                job_options={"started": False},
            )

    def test_job_total_count_must_match_input_batch(self) -> None:
        with self.assertRaisesRegex(TerminalBenchRolloutError, "trial count"):
            self._run(
                [{"id": "task-a"}],
                [{"task_id": "task-a"}],
                job_options={"total_count": 2},
            )

    def test_job_completed_count_must_match_input_batch(self) -> None:
        with self.assertRaisesRegex(TerminalBenchRolloutError, "incomplete trials"):
            self._run(
                [{"id": "task-a"}],
                [{"task_id": "task-a"}],
                job_options={"completed_count": 0},
            )

    def test_job_running_and_pending_counts_must_be_zero(self) -> None:
        for field in ("running_count", "pending_count"):
            with self.subTest(field=field):
                with self.assertRaisesRegex(TerminalBenchRolloutError, "nonzero"):
                    self._run(
                        [{"id": "task-a"}],
                        [{"task_id": "task-a"}],
                        job_options={field: 1},
                    )
                shutil.rmtree(self.root / "rollout")

    def test_zero_process_exit_without_job_artifacts_fails_closed(self) -> None:
        runner = self._runner()

        with patch.object(runner, "run", return_value=None):
            with self.assertRaisesRegex(TerminalBenchRolloutError, "job directory is missing"):
                run_terminalbench_rollout(
                    [{"id": "task-a"}],
                    skill_content="",
                    rollout_dir=self.root / "rollout",
                    runner=runner,
                    result_name="missing-job-artifacts",
                )

    def test_missing_trajectory_fails_after_valid_result(self) -> None:
        with self.assertRaisesRegex(TrajectoryConversionError, "missing"):
            self._run(
                [{"id": "task-a"}],
                [{"task_id": "task-a", "trajectory": False}],
            )

    def test_missing_trial_result_artifact_fails_closed(self) -> None:
        with self.assertRaisesRegex(TerminalBenchRolloutError, "trial result is missing"):
            self._run(
                [{"id": "task-a"}],
                [{"task_id": "task-a", "result_artifact": "missing"}],
            )

    def test_malformed_trial_result_artifact_fails_closed(self) -> None:
        with self.assertRaisesRegex(TerminalBenchRolloutError, "trial result is malformed"):
            self._run(
                [{"id": "task-a"}],
                [{"task_id": "task-a", "result_artifact": "malformed"}],
            )

    def test_infrastructure_invalid_trial_aborts_batch(self) -> None:
        with self.assertRaisesRegex(
            InfrastructureInvalidTrialError,
            "EnvironmentStartTimeoutError",
        ):
            self._run(
                [{"id": "task-a"}],
                [
                    {
                        "task_id": "task-a",
                        "reward": 0,
                        "exception_type": "EnvironmentStartTimeoutError",
                    }
                ],
            )
        self.assertFalse(
            (self.root / "rollout/predictions/task-a/conversation.json").exists()
        )

    def test_partial_conversation_artifact_does_not_produce_partial_results(self) -> None:
        with self.assertRaisesRegex(
            InfrastructureInvalidTrialError,
            "EnvironmentStartTimeoutError",
        ):
            self._run(
                [{"id": "task-a"}, {"id": "task-b"}],
                [
                    {"task_id": "task-a", "reward": 1},
                    {
                        "task_id": "task-b",
                        "reward": 0,
                        "exception_type": "EnvironmentStartTimeoutError",
                    },
                ],
            )

        self.assertTrue(
            (self.root / "rollout/predictions/task-a/conversation.json").is_file()
        )
        self.assertFalse(
            (self.root / "rollout/predictions/task-b/conversation.json").exists()
        )

    def test_n_attempts_other_than_one_fails_before_execution(self) -> None:
        runner = self._runner(n_attempts=2)

        with patch.object(runner, "run") as execute:
            with self.assertRaisesRegex(TerminalBenchRolloutError, "n_attempts=1"):
                run_terminalbench_rollout(
                    [{"id": "task-a"}],
                    skill_content="",
                    rollout_dir=self.root / "rollout",
                    runner=runner,
                    result_name="invalid-attempts",
                )
        execute.assert_not_called()


if __name__ == "__main__":
    unittest.main()
