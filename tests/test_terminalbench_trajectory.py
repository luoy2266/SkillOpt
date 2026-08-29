from __future__ import annotations

import copy
import json
import tempfile
import unittest
from pathlib import Path

from skillopt.envs.terminalbench.trajectory import (
    TrajectoryConversionError,
    conversation_output_path,
    convert_atif_trajectory,
)


class TerminalBenchTrajectoryTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary_directory.name)
        self.task_id = "fixture-task"
        self.atif_path = self.root / "trial" / "agent" / "trajectory.json"
        self.rollout_dir = self.root / "rollout"
        self.output_path = conversation_output_path(self.rollout_dir, self.task_id)

    def tearDown(self) -> None:
        self.temporary_directory.cleanup()

    def _agent_step(
        self,
        step_id: int,
        *,
        message: object = "Analysis: inspect the repository\nPlan: run one command",
        reasoning: object = "I should inspect before changing anything.",
        tool_calls: object | None = None,
        observation: object = "file.txt",
        copied: bool | None = None,
    ) -> dict:
        if tool_calls is None:
            tool_calls = [
                {
                    "tool_call_id": f"call_{step_id}_1",
                    "function_name": "bash_command",
                    "arguments": {"keystrokes": "ls", "duration": 0.1},
                }
            ]
        source_call_id = None
        if isinstance(tool_calls, list) and len(tool_calls) == 1:
            source_call_id = tool_calls[0].get("tool_call_id")
        step = {
            "step_id": step_id,
            "timestamp": f"2026-08-29T12:00:0{step_id}+00:00",
            "source": "agent",
            "model_name": "openai/DeepSeek-V4-Flash-0731",
            "message": message,
            "reasoning_content": reasoning,
            "tool_calls": tool_calls,
            "observation": {
                "results": [
                    {
                        "source_call_id": source_call_id,
                        "content": observation,
                    }
                ]
            },
            "metrics": {
                "prompt_tokens": 10,
                "completion_tokens": 5,
                "cost_usd": 0.01,
            },
        }
        if copied is not None:
            step["is_copied_context"] = copied
        return step

    def _fixture(self) -> dict:
        return {
            "schema_version": "ATIF-v1.7",
            "session_id": "terminus-session-not-a-task-id",
            "agent": {
                "name": "terminus-2",
                "version": "2.0.0",
                "model_name": "openai/DeepSeek-V4-Flash-0731",
                "extra": {"parser": "json"},
            },
            "steps": [
                {
                    "step_id": 1,
                    "timestamp": "2026-08-29T12:00:01+00:00",
                    "source": "user",
                    "message": "Complete the fixture task.\n\nInitial terminal state: ready",
                },
                self._agent_step(2),
                self._agent_step(
                    3,
                    message="Analysis: verification is complete\nPlan: finish",
                    reasoning=None,
                    tool_calls=[
                        {
                            "tool_call_id": "call_3_task_complete",
                            "function_name": "mark_task_complete",
                            "arguments": {},
                        }
                    ],
                    observation="Task completion confirmed.",
                ),
            ],
            "final_metrics": {
                "total_prompt_tokens": 20,
                "total_completion_tokens": 10,
                "total_steps": 3,
            },
        }

    def _write(self, fixture: object | None = None, path: Path | None = None) -> Path:
        destination = path or self.atif_path
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(
            json.dumps(self._fixture() if fixture is None else fixture),
            encoding="utf-8",
        )
        return destination

    def _convert(self, fixture: object | None = None) -> list[dict]:
        return convert_atif_trajectory(
            self._write(fixture),
            expected_task_id=self.task_id,
            output_path=self.output_path,
        )

    def test_success_preserves_task_action_observation_and_final_response(self) -> None:
        source_bytes = json.dumps(self._fixture()).encode()
        self.atif_path.parent.mkdir(parents=True)
        self.atif_path.write_bytes(source_bytes)

        conversation = convert_atif_trajectory(
            self.atif_path,
            expected_task_id=self.task_id,
            output_path=self.output_path,
        )

        self.assertEqual(conversation[0]["role"], "user")
        self.assertIn("Complete the fixture task", conversation[0]["content"])
        self.assertEqual(conversation[1]["role"], "assistant")
        self.assertEqual(
            conversation[2],
            {
                "step": 2,
                "reasoning": "I should inspect before changing anything.",
                "action": 'bash_command({"duration":0.1,"keystrokes":"ls"})',
                "env_feedback": "file.txt",
            },
        )
        self.assertEqual(conversation[-2]["role"], "assistant")
        self.assertEqual(conversation[-1]["action"], "mark_task_complete({})")
        self.assertEqual(conversation[-1]["env_feedback"], "Task completion confirmed.")
        self.assertEqual(self.atif_path.read_bytes(), source_bytes)

    def test_normal_failure_execution_is_not_discarded(self) -> None:
        fixture = self._fixture()
        fixture["steps"] = [
            fixture["steps"][0],
            self._agent_step(
                2,
                message="Analysis: the attempted implementation failed",
                reasoning="I need a different approach.",
                tool_calls=[
                    {
                        "tool_call_id": "call_2_1",
                        "function_name": "bash_command",
                        "arguments": {"keystrokes": "false", "duration": 0.1},
                    }
                ],
                observation="command exited with status 1",
            ),
        ]

        conversation = self._convert(fixture)

        self.assertIn("failed", conversation[1]["content"])
        self.assertIn("false", conversation[2]["action"])
        self.assertEqual(conversation[2]["env_feedback"], "command exited with status 1")

    def test_multiple_tool_calls_keep_order_and_one_shared_observation(self) -> None:
        fixture = self._fixture()
        fixture["steps"] = [
            fixture["steps"][0],
            self._agent_step(
                2,
                tool_calls=[
                    {
                        "tool_call_id": "call_2_1",
                        "function_name": "bash_command",
                        "arguments": {"keystrokes": "pwd", "duration": 0.1},
                    },
                    {
                        "tool_call_id": "call_2_2",
                        "function_name": "bash_command",
                        "arguments": {"keystrokes": "ls -la", "duration": 0.2},
                    },
                ],
                observation="/workspace\nfile.txt",
            ),
        ]

        conversation = self._convert(fixture)
        step_record = conversation[-1]

        self.assertLess(step_record["action"].index("pwd"), step_record["action"].index("ls -la"))
        self.assertEqual(step_record["action"].count("bash_command"), 2)
        self.assertEqual(step_record["env_feedback"], "/workspace\nfile.txt")
        self.assertEqual(
            sum(
                record.get("env_feedback") == "/workspace\nfile.txt"
                for record in conversation
            ),
            1,
        )

    def test_system_context_event_preserves_true_role(self) -> None:
        fixture = self._fixture()
        fixture["steps"] = [
            fixture["steps"][0],
            {
                "step_id": 2,
                "source": "system",
                "message": "Performed context summarization and handoff.",
                "observation": {
                    "results": [
                        {
                            "subagent_trajectory_ref": [
                                {
                                    "session_id": "summary-session",
                                    "trajectory_path": "trajectory.summarization-1-summary.json",
                                }
                            ]
                        }
                    ]
                },
            },
            self._agent_step(3),
        ]

        conversation = self._convert(fixture)
        self.assertEqual(
            conversation[1],
            {"role": "system", "content": "Performed context summarization and handoff."},
        )

    def test_partial_agent_timeout_trajectory_remains_convertible(self) -> None:
        fixture = self._fixture()
        fixture["steps"] = [
            fixture["steps"][0],
            self._agent_step(
                2,
                message="Analysis: long-running command is still active",
                reasoning=None,
                observation="partial output before Harbor AgentTimeoutError",
            ),
        ]

        conversation = self._convert(fixture)

        self.assertIn("long-running", conversation[1]["content"])
        self.assertIn("partial output", conversation[2]["env_feedback"])

    def test_explicit_continuation_chain_skips_copied_context(self) -> None:
        fixture = self._fixture()
        fixture["steps"] = fixture["steps"][:2]
        fixture["continued_trajectory_ref"] = "trajectory.cont-1.json"
        continuation = copy.deepcopy(fixture)
        continuation.pop("continued_trajectory_ref")
        continuation["session_id"] = "terminus-session-cont-1"
        continuation["steps"] = [
            {**copy.deepcopy(fixture["steps"][0]), "is_copied_context": True},
            {**copy.deepcopy(fixture["steps"][1]), "is_copied_context": True},
            self._agent_step(
                3,
                message="Analysis: continue after compaction",
                reasoning="The copied context is already represented.",
                observation="continued output",
            ),
        ]
        self._write(continuation, self.atif_path.parent / "trajectory.cont-1.json")

        conversation = self._convert(fixture)

        user_records = [record for record in conversation if record.get("role") == "user"]
        self.assertEqual(len(user_records), 1)
        self.assertEqual(
            sum("inspect the repository" in record.get("content", "") for record in conversation),
            1,
        )
        self.assertIn("continue after compaction", conversation[-2]["content"])
        self.assertEqual(conversation[-1]["env_feedback"], "continued output")

    def test_large_terminal_output_is_not_truncated(self) -> None:
        large_output = "terminal-output\n" * 20_000
        fixture = self._fixture()
        fixture["steps"] = [fixture["steps"][0], self._agent_step(2, observation=large_output)]

        conversation = self._convert(fixture)

        self.assertEqual(conversation[-1]["env_feedback"], large_output)

    def test_missing_invalid_and_wrong_top_level_artifacts_fail(self) -> None:
        with self.assertRaisesRegex(TrajectoryConversionError, "missing"):
            convert_atif_trajectory(
                self.atif_path,
                expected_task_id=self.task_id,
                output_path=self.output_path,
            )

        self.atif_path.parent.mkdir(parents=True)
        self.atif_path.write_text("{malformed", encoding="utf-8")
        with self.assertRaisesRegex(TrajectoryConversionError, "malformed"):
            convert_atif_trajectory(
                self.atif_path,
                expected_task_id=self.task_id,
                output_path=self.output_path,
            )

        with self.assertRaisesRegex(TrajectoryConversionError, "top-level object"):
            self._convert([])

    def test_empty_missing_and_user_only_steps_fail(self) -> None:
        malformed_fixtures = []
        empty = self._fixture()
        empty["steps"] = []
        malformed_fixtures.append(empty)
        missing = self._fixture()
        missing.pop("steps")
        malformed_fixtures.append(missing)
        user_only = self._fixture()
        user_only["steps"] = user_only["steps"][:1]
        malformed_fixtures.append(user_only)

        for fixture in malformed_fixtures:
            with self.subTest(fixture=fixture):
                with self.assertRaises(TrajectoryConversionError):
                    self._convert(fixture)

    def test_malformed_message_tool_and_observation_fail(self) -> None:
        malformed = []
        bad_message = self._fixture()
        bad_message["steps"][1]["message"] = None
        malformed.append(bad_message)

        bad_tool = self._fixture()
        bad_tool["steps"][1]["tool_calls"][0].pop("arguments")
        malformed.append(bad_tool)

        bad_observation = self._fixture()
        bad_observation["steps"][1]["observation"]["results"][0][
            "source_call_id"
        ] = "missing-call"
        malformed.append(bad_observation)

        for fixture in malformed:
            with self.subTest(fixture=fixture):
                with self.assertRaises(TrajectoryConversionError):
                    self._convert(fixture)

    def test_missing_continuation_fails_loudly(self) -> None:
        fixture = self._fixture()
        fixture["continued_trajectory_ref"] = "trajectory.cont-1.json"

        with self.assertRaisesRegex(TrajectoryConversionError, "missing"):
            self._convert(fixture)

    def test_output_path_must_match_expected_task_id(self) -> None:
        wrong_output = conversation_output_path(self.rollout_dir, "other-task")

        with self.assertRaisesRegex(TrajectoryConversionError, "expected-task-id"):
            convert_atif_trajectory(
                self._write(),
                expected_task_id=self.task_id,
                output_path=wrong_output,
            )

    def test_repeated_identical_conversion_is_idempotent(self) -> None:
        first = self._convert()
        first_stat = self.output_path.stat()
        first_bytes = self.output_path.read_bytes()

        second = convert_atif_trajectory(
            self.atif_path,
            expected_task_id=self.task_id,
            output_path=self.output_path,
        )

        self.assertEqual(second, first)
        self.assertEqual(self.output_path.read_bytes(), first_bytes)
        self.assertEqual(self.output_path.stat().st_ino, first_stat.st_ino)
        self.assertTrue(first_bytes.endswith(b"\n"))

    def test_conflicting_conversation_is_not_overwritten(self) -> None:
        self._write()
        self.output_path.parent.mkdir(parents=True)
        tampered = '[{"role":"user","content":"tampered"}]\n'
        self.output_path.write_text(tampered, encoding="utf-8")

        with self.assertRaisesRegex(TrajectoryConversionError, "conflicts"):
            convert_atif_trajectory(
                self.atif_path,
                expected_task_id=self.task_id,
                output_path=self.output_path,
            )

        self.assertEqual(self.output_path.read_text(encoding="utf-8"), tampered)

    def test_generated_artifact_loads_through_real_reflection_formatter(self) -> None:
        fmt_minibatch_trajectories = self._real_reflection_formatter()
        self._convert()

        formatted = fmt_minibatch_trajectories(
            [
                {
                    "id": self.task_id,
                    "task_description": "Complete the fixture task.",
                    "task_type": "terminalbench",
                    "n_turns": 2,
                }
            ],
            str(self.rollout_dir / "predictions"),
        )

        self.assertIn(f"id={self.task_id}", formatted)
        self.assertIn("[user] Complete the fixture task", formatted)
        self.assertIn("[step 2 think] I should inspect", formatted)
        self.assertIn("[step 2 action] bash_command", formatted)
        self.assertIn("[step 2 obs]    file.txt", formatted)
        self.assertIn("mark_task_complete", formatted)
        self.assertEqual(formatted.count("Analysis: inspect the repository"), 1)
        self.assertEqual(formatted.count("bash_command"), 1)
        self.assertEqual(formatted.count("[step 2 obs]    file.txt"), 1)

    def _real_reflection_formatter(self):
        try:
            from skillopt.gradient.reflect import fmt_minibatch_trajectories
        except ModuleNotFoundError as exc:
            if exc.name is None or exc.name.startswith("skillopt"):
                raise
            self.skipTest(
                "real SkillOpt reflection formatter compatibility requires "
                f"optional runtime dependency {exc.name!r}"
            )
        return fmt_minibatch_trajectories


if __name__ == "__main__":
    unittest.main()
