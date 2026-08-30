from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from skillopt.envs.terminalbench.result_parser import (
    InfrastructureInvalidTrialError,
    parse_trial_result,
)


class TerminalBenchResultParserTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary_directory.name)
        self.task_id = "fixture-task"
        self.result_path = self.root / "trial" / "result.json"

    def tearDown(self) -> None:
        self.temporary_directory.cleanup()

    def _fixture(
        self,
        reward: object = 0.0,
        *,
        exception_type: str | None = None,
    ) -> dict:
        exception_info = None
        if exception_type is not None:
            exception_info = {
                "exception_type": exception_type,
                "exception_message": "fixture exception",
                "exception_traceback": "fixture traceback",
                "occurred_at": "2026-08-29T12:00:01",
            }
        return {
            "id": "00000000-0000-0000-0000-000000000001",
            "task_name": f"terminal-bench/{self.task_id}",
            "trial_name": "fixture-task__trial",
            "trial_uri": "file:///fixture/trial",
            "task_id": {"path": f"/fixture/tasks/{self.task_id}"},
            "source": "fixture-dataset",
            "task_checksum": "fixture-checksum",
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

    def _write(self, fixture: object) -> Path:
        self.result_path.parent.mkdir(parents=True, exist_ok=True)
        self.result_path.write_text(json.dumps(fixture), encoding="utf-8")
        return self.result_path

    def _parse(self, fixture: object) -> dict[str, str | float]:
        return parse_trial_result(
            self._write(fixture),
            expected_task_id=self.task_id,
        )

    def test_success_maps_to_binary_hard_and_full_soft(self) -> None:
        result = self._parse(self._fixture(1))

        self.assertEqual(
            result,
            {
                "id": self.task_id,
                "hard": 1.0,
                "soft": 1.0,
                "raw_reward": 1.0,
                "trial_status": "completed",
                "harbor_result_path": str(self.result_path.resolve()),
            },
        )

    def test_real_terminalbench_canonical_identity_shape_is_accepted(self) -> None:
        task_id = "git-leak-recovery"
        fixture = self._fixture(1)
        fixture["task_name"] = f"terminal-bench/{task_id}"
        fixture["task_id"] = {"path": f"/fixture/tasks/{task_id}"}

        result = parse_trial_result(
            self._write(fixture),
            expected_task_id=task_id,
        )

        self.assertEqual(result["id"], task_id)

    def test_valid_failure_remains_normal_scored_outcome(self) -> None:
        result = self._parse(self._fixture(0))

        self.assertEqual(result["hard"], 0.0)
        self.assertEqual(result["soft"], 0.0)
        self.assertEqual(result["raw_reward"], 0.0)

    def test_partial_reward_is_soft_only(self) -> None:
        result = self._parse(self._fixture(0.5))

        self.assertEqual(result["hard"], 0.0)
        self.assertEqual(result["soft"], 0.5)
        self.assertEqual(result["raw_reward"], 0.5)

    def test_invalid_rewards_fail_closed(self) -> None:
        invalid_rewards = (
            float("nan"),
            float("inf"),
            float("-inf"),
            -0.1,
            1.1,
            None,
            "0.5",
            True,
            {},
        )
        for reward in invalid_rewards:
            with self.subTest(reward=reward):
                with self.assertRaises(InfrastructureInvalidTrialError):
                    self._parse(self._fixture(reward))

    def test_missing_result_artifact_fails_closed(self) -> None:
        with self.assertRaisesRegex(
            InfrastructureInvalidTrialError,
            "artifact is missing",
        ):
            parse_trial_result(
                self.result_path,
                expected_task_id=self.task_id,
            )

    def test_malformed_result_json_fails_closed(self) -> None:
        self.result_path.parent.mkdir(parents=True)
        self.result_path.write_text("{malformed", encoding="utf-8")

        with self.assertRaisesRegex(
            InfrastructureInvalidTrialError,
            "unreadable or malformed",
        ):
            parse_trial_result(
                self.result_path,
                expected_task_id=self.task_id,
            )

    def test_missing_verifier_result_fails_closed(self) -> None:
        for verifier_result in (None, "missing"):
            with self.subTest(verifier_result=verifier_result):
                fixture = self._fixture()
                fixture["verifier_result"] = verifier_result
                with self.assertRaisesRegex(
                    InfrastructureInvalidTrialError,
                    "missing verifier_result",
                ):
                    self._parse(fixture)

    def test_missing_verifier_timing_fails_closed(self) -> None:
        fixture = self._fixture()
        fixture["verifier"] = None

        with self.assertRaisesRegex(
            InfrastructureInvalidTrialError,
            "missing verifier timing",
        ):
            self._parse(fixture)

    def test_missing_rewards_or_reward_fails_closed(self) -> None:
        fixtures = (None, {}, {"rewards": None}, {"rewards": {}})
        for verifier_result in fixtures:
            with self.subTest(verifier_result=verifier_result):
                fixture = self._fixture()
                fixture["verifier_result"] = verifier_result
                with self.assertRaises(InfrastructureInvalidTrialError):
                    self._parse(fixture)

    def test_old_unprefixed_task_name_fails_closed(self) -> None:
        fixture = self._fixture()
        fixture["task_name"] = self.task_id

        with self.assertRaisesRegex(
            InfrastructureInvalidTrialError,
            "not the canonical Terminal-Bench name",
        ):
            self._parse(fixture)

    def test_wrong_canonical_task_name_prefix_fails_closed(self) -> None:
        fixture = self._fixture()
        fixture["task_name"] = f"other-bench/{self.task_id}"

        with self.assertRaisesRegex(
            InfrastructureInvalidTrialError,
            "not the canonical Terminal-Bench name",
        ):
            self._parse(fixture)

    def test_wrong_task_id_path_fails_closed(self) -> None:
        fixture = self._fixture()
        fixture["task_id"] = {"path": "/fixture/tasks/other-task"}

        with self.assertRaisesRegex(
            InfrastructureInvalidTrialError,
            "task_id path does not match",
        ):
            self._parse(fixture)

    def test_malformed_task_identity_fails_closed(self) -> None:
        fixtures = []
        missing_task_name = self._fixture()
        missing_task_name["task_name"] = None
        fixtures.append(missing_task_name)
        missing_task_id = self._fixture()
        missing_task_id["task_id"] = None
        fixtures.append(missing_task_id)
        missing_task_path = self._fixture()
        missing_task_path["task_id"] = {"path": None}
        fixtures.append(missing_task_path)

        for fixture in fixtures:
            with self.subTest(fixture=fixture):
                with self.assertRaises(InfrastructureInvalidTrialError):
                    self._parse(fixture)

    def test_infrastructure_exception_does_not_become_zero_score(self) -> None:
        fixture = self._fixture(0, exception_type="EnvironmentStartTimeoutError")

        with self.assertRaisesRegex(
            InfrastructureInvalidTrialError,
            "EnvironmentStartTimeoutError",
        ):
            self._parse(fixture)

    def test_cancelled_trial_fails_even_if_reward_is_present(self) -> None:
        fixture = self._fixture(0, exception_type="CancelledError")

        with self.assertRaisesRegex(
            InfrastructureInvalidTrialError,
            "CancelledError",
        ):
            self._parse(fixture)

    def test_incomplete_trial_fails_closed(self) -> None:
        fixtures = []
        missing_finished = self._fixture()
        missing_finished["finished_at"] = None
        fixtures.append(missing_finished)
        incomplete_verifier = self._fixture()
        incomplete_verifier["verifier"]["finished_at"] = None
        fixtures.append(incomplete_verifier)

        for fixture in fixtures:
            with self.subTest(fixture=fixture):
                with self.assertRaisesRegex(
                    InfrastructureInvalidTrialError,
                    "incomplete",
                ):
                    self._parse(fixture)

    def test_agent_timeout_with_trustworthy_reward_is_scored(self) -> None:
        result = self._parse(self._fixture(0, exception_type="AgentTimeoutError"))

        self.assertEqual(result["hard"], 0.0)
        self.assertEqual(result["soft"], 0.0)
        self.assertEqual(result["trial_status"], "AgentTimeoutError")

    def test_agent_timeout_without_reward_fails_closed(self) -> None:
        fixture = self._fixture(0, exception_type="AgentTimeoutError")
        fixture["verifier_result"] = None

        with self.assertRaises(InfrastructureInvalidTrialError):
            self._parse(fixture)

    def test_agent_nonzero_exit_with_trustworthy_reward_is_scored(self) -> None:
        result = self._parse(
            self._fixture(0, exception_type="NonZeroAgentExitCodeError")
        )

        self.assertEqual(result["hard"], 0.0)
        self.assertEqual(result["soft"], 0.0)
        self.assertEqual(result["trial_status"], "NonZeroAgentExitCodeError")


if __name__ == "__main__":
    unittest.main()
