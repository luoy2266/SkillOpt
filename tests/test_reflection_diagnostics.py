"""Offline tests for step-local optimizer/reflection diagnostics."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

import skillopt.gradient.reflect as reflect
import skillopt.model as model
from skillopt.gradient.optimizer_diagnostics import OptimizerDiagnosticsWriter
from skillopt.model import backend_config
from skillopt.model import openai_compatible_backend as backend
from skillopt.model.common import emit_model_diagnostic


def _write_conversation(prediction_dir: Path, task_id: str = "task-1") -> None:
    task_dir = prediction_dir / task_id
    task_dir.mkdir(parents=True)
    (task_dir / "conversation.json").write_text(
        json.dumps(
            [
                {"role": "user", "content": "Inspect the evidence."},
                {"role": "assistant", "content": "The attempt failed."},
            ]
        ),
        encoding="utf-8",
    )


def _backend_event(response: str, finish_reason: str = "stop") -> dict[str, Any]:
    encoded = response.encode("utf-8")
    return {
        "stage": "analyst",
        "role": "optimizer",
        "backend": "openai_compatible",
        "model": "optimizer-model",
        "endpoint": "https://provider.example",
        "reasoning_effort": "max",
        "caller_requested_max_tokens": 16384,
        "backend_configured_max_tokens": 8000,
        "effective_request_max_tokens": 8000,
        "request_chars": 123,
        "start_timestamp": "2026-08-30T00:00:00+00:00",
        "finish_timestamp": "2026-08-30T00:00:01+00:00",
        "latency_ms": 1000.0,
        "success": True,
        "usage": {
            "prompt_tokens": 10,
            "completion_tokens": 5,
            "total_tokens": 15,
        },
        "finish_reason": finish_reason,
        "response_bytes": len(encoded),
        "response_sha256": hashlib.sha256(encoded).hexdigest(),
        "response_id": "response-id",
        "provider_model": "provider-model",
        "backend_attempt_count": 1,
        "backend_attempts": [
            {"backend_attempt_index": 1, "outcome": "success", "latency_ms": 1000.0}
        ],
        "sdk_transport_attempts": "unknown",
    }


def _run_response_case(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    response: str,
) -> tuple[list[dict | None], Path, Path]:
    step_dir = tmp_path / "step_0001"
    prediction_dir = step_dir / "rollout" / "predictions"
    patches_dir = step_dir / "patches"
    diagnostics_dir = step_dir / "optimizer_diagnostics"
    _write_conversation(prediction_dir)

    def fake_chat_optimizer(*args: Any, **kwargs: Any):
        emit_model_diagnostic(_backend_event(response))
        return response, {
            "prompt_tokens": 10,
            "completion_tokens": 5,
            "total_tokens": 15,
        }

    monkeypatch.setattr(reflect, "chat_optimizer", fake_chat_optimizer)
    patches = reflect.run_minibatch_reflect(
        results=[{"id": "task-1", "hard": 0.0, "soft": 0.0}],
        skill_content="\n",
        prediction_dir=str(prediction_dir),
        patches_dir=str(patches_dir),
        diagnostics_dir=str(diagnostics_dir),
        workers=1,
        failure_only=True,
        minibatch_size=1,
        edit_budget=4,
        random_seed=42,
    )
    return patches, patches_dir, diagnostics_dir


def _attempts(diagnostics_dir: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in (diagnostics_dir / "attempts.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def test_valid_patch_persists_raw_response_parse_and_patch(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    response = json.dumps({
        "batch_size": 1,
        "failure_summary": [],
        "patch": {
            "reasoning": "test",
            "edits": [
                {
                    "op": "append",
                    "target": "",
                    "content": "Use evidence.",
                }
            ],
        },
    })

    patches, patches_dir, diagnostics_dir = _run_response_case(
        tmp_path,
        monkeypatch,
        response,
    )

    assert len(patches) == 1
    assert (patches_dir / "minibatch_fail_000.json").is_file()
    assert (diagnostics_dir / "response_0001.txt").read_text(encoding="utf-8") == response
    assert json.loads((diagnostics_dir / "parse_0001.json").read_text()) == json.loads(response)
    attempt = _attempts(diagnostics_dir)[0]
    assert attempt["parse_status"] == "patch_returned"
    assert attempt["edit_count_before_budget"] == 1
    assert attempt["edit_count_after_budget"] == 1
    assert attempt["returned_patch_truthy"] is True
    assert attempt["finish_reason"] == "stop"
    assert attempt["response_sha256"] == hashlib.sha256(response.encode()).hexdigest()
    assert attempt["system_prompt_chars"] > 0
    assert attempt["user_prompt_chars"] > 0
    assert attempt["trajectory_formatted_chars"] > 0
    assert not list(diagnostics_dir.glob("prompt_*.txt"))


def test_missing_patch_is_diagnosed_without_changing_return_semantics(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    response = '{"batch_size":1}'

    patches, patches_dir, diagnostics_dir = _run_response_case(
        tmp_path,
        monkeypatch,
        response,
    )

    assert patches == []
    assert not (patches_dir / "minibatch_fail_000.json").exists()
    attempt = _attempts(diagnostics_dir)[0]
    assert attempt["parse_status"] == "missing_patch"
    assert attempt["json_extraction_success"] is True
    assert attempt["patch_key_present"] is False
    assert json.loads((diagnostics_dir / "parse_0001.json").read_text()) == {
        "batch_size": 1
    }


@pytest.mark.parametrize(
    "response",
    [
        "not json",
        '{"patch":{"edits":[',
    ],
)
def test_invalid_or_truncated_json_records_extraction_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    response: str,
) -> None:
    patches, patches_dir, diagnostics_dir = _run_response_case(
        tmp_path,
        monkeypatch,
        response,
    )

    assert patches == []
    assert not (patches_dir / "minibatch_fail_000.json").exists()
    assert not (diagnostics_dir / "parse_0001.json").exists()
    attempt = _attempts(diagnostics_dir)[0]
    assert attempt["parse_status"] == "json_extract_failed"
    assert attempt["json_extraction_failure"] is True
    assert (diagnostics_dir / "response_0001.txt").read_text() == response


def test_empty_edits_are_distinguished_and_keep_current_patch_behavior(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    response = '{"patch":{"edits":[]}}'

    patches, patches_dir, diagnostics_dir = _run_response_case(
        tmp_path,
        monkeypatch,
        response,
    )

    assert len(patches) == 1
    assert (patches_dir / "minibatch_fail_000.json").is_file()
    attempt = _attempts(diagnostics_dir)[0]
    assert attempt["parse_status"] == "empty_edits"
    assert attempt["edit_count_before_budget"] == 0
    assert attempt["edit_count_after_budget"] == 0
    assert attempt["returned_patch_truthy"] is True


@pytest.mark.parametrize(
    ("response", "expected_status"),
    [
        ('{"patch":"invalid"}', "invalid_patch_type"),
        ('{"patch":{}}', "missing_edits"),
        ('{"patch":{"edits":"invalid"}}', "invalid_edits_type"),
    ],
)
def test_patch_schema_statuses_observe_without_rejecting_earlier(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    response: str,
    expected_status: str,
) -> None:
    patches, patches_dir, diagnostics_dir = _run_response_case(
        tmp_path,
        monkeypatch,
        response,
    )

    assert len(patches) == 1
    assert (patches_dir / "minibatch_fail_000.json").is_file()
    attempt = _attempts(diagnostics_dir)[0]
    assert attempt["parse_status"] == expected_status
    assert attempt["returned_patch_truthy"] is True


def test_empty_response_has_distinct_status(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    patches, patches_dir, diagnostics_dir = _run_response_case(
        tmp_path,
        monkeypatch,
        "",
    )

    assert patches == []
    assert not (patches_dir / "minibatch_fail_000.json").exists()
    assert (diagnostics_dir / "response_0001.txt").read_bytes() == b""
    attempt = _attempts(diagnostics_dir)[0]
    assert attempt["parse_status"] == "empty_response"
    assert attempt["response_received"] is True
    assert attempt["response_bytes"] == 0


def test_parser_exception_preserves_response_and_sanitizes_error(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    response = '{"patch":{"edits":[]}}'
    secret = "sk-THIS-MUST-NOT-APPEAR"

    def fail_extract_json(text: str):
        raise RuntimeError(f"parser failed with Authorization: Bearer {secret}")

    monkeypatch.setattr(reflect, "extract_json", fail_extract_json)
    patches, patches_dir, diagnostics_dir = _run_response_case(
        tmp_path,
        monkeypatch,
        response,
    )

    assert patches == []
    assert not (patches_dir / "minibatch_fail_000.json").exists()
    assert (diagnostics_dir / "response_0001.txt").read_text() == response
    attempt = _attempts(diagnostics_dir)[0]
    assert attempt["parse_status"] == "exception"
    assert attempt["exception_class"] == "RuntimeError"
    assert secret not in json.dumps(attempt)
    assert "[REDACTED]" in attempt["exception_message"]


def test_diagnostic_write_failure_warns_but_preserves_patch(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    response = '{"patch":{"edits":[{"op":"append","content":"Use evidence."}]}}'

    def fail_response_write(self: OptimizerDiagnosticsWriter, index: int, text: str):
        raise OSError("simulated diagnostics write failure")

    monkeypatch.setattr(OptimizerDiagnosticsWriter, "write_response", fail_response_write)
    with pytest.warns(RuntimeWarning, match="response_write failed"):
        patches, patches_dir, diagnostics_dir = _run_response_case(
            tmp_path,
            monkeypatch,
            response,
        )

    assert len(patches) == 1
    assert (patches_dir / "minibatch_fail_000.json").is_file()
    attempt = _attempts(diagnostics_dir)[0]
    assert attempt["parse_status"] == "patch_returned"
    assert attempt["diagnostics_write_errors"][0]["operation"] == "response_write"


def test_diagnostic_numbering_is_stable_and_never_overwrites(
    tmp_path: Path,
) -> None:
    root = tmp_path / "optimizer_diagnostics"
    first = OptimizerDiagnosticsWriter(root)
    assert first.reserve(1) == [1]
    first.write_response(1, "first")
    first.write_attempt(1, {"logical_request_id": "first"})

    resumed = OptimizerDiagnosticsWriter(root)
    assert resumed.reserve(1) == [2]
    resumed.write_response(2, "second")
    resumed.write_attempt(2, {"logical_request_id": "second"})

    assert (root / "response_0001.txt").read_text() == "first"
    assert (root / "response_0002.txt").read_text() == "second"
    assert [row["diagnostic_index"] for row in _attempts(root)] == [1, 2]
    with pytest.raises(RuntimeError, match="already exists"):
        resumed.write_response(1, "replacement")


@pytest.fixture
def isolate_openai_compatible_state():
    optimizer_backend = backend_config.get_optimizer_backend()
    optimizer_config = vars(backend.OPTIMIZER_CONFIG).copy()
    reasoning_effort = backend.REASONING_EFFORT
    yield
    vars(backend.OPTIMIZER_CONFIG).update(optimizer_config)
    backend.REASONING_EFFORT = reasoning_effort
    backend_config.set_optimizer_backend(optimizer_backend)
    backend._reset_clients()


def test_step_artifacts_exclude_optimizer_credentials(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    isolate_openai_compatible_state: None,
) -> None:
    secret = "sk-THIS-MUST-NOT-APPEAR"
    response = '{"patch":{"edits":[{"op":"append","content":"Use evidence."}]}}'
    calls: list[dict[str, Any]] = []

    class Completions:
        def create(self, **kwargs: Any) -> Any:
            calls.append(kwargs)
            message = SimpleNamespace(content=response, tool_calls=[])
            usage = SimpleNamespace(prompt_tokens=10, completion_tokens=5, total_tokens=15)
            return SimpleNamespace(
                id="response-id",
                model="provider-model",
                choices=[SimpleNamespace(message=message, finish_reason="length")],
                usage=usage,
            )

    client = SimpleNamespace(chat=SimpleNamespace(completions=Completions()))
    monkeypatch.setattr(backend, "_get_client", lambda role: client)
    model.set_optimizer_backend("openai_compatible")
    backend.OPTIMIZER_CONFIG.base_url = "https://provider.example/v1"
    backend.OPTIMIZER_CONFIG.api_key = secret
    backend.OPTIMIZER_CONFIG.deployment = "optimizer-model"
    backend.OPTIMIZER_CONFIG.max_tokens = 8000
    backend.set_reasoning_effort("max")

    step_dir = tmp_path / "step_0001"
    prediction_dir = step_dir / "rollout" / "predictions"
    _write_conversation(prediction_dir)
    reflect.run_minibatch_reflect(
        results=[{"id": "task-1", "hard": 0.0, "soft": 0.0}],
        skill_content="\n",
        prediction_dir=str(prediction_dir),
        patches_dir=str(step_dir / "patches"),
        diagnostics_dir=str(step_dir / "optimizer_diagnostics"),
        workers=1,
        failure_only=True,
        minibatch_size=1,
    )

    assert calls[0]["model"] == "optimizer-model"
    assert calls[0]["reasoning_effort"] == "max"
    assert calls[0]["max_tokens"] == 8000
    attempt = _attempts(step_dir / "optimizer_diagnostics")[0]
    assert attempt["finish_reason"] == "length"
    assert attempt["caller_requested_max_tokens"] == 16384
    assert attempt["backend_configured_max_tokens"] == 8000
    assert attempt["effective_request_max_tokens"] == 8000

    artifact_text = "\n".join(
        path.read_text(encoding="utf-8")
        for path in (step_dir / "optimizer_diagnostics").iterdir()
        if path.is_file()
    )
    assert secret not in artifact_text
