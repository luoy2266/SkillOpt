"""Tests for the generic OpenAI-compatible model backend."""

from __future__ import annotations

import hashlib
import json
from types import SimpleNamespace
from typing import Any

import pytest

import skillopt.model as model
from skillopt.model import backend_config
from skillopt.model import openai_compatible_backend as backend
from skillopt.model.common import capture_model_diagnostics


class _CompletionRecorder:
    def __init__(
        self,
        *,
        content: str = "ok",
        finish_reason: str = "stop",
    ) -> None:
        self.calls: list[dict[str, Any]] = []
        self.content = content
        self.finish_reason = finish_reason

    def create(self, **kwargs: Any) -> Any:
        self.calls.append(kwargs)
        message = SimpleNamespace(content=self.content, tool_calls=[])
        usage = SimpleNamespace(prompt_tokens=2, completion_tokens=3, total_tokens=5)
        return SimpleNamespace(
            id="response-id",
            model="provider-model",
            choices=[
                SimpleNamespace(
                    message=message,
                    finish_reason=self.finish_reason,
                )
            ],
            usage=usage,
        )


class _Client:
    def __init__(self, recorder: _CompletionRecorder) -> None:
        self.chat = SimpleNamespace(completions=recorder)


@pytest.fixture(autouse=True)
def isolate_backend_state(monkeypatch: pytest.MonkeyPatch):
    optimizer_backend = backend_config.get_optimizer_backend()
    target_backend = backend_config.get_target_backend()
    optimizer_config = vars(backend.OPTIMIZER_CONFIG).copy()
    target_config = vars(backend.TARGET_CONFIG).copy()
    reasoning_effort = backend.REASONING_EFFORT
    backend.reset_token_tracker()
    yield
    backend.reset_token_tracker()
    vars(backend.OPTIMIZER_CONFIG).update(optimizer_config)
    vars(backend.TARGET_CONFIG).update(target_config)
    backend.REASONING_EFFORT = reasoning_effort
    backend_config.set_optimizer_backend(optimizer_backend)
    backend_config.set_target_backend(target_backend)
    backend._reset_clients()


def test_configure_preserves_role_specific_values() -> None:
    model.configure_openai_compatible(
        base_url="https://shared.example/v1",
        api_key="shared-key",
        model="shared-model",
        optimizer_base_url="https://optimizer.example/v1",
        optimizer_api_key="optimizer-key",
        optimizer_model="optimizer-model",
        target_base_url="https://target.example/v1",
        target_api_key="target-key",
        target_model="target-model",
    )

    assert backend.OPTIMIZER_CONFIG.base_url == "https://optimizer.example/v1"
    assert backend.OPTIMIZER_CONFIG.api_key == "optimizer-key"
    assert backend.OPTIMIZER_CONFIG.deployment == "optimizer-model"
    assert backend.TARGET_CONFIG.base_url == "https://target.example/v1"
    assert backend.TARGET_CONFIG.api_key == "target-key"
    assert backend.TARGET_CONFIG.deployment == "target-model"


def test_optimizer_and_target_route_to_their_own_clients(monkeypatch: pytest.MonkeyPatch) -> None:
    optimizer_calls = _CompletionRecorder()
    target_calls = _CompletionRecorder()
    monkeypatch.setattr(
        backend,
        "_get_client",
        lambda role: _Client(optimizer_calls if role == "optimizer" else target_calls),
    )
    model.set_optimizer_backend("openai_compatible")
    model.set_target_backend("openai_compatible")
    backend.OPTIMIZER_CONFIG.deployment = "optimizer-model"
    backend.TARGET_CONFIG.deployment = "target-model"

    optimizer_text, optimizer_usage = model.chat_optimizer(
        "system",
        "user",
        max_completion_tokens=123,
        retries=1,
        reasoning_effort="max",
        timeout=7,
    )
    target_text, target_usage = model.chat_target_messages(
        [{"role": "user", "content": "question"}], retries=1
    )

    assert optimizer_calls.calls[0]["model"] == "optimizer-model"
    assert optimizer_calls.calls[0]["messages"] == [
        {"role": "system", "content": "system"},
        {"role": "user", "content": "user"},
    ]
    assert optimizer_calls.calls[0]["max_tokens"] == 123
    assert optimizer_calls.calls[0]["reasoning_effort"] == "max"
    assert optimizer_calls.calls[0]["timeout"] == 7
    assert target_calls.calls[0]["model"] == "target-model"
    assert "reasoning_effort" not in target_calls.calls[0]
    assert optimizer_text == target_text == "ok"
    assert optimizer_usage == target_usage == {
        "prompt_tokens": 2,
        "completion_tokens": 3,
        "total_tokens": 5,
    }


@pytest.mark.parametrize("effort", ["max", "high"])
def test_configured_reasoning_effort_is_forwarded(
    monkeypatch: pytest.MonkeyPatch,
    effort: str,
) -> None:
    calls = _CompletionRecorder()
    monkeypatch.setattr(backend, "_get_client", lambda role: _Client(calls))
    model.set_optimizer_backend("openai_compatible")

    backend.set_reasoning_effort(effort)
    model.chat_optimizer("system", "user", retries=1)

    assert calls.calls[0]["reasoning_effort"] == effort


def test_unset_reasoning_effort_is_omitted(monkeypatch: pytest.MonkeyPatch) -> None:
    calls = _CompletionRecorder()
    monkeypatch.setattr(backend, "_get_client", lambda role: _Client(calls))
    model.set_optimizer_backend("openai_compatible")

    backend.set_reasoning_effort(None)
    model.chat_optimizer("system", "user", retries=1)

    assert "reasoning_effort" not in calls.calls[0]


def test_client_creation_is_lazy(monkeypatch: pytest.MonkeyPatch) -> None:
    builds: list[str] = []

    def build(config: backend.OpenAICompatibleConfig) -> _Client:
        builds.append(config.deployment)
        return _Client(_CompletionRecorder())

    backend._reset_clients()
    monkeypatch.setattr(backend, "_build_client", build)
    assert builds == []

    backend._get_client("optimizer")
    backend._get_client("optimizer")

    assert builds == [backend.OPTIMIZER_CONFIG.deployment]


@pytest.mark.parametrize("finish_reason", ["stop", "length"])
def test_optimizer_diagnostics_capture_choice_and_token_metadata(
    monkeypatch: pytest.MonkeyPatch,
    finish_reason: str,
) -> None:
    response = '{"status":"ok"}'
    calls = _CompletionRecorder(content=response, finish_reason=finish_reason)
    monkeypatch.setattr(backend, "_get_client", lambda role: _Client(calls))
    model.set_optimizer_backend("openai_compatible")
    backend.OPTIMIZER_CONFIG.base_url = "https://provider.example/v1/chat/completions"
    backend.OPTIMIZER_CONFIG.api_key = "sk-THIS-MUST-NOT-APPEAR"
    backend.OPTIMIZER_CONFIG.deployment = "optimizer-model"
    backend.OPTIMIZER_CONFIG.max_tokens = 8000
    backend.set_reasoning_effort("max")
    events: list[dict[str, Any]] = []

    with capture_model_diagnostics(
        events.append,
        context={"logical_request_id": "reflection-0001"},
    ):
        text, usage = model.chat_optimizer(
            "system",
            "user",
            max_completion_tokens=16384,
            retries=1,
            stage="analyst",
        )

    assert text == response
    assert usage == {
        "prompt_tokens": 2,
        "completion_tokens": 3,
        "total_tokens": 5,
    }
    assert calls.calls[0]["max_tokens"] == 8000
    assert calls.calls[0]["reasoning_effort"] == "max"
    assert len(events) == 1
    event = events[0]
    assert event["logical_request_id"] == "reflection-0001"
    assert event["role"] == "optimizer"
    assert event["backend"] == "openai_compatible"
    assert event["model"] == "optimizer-model"
    assert event["endpoint"] == "https://provider.example"
    assert event["reasoning_effort"] == "max"
    assert event["caller_requested_max_tokens"] == 16384
    assert event["backend_configured_max_tokens"] == 8000
    assert event["effective_request_max_tokens"] == 8000
    assert event["finish_reason"] == finish_reason
    assert event["response_bytes"] == len(response.encode("utf-8"))
    assert event["response_sha256"] == hashlib.sha256(response.encode("utf-8")).hexdigest()
    assert event["usage"] == usage
    assert event["response_id"] == "response-id"
    assert event["provider_model"] == "provider-model"
    assert event["backend_attempt_count"] == 1
    assert event["backend_attempts"][0]["backend_attempt_index"] == 1
    assert event["sdk_transport_attempts"] == "unknown"
    assert "sk-THIS-MUST-NOT-APPEAR" not in json.dumps(event)


def test_optimizer_diagnostics_redact_credentials_from_errors(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    secret = "sk-THIS-MUST-NOT-APPEAR"

    class FailingCompletions:
        def create(self, **kwargs: Any) -> Any:
            raise RuntimeError(f"Authorization: Bearer {secret}; api_key={secret}")

    client = SimpleNamespace(chat=SimpleNamespace(completions=FailingCompletions()))
    monkeypatch.setattr(backend, "_get_client", lambda role: client)
    monkeypatch.setattr(backend.time, "sleep", lambda seconds: None)
    model.set_optimizer_backend("openai_compatible")
    backend.OPTIMIZER_CONFIG.api_key = secret
    events: list[dict[str, Any]] = []

    with capture_model_diagnostics(events.append):
        with pytest.raises(RuntimeError, match="failed after 1 retries"):
            model.chat_optimizer("system", "user", retries=1)

    assert len(events) == 1
    serialized = json.dumps(events[0])
    assert secret not in serialized
    assert "[REDACTED]" in serialized


def test_combined_token_summary_counts_each_backend_once(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def make(stage: str) -> dict[str, dict[str, int]]:
        return {
            stage: {
                "calls": 1,
                "prompt_tokens": 2,
                "completion_tokens": 3,
                "total_tokens": 5,
            },
            "_total": {
                "calls": 1,
                "prompt_tokens": 2,
                "completion_tokens": 3,
                "total_tokens": 5,
            },
        }

    monkeypatch.setattr(model._openai, "get_token_summary", lambda: make("azure"))
    monkeypatch.setattr(model._claude, "get_token_summary", lambda: make("claude"))
    monkeypatch.setattr(model._qwen, "get_token_summary", lambda: make("qwen"))
    monkeypatch.setattr(model._minimax, "get_token_summary", lambda: make("minimax"))
    monkeypatch.setattr(model._openai_compat, "get_token_summary", lambda: make("openai_compatible"))

    combined = model.get_token_summary()

    assert set(combined) - {"_total"} == {"azure", "claude", "qwen", "minimax", "openai_compatible"}
    expected_stage_total = {
        "calls": 1,
        "prompt_tokens": 2,
        "completion_tokens": 3,
        "total_tokens": 5,
    }
    for stage in {"azure", "claude", "qwen", "minimax", "openai_compatible"}:
        assert combined[stage] == expected_stage_total
    assert combined["_total"] == {
        "calls": 5,
        "prompt_tokens": 10,
        "completion_tokens": 15,
        "total_tokens": 25,
    }
