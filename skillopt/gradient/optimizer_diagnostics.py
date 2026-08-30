"""Step-local, secret-safe optimizer/reflection diagnostic artifacts."""
from __future__ import annotations

import copy
import hashlib
import json
import os
import re
import tempfile
import threading
import warnings
from contextlib import nullcontext
from pathlib import Path
from typing import Any

from skillopt.model import get_optimizer_backend
from skillopt.model.common import capture_model_diagnostics
from skillopt.optimizer.update_modes import (
    get_payload_items,
    normalize_update_mode,
    payload_key,
)


SCHEMA_VERSION = "skillopt-optimizer-diagnostics-v1"
_INDEXED_ARTIFACT = re.compile(r"^(?:response|parse)_(\d{4})\.(?:txt|json)$")


def text_identity(text: str) -> dict[str, int | str]:
    encoded = text.encode("utf-8")
    return {
        "chars": len(text),
        "bytes": len(encoded),
        "sha256": hashlib.sha256(encoded).hexdigest(),
    }


def _sanitize_error_message(exc: Exception) -> str:
    message = str(exc)
    message = re.sub(r"(?i)bearer\s+[^\s,;]+", "Bearer [REDACTED]", message)
    message = re.sub(
        r"(?i)(authorization\s*[:=]\s*)([^\s,;]+)",
        r"\1[REDACTED]",
        message,
    )
    message = re.sub(r"\bsk-[A-Za-z0-9_-]+", "[REDACTED]", message)
    return message[:1000]


class OptimizerDiagnosticsWriter:
    """Write deterministic reflection diagnostics without overwriting attempts."""

    def __init__(self, root: str | os.PathLike[str]) -> None:
        self.root = Path(root)
        if self.root.is_symlink():
            raise RuntimeError(f"Optimizer diagnostics directory must not be a symlink: {self.root}")
        self.root.mkdir(parents=True, exist_ok=True)
        if not self.root.is_dir() or self.root.is_symlink():
            raise RuntimeError(f"Optimizer diagnostics path is not a directory: {self.root}")
        self._lock = threading.Lock()
        self._attempts_path = self.root / "attempts.jsonl"
        self._records = self._load_records()
        self._next_index = self._discover_next_index()
        self._probe_writable()

    def reserve(self, count: int) -> list[int]:
        if count < 0:
            raise ValueError("Diagnostic reservation count must be non-negative")
        with self._lock:
            reserved = list(range(self._next_index, self._next_index + count))
            self._next_index += count
            return reserved

    def write_response(self, index: int, response: str) -> dict[str, Any]:
        path = self.root / f"response_{index:04d}.txt"
        encoded = response.encode("utf-8")
        self._atomic_create(path, encoded)
        return {
            "response_artifact": path.name,
            "response_bytes": len(encoded),
            "response_sha256": hashlib.sha256(encoded).hexdigest(),
        }

    def write_parse(self, index: int, extracted: dict[str, Any]) -> str:
        path = self.root / f"parse_{index:04d}.json"
        content = json.dumps(
            extracted,
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        ).encode("utf-8") + b"\n"
        self._atomic_create(path, content)
        return path.name

    def write_attempt(self, index: int, record: dict[str, Any]) -> None:
        payload = dict(record)
        payload["schema_version"] = SCHEMA_VERSION
        payload["diagnostic_index"] = index
        with self._lock:
            if index in self._records:
                raise RuntimeError(f"Optimizer diagnostic attempt already exists: {index:04d}")
            updated = dict(self._records)
            updated[index] = payload
            self._atomic_replace_attempts(updated)
            self._records = updated

    def _load_records(self) -> dict[int, dict[str, Any]]:
        if not self._attempts_path.exists():
            return {}
        if self._attempts_path.is_symlink() or not self._attempts_path.is_file():
            raise RuntimeError(
                f"Optimizer diagnostics ledger is not a regular file: {self._attempts_path}"
            )
        records: dict[int, dict[str, Any]] = {}
        with self._attempts_path.open(encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, 1):
                if not line.strip():
                    continue
                try:
                    record = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise RuntimeError(
                        f"Optimizer diagnostics ledger is malformed at line {line_number}"
                    ) from exc
                index = record.get("diagnostic_index") if isinstance(record, dict) else None
                if isinstance(index, bool) or not isinstance(index, int) or index <= 0:
                    raise RuntimeError(
                        f"Optimizer diagnostics ledger has invalid index at line {line_number}"
                    )
                if index in records:
                    raise RuntimeError(f"Duplicate optimizer diagnostic index: {index:04d}")
                records[index] = record
        return records

    def _discover_next_index(self) -> int:
        indexes = set(self._records)
        for path in self.root.iterdir():
            match = _INDEXED_ARTIFACT.fullmatch(path.name)
            if match:
                indexes.add(int(match.group(1)))
        return max(indexes, default=0) + 1

    def _probe_writable(self) -> None:
        descriptor, temporary_name = tempfile.mkstemp(
            dir=self.root,
            prefix=".diagnostics-probe-",
            suffix=".tmp",
        )
        os.close(descriptor)
        Path(temporary_name).unlink()

    def _atomic_create(self, path: Path, content: bytes) -> None:
        if path.exists() or path.is_symlink():
            raise RuntimeError(f"Optimizer diagnostic artifact already exists: {path}")
        descriptor, temporary_name = tempfile.mkstemp(
            dir=self.root,
            prefix=f".{path.name}.",
            suffix=".tmp",
        )
        temporary_path = Path(temporary_name)
        try:
            with os.fdopen(descriptor, "wb") as handle:
                handle.write(content)
                handle.flush()
                os.fsync(handle.fileno())
            try:
                os.link(temporary_path, path)
            except FileExistsError as exc:
                raise RuntimeError(
                    f"Optimizer diagnostic artifact already exists: {path}"
                ) from exc
        finally:
            temporary_path.unlink(missing_ok=True)

    def _atomic_replace_attempts(self, records: dict[int, dict[str, Any]]) -> None:
        if self._attempts_path.is_symlink():
            raise RuntimeError(
                f"Optimizer diagnostics ledger must not be a symlink: {self._attempts_path}"
            )
        descriptor, temporary_name = tempfile.mkstemp(
            dir=self.root,
            prefix=".attempts.jsonl.",
            suffix=".tmp",
        )
        temporary_path = Path(temporary_name)
        try:
            with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as handle:
                for index in sorted(records):
                    handle.write(json.dumps(records[index], ensure_ascii=False, sort_keys=True))
                    handle.write("\n")
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temporary_path, self._attempts_path)
        finally:
            temporary_path.unlink(missing_ok=True)


class AnalystDiagnosticSession:
    """Collect one reflection analyst request without changing parser decisions."""

    def __init__(
        self,
        *,
        writer: OptimizerDiagnosticsWriter | None,
        diagnostic_index: int | None,
        logical_request_id: str,
        system: str,
        user: str,
        trajectories_text: str,
        skill_content: str,
        items: list[dict],
        source_type: str,
        update_mode: str,
    ) -> None:
        self.writer = writer
        self.diagnostic_index = diagnostic_index
        self.logical_request_id = logical_request_id
        self.response: str | None = None
        self.extracted_snapshot = None
        self.backend_events: list[dict[str, Any]] = []
        self.parse_status = "transport_error"
        self.parse_fields = {
            "json_extraction_success": False,
            "json_extraction_failure": True,
            "top_level_type": None,
            "patch_key_present": False,
            "patch_type": None,
            "edits_present": False,
            "edits_type": None,
            "edit_count_before_budget": None,
            "edit_count_after_budget": None,
            "returned_patch_truthy": False,
        }
        self.record: dict[str, Any] = {}
        if self.enabled:
            system_identity = text_identity(system)
            user_identity = text_identity(user)
            self.record = {
                "logical_request_id": logical_request_id,
                "pipeline": "reflection",
                "request_role": "optimizer",
                "source_type": source_type,
                "update_mode": normalize_update_mode(update_mode),
                "batch_size": len(items),
                "task_ids": [str(item.get("id", "")) for item in items],
                "current_skill_chars": len(skill_content),
                "trajectory_formatted_chars": len(trajectories_text),
                "system_prompt_chars": system_identity["chars"],
                "system_prompt_bytes": system_identity["bytes"],
                "system_prompt_sha256": system_identity["sha256"],
                "user_prompt_chars": user_identity["chars"],
                "user_prompt_bytes": user_identity["bytes"],
                "user_prompt_sha256": user_identity["sha256"],
                "prompt_chars": len(system) + len(user),
            }

    @property
    def enabled(self) -> bool:
        return self.writer is not None and self.diagnostic_index is not None

    def capture_backend(self):
        if not self.enabled:
            return nullcontext()
        return capture_model_diagnostics(
            self.backend_events.append,
            context={"logical_request_id": self.logical_request_id},
        )

    def record_response(self, response: str) -> None:
        self.response = response

    def record_parse(self, result, update_mode: str) -> None:
        self.extracted_snapshot = copy.deepcopy(result)
        self.parse_fields = self._parse_fields(result, update_mode)
        self.parse_status = self._parse_status(self.response or "", result)

    def record_after_budget(self, patch, update_mode: str) -> None:
        self.parse_fields["edit_count_after_budget"] = len(
            get_payload_items(patch, update_mode)
        )

    def record_returned(self, patch) -> None:
        self.parse_fields["returned_patch_truthy"] = bool(patch)

    def record_exception(self, exc: Exception) -> None:
        self.parse_status = "transport_error" if self.response is None else "exception"
        if self.enabled:
            self.record["exception_class"] = type(exc).__name__
            self.record["exception_message"] = _sanitize_error_message(exc)

    def persist(self) -> None:
        if not self.enabled:
            return
        record = dict(self.record)
        record.update(self.parse_fields)
        record["parse_status"] = self.parse_status
        record["response_received"] = self.response is not None
        if self.backend_events:
            record.update(self.backend_events[-1])
            record["backend_metadata_available"] = True
        else:
            record.setdefault("backend", get_optimizer_backend())
            record["backend_metadata_available"] = False
            record.setdefault("sdk_transport_attempts", "unknown")

        if self.response is not None:
            try:
                record.update(
                    self.writer.write_response(self.diagnostic_index, self.response)
                )
            except Exception as exc:  # noqa: BLE001 - preserve parser semantics
                self._record_write_error(record, "response_write", exc)
        if isinstance(self.extracted_snapshot, dict):
            try:
                record["parse_artifact"] = self.writer.write_parse(
                    self.diagnostic_index,
                    self.extracted_snapshot,
                )
            except Exception as exc:  # noqa: BLE001 - preserve parser semantics
                self._record_write_error(record, "parse_write", exc)
        try:
            self.writer.write_attempt(self.diagnostic_index, record)
        except Exception as exc:  # noqa: BLE001 - preserve parser semantics
            self._record_write_error(record, "attempt_write", exc)

    def _parse_fields(self, result, update_mode: str) -> dict[str, Any]:
        fields = dict(self.parse_fields)
        fields.update({
            "json_extraction_success": result is not None,
            "json_extraction_failure": result is None,
            "top_level_type": type(result).__name__ if result is not None else None,
        })
        if not isinstance(result, dict):
            return fields
        fields["patch_key_present"] = "patch" in result
        if "patch" not in result:
            return fields
        patch = result["patch"]
        fields["patch_type"] = type(patch).__name__
        if not isinstance(patch, dict):
            return fields
        key = payload_key(update_mode)
        fields["payload_key"] = key
        fields["edits_present"] = key in patch
        if key not in patch:
            return fields
        edits = patch[key]
        fields["edits_type"] = type(edits).__name__
        if isinstance(edits, list):
            fields["edit_count_before_budget"] = len(edits)
        return fields

    def _parse_status(self, response: str, result) -> str:
        if response == "":
            return "empty_response"
        if result is None:
            return "json_extract_failed"
        if not self.parse_fields["patch_key_present"]:
            return "missing_patch"
        if self.parse_fields["patch_type"] != "dict":
            return "invalid_patch_type"
        if not self.parse_fields["edits_present"]:
            return "missing_edits"
        if self.parse_fields["edits_type"] != "list":
            return "invalid_edits_type"
        if self.parse_fields["edit_count_before_budget"] == 0:
            return "empty_edits"
        return "patch_returned"

    def _record_write_error(self, record: dict, label: str, exc: Exception) -> None:
        record.setdefault("diagnostics_write_errors", []).append({
            "operation": label,
            "error_type": type(exc).__name__,
            "error_message": str(exc)[:1000],
        })
        warnings.warn(
            f"Optimizer diagnostics {label} failed: {type(exc).__name__}: {exc}",
            RuntimeWarning,
            stacklevel=3,
        )
