"""Terminal-Bench task-ID split dataloader."""
from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from skillopt.datasets.base import SPLIT_NAMES, SplitDataLoader

MANIFEST_FILENAME = "split_manifest.json"
MANIFEST_CHECKSUM_FILENAME = "split_manifest.sha256"
SPLIT_ITEMS_FILENAME = "items.json"
LEGACY_MANIFEST_SCHEMA_VERSION = 1
MANIFEST_SCHEMA_VERSION = 2
SEMANTIC_IDENTITY_SCHEMA = "skillopt-terminalbench-split-v1"
BENCHMARK_NAME = "terminal-bench"
BENCHMARK_VERSION = "2.1"

_WINDOWS_RESERVED_NAMES = {
    "aux",
    "clock$",
    "con",
    "nul",
    "prn",
    *(f"com{index}" for index in range(1, 10)),
    *(f"lpt{index}" for index in range(1, 10)),
}
_INVALID_FILENAME_CHARACTERS = set('<>:"/\\|?*')


def validate_task_id(value: Any, *, context: str) -> str:
    """Return a stable, portable task ID or raise a descriptive error."""
    if not isinstance(value, str):
        raise ValueError(f"{context} must be a string, got {type(value).__name__}")
    if not value:
        raise ValueError(f"{context} must not be empty")
    if value != value.strip() or any(character.isspace() for character in value):
        raise ValueError(f"{context} must not contain whitespace: {value!r}")
    if value in {".", ".."}:
        raise ValueError(f"{context} is not a safe filesystem name: {value!r}")
    if value.endswith("."):
        raise ValueError(f"{context} must not end with '.': {value!r}")
    if any(ord(character) < 32 or ord(character) == 127 for character in value):
        raise ValueError(f"{context} must not contain control characters")
    invalid = sorted(set(value) & _INVALID_FILENAME_CHARACTERS)
    if invalid:
        raise ValueError(f"{context} contains unsafe filename characters {invalid}: {value!r}")

    windows_stem = value.split(".", 1)[0].casefold()
    if windows_stem in _WINDOWS_RESERVED_NAMES:
        raise ValueError(f"{context} is a reserved filesystem name: {value!r}")
    return value


def normalize_manifest_item(raw: Any, *, context: str) -> dict[str, Any]:
    """Validate one materialized item and normalize its ``id`` field."""
    if not isinstance(raw, Mapping):
        raise ValueError(f"{context} must be a JSON object, got {type(raw).__name__}")
    if "id" not in raw:
        raise ValueError(f"{context} is missing required field 'id'")
    if any(not isinstance(key, str) for key in raw):
        raise ValueError(f"{context} contains a non-string metadata key")

    item = dict(raw)
    item["id"] = validate_task_id(item["id"], context=f"{context}.id")
    return item


def validate_split_partition(splits: Mapping[str, Sequence[dict[str, Any]]]) -> list[str]:
    """Validate all split items and return the IDs in split-file order."""
    seen: dict[str, str] = {}
    ordered_ids: list[str] = []
    for split_name in SPLIT_NAMES:
        if split_name not in splits:
            raise ValueError(f"Terminal-Bench split is missing {split_name!r}")
        for index, item in enumerate(splits[split_name]):
            item_id = validate_task_id(
                item.get("id"),
                context=f"{split_name}/{SPLIT_ITEMS_FILENAME}[{index}].id",
            )
            previous_split = seen.get(item_id)
            if previous_split is not None:
                raise ValueError(
                    "Terminal-Bench split contains duplicate task ID "
                    f"{item_id!r} in {previous_split!r} and {split_name!r}"
                )
            seen[item_id] = split_name
            ordered_ids.append(item_id)
    return ordered_ids


def task_ids_sha256(task_ids: Sequence[str]) -> str:
    """Hash the canonical sorted task-ID set."""
    canonical = json.dumps(
        sorted(task_ids),
        ensure_ascii=False,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(canonical).hexdigest()


def build_semantic_split_identity(
    splits: Mapping[str, Sequence[dict[str, Any]]],
    *,
    terminalbench_revision: str,
    seed: int,
) -> dict[str, Any]:
    """Build the host-independent identity for one frozen split."""
    return {
        "schema": SEMANTIC_IDENTITY_SCHEMA,
        "benchmark": BENCHMARK_NAME,
        "benchmark_version": BENCHMARK_VERSION,
        "terminalbench_revision": str(terminalbench_revision),
        "seed": int(seed),
        "train_task_ids": [item["id"] for item in splits["train"]],
        "val_task_ids": [item["id"] for item in splits["val"]],
        "test_task_ids": [item["id"] for item in splits["test"]],
    }


def semantic_split_sha256(identity: Mapping[str, Any]) -> str:
    canonical = json.dumps(
        dict(identity),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(canonical).hexdigest()


def _sha256_path(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _load_checksum(path: Path) -> str:
    try:
        text = path.read_text(encoding="utf-8").strip()
    except FileNotFoundError as exc:
        raise FileNotFoundError(f"Missing Terminal-Bench manifest checksum: {path}") from exc
    if not text:
        raise ValueError(f"Terminal-Bench manifest checksum is empty: {path}")
    digest = text.split()[0].lower()
    if re.fullmatch(r"[0-9a-f]{64}", digest) is None:
        raise ValueError(f"Malformed SHA-256 checksum in {path}: {digest!r}")
    return digest


class TerminalBenchDataLoader(SplitDataLoader):
    """Load validated Terminal-Bench v2.1 task-ID manifests.

    Runtime ratio splitting is intentionally disabled. Use
    ``scripts/materialize_terminalbench_split.py`` to create a provenance- and
    checksum-pinned ``train/val/test`` directory, then load it with
    ``split_mode="split_dir"``.
    """

    def __init__(
        self,
        split_dir: str = "",
        data_path: str = "",
        split_mode: str = "split_dir",
        split_ratio: str = "1:1:8",
        split_seed: int = 42,
        split_output_dir: str = "",
        seed: int = 42,
        limit: int = 0,
        **kwargs,
    ) -> None:
        super().__init__(
            split_dir=split_dir,
            data_path=data_path,
            split_mode=split_mode,
            split_ratio=split_ratio,
            split_seed=split_seed,
            split_output_dir=split_output_dir,
            seed=seed,
            limit=limit,
        )

    def load_raw_items(self, data_path: str) -> list[dict]:
        raise ValueError(
            "TerminalBenchDataLoader does not materialize ratio splits at runtime. "
            "Run scripts/materialize_terminalbench_split.py, then use split_mode=split_dir."
        )

    def load_split_items(self, split_path: str) -> list[dict]:
        items_path = Path(split_path) / SPLIT_ITEMS_FILENAME
        try:
            raw_items = json.loads(items_path.read_text(encoding="utf-8"))
        except FileNotFoundError as exc:
            raise FileNotFoundError(f"Missing Terminal-Bench split items: {items_path}") from exc
        except json.JSONDecodeError as exc:
            raise ValueError(f"Malformed JSON in Terminal-Bench split file {items_path}: {exc}") from exc
        if not isinstance(raw_items, list):
            raise ValueError(
                f"Expected JSON array in {items_path}, got {type(raw_items).__name__}"
            )
        return [
            normalize_manifest_item(raw, context=f"{items_path}[{index}]")
            for index, raw in enumerate(raw_items)
        ]

    def _load_all_splits(self) -> None:
        full_splits: dict[str, list[dict[str, Any]]] = {}
        for split_name in SPLIT_NAMES:
            split_path = Path(self.split_dir) / split_name
            if not split_path.is_dir():
                raise ValueError(
                    f"Missing '{split_name}/' subdirectory in split_dir: {self.split_dir}"
                )
            full_splits[split_name] = self.load_split_items(str(split_path))

        task_ids = validate_split_partition(full_splits)
        self._validate_manifest(full_splits, task_ids)

        self._splits = {
            split_name: items[: self.limit] if self.limit else items
            for split_name, items in full_splits.items()
        }
        counts = " ".join(f"{name}={len(items)}" for name, items in self._splits.items())
        print(f"  [{type(self).__name__}] {counts}  (from {self.split_dir})")

    def _validate_manifest(
        self,
        splits: Mapping[str, Sequence[dict[str, Any]]],
        task_ids: Sequence[str],
    ) -> None:
        split_root = Path(self.split_dir)
        manifest_path = split_root / MANIFEST_FILENAME
        checksum_path = split_root / MANIFEST_CHECKSUM_FILENAME

        expected_checksum = _load_checksum(checksum_path)
        try:
            manifest_content = manifest_path.read_bytes()
        except FileNotFoundError as exc:
            raise FileNotFoundError(f"Missing Terminal-Bench split manifest: {manifest_path}") from exc
        try:
            manifest = json.loads(manifest_content.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise ValueError(f"Malformed Terminal-Bench split manifest {manifest_path}: {exc}") from exc
        if not isinstance(manifest, dict):
            raise ValueError(f"Terminal-Bench split manifest must be a JSON object: {manifest_path}")

        schema_version = manifest.get("schema_version")
        if schema_version == LEGACY_MANIFEST_SCHEMA_VERSION:
            actual_manifest_sha256 = hashlib.sha256(manifest_content).hexdigest()
            if actual_manifest_sha256 != expected_checksum:
                raise ValueError(
                    "Terminal-Bench manifest checksum mismatch: "
                    f"expected {expected_checksum}, got {actual_manifest_sha256}"
                )
        elif schema_version != MANIFEST_SCHEMA_VERSION:
            raise ValueError(
                "Terminal-Bench manifest field 'schema_version' must be "
                f"{LEGACY_MANIFEST_SCHEMA_VERSION!r} or {MANIFEST_SCHEMA_VERSION!r}, "
                f"got {schema_version!r}"
            )

        expected_identity = {
            "benchmark": BENCHMARK_NAME,
            "benchmark_version": BENCHMARK_VERSION,
            "manifest_type": "id_split",
        }
        for field, expected in expected_identity.items():
            if manifest.get(field) != expected:
                raise ValueError(
                    f"Terminal-Bench manifest field {field!r} must be {expected!r}, "
                    f"got {manifest.get(field)!r}"
                )

        counts = manifest.get("counts")
        files = manifest.get("files")
        input_provenance = manifest.get("input")
        item_fields = manifest.get("item_fields")
        if not isinstance(counts, dict):
            raise ValueError("Terminal-Bench manifest field 'counts' must be an object")
        if not isinstance(files, dict):
            raise ValueError("Terminal-Bench manifest field 'files' must be an object")
        if not isinstance(input_provenance, dict):
            raise ValueError("Terminal-Bench manifest field 'input' must be an object")
        if not isinstance(item_fields, list) or not all(isinstance(field, str) for field in item_fields):
            raise ValueError("Terminal-Bench manifest field 'item_fields' must be a string array")

        actual_fields = sorted({field for items in splits.values() for item in items for field in item})
        if sorted(item_fields) != actual_fields:
            raise ValueError(
                "Terminal-Bench manifest item_fields mismatch: "
                f"declared {sorted(item_fields)!r}, actual {actual_fields!r}"
            )

        for split_name in SPLIT_NAMES:
            actual_count = len(splits[split_name])
            if counts.get(split_name) != actual_count:
                raise ValueError(
                    f"Terminal-Bench manifest count mismatch for {split_name}: "
                    f"declared {counts.get(split_name)!r}, actual {actual_count}"
                )

            file_provenance = files.get(split_name)
            if not isinstance(file_provenance, dict):
                raise ValueError(
                    f"Terminal-Bench manifest files.{split_name} must be an object"
                )
            expected_relative_path = f"{split_name}/{SPLIT_ITEMS_FILENAME}"
            if file_provenance.get("path") != expected_relative_path:
                raise ValueError(
                    f"Terminal-Bench manifest files.{split_name}.path must be "
                    f"{expected_relative_path!r}"
                )
            if file_provenance.get("count") != actual_count:
                raise ValueError(
                    f"Terminal-Bench manifest files.{split_name}.count mismatch: "
                    f"declared {file_provenance.get('count')!r}, actual {actual_count}"
                )
            actual_file_sha256 = _sha256_path(split_root / expected_relative_path)
            if file_provenance.get("sha256") != actual_file_sha256:
                raise ValueError(
                    f"Terminal-Bench split checksum mismatch for {split_name}: "
                    f"declared {file_provenance.get('sha256')!r}, actual {actual_file_sha256}"
                )

        if input_provenance.get("count") != len(task_ids):
            raise ValueError(
                "Terminal-Bench manifest input count mismatch: "
                f"declared {input_provenance.get('count')!r}, actual {len(task_ids)}"
            )
        actual_ids_sha256 = task_ids_sha256(task_ids)
        if input_provenance.get("task_ids_sha256") != actual_ids_sha256:
            raise ValueError(
                "Terminal-Bench manifest task ID checksum mismatch: "
                f"declared {input_provenance.get('task_ids_sha256')!r}, "
                f"actual {actual_ids_sha256}"
            )

        if schema_version == MANIFEST_SCHEMA_VERSION:
            semantic_identity = manifest.get("semantic_identity")
            semantic_digest = manifest.get("semantic_sha256")
            if not isinstance(semantic_identity, dict):
                raise ValueError("Terminal-Bench manifest semantic_identity must be an object")
            if not isinstance(semantic_digest, str):
                raise ValueError("Terminal-Bench manifest semantic_sha256 must be a string")
            split_provenance = manifest.get("split")
            source_provenance = manifest.get("source")
            if not isinstance(split_provenance, dict) or not isinstance(source_provenance, dict):
                raise ValueError("Terminal-Bench manifest split/source provenance must be objects")
            expected_semantic_identity = build_semantic_split_identity(
                splits,
                terminalbench_revision=semantic_identity.get(
                    "terminalbench_revision", ""
                ),
                seed=semantic_identity.get("seed", -1),
            )
            if semantic_identity != expected_semantic_identity:
                raise ValueError("Terminal-Bench semantic split identity mismatch")
            if split_provenance.get("seed") != semantic_identity["seed"]:
                raise ValueError("Terminal-Bench split seed differs from semantic identity")
            if source_provenance.get("revision", "") != semantic_identity[
                "terminalbench_revision"
            ]:
                raise ValueError(
                    "Terminal-Bench source revision differs from semantic identity"
                )
            actual_semantic_digest = semantic_split_sha256(semantic_identity)
            if semantic_digest != actual_semantic_digest:
                raise ValueError(
                    "Terminal-Bench semantic split checksum mismatch: "
                    f"declared {semantic_digest!r}, actual {actual_semantic_digest}"
                )
            if expected_checksum != actual_semantic_digest:
                raise ValueError(
                    "Terminal-Bench semantic checksum file mismatch: "
                    f"expected {expected_checksum}, got {actual_semantic_digest}"
                )
