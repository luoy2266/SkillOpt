"""Materialize a deterministic Terminal-Bench v2.1 task-ID split."""

from __future__ import annotations

import argparse
import hashlib
import json
import random
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from skillopt.datasets.base import SPLIT_NAMES
from skillopt.envs.terminalbench.dataloader import (
    BENCHMARK_NAME,
    BENCHMARK_VERSION,
    MANIFEST_CHECKSUM_FILENAME,
    MANIFEST_FILENAME,
    MANIFEST_SCHEMA_VERSION,
    SPLIT_ITEMS_FILENAME,
    normalize_manifest_item,
    task_ids_sha256,
    validate_split_partition,
    validate_task_id,
)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_RATIO = "1:1:8"
DEFAULT_SEED = 42
_PROHIBITED_METADATA_FIELDS = {
    "dockerfile",
    "instruction",
    "solution",
    "solutions",
    "task",
    "task_content",
    "test",
    "tests",
}


def parse_ratio(text: str) -> tuple[int, int, int]:
    """Parse a positive train:val:test integer ratio."""
    parts = [part.strip() for part in str(text or "").split(":")]
    if len(parts) != 3 or any(not part for part in parts):
        raise ValueError(f"ratio must be in train:val:test form, got {text!r}")
    try:
        train, val, test = (int(part) for part in parts)
    except ValueError as exc:
        raise ValueError(f"ratio must contain integers, got {text!r}") from exc
    ratio = (train, val, test)
    if min(ratio) <= 0:
        raise ValueError(f"ratio parts must be positive, got {text!r}")
    return ratio


def compute_split_counts(total: int, ratio: tuple[int, int, int]) -> tuple[int, int, int]:
    """Match the pinned SplitDataLoader largest-remainder allocation."""
    if total < 0:
        raise ValueError(f"total must be non-negative, got {total}")
    weights = list(ratio)
    denominator = sum(weights)
    raw_counts = [total * weight / denominator for weight in weights]
    counts = [int(value) for value in raw_counts]
    remaining = total - sum(counts)
    order = sorted(
        range(len(raw_counts)),
        key=lambda index: (
            -(raw_counts[index] - counts[index]),
            -weights[index],
            index,
        ),
    )
    for index in order[:remaining]:
        counts[index] += 1
    return counts[0], counts[1], counts[2]


def _sha256_bytes(content: bytes) -> str:
    return hashlib.sha256(content).hexdigest()


def _json_bytes(value: Any) -> bytes:
    return (json.dumps(value, ensure_ascii=False, indent=2) + "\n").encode("utf-8")


def _write_bytes(path: Path, content: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(content)


def _validate_metadata_fields(id_field: str, metadata_fields: Sequence[str]) -> tuple[str, ...]:
    if not id_field:
        raise ValueError("id_field must not be empty")
    normalized: list[str] = []
    seen: set[str] = set()
    for field in metadata_fields:
        if not field:
            raise ValueError("metadata field names must not be empty")
        if field in {"id", id_field}:
            raise ValueError(f"metadata field {field!r} duplicates the task ID field")
        if field.casefold() in _PROHIBITED_METADATA_FIELDS:
            raise ValueError(
                f"metadata field {field!r} is task payload, not lightweight split metadata"
            )
        if field in seen:
            raise ValueError(f"duplicate metadata field: {field!r}")
        seen.add(field)
        normalized.append(field)
    return tuple(normalized)


def _normalize_source_entry(
    raw: Any,
    *,
    index: int,
    id_field: str,
    metadata_fields: Sequence[str],
) -> dict[str, Any]:
    context = f"task source item {index}"
    if isinstance(raw, str):
        if metadata_fields:
            raise ValueError(
                f"{context} is a bare ID but metadata fields were requested: {metadata_fields!r}"
            )
        return {"id": validate_task_id(raw, context=context)}
    if not isinstance(raw, Mapping):
        raise ValueError(f"{context} must be a string or JSON object, got {type(raw).__name__}")
    if id_field not in raw:
        raise ValueError(f"{context} is missing ID field {id_field!r}")

    item: dict[str, Any] = {
        "id": validate_task_id(raw[id_field], context=f"{context}.{id_field}"),
    }
    for field in metadata_fields:
        if field not in raw:
            raise ValueError(f"{context} is missing requested metadata field {field!r}")
        value = raw[field]
        if not isinstance(value, (str, int, float, bool, type(None))):
            raise ValueError(
                f"{context}.{field} must be scalar lightweight metadata, "
                f"got {type(value).__name__}"
            )
        item[field] = value
    return normalize_manifest_item(item, context=context)


def _unwrap_json_source(data: Any, *, source_path: Path) -> list[Any]:
    if isinstance(data, list):
        return data
    if isinstance(data, dict):
        present = [key for key in ("task_ids", "tasks", "data") if key in data]
        if len(present) != 1:
            raise ValueError(
                f"JSON task source {source_path} must contain exactly one of "
                "'task_ids', 'tasks', or 'data'"
            )
        items = data[present[0]]
        if not isinstance(items, list):
            raise ValueError(f"JSON task source field {present[0]!r} must be an array")
        return items
    raise ValueError(
        f"JSON task source {source_path} must be an array or supported wrapper object"
    )


def load_task_source(
    source_path: Path,
    *,
    id_field: str = "id",
    metadata_fields: Sequence[str] = (),
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Load IDs from a task directory, JSON, JSONL, or line-delimited file."""
    source_path = source_path.resolve()
    metadata_fields = _validate_metadata_fields(id_field, metadata_fields)
    if not source_path.exists():
        raise FileNotFoundError(f"Terminal-Bench task ID source does not exist: {source_path}")

    if source_path.is_dir():
        if metadata_fields:
            raise ValueError("directory task sources do not provide metadata fields")
        raw_items: list[Any] = [path.name for path in sorted(source_path.iterdir()) if path.is_dir()]
        source_format = "task_directory"
        source_checksum_scope = "canonical_task_ids"
        source_sha256 = task_ids_sha256(raw_items)
    else:
        suffix = source_path.suffix.casefold()
        if suffix == ".json":
            try:
                raw_items = _unwrap_json_source(
                    json.loads(source_path.read_text(encoding="utf-8")),
                    source_path=source_path,
                )
            except json.JSONDecodeError as exc:
                raise ValueError(f"Malformed JSON task source {source_path}: {exc}") from exc
            source_format = "json"
        elif suffix == ".jsonl":
            raw_items = []
            for line_number, line in enumerate(source_path.read_text(encoding="utf-8").splitlines(), start=1):
                if not line.strip():
                    continue
                try:
                    raw_items.append(json.loads(line))
                except json.JSONDecodeError as exc:
                    raise ValueError(
                        f"Malformed JSONL task source {source_path}:{line_number}: {exc}"
                    ) from exc
            source_format = "jsonl"
        else:
            raw_items = [line.strip() for line in source_path.read_text(encoding="utf-8").splitlines() if line.strip()]
            source_format = "text"
        source_checksum_scope = "file_bytes"
        source_sha256 = _sha256_bytes(source_path.read_bytes())

    if not raw_items:
        raise ValueError(f"Terminal-Bench task ID source is empty: {source_path}")
    items = [
        _normalize_source_entry(
            raw,
            index=index,
            id_field=id_field,
            metadata_fields=metadata_fields,
        )
        for index, raw in enumerate(raw_items)
    ]
    validate_split_partition({"train": items, "val": [], "test": []})

    provenance = {
        "path": str(source_path),
        "format": source_format,
        "sha256": source_sha256,
        "sha256_scope": source_checksum_scope,
        "id_field": id_field,
        "metadata_fields": list(metadata_fields),
    }
    return items, provenance


def split_task_items(
    items: Sequence[dict[str, Any]],
    *,
    ratio: tuple[int, int, int],
    seed: int,
) -> dict[str, list[dict[str, Any]]]:
    """Shuffle once and slice a deterministic train/val/test partition."""
    normalized = [
        normalize_manifest_item(item, context=f"task item {index}")
        for index, item in enumerate(items)
    ]
    input_ids = validate_split_partition({"train": normalized, "val": [], "test": []})
    shuffled = [dict(item) for item in normalized]
    random.Random(seed).shuffle(shuffled)
    train_count, val_count, test_count = compute_split_counts(len(shuffled), ratio)
    splits = {
        "train": shuffled[:train_count],
        "val": shuffled[train_count: train_count + val_count],
        "test": shuffled[train_count + val_count: train_count + val_count + test_count],
    }

    output_ids = validate_split_partition(splits)
    if set(output_ids) != set(input_ids) or len(output_ids) != len(input_ids):
        raise RuntimeError("Terminal-Bench split partition has missing or extra task IDs")
    return splits


def materialize_terminalbench_split(
    source_path: Path,
    output_dir: Path,
    *,
    ratio_text: str = DEFAULT_RATIO,
    seed: int = DEFAULT_SEED,
    id_field: str = "id",
    metadata_fields: Sequence[str] = (),
    source_revision: str = "",
) -> dict[str, Any]:
    """Write split files, provenance manifest, and manifest checksum."""
    ratio = parse_ratio(ratio_text)
    canonical_ratio = ":".join(str(value) for value in ratio)
    items, source_provenance = load_task_source(
        source_path,
        id_field=id_field,
        metadata_fields=metadata_fields,
    )
    if source_revision:
        source_provenance["revision"] = source_revision
    splits = split_task_items(items, ratio=ratio, seed=seed)

    output_dir = output_dir.resolve()
    file_provenance: dict[str, dict[str, Any]] = {}
    for split_name in SPLIT_NAMES:
        relative_path = Path(split_name) / SPLIT_ITEMS_FILENAME
        content = _json_bytes(splits[split_name])
        _write_bytes(output_dir / relative_path, content)
        file_provenance[split_name] = {
            "path": relative_path.as_posix(),
            "count": len(splits[split_name]),
            "sha256": _sha256_bytes(content),
        }

    all_ids = validate_split_partition(splits)
    counts = {split_name: len(splits[split_name]) for split_name in SPLIT_NAMES}
    item_fields = sorted({field for split in splits.values() for item in split for field in item})
    manifest = {
        "schema_version": MANIFEST_SCHEMA_VERSION,
        "benchmark": BENCHMARK_NAME,
        "benchmark_version": BENCHMARK_VERSION,
        "manifest_type": "id_split",
        "materializer": "scripts/materialize_terminalbench_split.py",
        "source": source_provenance,
        "input": {
            "count": len(items),
            "task_ids_sha256": task_ids_sha256(all_ids),
        },
        "split": {
            "ratio": canonical_ratio,
            "seed": int(seed),
            "algorithm": "python_random_shuffle_then_slice",
            "count_allocation": "largest_remainder; ties resolve train, then val, then test",
        },
        "counts": counts,
        "item_fields": item_fields,
        "files": file_provenance,
        "notes": [
            "Contains task identifiers and explicitly selected scalar metadata only.",
            "Does not contain Terminal-Bench solutions, tests, Dockerfiles, or full task payloads.",
        ],
    }
    manifest_content = _json_bytes(manifest)
    _write_bytes(output_dir / MANIFEST_FILENAME, manifest_content)
    manifest_sha256 = _sha256_bytes(manifest_content)
    checksum_content = f"{manifest_sha256}  {MANIFEST_FILENAME}\n".encode("utf-8")
    _write_bytes(output_dir / MANIFEST_CHECKSUM_FILENAME, checksum_content)
    return manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source",
        type=Path,
        required=True,
        help="Task-ID source: task directory, JSON, JSONL, or one-ID-per-line text file.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "data" / "terminalbench_split",
        help="Directory to write train/val/test items plus split_manifest files.",
    )
    parser.add_argument("--ratio", default=DEFAULT_RATIO, help="Train:val:test ratio.")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED, help="Deterministic split seed.")
    parser.add_argument(
        "--id-field",
        default="id",
        help="ID field for object entries in JSON/JSONL sources.",
    )
    parser.add_argument(
        "--metadata-field",
        action="append",
        default=[],
        help="Scalar metadata field to retain; repeat for multiple fields.",
    )
    parser.add_argument(
        "--source-revision",
        default="",
        help="Optional pinned source revision/version recorded in provenance.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    manifest = materialize_terminalbench_split(
        args.source,
        args.output_dir,
        ratio_text=args.ratio,
        seed=args.seed,
        id_field=args.id_field,
        metadata_fields=args.metadata_field,
        source_revision=args.source_revision,
    )
    counts = manifest["counts"]
    print(
        f"Wrote Terminal-Bench v2.1 split to {args.output_dir.resolve()}: "
        f"train={counts['train']} val={counts['val']} test={counts['test']}"
    )


if __name__ == "__main__":
    main()
