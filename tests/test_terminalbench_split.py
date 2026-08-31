from __future__ import annotations

import json
import hashlib
import tempfile
import unittest
from pathlib import Path

from scripts.materialize_terminalbench_split import (
    compute_split_counts,
    materialize_terminalbench_split,
    split_task_items,
)
from skillopt.envs.terminalbench.dataloader import TerminalBenchDataLoader


def _mock_ids(count: int, *, prefix: str = "mock-task") -> list[str]:
    return [f"{prefix}-{index:03d}" for index in range(count)]


def _write_json_source(path: Path, items: list[object]) -> None:
    path.write_text(json.dumps(items), encoding="utf-8")


def _load_split_ids(split_dir: Path, split_name: str) -> list[str]:
    items = json.loads((split_dir / split_name / "items.json").read_text(encoding="utf-8"))
    return [item["id"] for item in items]


class TerminalBenchSplitTests(unittest.TestCase):
    def test_ratio_allocation_maps_89_to_9_9_71(self) -> None:
        self.assertEqual(compute_split_counts(89, (1, 1, 8)), (9, 9, 71))

    def test_split_is_deterministic_for_same_seed(self) -> None:
        items = [{"id": item_id} for item_id in _mock_ids(30)]

        first = split_task_items(items, ratio=(1, 1, 8), seed=42)
        second = split_task_items(items, ratio=(1, 1, 8), seed=42)

        self.assertEqual(first, second)

    def test_split_has_no_duplicates_or_omissions(self) -> None:
        input_ids = _mock_ids(89)
        splits = split_task_items(
            [{"id": item_id} for item_id in input_ids],
            ratio=(1, 1, 8),
            seed=7,
        )

        split_sets = {
            split_name: {item["id"] for item in items}
            for split_name, items in splits.items()
        }
        self.assertEqual(
            {split_name: len(items) for split_name, items in splits.items()},
            {"train": 9, "val": 9, "test": 71},
        )
        self.assertTrue(split_sets["train"].isdisjoint(split_sets["val"]))
        self.assertTrue(split_sets["train"].isdisjoint(split_sets["test"]))
        self.assertTrue(split_sets["val"].isdisjoint(split_sets["test"]))
        self.assertEqual(set().union(*split_sets.values()), set(input_ids))

    def test_materializer_writes_manifest_provenance_and_checksums(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            source_path = root / "task_ids.json"
            output_dir = root / "split"
            _write_json_source(
                source_path,
                [
                    {"task_id": item_id, "category": "mock", "solution": "not copied"}
                    for item_id in _mock_ids(89)
                ],
            )

            manifest = materialize_terminalbench_split(
                source_path,
                output_dir,
                ratio_text="1:1:8",
                seed=42,
                id_field="task_id",
                metadata_fields=("category",),
                source_revision="fixture-revision",
            )

            self.assertEqual(manifest["counts"], {"train": 9, "val": 9, "test": 71})
            self.assertEqual(manifest["source"]["revision"], "fixture-revision")
            self.assertEqual(manifest["schema_version"], 2)
            self.assertEqual(
                manifest["semantic_identity"]["terminalbench_revision"],
                "fixture-revision",
            )
            self.assertEqual(manifest["item_fields"], ["category", "id"])
            self.assertTrue((output_dir / "split_manifest.json").is_file())
            self.assertTrue((output_dir / "split_manifest.sha256").is_file())
            for split_name in ("train", "val", "test"):
                items = json.loads(
                    (output_dir / split_name / "items.json").read_text(encoding="utf-8")
                )
                self.assertTrue(all(set(item) == {"id", "category"} for item in items))

    def test_semantic_hash_is_stable_across_source_paths(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            digests = []
            ordered_ids = []
            manifest_paths = []
            for name in ("a", "b"):
                source_dir = root / name / "terminal-bench" / "tasks"
                source_dir.mkdir(parents=True)
                for item_id in _mock_ids(89):
                    (source_dir / item_id).mkdir()
                output_dir = root / f"split-{name}"
                manifest = materialize_terminalbench_split(
                    source_dir,
                    output_dir,
                    seed=42,
                    source_revision="pinned-revision",
                )
                digests.append(manifest["semantic_sha256"])
                ordered_ids.append(
                    {
                        split: _load_split_ids(output_dir, split)
                        for split in ("train", "val", "test")
                    }
                )
                manifest_paths.append(manifest["source"]["path"])

            self.assertEqual(digests[0], digests[1])
            self.assertEqual(ordered_ids[0], ordered_ids[1])
            self.assertNotEqual(manifest_paths[0], manifest_paths[1])

    def test_dataloader_preserves_legacy_v1_manifest_validation(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            output_dir = root / "split"
            source_path = root / "source.json"
            _write_json_source(source_path, _mock_ids(10))
            materialize_terminalbench_split(source_path, output_dir)
            manifest_path = output_dir / "split_manifest.json"
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            manifest["schema_version"] = 1
            manifest.pop("semantic_identity")
            manifest.pop("semantic_sha256")
            manifest.pop("legacy_provenance", None)
            content = (json.dumps(manifest, ensure_ascii=False, indent=2) + "\n").encode()
            manifest_path.write_bytes(content)
            (output_dir / "split_manifest.sha256").write_text(
                f"{hashlib.sha256(content).hexdigest()}  split_manifest.json\n",
                encoding="utf-8",
            )

            loader = TerminalBenchDataLoader(split_dir=str(output_dir))
            loader.setup({})

            self.assertEqual(
                sum(len(loader.get_split_items(name)) for name in ("train", "val", "test")),
                10,
            )

    def test_committed_portable_split_has_stable_identity(self) -> None:
        split_dir = (
            Path(__file__).resolve().parents[1]
            / "configs"
            / "terminalbench"
            / "splits"
            / "v2.1-s42"
        )
        manifest = json.loads(
            (split_dir / "split_manifest.json").read_text(encoding="utf-8")
        )

        loader = TerminalBenchDataLoader(split_dir=str(split_dir))
        loader.setup({})

        self.assertEqual(
            manifest["semantic_sha256"],
            "bd36fe2f37a67cd2b46149263522d833166d3a4d036c8e9af082e742ad017500",
        )
        self.assertEqual(manifest["counts"], {"train": 9, "val": 9, "test": 71})

    def test_directory_task_source_uses_immediate_child_names_only(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            source_dir = root / "tasks"
            output_dir = root / "split"
            source_dir.mkdir()
            for item_id in _mock_ids(10, prefix="directory-task"):
                (source_dir / item_id).mkdir()
            (source_dir / "README.md").write_text("not a task directory", encoding="utf-8")

            manifest = materialize_terminalbench_split(source_dir, output_dir, seed=5)

            self.assertEqual(manifest["source"]["format"], "task_directory")
            self.assertEqual(manifest["counts"], {"train": 1, "val": 1, "test": 8})

    def test_dataloader_loads_replaceable_manifests_and_upstream_aliases(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            loaded_id_sets = []
            for name, prefix in (("first", "alpha"), ("second", "beta")):
                source_path = root / f"{name}.json"
                output_dir = root / name
                _write_json_source(source_path, _mock_ids(10, prefix=prefix))
                materialize_terminalbench_split(source_path, output_dir, seed=11)

                dataloader = TerminalBenchDataLoader(split_dir=str(output_dir))
                dataloader.setup({})
                loaded_id_sets.append(
                    {item["id"] for split in ("train", "val", "test") for item in dataloader.get_split_items(split)}
                )
                self.assertEqual(dataloader.get_split_items("valid_seen"), dataloader.val_items)
                self.assertEqual(dataloader.get_split_items("valid_unseen"), dataloader.test_items)

            self.assertNotEqual(loaded_id_sets[0], loaded_id_sets[1])

    def test_duplicate_source_ids_fail_loudly(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            source_path = root / "duplicates.json"
            _write_json_source(source_path, ["duplicate-task", "duplicate-task"])

            with self.assertRaisesRegex(ValueError, "duplicate task ID"):
                materialize_terminalbench_split(source_path, root / "split")

    def test_malformed_source_id_fails_loudly(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            source_path = root / "malformed.json"
            _write_json_source(source_path, ["valid-task", "unsafe/task"])

            with self.assertRaisesRegex(ValueError, "unsafe filename characters"):
                materialize_terminalbench_split(source_path, root / "split")

    def test_dataloader_rejects_duplicate_ids_across_materialized_splits(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            source_path = root / "task_ids.json"
            output_dir = root / "split"
            _write_json_source(source_path, _mock_ids(10))
            materialize_terminalbench_split(source_path, output_dir)

            train_ids = _load_split_ids(output_dir, "train")
            val_path = output_dir / "val" / "items.json"
            val_items = json.loads(val_path.read_text(encoding="utf-8"))
            val_items[0]["id"] = train_ids[0]
            val_path.write_text(json.dumps(val_items), encoding="utf-8")

            dataloader = TerminalBenchDataLoader(split_dir=str(output_dir))
            with self.assertRaisesRegex(ValueError, "duplicate task ID"):
                dataloader.setup({})

    def test_dataloader_rejects_tampered_split_checksum(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            source_path = root / "task_ids.json"
            output_dir = root / "split"
            _write_json_source(source_path, _mock_ids(10))
            materialize_terminalbench_split(source_path, output_dir)

            test_path = output_dir / "test" / "items.json"
            test_path.write_text(test_path.read_text(encoding="utf-8") + "\n", encoding="utf-8")

            dataloader = TerminalBenchDataLoader(split_dir=str(output_dir))
            with self.assertRaisesRegex(ValueError, "split checksum mismatch"):
                dataloader.setup({})

    def test_runtime_ratio_mode_is_disabled(self) -> None:
        dataloader = TerminalBenchDataLoader(
            data_path="unused.json",
            split_mode="ratio",
        )
        with self.assertRaisesRegex(ValueError, "does not materialize ratio splits at runtime"):
            dataloader.setup({})


if __name__ == "__main__":
    unittest.main()
