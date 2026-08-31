from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from scripts.freeze_terminalbench_skill import (
    FreezeFailure,
    freeze_training_skill,
    validate_frozen_skill,
)
from skillopt.envs.terminalbench.skill_pack import render_skill_artifact
from tests.terminalbench_lifecycle_fixtures import (
    sha256_bytes,
    write_completed_training_fixture,
)


class TerminalBenchFreezeTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary_directory.name)

    def tearDown(self) -> None:
        self.temporary_directory.cleanup()

    def _freeze(self, fixture: dict, *, output_name: str = "frozen") -> dict:
        return freeze_training_skill(
            experiment_id="experiment-001",
            training_output=fixture["output"],
            training_manifest_path=fixture["manifest_path"],
            output_root=self.root / output_name,
            expected_skillopt_head="a" * 40,
        )

    def test_completed_training_freezes_exact_raw_and_native_skill(self) -> None:
        fixture = write_completed_training_fixture(self.root)

        frozen = self._freeze(fixture)

        raw_bytes = (self.root / "frozen" / "best_skill.md").read_bytes()
        self.assertEqual(raw_bytes, fixture["best_content"].encode("utf-8"))
        native_bytes = (
            self.root / "frozen" / "terminalbench-skill" / "SKILL.md"
        ).read_bytes()
        self.assertEqual(native_bytes, render_skill_artifact(fixture["best_content"]))
        self.assertEqual(frozen["raw_sha256"], sha256_bytes(raw_bytes))
        self.assertEqual(frozen["native_sha256"], sha256_bytes(native_bytes))
        provenance = frozen["provenance"]
        self.assertEqual(provenance["source_training"]["expected_steps"], 4)
        self.assertEqual(provenance["split"]["split_identity_schema"], "v2")
        self.assertEqual(provenance["selection"]["best_version"], "skill_v0002")

    def test_blank_best_skill_is_valid_and_has_no_native_artifact(self) -> None:
        fixture = write_completed_training_fixture(
            self.root,
            best_content="\n",
            best_step=0,
        )

        frozen = self._freeze(fixture)

        self.assertTrue(frozen["is_blank"])
        self.assertIsNone(frozen["native_sha256"])
        self.assertFalse((self.root / "frozen" / "terminalbench-skill").exists())
        provenance = frozen["provenance"]
        self.assertTrue(provenance["raw_skill"]["is_blank"])
        self.assertEqual(
            provenance["native_skill"]["package_identity"], "harbor-skills-empty"
        )

    def test_missing_summary_blocks_freeze(self) -> None:
        fixture = write_completed_training_fixture(self.root)
        (fixture["output"] / "summary.json").unlink()

        with self.assertRaisesRegex(FreezeFailure, "missing training summary"):
            self._freeze(fixture)

    def test_incomplete_steps_block_freeze(self) -> None:
        fixture = write_completed_training_fixture(self.root)
        history_path = fixture["output"] / "history.json"
        history = json.loads(history_path.read_text(encoding="utf-8"))[:-1]
        history_path.write_text(json.dumps(history), encoding="utf-8")

        with self.assertRaisesRegex(FreezeFailure, "history is incomplete"):
            self._freeze(fixture)

    def test_inconsistent_runtime_state_blocks_freeze(self) -> None:
        fixture = write_completed_training_fixture(self.root)
        path = fixture["output"] / "runtime_state.json"
        state = json.loads(path.read_text(encoding="utf-8"))
        state["best_step"] = 3
        path.write_text(json.dumps(state), encoding="utf-8")

        with self.assertRaisesRegex(FreezeFailure, "best_step"):
            self._freeze(fixture)

    def test_missing_best_skill_blocks_freeze(self) -> None:
        fixture = write_completed_training_fixture(self.root)
        (fixture["output"] / "best_skill.md").unlink()

        with self.assertRaisesRegex(FreezeFailure, "missing training best skill"):
            self._freeze(fixture)

    def test_manifest_runtime_config_mismatch_blocks_freeze(self) -> None:
        fixture = write_completed_training_fixture(self.root)
        path = fixture["output"] / "config.json"
        config = json.loads(path.read_text(encoding="utf-8"))
        config["batch_size"] = 3
        path.write_text(json.dumps(config), encoding="utf-8")

        with self.assertRaisesRegex(FreezeFailure, "runtime batch_size"):
            self._freeze(fixture)

    def test_native_sha_is_deterministic_and_revalidated(self) -> None:
        fixture = write_completed_training_fixture(self.root)
        first = self._freeze(fixture, output_name="first")
        second = self._freeze(fixture, output_name="second")

        self.assertEqual(first["native_sha256"], second["native_sha256"])
        native_path = self.root / "first" / "terminalbench-skill" / "SKILL.md"
        native_path.write_text("tampered", encoding="utf-8")
        with self.assertRaisesRegex(FreezeFailure, "deterministic packaging"):
            validate_frozen_skill(self.root / "first" / "skill_provenance.json")

    def test_legacy_split_identity_is_preserved_without_v2_reinterpretation(self) -> None:
        fixture = write_completed_training_fixture(self.root)
        manifest_path = fixture["manifest_path"]
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        manifest["dataset"]["split_identity_type"] = "legacy_materialized_manifest_sha256"
        manifest["dataset"]["split_manifest_sha256"] = "8" * 64
        manifest["dataset"]["split_manifest"] = {"schema_version": 1}
        manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

        frozen = self._freeze(fixture)

        split = frozen["provenance"]["split"]
        self.assertEqual(split["split_identity_schema"], "v1")
        self.assertEqual(split["sha256"], "8" * 64)
        self.assertEqual(split["legacy_materialized_manifest_sha256"], "8" * 64)


if __name__ == "__main__":
    unittest.main()
