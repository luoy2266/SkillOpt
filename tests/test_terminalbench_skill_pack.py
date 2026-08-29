from __future__ import annotations

import hashlib
import re
import tempfile
import unittest
from pathlib import Path

from skillopt.envs.terminalbench.skill_pack import (
    ARTIFACTS_DIRNAME,
    SKILL_DESCRIPTION,
    SKILL_FILENAME,
    SKILL_NAME,
    SkillArtifactConflictError,
    SkillPackagingError,
    is_semantically_blank,
    package_skill_content,
    render_skill_artifact,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
INITIAL_SKILL = PROJECT_ROOT / "skillopt" / "envs" / "terminalbench" / "skills" / "initial.md"


class TerminalBenchSkillPackTests(unittest.TestCase):
    def test_packaging_metadata_meets_agent_skills_constraints(self) -> None:
        self.assertLessEqual(len(SKILL_NAME), 64)
        self.assertIsNotNone(re.fullmatch(r"[a-z0-9]+(?:-[a-z0-9]+)*", SKILL_NAME))
        self.assertTrue(SKILL_DESCRIPTION)
        self.assertLessEqual(len(SKILL_DESCRIPTION), 1024)

    def test_empty_string_maps_to_no_harbor_skills(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            output_root = Path(temporary_directory) / "artifacts"

            packaged = package_skill_content("", output_root)

            self.assertTrue(packaged.is_blank)
            self.assertIsNone(packaged.skill_dir)
            self.assertIsNone(packaged.skill_file)
            self.assertIsNone(packaged.sha256)
            self.assertEqual(packaged.harbor_skills, [])
            self.assertFalse(output_root.exists())

    def test_whitespace_only_maps_to_no_harbor_skills(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            packaged = package_skill_content(" \n\t\r\n", Path(temporary_directory) / "artifacts")

            self.assertTrue(packaged.is_blank)
            self.assertEqual(packaged.harbor_skills, [])

    def test_initial_markdown_exists_and_maps_to_no_harbor_skills(self) -> None:
        self.assertTrue(INITIAL_SKILL.is_file())
        content = INITIAL_SKILL.read_text(encoding="utf-8")
        self.assertTrue(is_semantically_blank(content))

        with tempfile.TemporaryDirectory() as temporary_directory:
            packaged = package_skill_content(content, Path(temporary_directory) / "artifacts")

        self.assertTrue(packaged.is_blank)
        self.assertEqual(packaged.harbor_skills, [])

    def test_non_blank_skill_writes_harbor_directory_with_exact_body(self) -> None:
        skill_content = "# General strategy\n\nKeep the task state explicit.\n"
        expected_artifact = render_skill_artifact(skill_content)

        with tempfile.TemporaryDirectory() as temporary_directory:
            output_root = Path(temporary_directory) / "run"
            packaged = package_skill_content(skill_content, output_root)

            self.assertFalse(packaged.is_blank)
            self.assertIsNotNone(packaged.skill_dir)
            self.assertIsNotNone(packaged.skill_file)
            self.assertIsNotNone(packaged.sha256)
            self.assertEqual(packaged.skill_file.read_bytes(), expected_artifact)
            self.assertEqual(
                packaged.skill_file,
                output_root
                / ARTIFACTS_DIRNAME
                / packaged.sha256
                / SKILL_NAME
                / SKILL_FILENAME,
            )
            self.assertEqual(packaged.harbor_skills, [str(packaged.skill_dir)])
            self.assertNotEqual(packaged.harbor_skills, [str(packaged.skill_file)])

            metadata_prefix = (
                "---\n"
                f"name: {SKILL_NAME}\n"
                f"description: {SKILL_DESCRIPTION}\n"
                "---\n\n"
            ).encode("utf-8")
            self.assertTrue(expected_artifact.startswith(metadata_prefix))
            self.assertEqual(expected_artifact[len(metadata_prefix):], skill_content.encode("utf-8"))

    def test_digest_uses_exact_written_artifact_bytes(self) -> None:
        skill_content = "Preserve trailing whitespace.  \n\n"

        with tempfile.TemporaryDirectory() as temporary_directory:
            packaged = package_skill_content(skill_content, Path(temporary_directory) / "run")

            self.assertEqual(
                packaged.sha256,
                hashlib.sha256(packaged.skill_file.read_bytes()).hexdigest(),
            )

    def test_same_content_is_idempotent_and_reuses_path(self) -> None:
        skill_content = "# Stable guidance\nDo the same thing every time."
        with tempfile.TemporaryDirectory() as temporary_directory:
            output_root = Path(temporary_directory) / "run"
            first = package_skill_content(skill_content, output_root)
            first_mtime = first.skill_file.stat().st_mtime_ns
            second = package_skill_content(skill_content, output_root)

            self.assertEqual(first.sha256, second.sha256)
            self.assertEqual(first.skill_dir, second.skill_dir)
            self.assertEqual(first.skill_file, second.skill_file)
            self.assertEqual(first_mtime, second.skill_file.stat().st_mtime_ns)

    def test_different_content_produces_different_digest_and_path(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            output_root = Path(temporary_directory) / "run"
            first = package_skill_content("First skill", output_root)
            second = package_skill_content("Second skill", output_root)

            self.assertNotEqual(first.sha256, second.sha256)
            self.assertNotEqual(first.skill_dir, second.skill_dir)

    def test_tampered_artifact_is_detected_without_overwrite(self) -> None:
        skill_content = "Original guidance"
        with tempfile.TemporaryDirectory() as temporary_directory:
            output_root = Path(temporary_directory) / "run"
            packaged = package_skill_content(skill_content, output_root)
            packaged.skill_file.write_text("tampered", encoding="utf-8")

            with self.assertRaisesRegex(
                SkillArtifactConflictError,
                "does not match its deterministic path",
            ):
                package_skill_content(skill_content, output_root)
            self.assertEqual(packaged.skill_file.read_text(encoding="utf-8"), "tampered")

    def test_unexpected_digest_directory_entry_is_detected(self) -> None:
        skill_content = "Reusable guidance"
        with tempfile.TemporaryDirectory() as temporary_directory:
            output_root = Path(temporary_directory) / "run"
            packaged = package_skill_content(skill_content, output_root)
            packaged.skill_dir.parent.joinpath("unexpected.txt").write_text(
                "conflict",
                encoding="utf-8",
            )

            with self.assertRaisesRegex(
                SkillArtifactConflictError,
                "unexpected entries",
            ):
                package_skill_content(skill_content, output_root)

    def test_output_root_path_traversal_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            unsafe_root = Path(temporary_directory) / "artifacts" / ".." / "escape"

            with self.assertRaisesRegex(ValueError, "path traversal"):
                package_skill_content("Non-blank", unsafe_root)

    def test_output_root_file_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            output_file = Path(temporary_directory) / "not-a-directory"
            output_file.write_text("occupied", encoding="utf-8")

            with self.assertRaisesRegex(SkillPackagingError, "must be a directory"):
                package_skill_content("Non-blank", output_file)

    def test_symlinked_artifact_root_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            output_root = root / "run"
            output_root.mkdir()
            outside = root / "outside"
            outside.mkdir()
            try:
                (output_root / ARTIFACTS_DIRNAME).symlink_to(outside, target_is_directory=True)
            except (OSError, NotImplementedError) as exc:
                self.skipTest(f"symlinks unavailable: {exc}")

            with self.assertRaisesRegex(SkillPackagingError, "must not be a symlink"):
                package_skill_content("Non-blank", output_root)

    def test_skill_content_cannot_control_artifact_path(self) -> None:
        skill_content = "../../outside\n\nThis remains Markdown body content."
        with tempfile.TemporaryDirectory() as temporary_directory:
            output_root = Path(temporary_directory) / "run"
            packaged = package_skill_content(skill_content, output_root)

            self.assertEqual(packaged.skill_dir.name, SKILL_NAME)
            self.assertTrue(packaged.skill_file.resolve().is_relative_to(output_root.resolve()))
            self.assertTrue(packaged.skill_file.read_bytes().endswith(skill_content.encode("utf-8")))

    def test_non_string_skill_content_fails_loudly(self) -> None:
        with self.assertRaisesRegex(TypeError, "skill_content must be a string"):
            package_skill_content(None, "unused")  # type: ignore[arg-type]


if __name__ == "__main__":
    unittest.main()
