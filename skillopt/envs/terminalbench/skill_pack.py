"""Deterministic SkillOpt-to-Harbor native skill packaging."""
from __future__ import annotations

import hashlib
import os
from dataclasses import dataclass
from pathlib import Path

SKILL_NAME = "terminalbench-skill"
SKILL_DESCRIPTION = "SkillOpt-generated reusable guidance for Terminal-Bench task execution."
SKILL_FILENAME = "SKILL.md"
ARTIFACTS_DIRNAME = "harbor_skills"

_FRONTMATTER = (
    "---\n"
    f"name: {SKILL_NAME}\n"
    f"description: {SKILL_DESCRIPTION}\n"
    "---\n\n"
).encode("utf-8")


class SkillPackagingError(RuntimeError):
    """Raised when a skill artifact cannot be created safely."""


class SkillArtifactConflictError(SkillPackagingError):
    """Raised when an existing deterministic artifact does not match its digest."""


@dataclass(frozen=True, slots=True)
class PackagedSkill:
    """Result of converting one SkillOpt skill string into Harbor input."""

    is_blank: bool
    skill_content: str
    skill_dir: Path | None
    skill_file: Path | None
    sha256: str | None

    @property
    def harbor_skills(self) -> list[str]:
        """Return the exact value for Harbor ``agents[].skills``."""
        if self.skill_dir is None:
            return []
        return [str(self.skill_dir)]


def is_semantically_blank(skill_content: str) -> bool:
    """Return whether SkillOpt content represents the no-skill baseline."""
    if not isinstance(skill_content, str):
        raise TypeError(
            f"skill_content must be a string, got {type(skill_content).__name__}"
        )
    return not skill_content.strip()


def render_skill_artifact(skill_content: str) -> bytes:
    """Return the exact Harbor ``SKILL.md`` bytes for non-blank content."""
    if is_semantically_blank(skill_content):
        raise ValueError("Blank skill content maps to Harbor skills=[] and has no artifact")
    return _FRONTMATTER + skill_content.encode("utf-8")


def package_skill_content(
    skill_content: str,
    output_root: str | os.PathLike[str],
) -> PackagedSkill:
    """Package SkillOpt content under a deterministic Harbor skill directory."""
    if is_semantically_blank(skill_content):
        return PackagedSkill(
            is_blank=True,
            skill_content=skill_content,
            skill_dir=None,
            skill_file=None,
            sha256=None,
        )

    artifact = render_skill_artifact(skill_content)
    artifact_sha256 = hashlib.sha256(artifact).hexdigest()
    resolved_output_root = _prepare_output_root(output_root)
    artifacts_root = resolved_output_root / ARTIFACTS_DIRNAME
    digest_dir = artifacts_root / artifact_sha256
    skill_dir = digest_dir / SKILL_NAME
    skill_file = skill_dir / SKILL_FILENAME

    _ensure_directory(artifacts_root, label="Harbor skill artifacts root")
    _ensure_directory(digest_dir, label="skill digest directory")
    _validate_only_child(digest_dir, allowed_name=SKILL_NAME)
    _ensure_directory(skill_dir, label="Harbor skill directory")
    _validate_only_child(skill_dir, allowed_name=SKILL_FILENAME)
    _write_or_verify_artifact(skill_file, artifact)
    _validate_only_child(skill_dir, allowed_name=SKILL_FILENAME)

    return PackagedSkill(
        is_blank=False,
        skill_content=skill_content,
        skill_dir=skill_dir,
        skill_file=skill_file,
        sha256=artifact_sha256,
    )


def _prepare_output_root(output_root: str | os.PathLike[str]) -> Path:
    try:
        raw_value = os.fspath(output_root)
    except TypeError as exc:
        raise TypeError("output_root must be a string or path-like value") from exc
    if not isinstance(raw_value, str):
        raise TypeError("output_root must resolve to a text filesystem path")
    if not raw_value.strip():
        raise ValueError("output_root must not be empty")

    lexical_path = Path(raw_value).expanduser()
    if lexical_path == Path("."):
        raise ValueError("output_root must name a dedicated artifact directory, not '.'")
    if ".." in lexical_path.parts:
        raise ValueError(f"output_root must not contain path traversal: {raw_value!r}")

    absolute_path = Path(os.path.abspath(lexical_path))
    if absolute_path == Path(absolute_path.anchor):
        raise ValueError("output_root must not be a filesystem root")
    _reject_symlink_path(absolute_path)
    _ensure_directory(absolute_path, label="output_root")
    _reject_symlink_path(absolute_path)
    return absolute_path


def _reject_symlink_path(path: Path) -> None:
    for candidate in (path, *path.parents):
        if candidate.is_symlink():
            raise SkillPackagingError(
                f"Skill artifact path must not contain symlinks: {candidate}"
            )


def _ensure_directory(path: Path, *, label: str) -> None:
    if path.is_symlink():
        raise SkillPackagingError(f"{label} must not be a symlink: {path}")
    if path.exists() and not path.is_dir():
        raise SkillPackagingError(f"{label} must be a directory: {path}")
    path.mkdir(parents=True, exist_ok=True)
    if path.is_symlink() or not path.is_dir():
        raise SkillPackagingError(f"Failed to create a safe {label}: {path}")


def _validate_only_child(parent: Path, *, allowed_name: str) -> None:
    unexpected = sorted(child.name for child in parent.iterdir() if child.name != allowed_name)
    if unexpected:
        raise SkillArtifactConflictError(
            f"Deterministic skill artifact path contains unexpected entries: {unexpected}"
        )


def _write_or_verify_artifact(skill_file: Path, expected_content: bytes) -> None:
    if skill_file.is_symlink():
        raise SkillPackagingError(f"Harbor skill file must not be a symlink: {skill_file}")
    if skill_file.exists():
        if not skill_file.is_file():
            raise SkillArtifactConflictError(
                f"Harbor skill artifact path is not a file: {skill_file}"
            )
        _verify_artifact_content(skill_file, expected_content)
        return

    try:
        with skill_file.open("xb") as handle:
            handle.write(expected_content)
    except FileExistsError:
        _verify_artifact_content(skill_file, expected_content)
        return
    _verify_artifact_content(skill_file, expected_content)


def _verify_artifact_content(skill_file: Path, expected_content: bytes) -> None:
    try:
        actual_content = skill_file.read_bytes()
    except OSError as exc:
        raise SkillPackagingError(f"Unable to read Harbor skill artifact: {skill_file}") from exc
    if actual_content != expected_content:
        expected_sha256 = hashlib.sha256(expected_content).hexdigest()
        actual_sha256 = hashlib.sha256(actual_content).hexdigest()
        raise SkillArtifactConflictError(
            "Existing Harbor skill artifact does not match its deterministic path: "
            f"expected sha256={expected_sha256}, actual sha256={actual_sha256}, "
            f"path={skill_file}"
        )
