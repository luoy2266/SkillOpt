"""Harbor 0.20.0 config preparation without job execution."""
from __future__ import annotations

import copy
import json
import os
import re
import shutil
import subprocess
import tempfile
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from skillopt.envs.terminalbench.dataloader import validate_task_id

EXPECTED_HARBOR_VERSION = "0.20.0"
RUNTIME_DIRNAME = "harbor_runtime"
CONFIGS_DIRNAME = "configs"
DRY_RUNS_DIRNAME = "dry_runs"
JOBS_DIRNAME = "jobs"
PARITY_ALLOWED_PATHS = ("job_name", "jobs_dir", "agents[*].skills")

_RESULT_NAME_PATTERN = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,127}")
_ENV_REFERENCE_PATTERN = re.compile(r"\$\{[A-Za-z_][A-Za-z0-9_]*\}")
_FNMATCH_METACHARACTERS = frozenset("*?[]")


class HarborConfigError(ValueError):
    """Raised when a Harbor config or runtime artifact violates the contract."""


class HarborVersionError(HarborConfigError):
    """Raised when the installed Harbor version is not pinned 0.20.0."""


class HarborParityError(HarborConfigError):
    """Raised when comparison configs differ outside the allowlist."""


class HarborArtifactConflictError(HarborConfigError):
    """Raised when a deterministic artifact already contains other bytes."""


class HarborExecutionDisabledError(RuntimeError):
    """Raised because real Harbor execution is outside Phase 3."""


@dataclass(frozen=True, slots=True)
class PreparedHarborRun:
    """A validated Harbor invocation prepared for audit or later execution."""

    resolved_config: dict[str, Any]
    resolved_config_path: Path
    dry_run_path: Path
    base_config_path: Path
    working_directory: Path
    task_ids: tuple[str, ...]
    harbor_skills: tuple[str, ...]
    result_name: str
    output_root: Path
    jobs_dir: Path
    expected_job_dir: Path
    harbor_version: str
    command: tuple[str, ...]


def load_harbor_base_config(path: str | os.PathLike[str]) -> dict[str, Any]:
    """Load and locally validate one Harbor YAML/JSON baseline config."""
    config_path = _path(path, label="Harbor base config", kind="file", must_exist=True)
    try:
        text = config_path.read_text(encoding="utf-8")
        if config_path.suffix.lower() in {".yaml", ".yml"}:
            raw = yaml.safe_load(text)
        elif config_path.suffix.lower() == ".json":
            raw = json.loads(text)
        else:
            raise HarborConfigError(
                f"Harbor base config must be .yaml, .yml, or .json: {config_path}"
            )
    except (OSError, yaml.YAMLError, json.JSONDecodeError) as exc:
        raise HarborConfigError(f"Unable to parse Harbor base config: {config_path}") from exc
    if not isinstance(raw, Mapping):
        raise HarborConfigError("Harbor base config must contain a top-level mapping")
    config = copy.deepcopy(dict(raw))
    _validate_base_contract(config)
    _validate_secret_references(config)
    return config


def build_harbor_config(
    *,
    base_config: Mapping[str, Any],
    task_ids: Sequence[str],
    harbor_skills: Sequence[str | os.PathLike[str]],
    result_name: str,
    jobs_dir: str | os.PathLike[str],
    n_attempts: int | None = None,
    n_concurrent_trials: int | None = None,
) -> dict[str, Any]:
    """Deep-copy a validated base config and apply the Phase 3 overlay."""
    if not isinstance(base_config, Mapping):
        raise TypeError("base_config must be a mapping")
    resolved = copy.deepcopy(dict(base_config))
    _validate_base_contract(resolved)
    _validate_secret_references(resolved)

    normalized_task_ids = _normalize_task_ids(task_ids)
    normalized_skills = _normalize_harbor_skills(harbor_skills)
    normalized_result_name = _validate_result_name(result_name)
    normalized_jobs_dir = _path(jobs_dir, label="Harbor jobs_dir", kind="directory")

    resolved["job_name"] = normalized_result_name
    resolved["jobs_dir"] = str(normalized_jobs_dir)
    resolved["agents"][0]["skills"] = list(normalized_skills)
    resolved["datasets"][0].update(
        task_names=list(normalized_task_ids),
        exclude_task_names=None,
        n_tasks=None,
    )
    if n_attempts is not None:
        resolved["n_attempts"] = _positive_int(n_attempts, "n_attempts")
    if n_concurrent_trials is not None:
        resolved["n_concurrent_trials"] = _positive_int(
            n_concurrent_trials,
            "n_concurrent_trials",
        )
    return resolved


def assert_harbor_config_parity(
    baseline_config: Mapping[str, Any],
    skill_config: Mapping[str, Any],
) -> None:
    """Require comparison configs to differ only at the Phase 3 allowlist."""
    baseline = copy.deepcopy(dict(baseline_config))
    candidate = copy.deepcopy(dict(skill_config))
    baseline_tasks = _resolved_task_ids(baseline, "baseline")
    candidate_tasks = _resolved_task_ids(candidate, "SkillOpt")
    if baseline_tasks != candidate_tasks:
        raise HarborParityError(
            "Harbor comparison configs must select the same task IDs: "
            f"baseline={list(baseline_tasks)}, SkillOpt={list(candidate_tasks)}"
        )
    if _single_agent(baseline, "baseline config").get("skills", []) != []:
        raise HarborParityError("Harbor baseline config must serialize agents[].skills as []")
    _single_agent(candidate, "SkillOpt config")

    for config in (baseline, candidate):
        config.pop("job_name", None)
        config.pop("jobs_dir", None)
        config["agents"][0].pop("skills", None)
    if baseline != candidate:
        raise HarborParityError(
            "Harbor configs differ outside the parity allowlist "
            f"{PARITY_ALLOWED_PATHS}"
        )


class HarborRunner:
    """Prepare Harbor 0.20.0 CLI configs while forbidding job execution."""

    def __init__(
        self,
        base_config_path: str | os.PathLike[str],
        harbor_executable: str | os.PathLike[str] = "harbor",
    ) -> None:
        self.base_config_path = _path(
            base_config_path,
            label="Harbor base config",
            kind="file",
            must_exist=True,
        )
        self.working_directory = self.base_config_path.parent
        self.harbor_executable = _resolve_executable(harbor_executable)
        self.harbor_version = self._validate_version()
        self._base_config = load_harbor_base_config(self.base_config_path)
        self._validate_config_path(self.base_config_path)

    def prepare(
        self,
        *,
        task_ids: Sequence[str],
        harbor_skills: Sequence[str | os.PathLike[str]],
        result_name: str,
        output_root: str | os.PathLike[str],
        n_attempts: int | None = None,
        n_concurrent_trials: int | None = None,
    ) -> PreparedHarborRun:
        """Build, schema-validate, and serialize one resolved config."""
        result_name = _validate_result_name(result_name)
        output_root_path = _path(
            output_root,
            label="output_root",
            kind="directory",
            create=True,
        )
        runtime_root = output_root_path / RUNTIME_DIRNAME
        configs_dir = runtime_root / CONFIGS_DIRNAME
        dry_runs_dir = runtime_root / DRY_RUNS_DIRNAME
        jobs_dir = runtime_root / JOBS_DIRNAME
        for directory in (configs_dir, dry_runs_dir, jobs_dir):
            directory.mkdir(parents=True, exist_ok=True)

        resolved_config = build_harbor_config(
            base_config=self._base_config,
            task_ids=task_ids,
            harbor_skills=harbor_skills,
            result_name=result_name,
            jobs_dir=jobs_dir,
            n_attempts=n_attempts,
            n_concurrent_trials=n_concurrent_trials,
        )
        config_bytes = _json_bytes(resolved_config)
        self._validate_config_bytes(config_bytes, configs_dir)
        resolved_config_path = configs_dir / f"{result_name}.json"
        _write_or_verify(resolved_config_path, config_bytes)

        expected_job_dir = jobs_dir / result_name
        if expected_job_dir.is_symlink():
            raise HarborConfigError(
                f"Expected Harbor job directory must not be a symlink: {expected_job_dir}"
            )
        command = (
            str(self.harbor_executable),
            "run",
            "--config",
            str(resolved_config_path),
        )
        return PreparedHarborRun(
            resolved_config=resolved_config,
            resolved_config_path=resolved_config_path,
            dry_run_path=dry_runs_dir / f"{result_name}.json",
            base_config_path=self.base_config_path,
            working_directory=self.working_directory,
            task_ids=tuple(resolved_config["datasets"][0]["task_names"]),
            harbor_skills=tuple(resolved_config["agents"][0]["skills"]),
            result_name=result_name,
            output_root=output_root_path,
            jobs_dir=jobs_dir,
            expected_job_dir=expected_job_dir,
            harbor_version=self.harbor_version,
            command=command,
        )

    def dry_run(self, prepared: PreparedHarborRun) -> dict[str, Any]:
        """Persist a secret-free description without starting a Harbor job."""
        manifest = {
            "schema_version": 1,
            "harbor_version": prepared.harbor_version,
            "base_config_path": str(prepared.base_config_path),
            "resolved_config_path": str(prepared.resolved_config_path),
            "working_directory": str(prepared.working_directory),
            "task_ids": list(prepared.task_ids),
            "skills": list(prepared.harbor_skills),
            "result_name": prepared.result_name,
            "output_root": str(prepared.output_root),
            "expected_job_dir": str(prepared.expected_job_dir),
            "command": list(prepared.command),
            "execution_started": False,
        }
        _write_or_verify(prepared.dry_run_path, _json_bytes(manifest))
        return manifest

    def run(self, prepared: PreparedHarborRun) -> None:
        """Refuse real execution until the later integration phase."""
        raise HarborExecutionDisabledError(
            "Real Harbor job execution is disabled in Phase 3; use dry_run(prepared)"
        )

    def _validate_version(self) -> str:
        completed = self._run_cli((str(self.harbor_executable), "--version"))
        version = completed.stdout.strip()
        if version != EXPECTED_HARBOR_VERSION:
            raise HarborVersionError(
                f"Expected Harbor {EXPECTED_HARBOR_VERSION}, found "
                f"{version or '<empty output>'} at {self.harbor_executable}"
            )
        return version

    def _validate_config_path(self, config_path: Path) -> None:
        completed = self._run_cli(
            (
                str(self.harbor_executable),
                "run",
                "--config",
                str(config_path),
                "--print-config",
            ),
            check=False,
        )
        if completed.returncode != 0:
            raise HarborConfigError(
                f"Harbor {EXPECTED_HARBOR_VERSION} rejected config: {config_path}"
            )

    def _validate_config_bytes(self, content: bytes, directory: Path) -> None:
        temporary_path: Path | None = None
        try:
            with tempfile.NamedTemporaryFile(
                mode="wb",
                suffix=".json",
                dir=directory,
                delete=False,
            ) as handle:
                handle.write(content)
                temporary_path = Path(handle.name)
            self._validate_config_path(temporary_path)
        finally:
            if temporary_path is not None:
                temporary_path.unlink(missing_ok=True)

    def _run_cli(
        self,
        command: tuple[str, ...],
        *,
        check: bool = True,
    ) -> subprocess.CompletedProcess[str]:
        try:
            return subprocess.run(
                command,
                cwd=self.working_directory,
                check=check,
                capture_output=True,
                text=True,
                timeout=30,
            )
        except (OSError, subprocess.SubprocessError) as exc:
            raise HarborConfigError(f"Unable to validate Harbor CLI: {command[0]}") from exc


def _validate_base_contract(config: dict[str, Any]) -> None:
    agent = _single_agent(config, "Harbor base config")
    if agent.get("name") != "terminus-2":
        raise HarborConfigError("Harbor base config must use one name: terminus-2 agent")
    if not isinstance(agent.get("model_name"), str) or not agent["model_name"].strip():
        raise HarborConfigError("Harbor base config Terminus-2 agent requires model_name")
    if agent.get("skills") not in (None, []):
        raise HarborConfigError("Harbor base config must be a baseline with skills empty")

    datasets = config.get("datasets")
    if not isinstance(datasets, list) or len(datasets) != 1 or not isinstance(datasets[0], dict):
        raise HarborConfigError("Harbor base config must contain exactly one dataset mapping")
    if not any(datasets[0].get(key) for key in ("path", "name", "repo")):
        raise HarborConfigError("Harbor base config dataset must define path, name, or repo")
    if config.get("tasks") not in (None, []):
        raise HarborConfigError("Harbor base config must not add top-level tasks")

    environment = config.get("environment", {})
    if not isinstance(environment, dict):
        raise HarborConfigError("Harbor base config environment must be a mapping")
    if environment.get("import_path") is not None or environment.get("type") not in (
        None,
        "docker",
    ):
        raise HarborConfigError("Terminal-Bench migration requires Harbor Docker")


def _single_agent(config: dict[str, Any], label: str) -> dict[str, Any]:
    agents = config.get("agents")
    if not isinstance(agents, list) or len(agents) != 1 or not isinstance(agents[0], dict):
        raise HarborConfigError(f"{label} must contain exactly one agent mapping")
    return agents[0]


def _normalize_task_ids(task_ids: Sequence[str]) -> tuple[str, ...]:
    if isinstance(task_ids, (str, bytes)):
        raise TypeError("task_ids must be a sequence of strings")
    normalized = []
    for index, task_id in enumerate(task_ids):
        if isinstance(task_id, str):
            metacharacters = sorted(set(task_id) & _FNMATCH_METACHARACTERS)
            if metacharacters:
                raise HarborConfigError(
                    "Terminal-Bench task IDs must not contain fnmatch metacharacters "
                    f"{metacharacters}: {task_id!r}"
                )
        task_id = validate_task_id(task_id, context=f"task_ids[{index}]")
        normalized.append(task_id)
    if not normalized:
        raise HarborConfigError("task_ids must not be empty")
    if len(set(normalized)) != len(normalized):
        raise HarborConfigError("task_ids contains duplicate Terminal-Bench task IDs")
    return tuple(sorted(normalized))


def _resolved_task_ids(config: dict[str, Any], label: str) -> tuple[str, ...]:
    datasets = config.get("datasets")
    if not isinstance(datasets, list) or len(datasets) != 1 or not isinstance(datasets[0], dict):
        raise HarborParityError(f"{label} must contain exactly one dataset mapping")
    task_names = datasets[0].get("task_names")
    if not isinstance(task_names, list) or not task_names:
        raise HarborParityError(f"{label} must contain explicit datasets[0].task_names")
    return _normalize_task_ids(task_names)


def _normalize_harbor_skills(
    harbor_skills: Sequence[str | os.PathLike[str]],
) -> tuple[str, ...]:
    if isinstance(harbor_skills, (str, bytes, os.PathLike)):
        raise TypeError("harbor_skills must be a sequence of directory paths")
    if len(harbor_skills) > 1:
        raise HarborConfigError("Terminal-Bench accepts zero or one generated skill")
    normalized = []
    for index, skill in enumerate(harbor_skills):
        skill_dir = _path(
            skill,
            label=f"harbor_skills[{index}]",
            kind="directory",
            must_exist=True,
        )
        skill_file = skill_dir / "SKILL.md"
        if skill_file.is_symlink() or not skill_file.is_file():
            raise HarborConfigError(
                f"Harbor skill directory must directly contain SKILL.md: {skill_dir}"
            )
        normalized.append(str(skill_dir))
    return tuple(normalized)


def _validate_result_name(value: str) -> str:
    if not isinstance(value, str):
        raise TypeError("result_name must be a string")
    if _RESULT_NAME_PATTERN.fullmatch(value) is None:
        raise HarborConfigError(
            "result_name must start alphanumeric and use only letters, digits, '.', '_', or '-'"
        )
    return value


def _positive_int(value: int, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise HarborConfigError(f"{label} must be a positive integer")
    return value


def _validate_secret_references(
    value: Any,
    *,
    path: str = "config",
    sensitive: bool = False,
) -> None:
    if isinstance(value, Mapping):
        for key, child in value.items():
            key_text = str(key)
            _validate_secret_references(
                child,
                path=f"{path}.{key_text}",
                sensitive=sensitive or _is_sensitive_key(key_text),
            )
        return
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        for index, child in enumerate(value):
            _validate_secret_references(
                child,
                path=f"{path}[{index}]",
                sensitive=sensitive,
            )
        return
    if not isinstance(value, str) or not value:
        return
    assignment_key, separator, assignment_value = value.partition("=")
    assignment_is_sensitive = bool(separator) and _is_sensitive_key(
        assignment_key.strip()
    )
    if (sensitive or assignment_is_sensitive) and _ENV_REFERENCE_PATTERN.search(
        assignment_value if assignment_is_sensitive else value
    ) is None:
        raise HarborConfigError(
            "Harbor config contains a plaintext value in a sensitive field; "
            f"use an environment reference such as '${{VAR}}': {path}"
        )


def _is_sensitive_key(key: str) -> bool:
    normalized = re.sub(r"[^a-z0-9]+", "_", key.casefold()).strip("_")
    return (
        normalized
        in {
            "api_key",
            "apikey",
            "authorization",
            "credential",
            "credentials",
            "password",
            "passwd",
            "secret",
            "token",
        }
        or normalized.endswith(
            (
                "_api_key",
                "_authorization",
                "_credential",
                "_credentials",
                "_password",
                "_secret",
                "_token",
            )
        )
    )


def _resolve_executable(value: str | os.PathLike[str]) -> Path:
    raw = os.fspath(value)
    if not isinstance(raw, str) or not raw.strip():
        raise HarborConfigError("harbor_executable must not be empty")
    candidate = shutil.which(raw)
    if candidate is None:
        raise FileNotFoundError(f"Harbor executable not found: {raw}")
    path = Path(candidate).resolve(strict=True)
    if not path.is_file() or not os.access(path, os.X_OK):
        raise HarborConfigError(f"Harbor executable is not executable: {path}")
    return path


def _path(
    value: str | os.PathLike[str],
    *,
    label: str,
    kind: str,
    must_exist: bool = False,
    create: bool = False,
) -> Path:
    raw = os.fspath(value)
    if not isinstance(raw, str) or not raw.strip():
        raise HarborConfigError(f"{label} must not be empty")
    lexical = Path(raw).expanduser()
    if lexical == Path(".") or ".." in lexical.parts:
        raise HarborConfigError(f"{label} must be explicit and contain no traversal: {raw!r}")
    path = Path(os.path.abspath(lexical))
    if path == Path(path.anchor) or path.is_symlink():
        raise HarborConfigError(f"Unsafe {label}: {path}")
    if create:
        path.mkdir(parents=True, exist_ok=True)
    if must_exist and not path.exists():
        raise FileNotFoundError(f"{label} does not exist: {path}")
    if path.exists() and kind == "file" and not path.is_file():
        raise FileNotFoundError(f"{label} is not a file: {path}")
    if path.exists() and kind == "directory" and not path.is_dir():
        raise HarborConfigError(f"{label} is not a directory: {path}")
    return path.resolve() if path.exists() else path


def _json_bytes(value: Any) -> bytes:
    try:
        text = json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True)
    except (TypeError, ValueError) as exc:
        raise HarborConfigError("Harbor config contains non-JSON values") from exc
    return (text + "\n").encode("utf-8")


def _write_or_verify(path: Path, content: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.is_symlink():
        raise HarborConfigError(f"Harbor runtime artifact must not be a symlink: {path}")
    if path.exists():
        if not path.is_file() or path.read_bytes() != content:
            raise HarborArtifactConflictError(
                f"Existing deterministic Harbor artifact conflicts: {path}"
            )
        return
    try:
        with path.open("xb") as handle:
            handle.write(content)
    except FileExistsError:
        if not path.is_file() or path.read_bytes() != content:
            raise HarborArtifactConflictError(
                f"Existing deterministic Harbor artifact conflicts: {path}"
            )
