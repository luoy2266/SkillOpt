# Harbor 0.20.0 Runtime Configuration Boundary

Phase 3 audit and implementation notes for the SkillOpt → Harbor boundary.
Phase 3 prepared and validated configs only. Phase 6 later enabled the exact
prepared command through the public CLI; see `ROLLOUT_CONTRACT.md`. No migration
test in either phase started Docker, resolved a real Terminal-Bench task, or
called a model.

## Provenance

- Audit date: 2026-08-29
- Harbor executable audited: the local `harbor==0.20.0` uv tool
- Public command boundary: `harbor run --config <path>`
- Schema: `harbor.models.job.config.JobConfig`
- Target agent: `agents[0].name: terminus-2`
- Target environment: Harbor Docker environment

The installed Harbor package is external to this repository and is neither
vendored nor imported by SkillOpt's Python environment.

## Harbor 0.20.0 contract

### CLI and schema validation

`harbor run` is an alias for `harbor job start`. It accepts YAML or JSON via:

```text
harbor run --config <config.yaml-or-json>
```

`--print-config` parses the input with `JobConfig`, prints resolved config JSON,
and returns before `Job.create()`. Phase 3 uses this mode for schema validation:

```text
harbor run --config <config> --print-config
```

This starts no job, performs no dataset resolution, runs no Docker preflight,
and makes no target-model request.

### Relevant `JobConfig` fields

The actual Harbor 0.20.0 names are:

- `job_name`: job/result directory name;
- `jobs_dir`: parent directory for job results;
- `n_attempts`: number of attempts per task/agent combination;
- `n_concurrent_trials`: maximum concurrent trials;
- `agents`: list of `AgentConfig` objects;
- `datasets`: list of dataset sources and filters;
- `tasks`: explicit ad-hoc task sources;
- `environment`: shared environment configuration;
- `verifier`, timeout multipliers, retry, metrics, artifacts, and extra
  instruction paths: runtime inputs preserved from the base config.

There is no separate `result_name` or generic `output_dir` field in
`JobConfig`. The job directory is:

```text
<jobs_dir>/<job_name>/
```

### Task selection

`DatasetConfig.task_names` is a list of `fnmatch` patterns applied to each
Harbor task identifier's `get_name()`. For a local dataset this is the task
directory basename. Phase 3 accepts Terminal-Bench IDs already validated by
the Phase 1 data layer and additionally rejects all `fnmatch` metacharacters
`*`, `?`, `[`, and `]`, so each value is an exact name rather than a pattern.

The builder requires exactly one dataset source and no top-level `tasks`.
For each rollout batch it writes:

```yaml
datasets:
  - <unchanged source fields>
    task_names: [<sorted exact task IDs>]
    exclude_task_names: null
    n_tasks: null
```

Clearing the exclusion and truncation fields is part of the task-selection
overlay; otherwise a validated base config could silently remove selected
batch tasks. No Terminal-Bench task content is copied into SkillOpt.

### Agent skills and Docker

`AgentConfig.skills` validates `str | Path` entries and serializes them as
strings. The Phase 3 builder consumes the Phase 2 `harbor_skills` value
directly:

```text
blank package    -> agents[0].skills = []
nonblank package -> agents[0].skills = [absolute skill directory]
```

It does not reinterpret blank content. The base config must use one
`terminus-2` agent, provide its `model_name`, and contain no pre-existing
skills. A nonblank runtime skill must be an absolute local directory directly
containing a regular `SKILL.md`.

`EnvironmentConfig.type` defaults to Docker when both `type` and
`import_path` are absent. The migration base contract accepts that default or
explicit `type: docker` and rejects a custom environment import or any other
environment type. All Docker/resource fields remain base-config passthrough.

### Result layout

Static inspection of Harbor 0.20.0 records the following host layout after a
future successful run:

```text
<jobs_dir>/<job_name>/
├── config.json
├── lock.json
├── result.json
├── job.log
└── <trial_name>/
    ├── config.json
    ├── lock.json
    ├── result.json
    ├── trial.log
    ├── agent/
    │   └── trajectory.json
    ├── verifier/
    │   ├── reward.json or reward.txt
    │   └── test output/logs
    └── artifacts/
```

Terminus-2 declares ATIF support and writes its primary trajectory to
`agent/trajectory.json`; continuation/summarization files may also exist.
Phase 3 records these paths for later parser design but does not read them.

The `TrialPaths` docstring says `results.json`, but its actual `result_path`
property and all current readers/writers use singular `result.json`.

## Boundary decision: CLI, not Python API

The migration uses the public CLI/subprocess boundary because:

1. Harbor is a separately pinned external tool and is not installed in the
   SkillOpt Python environment;
2. the project owner's existing validated YAML can be passed through directly;
3. `harbor run --config` is the documented launch interface;
4. resolved config files remain inspectable and replayable;
5. the Python path requires async `Job.create()` and internal objects that
   resolve tasks/skills and mutate runtime config in place;
6. the CLI is easier to execute consistently under a future systemd/Docker
   server setup.

Phase 3 tests invoke only `harbor --version` and `harbor run --print-config`.
Phase 6 intentionally enables `HarborRunner.run()` for the exact prepared
command and validates job artifacts separately after process completion.

## Base config input contract

`HarborRunner` accepts one local `.yaml`, `.yml`, or `.json` path. The file must
already represent the project owner's validated Terminal-Bench v2.1 baseline:

- exactly one `terminus-2` agent with a non-empty `model_name`;
- `agents[0].skills` absent or empty;
- exactly one dataset source in `datasets`;
- no top-level `tasks` entries;
- Docker environment, either explicit or by Harbor default;
- all model, agent kwargs, Docker/resource, timeout, retry, proxy/cache,
  verifier, instruction/template, and build policy values set by the owner.

The migration does not guess or repair an invalid config. Harbor 0.20.0 must
accept both the base and the resolved config through `--print-config`.

Relative paths retained from the base config are interpreted with the base
config's parent directory as the prepared subprocess working directory. New
skill, config, jobs, and output paths are serialized as absolute paths.
Harbor's loader does not rebase nested Path values against the resolved config
file location, so both `--print-config` validation and execution must retain
that same working directory.

Harbor's YAML/JSON loader does not expand environment references.
`--print-config` serializes agent/environment/verifier `env` mappings with
Harbor's sensitive-env serializer: `${VAR}` references remain unchanged;
sensitive literal values matching the host environment are replaced by a
reference, and other sensitive literals are partially redacted. This protection
does not cover arbitrary nested kwargs or headers, and the migration writes its
own raw resolved mapping rather than Harbor's printed representation.

The migration therefore recursively rejects non-empty plaintext string values
under secret-like keys such as API key, token, secret, password, credential,
and Authorization fields. Such values must contain an unexpanded `${VAR}`
reference; forms such as `Bearer ${TOKEN}` are allowed. Sensitive references
with inline defaults are intentionally not accepted because the default would
also be persisted. The resolved artifact preserves the reference, never reads
the host secret, and the dry-run manifest does not embed the resolved config or
environment values.

## Minimal overlay and immutability

`build_harbor_config()` deep-copies the input mapping. It changes only:

- `datasets[0].task_names`;
- `datasets[0].exclude_task_names` and `datasets[0].n_tasks`, cleared to make
  the selected set exact;
- `agents[0].skills`;
- `job_name`;
- `jobs_dir`;
- `n_attempts` and `n_concurrent_trials` only when explicitly overridden.

It does not modify the caller's mapping, and each build starts from the stored
baseline copy. Duplicate task IDs fail before serialization; no deduplication
changes rollout cardinality. Valid IDs are sorted only for deterministic config
serialization. This sorting is not a Harbor execution-order contract.

## Parity contract

`assert_harbor_config_parity()` requires the same explicit task IDs and an
empty baseline skills list. Its complete config-field allowlist is:

```text
job_name
jobs_dir
agents[*].skills
```

Skill provenance and packaging digest stay outside `JobConfig`, in the Phase 2
artifact path and dry-run metadata, so no additional runtime-config exception
is necessary. Every other difference fails, including model, agent kwargs,
environment/resources, timeouts, retry, task source/instruction, concurrency,
and attempts.

## Prepared and dry-run artifacts

For `output_root=<root>` and `result_name=<name>`:

```text
<root>/
└── harbor_runtime/
    ├── configs/<name>.json
    ├── dry_runs/<name>.json
    └── jobs/
        └── <name>/          # created by a future real Harbor execution
```

The resolved config and dry-run JSON use UTF-8, sorted keys, two-space
indentation, and one trailing newline. Existing mismatched deterministic files
raise an error instead of being overwritten. `PreparedHarborRun` also retains
the original resolved-config SHA-256; execution rejects later byte changes even
if the exposed in-memory config mapping was changed to match them.

The dry-run manifest contains only paths, selected task IDs, the native skills
list, result name, expected job directory, prepared command, working directory,
Harbor version, and `execution_started: false`.

## Remaining integration verification

No Phase 3 test resolves or executes real Terminal-Bench tasks. Later phases
must still verify with one real job that:

1. the owner's real Terminal-Bench v2.1 dataset source exposes task names that
   exactly match the Phase 1 manifest IDs;
2. Docker preflight and the full Terminus-2/model config are valid on the
   target server;
3. injected skills upload to the effective `skills_dir` and Terminus-2 reads
   their `SKILL.md`;
4. the real job lock contains the expected skill digest;
5. the documented job/trial/result/ATIF paths are emitted for this exact
   Terminal-Bench configuration;
6. later result and trajectory readers correctly handle failures, retries,
   multi-attempt jobs, and any continuation ATIF files.
