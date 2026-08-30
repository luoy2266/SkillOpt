# Terminal-Bench Harbor Rollout Contract

Phase 6 audit and implementation notes for composing one SkillOpt rollout from
Harbor 0.20.0 job artifacts. This phase was tested only with synthetic job
artifacts and mocked process execution; it did not start Harbor, Docker, or a
model.

## Provenance

- Audit date: 2026-08-29
- Harbor source: the local pinned `harbor==0.20.0` uv tool
- Harbor entrypoint: `harbor run --config <resolved-config>`
- Job implementation: `harbor.job.Job`
- Job schema: `harbor.models.job.result.JobResult` and `JobStats`
- Trial paths: `harbor.models.trial.paths.TrialPaths`
- Trial result and trajectory contracts:
  `RESULT_CONTRACT.md` and `TRAJECTORY_CONTRACT.md`

Production migration code does not import Harbor. The external uv tool is
validated and invoked through its public CLI.

## Harbor 0.20.0 job layout

For `jobs_dir` and `job_name`, `Job.job_dir` is exactly:

```text
<jobs_dir>/<job_name>/
```

The current single-step Terminal-Bench layout is:

```text
<jobs_dir>/<job_name>/
├── config.json
├── lock.json
├── result.json
├── job.log
└── <random-trial-name>/
    ├── config.json
    ├── lock.json
    ├── result.json
    ├── trial.log
    ├── agent/
    │   └── trajectory.json
    ├── verifier/
    └── artifacts/
```

`TrialPaths.result_path` uses singular `result.json`; an older docstring in
Harbor says `results.json` but does not match the implementation. Terminus-2
writes the ATIF artifact at `agent/trajectory.json`.

Trial directory names are generated from a task-derived prefix plus a random
short UUID. They are not a task identity contract and rollout discovery must
not parse or sort them.

## Trial cardinality

`Job._init_trial_configs()` creates trials with the nested product:

```text
n_attempts × resolved tasks × agents
```

The migration base contract permits exactly one Terminus-2 agent. Phase 6 also
requires `n_attempts == 1`, so every selected task must have exactly one trial.
An omitted `n_attempts` has Harbor's default value `1`; any resolved non-`1`
value fails before execution. `n_concurrent_trials` is only scheduling and may
vary.

No averaging, voting, best-of-k selection, or repeat aggregation exists in
Phase 6.

## Job result schema and completion

Harbor's modeled `JobResult` fields are:

```text
id
started_at
updated_at | null
finished_at | null
n_total_trials
stats
trial_results
```

The persisted job `result.json` is always written with `trial_results`
excluded. Individual trial directories are therefore the only source for
per-task rewards and trajectories.

Relevant persisted `JobStats` fields are:

```text
n_completed_trials
n_errored_trials
n_running_trials
n_pending_trials
n_cancelled_trials
n_retries
evals
token/cost totals
```

`Job.run()` writes an initial result with `finished_at: null`, updates progress
during trial execution, and only persists a non-null `finished_at` after its
queue returns and final stats are computed. If `Job.run()` raises, its exception
path intentionally leaves persisted `finished_at` unset. A cancelled trial can
still be included in a normally finalized job and is represented by trial
`exception_info` plus `n_cancelled_trials`.

For an input batch of size `N`, Phase 6 requires:

```text
job result.json is a valid object
started_at is a valid timestamp
finished_at is a valid non-null timestamp
n_total_trials == N
stats.n_completed_trials == N
stats.n_running_trials == 0
stats.n_pending_trials == 0
```

It deliberately does not require `n_errored_trials == 0`: Harbor counts valid
agent/model outcomes such as `AgentTimeoutError` as errors, while the Phase 4
parser can still accept them when verifier evidence is trustworthy. Cancelled,
infrastructure-invalid, and unclassified trials fail later through the strict
per-trial result contract.

This division follows the pinned implementation rather than assuming that
"completed" means "successful". `JobStats.increment()` increments
`n_completed_trials` for every finalized `TrialResult`, then independently
increments `n_errored_trials` when `exception_info` is present. The two counts
can therefore both include the same trial. `Job.run()` sets persisted
`finished_at` only after the trial queue returns and final stats are recomputed.
Job-level checks establish that every planned trial finalized; Phase 4 then
decides whether each finalized artifact is a trustworthy scored outcome.

## Process completion boundary

`HarborRunner.run()` executes the exact command prepared in Phase 3:

```text
harbor run --config <resolved-config>
```

It uses an explicit argument list, `shell=False`, and the base config's parent
as the working directory. It does not expand secret references, retry, impose a
job timeout, or capture stdout/stderr in memory. A nonzero process return raises
immediately.

Before execution it verifies that the prepared version, base config, working
directory, command, and resolved config bytes still match the runner. The
original serialized config SHA-256 is retained at prepare time, so changing both
the mutable in-memory mapping and its file cannot redefine the prepared run. It
also refuses an existing expected job directory, so Phase 6 does not implicitly
resume or rerun a Harbor job.

Harbor's config loader reads the explicit config file but does not rebase Path
fields against that file's directory. Relative values therefore follow process
working-directory semantics. Base `--print-config` validation, resolved
`--print-config` validation, and the real command all use the base config's
parent directory as `cwd`; moving the resolved JSON under the rollout output
does not change a retained relative dataset/task path.

A zero process return is necessary but never sufficient. The rollout validates
the persisted completion contract and exact trial set after `run()` returns.

## Trial discovery and identity

Phase 6 inspects only immediate child directories of the explicit prepared job
directory. For every child directory it loads that directory's `result.json`
and validates both persisted identity fields. It never uses filesystem order,
completion order, job aggregate ordering, or a trial directory name as task
identity.

Discovery constructs:

```text
basename(task_id.path) selector ID -> trial directory
```

Before using that selector key, the shared Phase 4 identity validator requires:

```text
task_name == terminal-bench/<basename(task_id.path)>
```

The expected/discovered selector sets must then match exactly, and the Phase 4
parse requires the selected path basename to equal the expected manifest ID:

```text
split manifest selector ID == basename(Harbor task_id.path)
Harbor task_name            == terminal-bench/<selector-id>
```

Missing, malformed, unexpected, or duplicate identities fail the whole batch.

## Result and trajectory pairing

For each discovered task, Phase 6 reads both artifacts from the same directory:

```text
<trial>/result.json
<trial>/agent/trajectory.json
```

It first calls `parse_trial_result()` with the expected task id. Only an
accepted agent/model outcome proceeds to `convert_atif_trajectory()`. This is
the trust boundary for Phase 5 because Terminus-2 ATIF does not contain a
canonical Terminal-Bench task id.

An infrastructure-invalid result aborts the entire batch. A valid verifier
reward of zero still converts its trajectory because failed execution is
reflection input. Missing or malformed ATIF also aborts the batch. Artifacts
already written before a later task fails may remain for diagnosis, but no
partial result list is returned.

The pinned trainer assigns the return value of `adapter.rollout()` before it
calls `compute_score()` or `adapter.reflect()`. It does not catch rollout
exceptions on that path. A Phase 6 exception therefore aborts the current
trainer batch before scoring or reflection; any earlier conversation files are
debug/provenance residue only and are not consumed as a successful rollout.

## SkillOpt output and order

The orchestration sequence is:

```text
items + skill_content
  -> package_skill_content()
  -> HarborRunner.prepare()
  -> HarborRunner.run()
  -> validate completed job
  -> discover exact trials
  -> parse_trial_result()
  -> convert_atif_trajectory()
  -> restore input order
  -> list[dict]
```

Only `PackagedSkill.harbor_skills` enters the Harbor config. Blank content stays
`agents[].skills=[]`; nonblank content uses the single Phase 2 skill directory.

Harbor may complete and persist trials in any order. Returned rollout entries
are reconstructed by task id so that:

```text
len(results) == len(items)
results[i]["id"] == items[i]["id"]
```

Conversation artifacts are written only under:

```text
<rollout_dir>/predictions/<task-id>/conversation.json
```

Passing the same `rollout_dir` to Phase 2 and Phase 3 creates separate sibling
namespaces:

```text
<rollout_dir>/harbor_skills/...
<rollout_dir>/harbor_runtime/configs/...
<rollout_dir>/harbor_runtime/jobs/<result-name>/...
<rollout_dir>/predictions/<task-id>/conversation.json
```

SkillOpt may create `<rollout_dir>` before rollout. Harbor execution rejects
only an existing concrete `harbor_runtime/jobs/<result-name>` directory, not
the rollout root or the `predictions` directory.

The Harbor job tree remains under the prepared jobs directory and is not copied.
Each result retains Phase 4 fields plus lightweight paths to the resolved config
and job directory and the Phase 2 skill digest. The persisted resolved config,
skill artifact, job path, result path, and selected task IDs provide the Phase 6
provenance chain without a separate database.

## Deferred real integration checks

The first real single-task smoke must verify:

1. the persisted result follows the statically audited two-level identity
   contract: selector/path basename `X` and canonical `task_name`
   `terminal-bench/X`;
2. a successful CLI return produces the audited complete job result fields;
3. exactly one trial directory is produced for `n_attempts=1`;
4. the real trial writes both `result.json` and `agent/trajectory.json`;
5. an agent timeout with verifier reward zero has the expected complete job
   counts and is accepted without rerun;
6. a real infrastructure or cancelled case remains fail-closed;
7. large Harbor output is adequately handled by the target service/logging
   setup, since Phase 6 intentionally does not capture it.

Retry/rerun behavior, multi-attempt semantics, adapter registration, and
server hardening remain outside Phase 6.
