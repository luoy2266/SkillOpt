# Harbor 0.20.0 Result and Reward Contract

Phase 4 audit and parser contract for Terminal-Bench v2.1 trial results. This
phase reads one explicit trial `result.json` path. It does not scan a Harbor job
directory, run Harbor, start Docker, call a model, read ATIF, or retry a trial.

## Provenance

- Audit date: 2026-08-29
- Executable: local `harbor==0.20.0` uv tool
- Distribution metadata: `harbor-0.20.0.dist-info`
- Audited result, trial, verifier, and Terminal-Bench mapper files match their
  wheel `RECORD` SHA-256 values.
- Production parsing remains independent of Harbor's Python package and uses
  only the persisted JSON contract.

## Job result

Harbor writes `<jobs_dir>/<job_name>/result.json` as `JobResult` with:

```text
id
started_at
updated_at
finished_at
n_total_trials
stats
```

`stats` contains completed, errored, running, pending, cancelled, and retry
counts plus per-evaluation reward distributions and exception statistics. The
model also defines `trial_results`, but Harbor 0.20.0 writes the job artifact
with that field excluded, including at final completion. The job result is
therefore not the Phase 4 source of an individual task reward.

A set job `finished_at` represents completed job orchestration. An interrupted
job leaves persisted `finished_at` unset. Phase 6 must enforce job-level
completion when it scans and associates trials.

## Trial result

Each trial writes `<trial_dir>/result.json` as `TrialResult`. Relevant fields
are:

```text
id
task_name
trial_name
trial_uri
task_id
source
task_checksum
config
agent_info
agent_result
verifier_result
exception_info
started_at
finished_at
environment_setup
agent_setup
agent_execution
verifier
step_results
```

There is no explicit trial `status` field. Completion and failure must be
classified from `finished_at`, verifier timing/result, and `exception_info`.
Cancellation is represented by `exception_info.exception_type ==
"CancelledError"` and is also counted in job stats.

For the current local Terminal-Bench dataset route, `task_id` serializes as:

```json
{"path": "/absolute/task/source/<task-id>"}
```

The actual task ID is the basename of that path. Harbor's Terminal-Bench mapper
also preserves the source task directory name as `task_name`. Phase 4 requires
both values to equal the expected manifest task ID.

## Verifier reward

`VerifierResult` contains:

```python
rewards: dict[str, float | int] | None
```

The Terminal-Bench reward is stored at:

```python
trial_result["verifier_result"]["rewards"]["reward"]
```

Harbor's generic verifier prefers `verifier/reward.json`; otherwise it parses
`verifier/reward.txt` as a float and wraps it as `{"reward": value}`. Missing,
empty, or malformed reward files raise verifier exceptions and do not produce
a trustworthy `verifier_result`.

Harbor's Terminal-Bench mapper appends exit-code handling to the released test
script and writes `1` when it exits zero and `0` otherwise. The current mapped
Terminal-Bench reward is therefore strictly binary. Phase 4 still validates
the general safe range `[0, 1]` and preserves partial values for compatible
fixtures or later verifier output.

There is no scalar aggregate reward on `TrialResult`. Job stats group the
individual values by reward key and value; those distributions are not a
substitute for the per-trial `rewards["reward"]` field.

## Reward mapping

For a finite numeric verifier reward in `[0, 1]`:

```python
raw_reward = float(verifier_reward)
soft = raw_reward
hard = 1.0 if raw_reward == 1.0 else 0.0
```

Examples:

```text
0.0 -> hard=0.0, soft=0.0
0.5 -> hard=0.0, soft=0.5
1.0 -> hard=1.0, soft=1.0
```

Boolean, null, string, non-finite, negative, and greater-than-one values are
invalid. There is no configurable threshold.

## Failure classification

Harbor records exceptions in:

```text
exception_info.exception_type
exception_info.exception_message
exception_info.exception_traceback
exception_info.occurred_at
```

The current single-step execution flow treats these two agent outcomes
specially:

- `AgentTimeoutError`
- `NonZeroAgentExitCodeError`

It records the exception and then continues to run the verifier. If the trial
is finished and verifier timing plus `rewards["reward"]` are valid, Phase 4
returns a normal scored rollout and exposes the exception type as the derived
`trial_status`. It does not retry. In particular, an agent timeout followed by
a trustworthy reward of zero is a normal model/agent failure.

Every other recorded exception fails closed. This includes cancellation,
environment or Docker startup/setup failures, verifier timeout/output failure,
and cleanup failures recorded after verification. Harbor exposes these through
the exception type and message rather than a separate infrastructure-status
field.

The following also raise `InfrastructureInvalidTrialError` instead of
returning `hard=0, soft=0`:

- missing, unreadable, non-UTF-8, malformed, or non-object `result.json`;
- missing or mismatched task identity;
- missing or invalid completion/verifier timing;
- missing `verifier_result`, `rewards`, or `reward`;
- invalid reward values;
- malformed exception information;
- cancelled or otherwise unclassified incomplete trials.

## Parser API

```python
parse_trial_result(
    result_path,
    *,
    expected_task_id,
) -> dict
```

The returned mapping is:

```python
{
    "id": expected_task_id,
    "hard": hard,
    "soft": soft,
    "raw_reward": raw_reward,
    "trial_status": "completed" | agent_exception_type,
    "harbor_result_path": str(result_path),
}
```

Phase 6, not this parser, will map `PreparedHarborRun` to the job directory,
discover the correct trials, and call the result and trajectory readers.

## Remaining integration verification

A later real single-task run must still confirm that the project owner's exact
Terminal-Bench v2.1 source was mapped with the audited Harbor mapper, that task
names and local `task_id.path` basenames match the Phase 1 manifest, and that
the documented AgentTimeout and infrastructure-failure artifacts are emitted
unchanged under the target server configuration.
