# Terminal-Bench SkillOpt Configuration

Phase 8 connects repository configuration to the Phase 7
`TerminalBenchAdapter`. It performs no Harbor job, Docker, or model execution.

## Configuration entry points

- `configs/terminalbench/default.yaml` is the normal train/eval handoff.
- `configs/terminalbench/smoke.yaml` inherits `default.yaml` and limits each
  loaded split to one item for the Phase 9 single-task smoke.
- Both files use the pinned SkillOpt `_base_` contract: one string path,
  resolved relative to the child YAML file.

Run these commands from the repository root. `_base_` is config-relative, but
ordinary values such as `skill_init` and relative CLI path overrides retain
the caller's working-directory semantics. Absolute server paths are therefore
recommended for the external split, Harbor config, and output root.

## Required external inputs

The checked-in values
`REPLACE_WITH_MATERIALIZED_TERMINALBENCH_SPLIT` and
`REPLACE_WITH_VALIDATED_HARBOR_BASE_CONFIG.yaml` are deliberate non-runnable
placeholders. The config loader does not expand `${VARIABLE}` expressions.
Supply real paths with the supported `--cfg-options section.key=value` syntax.

### Materialized split

The repository does not contain an authoritative Terminal-Bench v2.1 task-ID
source or a formal 89-task manifest. First materialize the real split:

```bash
python3 -m scripts.materialize_terminalbench_split \
  --source /srv/terminal-bench-v2.1/tasks \
  --source-revision '<pinned-source-revision>' \
  --ratio 1:1:8 \
  --seed 42 \
  --output-dir /srv/skillopt-inputs/terminalbench-split
```

The resulting directory must satisfy the Phase 1 manifest contract documented
in `docs/terminalbench/SPLIT_MANIFEST.md`. `data/terminalbench_split/` contains
only instructions and is not a runnable formal split.

### Harbor base config

Provide a baseline YAML already validated independently with Harbor 0.20.0 and
Terminal-Bench v2.1. It remains the source of truth for Terminus-2, Docker,
DeepSeek-V4-Flash-0731 target execution, resources, timeouts, and provider
runtime settings. It must satisfy `docs/terminalbench/HARBOR_RUNTIME.md`,
including one Terminus-2 agent, `n_attempts: 1`, and baseline `skills: []`.

Do not add credentials to either SkillOpt YAML. Harbor credentials remain in
the base config's supported environment references/runtime environment.

## Optimizer and target boundary

The checked-in config selects SkillOpt's native `openai_compatible` optimizer
backend. Keep its API request identifier separate from experiment provenance:

```text
Optimizer API request model:          deepseek-v4-flash
Underlying experiment/model identity: DeepSeek-V4-Flash-0731
Optimizer reasoning effort:           max
```

The request model is passed directly to the OpenAI-compatible Chat Completions
API; it does not use a LiteLLM provider prefix. Provide the endpoint and
credential through the existing runtime variables:

```text
OPTIMIZER_OPENAI_COMPATIBLE_BASE_URL=https://api.deepseek.com
OPTIMIZER_OPENAI_COMPATIBLE_API_KEY
```

For a local process that already has `DEEPSEEK_API_KEY`, the credential may be
bridged without persisting it:

```bash
OPTIMIZER_OPENAI_COMPATIBLE_API_KEY="$DEEPSEEK_API_KEY" \
python3 scripts/train.py ...
```

The shared `OPENAI_COMPATIBLE_*` variables remain an upstream fallback. The
inherited SkillOpt `model.target` and `model.target_backend` fields are not
consumed by `TerminalBenchAdapter`; they do not configure the Terminal-Bench
target. Its separate identity contract is:

```text
Harbor/LiteLLM request model:          deepseek/deepseek-v4-flash
Underlying experiment/model identity: DeepSeek-V4-Flash-0731
Terminus reasoning effort:             max
```

The actual target path is exclusively:

```text
TerminalBenchAdapter -> Harbor -> Terminus-2 -> DeepSeek-V4-Flash-0731
```

No Harbor target-model backend is registered in SkillOpt.

## Optimizer reflection diagnostics

Trainer reflection calls write SkillOpt-owned diagnostics under the current
step directory:

```text
<out_root>/
  steps/
    step_0001/
      optimizer_diagnostics/
        attempts.jsonl
        response_0001.txt
        parse_0001.json
```

This namespace is separate from
`harbor_runtime/.../agent/request_events.jsonl`. Harbor request events describe
Terminus target-model traffic; `optimizer_diagnostics/` describes the SkillOpt
Python process calling its configured reflection/optimizer backend.

`attempts.jsonl` contains one deterministic record per logical reflection
request. Records include the stage, task IDs, backend and request model,
scheme/host-only endpoint, reasoning effort, prompt character counts and
SHA-256 identities, request/completion token limits, usage, application-level
attempts, latency, finish reason, response byte count/hash, and parser status.
SDK-internal transport retry counts are recorded as `unknown` when the SDK does
not expose them.

The raw response file is the exact UTF-8 text returned by `chat_optimizer()` to
the production parser. It is not stripped, repaired, normalized, or
pretty-printed. An empty response is persisted as a zero-byte file. When
`extract_json()` succeeds, `parse_NNNN.json` contains that exact extracted JSON
structure before source tagging or edit-budget truncation. Failed extraction
does not create a synthetic parse artifact.

Prompt text is not persisted by default. The ledger records system/user prompt
character and byte counts plus SHA-256 identities, along with formatted
trajectory and current-skill character counts. This provides prompt identity
without making every generic OpenAI-compatible caller persist potentially
sensitive prompt content.

The stable parser statuses are:

```text
transport_error
empty_response
json_extract_failed
missing_patch
invalid_patch_type
missing_edits
invalid_edits_type
empty_edits
patch_returned
exception
```

Diagnostics never contain API keys, Authorization headers, bearer tokens, or
cookies. Endpoint metadata is limited to scheme plus host. Diagnostic directory
creation and writability are checked before reflection requests start. A later
individual artifact-write failure emits an explicit warning but preserves the
existing model response and parser/update semantics.

## Phase 9 single-task smoke

Use `eval_only.py`, one explicit split, and a clean unique output root:

```bash
python3 scripts/eval_only.py \
  --config configs/terminalbench/smoke.yaml \
  --skill skillopt/envs/terminalbench/skills/initial.md \
  --split valid_seen \
  --cfg-options \
    env.split_dir=/srv/skillopt-inputs/terminalbench-split \
    env.harbor_base_config=/srv/harbor/tbench-dsv4-flash-docker.yaml \
    env.out_root=/srv/skillopt-runs/tbench-smoke-baseline
```

`env.limit=1` makes `valid_seen` contain one item and
`env.n_concurrent_trials=1` keeps Harbor concurrency at one. The external
Harbor base config must set `n_attempts: 1`. This command therefore describes
one task, one attempt, and one adapter rollout. Running `scripts/train.py` is
not a one-rollout smoke because the pinned trainer also performs baseline and
validation rollouts.

## Training handoff

After the smoke succeeds, a normal training invocation is:

```bash
python3 scripts/train.py \
  --config configs/terminalbench/default.yaml \
  --cfg-options \
    env.split_dir=/srv/skillopt-inputs/terminalbench-split \
    env.harbor_base_config=/srv/harbor/tbench-dsv4-flash-docker.yaml \
    env.out_root=/srv/skillopt-runs/tbench-train
```

Use additional supported SkillOpt overrides for epochs, batch sizes,
validation counts, or optimizer settings as needed. Do not move Harbor server
hardening fields into `TerminalBenchAdapter`; keep them in the validated Harbor
base config and server runtime.

## Baseline and skill evaluation

Baseline evaluation keeps the upstream `--skill` requirement and passes the
real empty initialization file:

```bash
python3 scripts/eval_only.py \
  --config configs/terminalbench/default.yaml \
  --skill skillopt/envs/terminalbench/skills/initial.md \
  --split valid_unseen \
  --cfg-options \
    env.split_dir=/srv/skillopt-inputs/terminalbench-split \
    env.harbor_base_config=/srv/harbor/tbench-dsv4-flash-docker.yaml \
    env.out_root=/srv/skillopt-runs/tbench-eval-baseline
```

The file is read as semantically blank content, and Phase 2 maps it to Harbor
`agents[].skills=[]`. Evaluate a generated or best skill with the same config,
split, Harbor base config, and runtime settings, changing only `--skill` and a
clean output root:

```bash
python3 scripts/eval_only.py \
  --config configs/terminalbench/default.yaml \
  --skill /srv/skillopt-runs/tbench-train/best_skill.md \
  --split valid_unseen \
  --cfg-options \
    env.split_dir=/srv/skillopt-inputs/terminalbench-split \
    env.harbor_base_config=/srv/harbor/tbench-dsv4-flash-docker.yaml \
    env.out_root=/srv/skillopt-runs/tbench-eval-skill
```

## Phase 9 pre-smoke checklist

- [ ] Harbor exactly 0.20.0
- [ ] Docker available
- [ ] Real Terminal-Bench v2.1 source available
- [ ] Authoritative/materialized split available
- [ ] Selected smoke task exists
- [ ] Harbor base config validated independently
- [ ] Terminus-2 2.0.0
- [ ] DeepSeek-V4-Flash-0731 target config present in Harbor
- [ ] SkillOpt optimizer endpoint and credential available
- [ ] Optimizer request model is `deepseek-v4-flash`
- [ ] `skillopt/envs/terminalbench/skills/initial.md` exists
- [ ] Expected SkillOpt and Harbor output roots are clean

The 89-task source audit has confirmed the selector ID `X` maps to canonical
Harbor `task_name` `terminal-bench/X`. The first real smoke must still confirm
that the persisted trial artifacts follow that audited contract. Native skill
discovery, job completion artifacts, trial cardinality, trajectory location,
AgentTimeout, and infrastructure-invalid artifacts also remain unverified.
