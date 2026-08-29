# Terminal-Bench v2.1 Protocol Mapping

Phase 0 interface audit for the SkillOpt → Harbor → Terminus-2 → Docker →
DeepSeek-V4-Flash-0731 migration.

This document records the contract of the pinned SkillOpt source. It does not
implement a Terminal-Bench adapter, invoke Harbor, start Docker, or call the
target model.

## Provenance and audit scope

- Audit date: 2026-08-29
- Pinned upstream commit: `eb8c1e7bcbccdd80f9d422f12018fcd8e84ce19a`
- Migration branch: `terminalbench-v2.1`
- Audit HEAD: `b2415e260bd7be0df5a7f7821c92b3e98cc794a3`
- `git diff eb8c1e7...HEAD -- skillopt scripts configs tests` is empty.
  The migration HEAD adds only migration documents, so the interfaces below
  are the pinned upstream interfaces rather than locally modified behavior.
- Harbor is an external dependency pinned to `0.20.0`. Its result and ATIF
  file schemas were not exercised in Phase 0; exact Harbor-side paths and
  field names remain implementation-time checks against that pinned version.

## Key files read

### Migration and benchmark guidance

- `AGENTS.md`
- `UPSTREAM.md`
- `docs/terminalbench/CODEX_MIGRATION_TASK.md`
- `docs/guide/new-benchmark.md`
- `docs/reference/api.md`

### Pinned SkillOpt contracts

- `skillopt/envs/base.py`
- `skillopt/datasets/base.py`
- `skillopt/types.py`
- `skillopt/config.py`
- `skillopt/engine/trainer.py`
- `skillopt/gradient/reflect.py`
- `skillopt/evaluation/gate.py`
- `skillopt/utils/scoring.py`
- `scripts/train.py`
- `scripts/eval_only.py`
- `configs/_base_/default.yaml`

### Closest exec-style reference

- `skillopt/envs/searchqa/adapter.py`
- `skillopt/envs/searchqa/dataloader.py`
- `skillopt/envs/searchqa/rollout.py`
- `skillopt/model/codex_harness.py`

`SearchQA` is the closest structural reference because it uses a thin
dataset-backed adapter, delegates execution to a rollout helper, creates
per-task prediction directories, supports an exec target path, persists
`conversation.json`, and returns normal SkillOpt result dictionaries.

It is not an execution implementation to reuse for Terminal-Bench: SearchQA
calls SkillOpt's target exec backend directly and prepares its own
`.agents/skills/.../SKILL.md` workspace. Terminal-Bench must instead execute
through Harbor and Terminus-2 and inject skills only through Harbor's native
`agents[].skills` field.

## End-to-end protocol mapping

| Concern | Pinned SkillOpt side | Terminal-Bench migration side |
| --- | --- | --- |
| Dataset | `SplitDataLoader` with `train/`, `val/`, `test/` | Replaceable task-ID manifests under the same split layout |
| Batch payload | `BatchSpec.payload`, normally `list[dict]` | Item dictionaries containing at least a stable string `id` |
| Skill input | `skill_content: str` | Blank/whitespace-only content maps to Harbor `skills=[]`; non-blank content maps to one generated skill directory |
| Execution entry | `EnvAdapter.rollout(env_manager, skill_content, out_dir, **kwargs)` | One shared Harbor runner invokes Terminus-2 in Docker |
| Target model route | Environment-specific rollout helper | Harbor → Terminus-2 → DeepSeek-V4-Flash-0731; never `chat_target()` or `run_target_exec()` |
| Reward | Result dict fields `hard` and `soft` | Terminal-Bench verifier reward, without a custom scorer |
| Trajectory | `<out_dir>/predictions/<id>/conversation.json` | Convert Harbor/Terminus-2 ATIF into a supported conversation record list |
| Reflection | Inherited `EnvAdapter.reflect()` | Shared `run_minibatch_reflect`; no Terminal-Bench-specific reflection algorithm initially |
| Registration | Lazy `_ENV_REGISTRY` in both CLI scripts | Add `terminalbench` to both registries in a later phase |
| Config | Structured YAML flattened to the trainer's flat dict | `env.name: terminalbench` plus explicit adapter constructor parameters |

## `EnvAdapter` contract

The abstract methods at the pinned commit are:

```python
def build_train_env(self, batch_size: int, seed: int, **kwargs)
def build_eval_env(self, env_num: int, split: str, seed: int, **kwargs)
def rollout(
    self,
    env_manager,
    skill_content: str,
    out_dir: str,
    **kwargs,
) -> list[dict]
def get_task_types(self) -> list[str]
```

Source: `skillopt/envs/base.py:187`, `skillopt/envs/base.py:197`,
`skillopt/envs/base.py:216`, and `skillopt/envs/base.py:274`.

Relevant default hooks are:

- `setup(cfg)`: stores a shallow copy on `self._cfg`. Dataset-backed adapters
  normally override it, call `super().setup(cfg)`, then call
  `self.dataloader.setup(cfg)`.
- `get_dataloader()`: returns `None` by default. Returning the loader activates
  the trainer's `BatchSpec`/epoch-planning path.
- `requires_ray()`: defaults to `False`.
- `build_reference_text`, `get_reference_metadata`,
  `attach_reference_context`, and `select_representative_items`: available
  helper hooks, but the pinned trainer has no call site for them. A future
  Terminal-Bench rollout must not rely on automatic reference/metadata
  attachment.
- `build_env_from_batch(batch, **kwargs)`: defaults to routing back through
  `build_train_env` or `build_eval_env`.
- `reflect(...)`: defaults to `run_minibatch_reflect(...)` and reads
  `analyst_workers`, `failure_only`, `minibatch_size`, and `edit_budget` from
  adapter attributes.

### Practical requirement for dataset-backed adapters

Although `build_env_from_batch` is not abstract, the current trainer calls it
whenever `get_dataloader()` returns a loader. Existing dataset-backed adapters
override it as:

```python
def build_env_from_batch(self, batch: BatchSpec, **kwargs):
    return list(batch.payload or [])
```

Without that override, the default implementation calls `build_train_env` or
`build_eval_env` again and can resample/rebuild instead of consuming the batch
already planned by the dataloader. A future `TerminalBenchAdapter` therefore
needs this method in addition to the four abstract methods.

### Actual trainer call sequence

Training uses the following sequence:

```text
scripts/train.py:get_adapter
→ ReflACTTrainer.train
→ adapter.setup(cfg)
→ adapter.get_dataloader()
→ dataloader.plan_train_epoch(...)
→ adapter.build_env_from_batch(batch)
→ adapter.rollout(items, current_skill, rollout_dir, use_eval_feedback=True)
→ compute_score(results)
→ adapter.reflect(results, current_skill, batch_dir,
                  prediction_dir=<rollout_dir>/predictions, ...)
```

Candidate validation and final evaluation call the same adapter and rollout
method with a `val`/`test` batch. There is no separate `EnvAdapter.evaluate()`
method; verifier parsing and score assignment belong inside rollout handling.

`eval_only.py` differs slightly: after `adapter.setup(cfg)`, it calls
`adapter.build_eval_env(...)` directly rather than using
`dataloader.build_eval_batch(...)` plus `build_env_from_batch(...)`.

## `SplitDataLoader` contract

`SplitDataLoader` supports:

- `split_mode="split_dir"`: load an existing directory tree.
- `split_mode="ratio"`: load raw JSON/JSONL, shuffle deterministically, and
  materialize a split under the run output unless `split_output_dir` is set.

The required split layout is:

```text
<split_dir>/
├── train/
├── val/
└── test/
```

The base `load_split_items(split_path)` reads the lexicographically first
`*.json` file and requires a top-level JSON array. A future
`train/tasks.json`, `val/tasks.json`, and `test/tasks.json` layout is therefore
compatible without changing the base loader. A subclass only needs to
override loading when normalization or a different format is required.

Other current behavior:

- Split aliases are `valid_seen → val`, `selection → val`, and
  `valid_unseen → test`.
- Unknown split names fall back to `val` through `get_split_items()`.
- `limit > 0` truncates every loaded split independently.
- `build_train_batch` shuffles then takes the requested prefix.
- `plan_train_epoch` shuffles the full train split and emits sequential
  `BatchSpec.payload` slices, so a normal epoch covers the loaded train split.
- `build_eval_batch` preserves split order and takes the first `env_num` items;
  `env_num=0` means all items.
- If `train.train_size` is nonzero, the trainer requires it to equal the
  loaded train split size. `train_size: 0` lets the trainer infer it.

`SplitDataLoader` does not validate a universal item schema. However, rollout
and reflection directly use `item["id"]`, so the Terminal-Bench loader must
normalize every manifest entry to a non-empty, stable, filesystem-safe string
`id`. The same exact string must identify the Harbor trial, rollout result,
and prediction directory.

## Rollout return schema

The actual runtime contract is a plain `list[dict]`. The trainer does not
instantiate `RolloutResult` before scoring or reflection.

Minimum fields needed by the current pipeline are:

```python
{
    "id": str,
    "hard": float,
    "soft": float,
}
```

Phase 4 Terminal-Bench result shape:

```python
{
    "id": task_id,
    "raw_reward": float(verifier_reward),
    "hard": 1.0 if verifier_reward == 1.0 else 0.0,
    "soft": float(verifier_reward),
    "trial_status": derived_trial_status,
    "harbor_result_path": harbor_result_path,
}
```

The common `RolloutResult` dataclass accepts fractional `hard` values, but the
Terminal-Bench mapping intentionally keeps `hard` binary. It has named optional
fields plus `extras`. `RolloutResult.from_dict()` moves unknown keys into
`extras`, and `to_dict()` merges them back. That conversion is a utility only
at the pinned commit: the trainer and built-in rollouts continue to pass
dictionaries directly. Extra Terminal-Bench metadata will therefore remain
ordinary dict keys unless later code explicitly constructs a `RolloutResult`.

No current layer validates score bounds. `compute_score()` converts `hard` and
`soft` to floats and returns their arithmetic means.

## Reward and failure semantics

The migration must use Terminal-Bench's verifier reward and must not add a
text-based scorer.

The original direct mapping `hard = reward` is superseded. Reflection currently
classifies outcomes as:

```python
failure = not hard or float(hard) < 1e-9
success = bool(hard)
```

Therefore any positive fractional `hard` value would be treated as a successful
trajectory. The frozen Terminal-Bench mapping is instead:

```python
raw_reward = float(verifier_reward)
soft = raw_reward
hard = 1.0 if raw_reward == 1.0 else 0.0
```

The reward must be a finite numeric value in `[0, 1]`. This preserves partial
credit in `soft` while only complete verifier success enters reflection as a
successful rollout. Harbor 0.20.0's Terminal-Bench mapper currently emits a
strict binary `reward` value, but the parser retains the safe partial-credit
mapping.

The trainer also does not inspect `infrastructure_valid`. Phase 4 therefore
fails closed before returning a scored result: missing or corrupt artifacts,
missing or invalid verifier output, cancelled/incomplete trials, and explicit
infrastructure exceptions raise `InfrastructureInvalidTrialError`. A legitimate
verifier reward of zero remains a model/task failure, not an infrastructure
failure. `AgentTimeoutError` and agent nonzero exit are scored only when Harbor
still completed the verifier and persisted a trustworthy reward.

The validation gate consumes aggregate `hard` and `soft` means and can compare
`hard`, `soft`, or a configured weighted `mixed` score. No Terminal-Bench
specific gate is needed for the initial migration.

## `conversation.json` and reflection

For each returned result, rollout learning artifacts belong at:

```text
<rollout out_dir>/
└── predictions/
    └── <result-id>/
        └── conversation.json
```

During training, `<rollout out_dir>` is the step's `rollout/` directory. The
trainer passes its `predictions/` path explicitly to the inherited reflection
method.

`conversation.json` must contain a non-empty JSON list. The formatter accepts
more than ordinary role/content chat messages:

1. Tool-call record:

   ```json
   {"type": "tool_call", "cmd": "...", "obs": "..."}
   ```

2. Environment step record:

   ```json
   {
     "step": 1,
     "reasoning": "...",
     "action": "...",
     "env_feedback": "..."
   }
   ```

3. Generic message record:

   ```json
   {"role": "assistant", "content": "..."}
   ```

Any other dictionary is rendered through its `content` field, with a default
display role of `agent`. A `role: "system"` record is rendered as
post-execution verification information, which is suitable for a verifier
summary but should not be used to simulate Harbor skill injection.

This means ATIF conversion does not need to collapse all terminal activity
into one role/content string. Commands and observations can be mapped to the
supported tool-call or step records while preserving order. The converter
should retain the task instruction, assistant messages/reasoning when
available, terminal commands, outputs, observations, final response, and a
verifier summary.

Optional reflection context can also come from result fields or sibling files:

- `task_description`, `task_type`, `fail_reason`, `n_turns`
- `reference_text`
- `target_system_prompt` or `target_system_prompt.txt`
- `target_user_prompt` or `target_user_prompt.txt`
- selected backend trace summaries

Missing or empty conversation files are skipped while formatting the analyst
input. If a whole minibatch has no usable trajectories, the analyst function
returns `None`, so results can still be scored but cannot produce learning
patches.

## Skill mapping and baseline semantics

SkillOpt exposes one `skill_content: str`; Harbor expects a list of skill
directories. The packaging boundary is:

```text
not skill_content.strip() → agents[].skills = []
skill_content.strip()     → package one stable skill directory containing SKILL.md
```

Blank content must not be packaged as a placeholder skill. This is different
from `codex_harness.render_skill_md()`, which turns blank content into a
non-empty fallback message and still installs a skill. That behavior is valid
for existing exec backends but would violate the Terminal-Bench baseline
contract.

The trainer's built-in “baseline” is the initial skill `S_0`, not an intrinsic
no-skill mode. For Terminal-Bench, `env.skill_init` must point to an existing
empty Markdown file so `S_0` maps to Harbor `skills=[]`.

The pinned trainer has an edge case for `skill_init: ""`: `os.path.abspath("")`
resolves to the current directory, which exists, and the trainer then attempts
to open that directory. Therefore the config must not use an empty path;
`skillopt/envs/terminalbench/skills/initial.md` now exists with a whitespace-only
body and packages to Harbor `skills=[]`.

`eval_only.py` does not read `env.skill_init`. It requires an explicit
`--skill <path>` and reads that file. Baseline evaluation will therefore also
need a real empty Markdown file; best-skill evaluation will pass
`best_skill.md`.

### Phase 2 Harbor 0.20.0 skill audit

Static inspection of the pinned local Harbor `0.20.0` distribution confirmed:

- `agents[].skills` is a list of string/path values;
- a local value points to a skill directory containing `SKILL.md`, or to a
  parent whose immediate child directories each contain `SKILL.md`;
- Harbor resolves and uploads directories, not direct `SKILL.md` paths;
- Harbor records its own directory-level skill digest in the job lock;
- Terminus-2 discovers `<skills_dir>/*/SKILL.md` and requires YAML frontmatter
  containing `name` and `description` before advertising a skill;
- the generated skill body is not concatenated into SkillOpt's system prompt.

The exact packaging metadata, artifact digest, integrity policy, and remaining
runtime checks are recorded in `docs/terminalbench/SKILL_PACKAGING.md`.

### Phase 3 Harbor 0.20.0 runtime-config audit

Static inspection and public CLI parsing against the pinned local Harbor
`0.20.0` distribution confirmed:

- the documented launch boundary is `harbor run --config <yaml-or-json>`, an
  alias for `harbor job start`;
- `harbor run --config <path> --print-config` validates with `JobConfig` and
  returns before `Job.create()`, so it starts no job or Docker environment;
- the real output identity fields are `job_name` and `jobs_dir`, producing
  `<jobs_dir>/<job_name>/`; there is no `result_name` or generic `output_dir`
  field in `JobConfig`;
- repeat and concurrency are `n_attempts` and `n_concurrent_trials`;
- dataset task selection is `datasets[].task_names`, a list of `fnmatch`
  patterns applied to Harbor task names; Terminal-Bench exact IDs therefore
  require clearing `exclude_task_names` and `n_tasks` in the resolved overlay;
- the Docker runtime is `environment.type: docker`, or Harbor's same Docker
  default when both environment type and import path are absent;
- a future job writes job-level `config.json`, `lock.json`, `result.json`, and
  `job.log`, plus per-trial directories containing singular `result.json`;
- Terminus-2 declares ATIF support and writes its primary trajectory under the
  trial's `agent/trajectory.json`.

The migration therefore uses a validated base config plus a minimal overlay,
and selects Harbor's public CLI rather than importing its async/internal job
API into SkillOpt. Phase 3 invokes only `--version` and `--print-config`; real
execution remains disabled. Full details, parity rules, and artifact layout are
recorded in `docs/terminalbench/HARBOR_RUNTIME.md`.

### Phase 4 Harbor 0.20.0 result audit

Wheel-matching Harbor source inspection confirmed that each trial's verifier
reward is `verifier_result.rewards["reward"]`. `TrialResult` has no explicit
status field; completion and failure classification use `finished_at`, verifier
timing/result, and `exception_info`. Job-level `result.json` contains aggregate
stats but is written without its modeled `trial_results` list, so it is not the
source of individual rollout rewards.

Harbor's Terminal-Bench mapper writes binary `0` or `1` to `reward.txt` from
the released test command's exit code. `AgentTimeoutError` and agent nonzero
exit are recorded while still allowing verifier execution; all other recorded
exceptions fail closed in Phase 4. The exact persisted schema, reward mapping,
and parser boundary are recorded in `docs/terminalbench/RESULT_CONTRACT.md`.

## Registration and CLI contract

There is no global benchmark registry in `skillopt/envs/__init__.py`.

Both entry points own a lazy `_ENV_REGISTRY` and a `_register_builtins()`
function:

- `scripts/train.py`
- `scripts/eval_only.py`

A later phase must add the same guarded `terminalbench` import to both.
Adapter construction inspects the adapter class's `__init__` signature and
passes only flattened config keys whose names appear explicitly in that
signature. A catch-all `**kwargs` does not cause arbitrary environment keys to
be forwarded, because the factory only sees the parameter name `kwargs`.
Terminal-Bench runtime options must therefore be explicit constructor
parameters, or be grouped under an explicitly named accepted parameter.

Actual commands are:

```bash
python scripts/train.py --config <config.yaml>

python scripts/eval_only.py \
  --config <config.yaml> \
  --skill <skill.md> \
  --split <train|valid_seen|valid_unseen|all>
```

For `--split all`, `eval_only.py` concatenates train, val (`valid_seen`), and
test (`valid_unseen`) items. For a specific split, it uses `test_env_num` as
the item limit.

## Config contract

The config loader supports structured and legacy flat YAML. Structured
sections include `model`, `train`, `gradient`, `optimizer`, `evaluation`, and
`env`.

Important current behavior:

- `_base_` is one string path resolved relative to the child config. List-form
  inheritance is not supported.
- `env.name` flattens to the trainer key `env`.
- `env.skill_init` flattens to `skill_init`.
- `env.out_root` flattens to `out_root`.
- Other `env.*` keys pass through using their leaf names, making them
  available to adapter construction when the names match explicit constructor
  parameters.
- `--cfg-options env.<key>=<value>` is supported.

The migration plan's prose label `benchmark = terminalbench` is experimental
terminology, not a current config key. The actual registration/config selector
is:

```yaml
env:
  name: terminalbench
```

Target execution settings that belong to Harbor/Terminus-2 should remain
environment-specific config values. They must not cause the adapter to call
SkillOpt's configured target chat/exec backend directly. The optimizer model
continues to use SkillOpt's normal model/config route.

## Differences from migration-plan assumptions

1. **`build_env_from_batch` is practically required.** The plan lists the four
   abstract methods, but the current dataloader-backed trainer consumes planned
   batches through this non-abstract hook.
2. **Rollouts remain dictionaries.** The current trainer does not call
   `RolloutResult.from_dict()` automatically, despite wording in
   `docs/reference/api.md`. Extra fields are not automatically moved into
   `RolloutResult.extras`.
3. **The trajectory schema is broader than role/content.** Native tool-call
   and environment-step records are already accepted, so ATIF should preserve
   structure instead of flattening everything preemptively.
4. **Terminal-Bench `hard` must remain binary.** Positive partial verifier
   reward is retained in `soft`, while `hard` is `1.0` only for exact complete
   reward `1.0`, avoiding false reflection success.
5. **Infrastructure validity cannot be metadata only.** The current trainer
   will not filter broken trials, so the result parser raises before returning
   a scored result when verifier validity is not trustworthy.
6. **Blank baseline requires an empty file and adapter translation.** The
   trainer baseline is `S_0`; it does not natively know Harbor `skills=[]`.
7. **`eval_only.py` requires `--skill`.** The migration plan's abbreviated
   `python scripts/eval_only.py --config ...` command is incomplete for the
   pinned CLI.
8. **The actual selector is `env.name`, not a `benchmark` key.** It flattens to
   `cfg["env"]` for registry lookup.
9. **Exec-style examples are structural references only.** Copying SearchQA's
   direct `run_target_exec()` route, workspace skill installation, or prompt
   instruction would bypass the required Harbor/Terminus-2 path and violate
   the migration constraints.
10. **Harbor result/ATIF paths must follow current code, not stale labels.**
    Phase 3 static audit established `<jobs_dir>/<job_name>/`, singular
    `result.json` at both job and trial level, and Terminus-2's primary
    `agent/trajectory.json`. Phase 4 established the static result schema; the
    artifacts' actual presence for the real Terminal-Bench v2.1 baseline
    remains a later single-job integration check.

## Phase 0 conclusion

The pinned SkillOpt core already provides the required extension points. The
planned migration can remain an environment extension with a
`SplitDataLoader`, thin adapter, Harbor-backed rollout helper, verifier result
parser, and ATIF converter. No core trainer, reflection algorithm, optimizer,
gate, Terminal-Bench task, Harbor source, or Terminus-2 source change is
justified by this audit.
