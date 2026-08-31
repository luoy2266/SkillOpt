# Terminal-Bench v2.1 Formal SkillOpt Experiment

This runbook freezes the post-Phase-9 formal experiment contract. It does not
authorize automatic retries, task replacement, test-set tuning, or adaptive
changes to the optimizer completion limit.

## Frozen identities

- SkillOpt branch: `terminalbench-v2.1`.
- Terminal-Bench checkout:
  `/home/yunl/projects/skillopt-runtime/datasets/terminal-bench-2-1`.
- Terminal-Bench revision: `7131e4375048a0e408a8fb404b5f499d726b695b`.
- Split: `/home/yunl/projects/skillopt-runtime/splits/tbench-v2.1-1-1-8`.
- Split manifest SHA-256:
  `8fa19aa350b90a7c39c3cde56f87a93bbfcb450586b416dc700c4c0b35827894`.
- Harbor `0.20.0`, Terminus-2 `2.0.0`, Docker execution.

The final SkillOpt HEAD is pinned only after this Phase 10A change set is
reviewed and committed. Pass that exact commit to
`scripts/preflight_terminalbench.py --expected-skillopt-head`.

## Model contracts

Target traffic remains:

```text
SkillOpt -> TerminalBenchAdapter -> Harbor -> Terminus-2 -> LiteLLM
request model: deepseek/deepseek-v4-flash
reasoning_effort: max
```

Optimizer traffic remains:

```text
run_minibatch_reflect() -> chat_optimizer() -> openai_compatible
request model: deepseek-v4-flash
reasoning_effort: max
endpoint: https://api.deepseek.com
configured completion cap: 16384
```

Both roles have underlying experiment identity `DeepSeek-V4-Flash-0731`, but
their request model identifiers are intentionally different.

The reflection completion contract is fixed at:

```text
caller requested = 16384
backend configured cap = 16384
effective max = min(caller requested, configured cap)
```

`finish_reason=length` is recorded as an experimental event. It does not
authorize changing the cap or rerunning the step.

## Formal training configuration

`configs/terminalbench/formal.yaml` explicitly freezes the pinned SkillOpt
defaults instead of inheriting them silently:

| Field | Value | Source |
|---|---:|---|
| `train.num_epochs` | 4 | ORIGINAL SKILLOPT |
| `train.train_size` | 0 (resolve to 9) | TERMINALBENCH MIGRATION OVERRIDE |
| `train.batch_size` | 40 | ORIGINAL SKILLOPT |
| `train.accumulation` | 1 | ORIGINAL SKILLOPT |
| `train.seed` | 42 | ORIGINAL SKILLOPT |
| `gradient.minibatch_size` | 8 | ORIGINAL SKILLOPT |
| `gradient.merge_batch_size` | 8 | ORIGINAL SKILLOPT |
| `gradient.analyst_workers` | 16 | ORIGINAL SKILLOPT |
| `gradient.failure_only` | false | ORIGINAL SKILLOPT |
| `optimizer.learning_rate` / edit budget | 4 | ORIGINAL SKILLOPT |
| `evaluation.sel_env_num` | 0 (full 9-task val) | ORIGINAL SKILLOPT |
| `optimizer.use_slow_update` | true | ORIGINAL SKILLOPT |
| `optimizer.use_meta_skill` | true | ORIGINAL SKILLOPT |
| `env.limit` | 0 | TERMINALBENCH MIGRATION OVERRIDE |
| `env.n_concurrent_trials` | 1 | VALIDATED HARBOR BASE POLICY |
| `evaluation.eval_test` | false | FORMAL ORCHESTRATION OVERRIDE |

The checked-in YAML remains the conservative local profile. Server delivery
sets one positive `SKILLOPT_TBENCH_CONCURRENCY` value and applies the same
override to training, baseline-test, and skill-test. Preflight requires the
requested value, resolved SkillOpt config, and rendered Harbor config to match;
attempts remain `1` and Harbor retries remain `0`.

The separate test commands prevent Trainer from mixing held-out evaluation
artifacts into the training namespace. Formal learned-skill evaluation uses
only `training/best_skill.md`; the final force-injected slow-update state is not
manually promoted or selected from test results.

## Dataset semantics

```text
train         -> 9 tasks
valid_seen    -> val  -> 9 tasks
valid_unseen  -> test -> 71 tasks
```

The formal config keeps `env.limit=0`. The materialized task IDs are immutable;
timeouts, difficulty, reward, or network behavior never trigger task replacement.

## Expected cardinality

With `train_size=9`, `batch_size=40`, `accumulation=1`, and four epochs:

```text
steps_per_epoch = ceil(9 / (40 * 1)) = 1
total_steps = 4
```

Training Harbor jobs/trials:

| Source | Jobs | Trials |
|---|---:|---:|
| Initial val baseline | 1 | 9 |
| Four train rollouts | 4 | 36 |
| Slow update, epochs 2-4, previous/current | 6 | 54 |
| Candidate selection | 0-4 | 0-36 |
| Training total | 11-15 | 99-135 |

Separate held-out evaluation adds two jobs and 142 trials: one 71-task blank
baseline and one 71-task frozen learned-skill condition. Full experiment total
is therefore 13-17 Harbor jobs and 241-277 trials.

For every valid nine-trajectory training step, `failure_only=false` and
`minibatch_size=8` produce two analyst groups in total. Across four steps:

- failure analyst calls: 0-8;
- success analyst calls: 0-8;
- failure + success analyst calls: 8;
- merge calls: 0-4, normally 4 when both analyst patches are usable;
- ranking calls: 0-4, only when merged edits exceed the edit budget;
- slow-update calls: 3;
- meta-skill calls: 3;
- total optimizer logical calls: 14-22, normally 18 without ranking calls.

Each optimizer logical call currently allows up to three application attempts.
SDK-internal transport attempts remain `unknown` unless the SDK exposes them.

## Fair baseline and learned-skill test

Both held-out conditions use the same 71 task IDs, Harbor base config,
Terminus-2, Docker environment, target model, reasoning effort, timeouts,
verifier, `n_attempts=1`, `retry.max_retries=0`, and concurrency. The only
semantic difference is:

```text
baseline: agents[0].skills=[]
skill:    agents[0].skills=[deterministically packaged best_skill.md]
```

No skill text is concatenated into the agent system prompt.

## Skill freeze

- `candidate_skill.md` is an attempted step candidate and is never selected by
  hand.
- `skill_vNNNN.md` is the retained current skill after that step's gate.
- `best_skill.md` is the validation-best retained state and is the only formal
  learned skill used on test.

Before skill-test, the preflight records the exact `best_skill.md` bytes,
SHA-256, deterministic native `SKILL.md` SHA-256, `best_step`, `best_score`, and
`best_origin` from `runtime_state.json`.

## Output namespace

Use one immutable experiment ID:

```text
/home/yunl/projects/skillopt-runtime/outputs/formal/
  tbench-v2.1-dsv4flash-s42-formal-001/
    training/
    baseline-test/
    skill-test/
    manifests/
    logs/
```

Every condition has a distinct output root, log, and manifest. An existing
condition output root or manifest is a hard refusal; never delete it to rerun.

## Formal public-cache contract

The formal cache is a curated, immutable input, not a generic claim that host
tool caches happen to exist:

```text
host root:
/home/yunl/projects/skillopt-runtime/cache/terminal-bench-v2.1

container root:
/opt/skillopt-cache/terminal-bench-v2.1

integrity manifest:
MANIFEST.tsv
```

`MANIFEST.tsv` uses the checked example at
`configs/terminalbench/formal_cache_manifest.example.tsv`. Every
`required-cache` entry points to a readable file or directory below
`huggingface/` and carries the canonical file/tree SHA-256. The tree digest
includes relative paths, file content digests, and internal symlink targets plus
their resolved file content. This supports the Hugging Face hub cache layout
while rejecting broken links, directory links, and links escaping the declared
asset directory. The current required cache entries are:

- DistilBERT SST-2 model/tokenizer for `hf-model-inference`;
- Qwen2.5-1.5B-Instruct tokenizer files for `count-dataset-tokens`;
- OpenThoughts-1k-sample hub snapshot and prepared dataset cache;
- BAAI/bge-small-zh-v1.5 at revision
  `7999e1d3359715c523056ef9478215996d62a620` for `mteb-retrieve`.

The following remain explicitly `runtime-network-only` because the tasks use
direct task-time download semantics that cannot consume a transparent immutable
cache without changing the task: CIFAR-10/Caffe, POV-Ray 2.2 archives, QEMU
5.2.0 source, and the pinned MTEB results Git repository.

Alpine QEMU ISO, OEWN SQLite, the C4 shard, and Yelp parquet files are
`image-build-only`: they are baked into pinned task images. Pre-pulling those
images is recommended preparation, but is not a formal requirement that all 89
images be local before launch.

Harbor 0.20.0 must expose the cache to the task container with this exact
operational overlay in the external base config:

```yaml
environment:
  mounts:
    - type: bind
      source: /home/yunl/projects/skillopt-runtime/cache/terminal-bench-v2.1
      target: /opt/skillopt-cache/terminal-bench-v2.1
      read_only: true
      bind:
        create_host_path: false
  env:
    SKILLOPT_TERMINALBENCH_CACHE_ROOT: /opt/skillopt-cache/terminal-bench-v2.1
    HF_HOME: /opt/skillopt-cache/terminal-bench-v2.1/huggingface
    HF_HUB_CACHE: /opt/skillopt-cache/terminal-bench-v2.1/huggingface/hub
    HF_DATASETS_CACHE: /opt/skillopt-cache/terminal-bench-v2.1/huggingface/datasets
```

The preflight validates host readability, manifest completeness, SHA-256,
canonical mount source, container target, `read_only: true`, and
`create_host_path: false`. A missing cache root, required asset, manifest, hash,
or Harbor mount is a hard blocker.

## Runtime environment

Store secret and proxy values outside the repository, for example in:

```text
%h/.config/skillopt/terminalbench-formal.env
```

The EnvironmentFile must define `SKILLOPT_FORMAL_HEAD`, `DEEPSEEK_API_KEY`,
`HTTP_PROXY`, `HTTPS_PROXY`, and `NO_PROXY`. It does not rely on shell-style
variable expansion: systemd EnvironmentFile values are passed literally. The
wrapper creates the optimizer credential and lowercase proxy variables inside
the stage process:

```bash
export OPTIMIZER_OPENAI_COMPATIBLE_BASE_URL=https://api.deepseek.com
export OPTIMIZER_OPENAI_COMPATIBLE_API_KEY="$DEEPSEEK_API_KEY"
export http_proxy="$HTTP_PROXY"
export https_proxy="$HTTPS_PROXY"
DOCKER_NETWORK_IDS="$(docker network ls -q)"
# The wrapper reads every actual Docker IPAM subnet and gateway.
export NO_PROXY="${NO_PROXY:+$NO_PROXY,}localhost,127.0.0.1,127.0.0.11,::1,<actual Docker IPAM entries>"
export no_proxy="$NO_PROXY"
```

The validated formal Harbor base config must pass these six variables to task
containers through `environment.env` using `${VAR}` references, never literal
values.

## Persistent execution

Do not run formal work in an SSH-attached shell. Recommended user unit:

```ini
[Unit]
Description=SkillOpt Terminal-Bench formal training
After=network-online.target docker.service
Wants=network-online.target

[Service]
Type=exec
WorkingDirectory=/home/yunl/projects/SkillOpt
EnvironmentFile=%h/.config/skillopt/terminalbench-formal.env
ExecStart=/usr/bin/sg docker -c 'SKILLOPT_FORMAL_DOCKER_MODE=sg exec /home/yunl/projects/SkillOpt/scripts/run_terminalbench_formal_stage.sh training'
Restart=no
StandardOutput=journal
StandardError=journal

[Install]
WantedBy=default.target
```

The operator must have an active user manager and non-interactive Docker
access. Establish group membership first:

```bash
sudo usermod -aG docker "$USER"
getent group docker
```

Do not assume an existing user manager receives the new supplementary group.
The frozen service path always enters the group explicitly:

```text
systemd --user → sg docker -c → formal wrapper → preflight-only or preflight → stage
```

This is valid only after the user is a Docker-group member, so `sg docker` does
not request a group password or TTY. The service probe below verifies this path
non-interactively.

Before any formal stage, run the one-shot probe:

```bash
/home/yunl/projects/SkillOpt/scripts/probe_terminalbench_formal_systemd.sh
```

It uses the same EnvironmentFile, systemd user manager, `sg docker`, wrapper,
systemd invocation identity, credential bridge, proxy derivation, Docker access,
committed HEAD, and cache contract as the real stages. It prints only
`SET`/`MISSING` and `PASS`/`FAIL`, starts no Trainer/Harbor/model work, waits for
completion, and uses `--collect` so the transient unit does not remain installed.

After installing the environment file and setting the committed HEAD, launch
the preflight-only stage first:

```bash
systemd-run --user --unit=skillopt-tbench-formal-preflight \
  --no-ask-password \
  --property=Type=exec \
  --property=EnvironmentFile=/home/yunl/.config/skillopt/terminalbench-formal.env \
  --collect \
  /usr/bin/sg docker -c \
  'SKILLOPT_FORMAL_DOCKER_MODE=sg exec /home/yunl/projects/SkillOpt/scripts/run_terminalbench_formal_stage.sh preflight'
```

This runs the same fail-closed `scripts/preflight_terminalbench.py` invocation
and wrapper bootstrap as the execution stages, writes the preflight manifest and
log, and exits without starting Trainer, evaluation, or Harbor benchmark work.
It validates the prospective training condition with the blank initial skill,
so the unchanged manifest schema records `condition: training`; the dedicated
preflight artifact paths below identify it as the preflight-only invocation.
For a formal root `${FORMAL_ROOT}`, its reserved identities are:

```text
output identity: ${FORMAL_ROOT}/preflight
manifest:        ${FORMAL_ROOT}/manifests/preflight.experiment_manifest.json
log:             ${FORMAL_ROOT}/logs/preflight.console.log
```

These paths do not occupy `${FORMAL_ROOT}/training`, `baseline-test`, or
`skill-test`. Reusing the same preflight identity fails closed through the
existing fresh-manifest/output checks.

Inspect the successful preflight manifest before launching exactly one
execution stage per transient user unit:

```bash
systemd-run --user --unit=skillopt-tbench-formal-training \
  --no-ask-password \
  --property=Type=exec \
  --property=EnvironmentFile=/home/yunl/.config/skillopt/terminalbench-formal.env \
  --collect \
  /usr/bin/sg docker -c \
  'SKILLOPT_FORMAL_DOCKER_MODE=sg exec /home/yunl/projects/SkillOpt/scripts/run_terminalbench_formal_stage.sh training'

systemd-run --user --unit=skillopt-tbench-formal-baseline-test \
  --no-ask-password \
  --property=Type=exec \
  --property=EnvironmentFile=/home/yunl/.config/skillopt/terminalbench-formal.env \
  --collect \
  /usr/bin/sg docker -c \
  'SKILLOPT_FORMAL_DOCKER_MODE=sg exec /home/yunl/projects/SkillOpt/scripts/run_terminalbench_formal_stage.sh baseline-test'

systemd-run --user --unit=skillopt-tbench-formal-skill-test \
  --no-ask-password \
  --property=Type=exec \
  --property=EnvironmentFile=/home/yunl/.config/skillopt/terminalbench-formal.env \
  --collect \
  /usr/bin/sg docker -c \
  'SKILLOPT_FORMAL_DOCKER_MODE=sg exec /home/yunl/projects/SkillOpt/scripts/run_terminalbench_formal_stage.sh skill-test'
```

Run them in order: preflight-only, inspect its PASS manifest, training, freeze
the retained best skill and hashes, baseline-test, then skill-test. Training,
baseline-test, and skill-test each rerun the same fail-closed preflight in their
own current service context before executing. This defense-in-depth catches
HEAD, configuration, credential, Docker, cache, output, or other runtime drift
between the earlier audit and the actual stage; it is not a repeated experiment
or benchmark retry. Each stage uses one unique condition output and the wrapper
contains no retry or cleanup path.

Baseline-test and skill-test remain two sequential Harbor jobs. The adapter has
no natural cross-process global queue to share without adding new orchestration.
Sequential execution with `n_concurrent_trials=1` preserves the same 71 test
IDs, Harbor config, attempts, retries, and concurrency; only the native skill
directory and result/provenance identity differ.

## Diagnostics and accounting

Reflection diagnostics remain separate from Harbor target request logs:

```text
steps/step_NNNN/optimizer_diagnostics/
harbor_runtime/jobs/.../agent/request_events.jsonl
```

The final aggregation records target and optimizer prompt/completion tokens,
optimizer logical calls/application attempts, finish-reason and parse-status
counts, Harbor jobs/trials/retries, agent timeouts, provider failures, verifier
failures, and infrastructure-invalid trials. Failure categories remain
separate; reward-zero cases are never automatically rerun.

Each condition manifest embeds the exact source SkillOpt YAML, the exact
secret-free Harbor base YAML, the parsed split manifest, the resolved flattened
configuration, CLI overrides, all 89 task agent/verifier/build timeouts, cache
manifest/hash/asset classifications, and Docker network/address-pool state.
Hashes and paths are retained alongside these snapshots.

## Static preflight

`scripts/preflight_terminalbench.py` refuses dirty or wrong revisions, split
drift, config/model drift, missing credentials/proxy entries, Docker failure,
insufficient declared task storage, reused outputs/manifests, nonblank baseline
skills, and learned skills not sourced from `training/best_skill.md`. On success
it writes a secret-free manifest conforming to
`configs/terminalbench/experiment_manifest.schema.json`.

The preflight does not launch Trainer, Harbor jobs, Docker tasks, or models.

## Known pre-launch gaps from Phase 10A

- The final SkillOpt commit is not available until this change set is reviewed
  and committed.
- The Docker group database lists `yunl`, but the current session does not have
  the supplementary group, non-interactive `sg docker` fails, and the user
  systemd manager is unavailable.
- The formal EnvironmentFile is not installed. The current shell lacks
  lowercase `no_proxy`, and its no-proxy list does not include all effective
  Docker/local entries. The wrapper fixes these values, but the service probe
  must prove the resulting environment.
- The formal cache root, integrity manifest, required HF assets, and read-only
  Harbor mount are not established. This is a hard blocker.
- `/etc/docker/daemon.json` has no explicit `default-address-pools`; concurrency
  is frozen at one, so this is a capacity warning rather than an automatic
  blocker.
- Only 2 of 89 pinned task images are local. The other 87 must be staged or
  pulled during a separately approved preparation phase.
- A static scan of task build/runtime files found public-download markers in 83
  of 89 tasks: 67 use apt install paths, 33 pip installs, 14 git clones, 11
  Hugging Face-related paths, and 41 literal HTTP(S) references. This makes
  proxy and upstream availability a material preparation risk, but does not
  authorize task replacement.
- Host Hugging Face, pip, and uv caches exist, but they are not the formal cache
  and are not exposed to task containers. They do not satisfy readiness.

Do not start the formal experiment until the hard blockers are cleared and a
fresh preflight writes all three condition manifests.
