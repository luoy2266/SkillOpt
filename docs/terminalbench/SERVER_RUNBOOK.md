# Terminal-Bench v2.1 Server Handoff Runbook

This runbook deploys the pinned SkillOpt → Harbor 0.20.0 → Terminus-2 2.0.0
Terminal-Bench v2.1 lifecycle on a fresh Linux Docker server. The repository
does not install host packages, change Docker daemon settings, download the
large formal cache, or store credentials.

## First command

From the fresh SkillOpt checkout:

```bash
scripts/bootstrap_terminalbench_server.sh inspect
```

The command is read-only. It reports missing dependencies and host blockers;
it does not repair them.

## Frozen identities

The `terminalbench-v2.1-delivery` branch is the server handoff entry point.
Before each formal experiment, update it with a fast-forward-only pull and
record the exact checkout commit:

```bash
git switch terminalbench-v2.1-delivery
git pull --ff-only
git status --short
git rev-parse HEAD
```

`git status --short` must produce no output. Put the `git rev-parse HEAD`
result in `SKILLOPT_FORMAL_HEAD`; that commit SHA is the provenance lock for
this specific formal experiment. A delivery tag is not required, and the
repository owner does not need to predeclare one permanent delivery SHA.

The server Terminal-Bench checkout must be exactly:

```text
7131e4375048a0e408a8fb404b5f499d726b695b
```

The checked-in portable split is:

```text
configs/terminalbench/splits/v2.1-s42/
semantic SHA-256 = bd36fe2f37a67cd2b46149263522d833166d3a4d036c8e9af082e742ad017500
train/val/test = 9/9/71
```

Do not create a new random split for the formal server experiment.

## Required and optional dependencies

Manually required:

- Linux with Docker Engine and noninteractive Docker access for the service user.
- Python 3.12-compatible environment and the repository dependencies.
- Harbor exactly 0.20.0. Terminus-2 2.0.0 is discovered through Harbor's
  bundled runtime registry; do not install a separate Terminus distribution.
- A clean Terminal-Bench checkout at the pinned revision.
- A prepared formal cache with its reviewed `MANIFEST.tsv`.
- A DeepSeek API credential.
- A working `systemd --user` manager and `flock`.

Optional:

- `uv` for creating the repository environment and installing Harbor.
- HTTP/HTTPS proxy variables when the server cannot use direct networking.
- NVIDIA tooling for reporting GPU inventory. Terminal-Bench task resources
  remain defined by the reviewed Harbor/task contract.

Example installation flow; adapt the Python environment to the server's
standard packaging policy:

```bash
uv venv .venv --python 3.12
uv pip install --python .venv/bin/python -e .
uv tool install harbor==0.20.0
harbor --version
```

The EnvironmentFile `PATH` must include the newly installed Harbor executable.
Do not rely on a Harbor tool left in another user's home directory.

## Runtime root

Before `probe` or `preflight`, the server operator must explicitly choose all
three formal runtime inputs below. The repository intentionally does not
select defaults for them:

1. `SKILLOPT_RUNTIME_ROOT`: the absolute server experiment/data root.
2. `SKILLOPT_FORMAL_EXPERIMENT_ID`: a unique identity for this formal run.
3. `SKILLOPT_TBENCH_CONCURRENCY`: the positive-integer Harbor trial
   concurrency selected for this server.

Export the operator-selected values; the placeholders below are not runnable
values:

```bash
export SKILLOPT_RUNTIME_ROOT=<OPERATOR_SELECTED_ABSOLUTE_PATH>
export SKILLOPT_FORMAL_EXPERIMENT_ID=<OPERATOR_SELECTED_UNIQUE_ID>
export SKILLOPT_TBENCH_CONCURRENCY=<OPERATOR_SELECTED_POSITIVE_INTEGER>
```

The formal lifecycle fails closed with `OPERATOR INPUT REQUIRED` when any of
these values is missing or still contains a template placeholder. For
`SKILLOPT_TBENCH_CONCURRENCY=N` greater than `1`, the fail-closed Docker
`default-address-pools` and remaining subnet-capacity checks apply.

Derived directories are:

```text
datasets/
splits/
harbor-configs/
outputs/
skills/
cache/
logs/
locks/
```

Explicit granular variables such as `TERMINALBENCH_ROOT`,
`TERMINALBENCH_SPLIT_DIR`, `TERMINALBENCH_HARBOR_BASE_CONFIG`, and
`TERMINALBENCH_FORMAL_CACHE_ROOT` override these defaults.

## Prepare the Terminal-Bench checkout

```bash
mkdir -p "$SKILLOPT_RUNTIME_ROOT/datasets"
git clone https://github.com/laude-institute/terminal-bench.git \
  "$SKILLOPT_RUNTIME_ROOT/datasets/terminal-bench-2-1"
git -C "$SKILLOPT_RUNTIME_ROOT/datasets/terminal-bench-2-1" \
  checkout --detach 7131e4375048a0e408a8fb404b5f499d726b695b
git -C "$SKILLOPT_RUNTIME_ROOT/datasets/terminal-bench-2-1" status --short
git -C "$SKILLOPT_RUNTIME_ROOT/datasets/terminal-bench-2-1" rev-parse HEAD
```

The status output must be empty and the revision must match exactly.

## Initialize secret-free runtime files

After the repository Python environment exists:

```bash
SKILLOPT_RUNTIME_ROOT="$SKILLOPT_RUNTIME_ROOT" \
SKILLOPT_FORMAL_EXPERIMENT_ID="$SKILLOPT_FORMAL_EXPERIMENT_ID" \
SKILLOPT_TBENCH_CONCURRENCY="$SKILLOPT_TBENCH_CONCURRENCY" \
scripts/bootstrap_terminalbench_server.sh init
```

`init` creates the runtime directory skeleton, copies the portable split,
renders a secret-free Harbor config, and copies the EnvironmentFile example.
It refuses to overwrite an existing Harbor config or EnvironmentFile template.
It does not download the cache or run a benchmark.

`SKILLOPT_RUNTIME_ROOT` and `SKILLOPT_TBENCH_CONCURRENCY` are required for
`init` because they determine the generated paths and Harbor config.
`SKILLOPT_FORMAL_EXPERIMENT_ID` is not invented when absent; `init` leaves the
committed placeholder unchanged and reports that the operator must set it
before `probe` or `preflight`.

## EnvironmentFile

Copy the generated example to a protected file outside the repository:

```bash
mkdir -p "$HOME/.config/skillopt"
cp "$SKILLOPT_RUNTIME_ROOT/terminalbench-formal.env.example" \
  "$HOME/.config/skillopt/terminalbench-formal.env"
chmod 600 "$HOME/.config/skillopt/terminalbench-formal.env"
```

Set absolute values for:

```text
PATH
SKILLOPT_RUNTIME_ROOT
SKILLOPT_FORMAL_EXPERIMENT_ID
SKILLOPT_FORMAL_HEAD
SKILLOPT_FORMAL_DOCKER_MODE=sg
SKILLOPT_TBENCH_CONCURRENCY
DEEPSEEK_API_KEY
```

Set `SKILLOPT_FORMAL_HEAD` to the current clean delivery checkout's
`git rev-parse HEAD` output, not to a tag name or a previously documented
delivery commit.

Do not run the copied file with its `<REQUIRED_...>` placeholders. The formal
launcher rejects missing or unchanged operator-owned values before creating a
systemd unit.

Set both `HTTP_PROXY` and `HTTPS_PROXY`, or leave both unset. `NO_PROXY` is
optional input; the wrapper adds localhost, Docker DNS, and actual Docker
subnets/gateways and exports the lowercase bridge. Never print the completed
EnvironmentFile in logs or audit reports.

Point the launcher at a non-default EnvironmentFile with:

```bash
export SKILLOPT_FORMAL_ENV_FILE=/absolute/path/terminalbench-formal.env
```

## Formal cache

Prepare the reviewed cache under:

```text
${SKILLOPT_RUNTIME_ROOT}/cache/terminal-bench-v2.1
```

It must contain `MANIFEST.tsv` and the existing required assets. The Harbor
mount remains read-only at `/opt/skillopt-cache/terminal-bench-v2.1` with
`create_host_path=false`. Bootstrap does not download the approximately
hundreds-of-megabytes asset set.

Validate an existing cache without running Harbor:

```bash
.venv/bin/python -c '
from pathlib import Path
from scripts.preflight_terminalbench import _validate_cache_contract
import os
state = _validate_cache_contract(
    Path(os.environ["SKILLOPT_RUNTIME_ROOT"]) / "cache" / "terminal-bench-v2.1"
)
print(state["manifest_sha256"])
'
```

## High-concurrency Docker prerequisite

`SKILLOPT_TBENCH_CONCURRENCY=N` is selected by the server operator. Training,
baseline-test, and skill-test use the same frozen `N`. It never changes
`n_attempts=1`, `retry.max_retries=0`, task timeouts, models, or skills.

For `N=1`, missing Docker `default-address-pools` remains a warning. For
`N>1`, missing, invalid, unresolved, or insufficient address-pool capacity is
a fail-closed preflight blocker.

**EXAMPLE — adapt to host networking.** An administrator may choose a daemon
configuration shaped like:

```json
{
  "default-address-pools": [
    {"base": "172.30.0.0/16", "size": 24}
  ]
}
```

This is not a universal allocation and repository scripts never write it.
The administrator must avoid overlap with host, VPN, cloud, and corporate
routes, apply the site's normal Docker restart procedure, and verify:

```bash
docker info
docker network ls
docker network inspect <reviewed-network>
```

Preflight records logical CPUs, total/available RAM, GPU inventory, Docker
root free space, runtime output free space, dataset/cache free space, requested
concurrency, and Harbor resource overrides. It does not invent task-specific
quotas or automatically lower concurrency when capacity is unresolved.

## Offline clean-clone acceptance

Before using credentials or executing a benchmark:

```bash
.venv/bin/python scripts/accept_terminalbench_server_handoff.py
```

This creates a temporary runtime root and validates portable paths, split
identity, Harbor rendering, concurrency propagation, systemd identities,
manifest schema, nonblank/blank freeze fixtures, and aggregate fixtures. It
does not call DeepSeek, launch Harbor, execute Docker tasks, or run Trainer.

## Formal lifecycle

Use the committed launcher for every formal stage:

```bash
scripts/run_terminalbench_formal_systemd.sh probe
scripts/run_terminalbench_formal_systemd.sh preflight
scripts/run_terminalbench_formal_systemd.sh training
scripts/run_terminalbench_formal_systemd.sh freeze-skill
scripts/run_terminalbench_formal_systemd.sh baseline-test
scripts/run_terminalbench_formal_systemd.sh skill-test
scripts/run_terminalbench_formal_systemd.sh aggregate
```

Run them sequentially and inspect each completed stage before starting the
next. `probe`, `preflight`, `freeze-skill`, and `aggregate` wait for completion.
Training and the two evaluations remain persistent systemd user services.

Training, baseline-test, and skill-test each rerun the same fail-closed formal
preflight immediately before execution. A prior preflight-only PASS does not
weaken this defense against runtime drift.

### Freeze-skill

`freeze-skill` requires a normally completed training output. It derives the
expected Trainer step count from the frozen configuration and split size,
verifies summary/history/runtime state and every step record, confirms
`evaluation.eval_test=false`, and rejects inconsistent retained/best state.

Outputs are:

```text
${SKILLOPT_RUNTIME_ROOT}/skills/${SKILLOPT_FORMAL_EXPERIMENT_ID}/
  best_skill.md
  terminalbench-skill/SKILL.md
  skill_provenance.json
```

`best_skill.md` is a byte-for-byte copy. A blank final best skill is valid and
has no native file; its provenance records `harbor-skills-empty`. Rejected
candidates are never selected manually.

### Baseline and learned-skill tests

Baseline uses the blank initial skill and Harbor `skills=[]`. Skill-test uses
only the frozen `best_skill.md` plus `skill_provenance.json`; it verifies raw
and native SHA-256 identities before any task runs. Both conditions use the
same 71 frozen test IDs, concurrency, models, reasoning, attempts, retries,
timeouts, cache, Harbor config, tasks, and verifiers.

The test set is not used for training, selection, or freeze decisions.

### Aggregate

`aggregate` is read-only over the two evaluation outputs, manifests, and
frozen provenance. It blocks on any parity mismatch, missing/duplicate task,
invalid verifier reward, unclassified trial status, or infrastructure-invalid
trial. It does not reinterpret infrastructure failure as reward zero.

Outputs are:

```text
${SKILLOPT_RUNTIME_ROOT}/outputs/formal/${SKILLOPT_FORMAL_EXPERIMENT_ID}/aggregate/
  summary.json
  results.tsv
```

The summary contains paired task rewards, baseline/skill raw scores, absolute
and defined relative delta, wins/ties/losses, timeout/nonzero-exit counts, and
the full frozen experiment identity. No significance test is added.

## Monitoring

Get the deterministic unit name without guessing:

```bash
UNIT=$(.venv/bin/python scripts/terminalbench_formal_identity.py unit-name \
  --experiment-id "$SKILLOPT_FORMAL_EXPERIMENT_ID" --stage training)
systemctl --user status "$UNIT"
journalctl --user -fu "$UNIT"
```

The formal console log defaults below the runtime `logs/formal/<experiment>/`
tree. Harbor job directories are below the generated Harbor `jobs_dir`.
Read logs and artifacts only; do not edit them while a stage is active.

## Scheduling and locks

One experiment-scoped `flock` protects training, freeze-skill, baseline-test,
skill-test, and aggregate. Baseline and skill evaluation therefore cannot
accidentally run together and consume `2N` capacity. Different experiment IDs
have independent locks; the operator remains responsible for total server
capacity across experiments.

The recommended order is always:

```text
training → freeze-skill → baseline-test → skill-test → aggregate
```

## Failure and rerun policy

Do not manually rerun a task or Trainer step because of:

- reward `0`;
- `AgentTimeoutError`;
- `NonZeroAgentExitCodeError` with a trustworthy verifier result;
- rejected candidate;
- no-patch update;
- optimizer `finish_reason=length`.

These are formal outcomes under attempts `1` and retries `0`. An
infrastructure-invalid trial must stop the stage and be reported; it must not
be converted to reward zero.

If an infrastructure failure requires a whole experiment restart, preserve all
old artifacts and choose a new `SKILLOPT_FORMAL_EXPERIMENT_ID`. Never delete an
old formal output and rerun in place.
