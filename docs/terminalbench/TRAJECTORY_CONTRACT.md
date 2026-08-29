# Harbor / Terminus-2 ATIF to SkillOpt Trajectory Contract

Phase 5 audit and conversion contract for the pinned migration target:

- Harbor `0.20.0` installed as the external uv tool;
- Terminus-2 `2.0.0` bundled with that Harbor distribution;
- ATIF `v1.7` emitted at `<trial>/agent/trajectory.json`;
- SkillOpt pinned at the commit recorded in `UPSTREAM.md`.

This phase reads existing artifacts only. It does not find trials, parse
rewards, start Harbor or Docker, call a model, or perform reflection.

## Audited sources

The Harbor side was audited from the local pinned installation:

- `harbor/models/trajectories/trajectory.py`
- `harbor/models/trajectories/step.py`
- `harbor/models/trajectories/content.py`
- `harbor/models/trajectories/tool_call.py`
- `harbor/models/trajectories/observation.py`
- `harbor/models/trajectories/observation_result.py`
- `harbor/models/trajectories/metrics.py`
- `harbor/models/trajectories/final_metrics.py`
- `harbor/models/agent/trajectory_config.py`
- `harbor/agents/terminus_2/terminus_2.py`

The installed ATIF model files match the Harbor `0.20.0` wheel record. The
active Terminus-2 file has owner-local request-event logging instrumentation;
comparison with its wheel-matching backup confirms that this instrumentation
does not change ATIF construction or writing.

The SkillOpt side was audited from:

- `skillopt/gradient/reflect.py::fmt_trajectory`
- `skillopt/gradient/reflect.py::fmt_minibatch_trajectories`

Production conversion code deliberately does not `import harbor`. Harbor
remains an external uv tool, and the converter validates the pinned persisted
JSON contract with the Python standard library.

## Harbor / Terminus-2 persisted schema

Terminus-2 declares `SUPPORTS_ATIF = True` and writes the primary artifact to:

```text
<trial>/agent/trajectory.json
```

The top-level object is a Harbor `Trajectory` serialized with null fields
omitted. Terminus-2 uses the default `schema_version: "ATIF-v1.7"`:

```text
schema_version                 required by this converter; ATIF-v1.7
session_id                     optional string; agent-run identity
trajectory_id                  optional string; absent from normal Terminus-2 roots
agent                          required Agent object
steps                          required non-empty Step list
notes                          optional string
final_metrics                  optional aggregate token/cost metrics
continued_trajectory_ref       optional relative continuation filename
extra                          optional producer metadata
subagent_trajectories          optional embedded trajectories
```

`agent` contains required `name` and `version`, plus optional `model_name`,
OpenAI-style `tool_definitions`, and `extra`. The current target emits
`name: "terminus-2"`, `version: "2.0.0"`, and parser/config bookkeeping in
`extra`.

`steps` must be sequential from `step_id == 1`. Each step has:

```text
step_id                        required positive integer
source                         required: system | user | agent
message                        required string or ContentPart list
timestamp                      optional ISO-8601 string
model_name                     optional; agent steps only
reasoning_effort               optional; agent steps only
reasoning_content              optional string; agent steps only
tool_calls                     optional ToolCall list; agent steps only
observation                    optional Observation object
metrics                        optional token/cost/logprob data; agent steps only
is_copied_context              optional bool
llm_call_count                 optional non-negative integer
extra                          optional step metadata
```

A `ToolCall` requires `tool_call_id`, `function_name`, and an `arguments`
mapping. An `Observation` requires a `results` list. Each result may contain a
`source_call_id`, text or multimodal `content`, subagent trajectory references,
and `extra`. A non-null `source_call_id` must name a tool call in the same step.

ATIF permits `ContentPart` lists containing text or image references. The
current Terminus-2 writer supplies strings for its prompts, responses, and
terminal observations. The converter nevertheless preserves valid text parts
and minimally serializes an image reference because SkillOpt's reflection
records are text-only; the original ATIF remains the provenance artifact.

## Actual Terminus-2 step production

The first step is a `source: "user"` message containing Terminus-2's rendered
initial prompt. It includes the task instruction, initial terminal state, and,
when present, native MCP/skill context. It is the trustworthy task context for
reflection, but it does not provide a canonical Terminal-Bench task id.

Each normal agent episode appends one `source: "agent"` step:

- `message` contains parsed `Analysis:` and `Plan:` text, or the raw model
  response when `trajectory_config.raw_content` is enabled;
- `reasoning_content` contains backend-provided reasoning when available;
- parsed terminal commands become synthetic `bash_command` tool calls with
  `keystrokes` and `duration` arguments;
- task completion becomes a `mark_task_complete` tool call;
- terminal/environment feedback is stored in `observation.results[].content`;
- timestamp and per-call token/cost metrics are recorded when available.

For a single terminal command, the observation is linked by
`source_call_id`. For a multi-command batch, Terminus-2 intentionally writes
one shared terminal observation with no `source_call_id`; a consumer must not
invent separate outputs for each command.

Parser-error episodes store the raw response in `message` and the repair
prompt in an unlinked observation. In `raw_content` mode, tool calls are not
constructed and the observation remains unlinked.

There is no dedicated top-level final-response field. The final agent step,
its text/reasoning, any `mark_task_complete` call, and its observation are the
final execution record.

## Exception, timeout, and incomplete artifacts

ATIF `Trajectory` and `Step` have no trial status or exception field.
Harbor records an `AgentTimeoutError` or other trial exception in the trial's
`result.json`, not in `agent/trajectory.json`. Terminus-2 writes the current
trajectory in `run()`'s `finally` block, so a timed-out or failed agent may
leave a valid partial ATIF containing all completed execution steps.

A terminal command send timeout inside Terminus-2 is represented as ordinary
observation text rendered from its timeout template; it is not a special ATIF
exception type.

Consequently the converter does not classify reward, infrastructure validity,
or timeout type. Phase 4/6 must keep infrastructure-invalid trials out. A valid
partial trajectory with user context and at least one agent execution record
is converted, including an AgentTimeout outcome. Missing, malformed, empty, or
user-only ATIF fails loudly instead of producing a fake conversation.

## Continuations and subagents

`trajectory_config.linear_history` defaults to `false`. With the default,
context summarization does not split the main trajectory. With
`linear_history: true`, Terminus-2 writes:

```text
trajectory.json
trajectory.cont-1.json
trajectory.cont-2.json
...
```

Each segment points to the next through `continued_trajectory_ref`.
Continuation files restart `step_id` at one and may contain
`is_copied_context: true` records copied from the prior history. The converter
follows only this explicit, same-directory filename chain and skips copied
records after the first segment. It does not scan the trial or job directory.

Summarization subagents are written separately as
`trajectory.summarization-<n>-<kind>.json` and referenced from observations.
They are context-management provenance, not additional main-agent execution
steps, so Phase 5 does not recursively inline them. Relevant main-trajectory
system/handoff messages remain preserved.

## Task identity trust boundary

The emitted ATIF does not contain `task_name`, Harbor `task_id`, or another
canonical Terminal-Bench identifier. Terminus-2's internal trajectory
`session_id` is generated for its model run and must not be treated as a task
id. The task instruction embedded in the initial prompt is text, not a stable
identity field.

Therefore `expected_task_id` is validated as the output artifact id but is not
compared with an invented ATIF field. Phase 6 must guarantee that:

```text
trial result + agent/trajectory.json + expected task id
```

come from the same explicit trial directory. Phase 4 independently validates
the task id in `result.json`.

## SkillOpt reflection input

`fmt_minibatch_trajectories()` reads:

```text
<prediction_dir>/<id>/conversation.json
```

It skips a missing file or a falsey decoded value. It otherwise passes the
decoded object to `fmt_trajectory()` without schema validation. Malformed JSON
propagates an error; malformed list entries are stringified or rendered with
empty/default fields rather than rejected.

The useful native record shapes are:

```json
{"role": "assistant", "content": "text"}
```

```json
{"type": "tool_call", "cmd": "command", "obs": "output"}
```

```json
{
  "step": 2,
  "reasoning": "why",
  "action": "bash_command(...) ",
  "env_feedback": "terminal output"
}
```

`fmt_trajectory()` preserves full strings and performs no truncation. It
explicitly accepts `role: "system"`, rendering it with a `[verification]`
display label because upstream primarily uses that role for post-execution
enrichment. Phase 5 nevertheless preserves a real ATIF `source: "system"` as
`role: "system"`; changing the role to assistant would disguise provenance.
The display label remains pinned upstream behavior and is not changed here.

Phase 5 validates more strictly than the upstream reader: the top level must
be a non-empty list, every record must be one of the supported dict shapes,
all text fields must be strings, and the result must contain both user/task
context and agent execution information.

Converter unit tests use only the repository's normal lightweight runtime.
The separate compatibility test imports the real upstream reflection formatter
inside the test. If that import cannot complete because an optional model
provider dependency such as `openai` is absent, unittest reports the
compatibility check as skipped rather than misreporting a converter failure.
Such a skip is not evidence that real formatter compatibility was verified.

## Mapping

| ATIF source | SkillOpt conversation record | Notes |
| --- | --- | --- |
| user `message` | `{"role":"user","content":...}` | Preserves initial task prompt and later handoffs. |
| agent `message` | `{"role":"assistant","content":...}` | Preserves parsed analysis/plan or raw response. |
| agent reasoning + tool calls + observation | `step/reasoning/action/env_feedback` | One structured record per ATIF agent episode. |
| multiple tool calls | deterministic ordered calls in `action` | Kept in one step because Terminus may provide one shared observation. |
| terminal/tool output | `env_feedback` | Preserves content without truncation. |
| system `message` | `{"role":"system","content":...}` | Preserves ATIF provenance; pinned reflection displays it as verification. |
| timestamps and metrics | omitted | Reflection does not read them; original ATIF retains them. |
| subagent refs/bookkeeping | omitted | Main handoff/system text remains; provenance stays in ATIF. |

The structured step is used instead of flattening an action and observation
into a generic chat string. Tool arguments are serialized deterministically,
including command `keystrokes` and `duration`. Reasoning is copied only when
`reasoning_content` is actually present; it is never reconstructed.

## Output and integrity

The converter accepts one explicit primary ATIF path, one
`expected_task_id`, and one explicit output path. The output must be:

```text
<rollout_dir>/predictions/<task-id>/conversation.json
```

The helper `conversation_output_path()` constructs this path. Conversion:

- creates parent directories;
- writes UTF-8 deterministic JSON with a trailing newline;
- writes a complete temporary file and atomically links it into place;
- reuses an existing semantically identical conversation without rewriting;
- raises `TrajectoryConversionError` if the destination is malformed,
  symlinked, or contains different content;
- never rewrites, truncates, or deletes the source ATIF.

No output truncation is applied. This matches upstream reflection's current
full-content behavior.

## Phase 6 boundary

Phase 5 does not discover jobs or trials. Phase 6 remains responsible for:

```text
PreparedHarborRun
-> explicit job directory
-> explicit trial result and trajectory paths
-> result parser
-> trajectory converter
-> rollout result assembly
```

Real Harbor integration still needs to confirm the exact artifact contents
produced by one pinned Terminal-Bench v2.1 trial, especially timeout timing,
continuation settings in the owner config, and whether backend reasoning is
present for DeepSeek-V4-Flash-0731.
