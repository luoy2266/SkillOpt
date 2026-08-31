# Terminal-Bench v2.1 Split Manifest

Phase 1 represents the Terminal-Bench split as lightweight, replaceable task-ID
files. It does not include task solutions, verifier tests, Dockerfiles, or full
task payloads, and it does not invoke Harbor, Docker, or a model API.

## Frozen delivery split

The reviewed 89-task `9/9/71` partition is checked in at
`configs/terminalbench/splits/v2.1-s42/`. It contains task IDs only and is
pinned to Terminal-Bench revision
`7131e4375048a0e408a8fb404b5f499d726b695b`. Its portable semantic SHA-256 is:

```text
bd36fe2f37a67cd2b46149263522d833166d3a4d036c8e9af082e742ad017500
```

Server bootstrap may copy this directory below `SKILLOPT_RUNTIME_ROOT/splits/`
without changing its identity. It must not rerandomize or replace the task IDs.

The materializer accepts any of these external sources:

- a directory whose immediate child directory names are task IDs;
- a JSON array of task-ID strings or objects;
- a JSON object containing exactly one array field named `task_ids`, `tasks`,
  or `data`;
- JSONL containing one task-ID string or object per line;
- a text file containing one task ID per non-empty line.

## Materialization

For an installed Terminal-Bench task directory:

```bash
python -m scripts.materialize_terminalbench_split \
  --source /path/to/terminal-bench-v2.1/tasks \
  --source-revision '<pinned-source-revision>' \
  --ratio 1:1:8 \
  --seed 42 \
  --output-dir data/terminalbench_split
```

For a JSON source whose ID field is `task_id`, with optional lightweight
metadata retained explicitly:

```bash
python -m scripts.materialize_terminalbench_split \
  --source /path/to/terminalbench-v2.1-task-ids.json \
  --id-field task_id \
  --metadata-field category \
  --source-revision '<pinned-source-revision>' \
  --ratio 1:1:8 \
  --seed 42 \
  --output-dir data/terminalbench_split
```

Object sources copy only `id` plus explicitly selected scalar metadata fields.
Known task-payload fields such as `solution`, `tests`, `Dockerfile`, `task`, and
`instruction` are rejected as metadata.

## Materialized layout

The pinned `SplitDataLoader` consumes directories rather than a single split
JSON document, so Phase 1 uses the repository's current `items.json`
convention:

```text
data/terminalbench_split/
├── split_manifest.json
├── split_manifest.sha256
├── train/
│   └── items.json
├── val/
│   └── items.json
└── test/
    └── items.json
```

Each split file is a JSON array. The minimum item schema is:

```json
{
  "id": "<terminalbench-task-id>"
}
```

Additional fields must be lightweight metadata selected during
materialization. `id` must be a non-empty, stable, portable filesystem name
because later phases must use the same exact value for the Harbor trial,
rollout result, and prediction directory.

## Root manifest schema

`split_manifest.json` is a JSON object with these fields:

| Field | Meaning |
|---|---|
| `schema_version` | `2` for the host-independent portable manifest. |
| `benchmark` | Fixed to `terminal-bench`. |
| `benchmark_version` | Fixed to `2.1`. |
| `manifest_type` | Fixed to `id_split`. |
| `materializer` | Repository script that created the files. |
| `semantic_identity` | Stable schema, benchmark/version, pinned Terminal-Bench revision, seed, and ordered IDs for all three splits. |
| `semantic_sha256` | SHA-256 of canonical JSON for `semantic_identity`. |
| `source` | Resolved source path, source format, source checksum and scope, ID field, retained metadata fields, and optional pinned revision. |
| `input` | Input task count and SHA-256 of the canonical sorted task-ID set. |
| `split` | Ratio, deterministic seed, shuffle algorithm, and count-allocation rule. |
| `counts` | Declared `train`, `val`, and `test` item counts. |
| `item_fields` | Union of fields present in materialized items. |
| `files` | Relative path, count, and SHA-256 for each split file. |
| `notes` | Data-minimization reminders. |

`split_manifest.sha256` contains the portable semantic SHA-256, not a hash of
the whole manifest bytes. Absolute source checkout paths remain useful runtime
provenance, but they do not affect split identity. `TerminalBenchDataLoader`
validates the semantic identity, all split-file checksums and counts, the
task-ID set checksum, item fields, ID safety, and uniqueness across all three
splits before applying the inherited per-split `limit`.

Legacy schema-v1 materializations remain valid only through their original
whole-manifest checksum path. The previously completed local formal experiment
retains legacy SHA-256
`8fa19aa350b90a7c39c3cde56f87a93bbfcb450586b416dc700c4c0b35827894`;
the loader and formal preflight do not silently reinterpret it as a portable
semantic hash.

## Count allocation

The materializer matches the pinned `SplitDataLoader` largest-remainder rule:

1. compute each ideal count from the integer ratio;
2. take the floor of each ideal count;
3. distribute remaining items by descending fractional remainder;
4. break a remaining tie by larger ratio weight, then `train`, `val`, `test`
   order.

For 89 tasks with `1:1:8`, the ideal counts are `8.9`, `8.9`, and `71.2`.
Flooring gives `8/8/71`, and the two remaining tasks go to `train` and `val`,
producing `9/9/71`.

## Dataloader mapping

`TerminalBenchDataLoader` subclasses the pinned `SplitDataLoader` and preserves
its batch behavior:

- `train` loads `train/items.json`;
- `valid_seen` and `selection` resolve to `val`;
- `valid_unseen` resolves to `test`;
- train batches use the inherited deterministic shuffle and epoch planning;
- evaluation preserves manifest order and applies `env_num` as an optional
  prefix limit;
- `limit > 0` truncates each validated split independently.

Runtime `split_mode="ratio"` is intentionally rejected for Terminal-Bench.
Changing the task partition requires rerunning the materializer against the
replacement source, not changing Python code.
