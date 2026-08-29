# Terminal-Bench v2.1 Split Placeholder

This directory intentionally does not contain a formal task-ID split because
the repository has no authoritative Terminal-Bench v2.1 89-task ID source.
Do not replace it with invented IDs.

Materialize `train/`, `val/`, `test/`, `split_manifest.json`, and
`split_manifest.sha256` from the real pinned task source with:

```bash
python -m scripts.materialize_terminalbench_split \
  --source /path/to/terminal-bench-v2.1/tasks \
  --source-revision '<pinned-source-revision>' \
  --ratio 1:1:8 \
  --seed 42 \
  --output-dir data/terminalbench_split
```

See `docs/terminalbench/SPLIT_MANIFEST.md` for the input formats, output schema,
validation rules, and the `89 -> 9/9/71` allocation rule.
