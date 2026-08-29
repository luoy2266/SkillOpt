# SkillOpt → Terminal-Bench v2.1 Migration

Before making changes, read:

- `docs/terminalbench/CODEX_MIGRATION_TASK.md`
- `docs/guide/new-benchmark.md`
- `UPSTREAM.md`

## Hard constraints

- Preserve the pinned SkillOpt upstream behavior and avoid modifying core training algorithms unless strictly necessary.
- Add Terminal-Bench as a SkillOpt benchmark/environment extension.
- Harbor is an external dependency pinned to 0.20.0. Do not fork or vendor Harbor.
- Use Docker, not Daytona.
- Target execution must remain:
  SkillOpt -> Harbor -> Terminus-2 -> DeepSeek-V4-Flash-0731 -> Terminal-Bench v2.1.
- Do not bypass Terminus-2 by calling the target model directly.
- Inject skills only through Harbor native `agents[].skills`.
- Do not manually concatenate SkillOpt skills into the system prompt.
- Baseline and SkillOpt must share the same runner/config; the main experimental difference is `skills=[]` versus generated skills.
- Do not modify Terminal-Bench tasks or verifier.
- Prefer minimal, phase-by-phase changes.
- Run relevant tests after each phase.
- Do not start full-scale training/evaluation from this development repository.
- Do not commit automatically unless explicitly requested.

## Implementation style

- Keep migration code minimal and reviewable.
- Prefer existing SkillOpt/Harbor abstractions over new framework layers.
- Do not generalize for hypothetical future benchmarks, Harbor versions, or agents.
- Avoid factories, registries, wrappers, and dataclasses unless they solve a concrete current requirement.
- Minimize diff size while preserving correctness and tests.
