# Terminal-Bench Skill Packaging

Phase 2 converts SkillOpt's current `skill_content: str` into the native skill
input consumed by Harbor 0.20.0. It does not build or execute a Harbor job,
start Docker, call a model, or modify Terminal-Bench tasks.

## Harbor 0.20.0 audit

Audit date: 2026-08-29.

The locally installed `harbor` executable resolves to a uv tool environment
whose distribution metadata reports exactly `harbor==0.20.0`. The relevant
`harbor/skills.py`, trial config, trial upload, job resolution, and job-lock
files match their wheel `RECORD` hashes.

The active local Terminus-2 source contains an unrelated request-event logging
patch. A side-by-side backup matches the Harbor 0.20.0 wheel `RECORD`, and its
skill discovery/frontmatter implementation is identical to the active file.
The contract below therefore uses the wheel-matching source.

Verified Harbor behavior:

- `AgentConfig.skills` is `list[str | Path]`, serialized as strings.
- A local entry points to a directory, not directly to `SKILL.md`.
- The directory may be one skill directory containing `SKILL.md`, or a root
  whose immediate non-hidden child directories each contain `SKILL.md`.
- Harbor identifies a resolved skill by the skill directory basename. Duplicate
  directory names use last-wins resolution.
- Harbor uploads each resolved skill directory under
  `<environment skills root>/<directory basename>`. With injected skills and
  no task override, the default environment skills root is `/harbor/skills`.
- Harbor independently records an `AgentSkillLock` digest. Its digest covers
  every file under the skill directory using sorted relative paths and each
  file's SHA-256; the result is stored with a `sha256:` prefix.
- Terminus-2 scans `<skills_dir>/*/SKILL.md`. It skips files without YAML
  frontmatter containing both `name` and `description`.
- The Agent Skills format additionally requires a lowercase alphanumeric/hyphen
  `name` of at most 64 characters that matches the parent directory, plus a
  non-empty `description` of at most 1024 characters. The fixed Phase 2 values
  satisfy those constraints.
- For each valid skill, Terminus-2 exposes the frontmatter name, description,
  and absolute `SKILL.md` location in an `<available_skills>` block. Actual
  model use of that location remains a Phase 3 runtime integration check.

## Blank semantics

SkillOpt's trainer baseline is the initial skill `S_0`; it is not intrinsically
a no-skill run. The Terminal-Bench experimental baseline instead requires
Harbor `skills=[]`.

The packaging boundary therefore defines:

```text
not skill_content.strip() -> semantically blank -> harbor_skills == []
skill_content.strip()     -> package exactly one native skill directory
```

`skillopt/envs/terminalbench/skills/initial.md` is a real, whitespace-only
Markdown file so the pinned trainer can open it. Packaging its contents creates
no directory and returns no Harbor skill entry.

## Non-blank artifact

Non-blank content is written to:

```text
<output_root>/
└── harbor_skills/
    └── <artifact-sha256>/
        └── terminalbench-skill/
            └── SKILL.md
```

`harbor_skills` for the returned package is a one-element list containing the
absolute `terminalbench-skill/` directory path. It never points directly to
`SKILL.md`.

The exact artifact is:

```markdown
---
name: terminalbench-skill
description: SkillOpt-generated reusable guidance for Terminal-Bench task execution.
---

<exact SkillOpt skill_content bytes encoded as UTF-8>
```

The fixed frontmatter is Harbor/Terminus-2 packaging metadata. It is not
optimizer output. The body after the frontmatter is the original
`skill_content` without stripping, newline conversion, Markdown rewriting,
summarization, task-specific solutions, or additional Terminal-Bench prompts.

## Digest and canonicalization

The package `sha256` is the lowercase SHA-256 hex digest of the exact bytes
written to `SKILL.md`, including the fixed frontmatter and the untouched UTF-8
body. The digest directory is named with this value.

Consequences:

- identical input and fixed metadata produce the same digest and path;
- trailing whitespace and newline differences remain significant;
- semantically different content produces a different digest;
- packaging never normalizes the optimizer-generated body;
- Harbor later computes its own directory-level, `sha256:`-prefixed lock
  digest. The package SHA and Harbor lock digest intentionally have different
  algorithms and provenance roles.

## Integrity and path safety

- Skill content never controls a path component.
- `output_root` must be explicit, cannot be `.` or a filesystem root, and
  cannot contain lexical `..` traversal.
- Existing symlink path components and symlink artifact entries are rejected.
- The deterministic digest directory may contain only
  `terminalbench-skill/SKILL.md`.
- Repackaging matching content reuses the same artifact without rewriting it.
- Existing mismatched or extra content raises an error and is not overwritten.
- Source `initial.md` is never modified by packaging.

## Phase 3 integration checks

Phase 2 validates the filesystem artifact and the audited Harbor 0.20.0 schema
without running Harbor. Phase 3 must still verify:

1. a generated directory survives the real job-config serialization and
   Harbor upload path;
2. Terminal-Bench task configuration does not override `skills_dir` with an
   incompatible relative path;
3. Terminus-2 lists the generated frontmatter and the target model reads the
   referenced `SKILL.md` during an actual single-task run;
4. blank packaging reaches the real job as exactly `agents[].skills: []`;
5. Harbor's emitted job-lock skill digest matches the uploaded directory;
6. baseline and SkillOpt runs differ only in the native `skills` list.
