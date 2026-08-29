# Dev mode: work on Hitchcock

Use this mode to inspect, change, test, or document the repository itself.

## Scope and workflow

- Make only the repository changes requested by the user. Preserve unrelated
  work in the dirty worktree.
- Read `README.md`, `pyproject.toml`, and the relevant module before editing.
- Package code lives under `src/hitchcock/`; the CLI entry point is
  `hitchcock.cli:main`.
- Use `pip install -e .` for an editable local install when needed.
- Discover and run the smallest relevant test set. If no test covers the
  change, perform a focused CLI or module-level smoke check and state the gap.
- Do not call MIMO, Gemini, image, TTS, or video generation APIs for routine
  tests. Mock clients or use validation-only paths. Paid live tests require
  explicit user approval.
- Never commit `.env`, credentials, generated story media, or large outputs.
- When changing CLI behavior, update the relevant user-facing documentation.

## Architecture invariants

- Story-specific titles, beats, dialogue, camera choices, art prompts, video
  prompts, music descriptions, and voice selections belong in generated
  artifacts—not source code.
- Source code may contain story-agnostic meta-prompts, schemas, model clients,
  validation, orchestration, and file I/O.
- Preserve the gated pending/approved/history lifecycle and machine-readable
  `hitchcock-error: <code>: <message>` failures.
- A refine works from the current pending artifact. Approval promotes pending
  state and unlocks downstream work.

## Script and storyboard invariants

- Every dialogue speaker must be present in `characters_in_scene`. Validation
  appends a missing speaker and logs a warning rather than dropping dialogue.
- Each Seedance character uses one Chinese canonical role label consistently in
  the cast prompt, image binding, action, scene summary, and VO speaker tag.
  Dialogue text is exempt.
- Duplicate canonical labels within a scene receive stable `甲/乙/丙/...`
  suffixes in character order. The same disambiguated label must appear on
  every surface.
- Seedance prompt assembly keeps per-line timing and delivery, targets about
  15 seconds per scene with at least 3 seconds per shot, and caps VO at 4 lines
  per shot and 8 per scene.
- Every Seedance prompt explicitly forbids BGM, subtitles, and captions.
  Compression must preserve timing brackets, delivery notes, VO, shot headers,
  and the negation trailer.

## Verification guidance

At minimum, syntax-check changed Python and exercise relevant parser/help paths
without contacting external services. For example:

```bash
python -m compileall -q src
hitchcock --help
hitchcock <changed-stage> --help
```

Use stronger tests when the change affects state transitions, validation,
prompt assembly, path cleanup, audio mixing, or paid-service boundaries.
