# User mode: operate Hitchcock

Use this mode as a director or operator of the installed Hitchcock CLI. Do not
develop or modify the software.

## Boundaries

- Do not edit `src/`, project configuration, packaging, documentation, scripts,
  or tests.
- Reading code for diagnosis is allowed, but stop at reporting a software bug.
  Offer dev mode if the user wants it fixed.
- Hitchcock may write normal story state and generated output under `bible/`.
  Restrict changes to the story named by the user.
- Never hand-edit generated scripts, storyboards, art prompts, or Seedance
  prompts. Use the stage's `refine --feedback "..."` command.
- The only manual-prompt exception is an intentional packaged `prompt.txt`
  workflow used with `render seedance --use-package-prompt`.
- Generation and rendering can use paid APIs. Show or inspect the estimated
  cost and get the user's approval before bypassing a cost prompt with `--yes`
  or `--allow-cost`.

## Operating loop

Start with:

```bash
hitchcock status -s <story-id> --json
```

Follow `next_action` unless the user asks for a different stage. For gated
stages, use:

```text
generate -> show -> refine (repeat) -> approve
```

Approve a stage only after the user accepts it or explicitly delegates that
decision. A refine requires the current pending artifact. To restart from the
approved baseline, run `generate` first.

Use `hitchcock --help` and `hitchcock <stage> --help` as the current command
reference. The public quickstart is in [README.md](../README.md).

## Pipeline

```text
source -> brief -> style -> cast -> script -> storyboard -> art -> render
```

Each downstream stage consumes approved upstream state. Prefer scene- or
shot-scoped commands when the user requests a local change; this limits cost
and avoids disturbing accepted work.

For art with one candidate, Hitchcock selects it automatically. With multiple
candidates, inspect them and run `art pick` before approval. After a partial
`art approve --allow-unpicked`, newly added picks require another `art approve`
before packaging.

`render package` prepares Jimeng upload bundles. `render seedance` calls Ark and
writes clips. Use `--use-package-prompt` only when the package prompt was
deliberately adjusted and its reference-image order has been preserved.

## State and errors

Canonical story state lives in `bible/stories/<story-id>/`. Pending artifacts
are drafts; approved artifacts are canonical; `history/` and `feedback.log`
record provenance.

Errors use this stderr form:

```text
hitchcock-error: <code>: <message>
```

Common codes include `NO_UPSTREAM_APPROVED`, `NO_PENDING`, `UNKNOWN_SCENE`,
`MIMO_PARSE_FAIL`, `IMAGE_GEN_FAIL`, and `OVER_BUDGET`. A model parse failure
is normally safe to retry. For API failures, check credentials, quota, and the
requested cost before retrying.

## Operational caveats

- Changing scene IDs can leave obsolete art candidates and render packages.
  Identify exact orphaned scene paths before removing anything.
- Style or brief changes do not reliably mark downstream output stale. Expect
  to rebuild and review affected cast, storyboard, art, and render stages.
- `render package` may retain packages for deleted scenes after structural
  changes. Inspect the story's package directory before regenerating.
- Prompt compression is best effort. Before Jimeng upload, check the 2000
  character limit and stray CJK text in English prose; preserve VO, timing, and
  shot headers when trimming.
- Before mixing TTS, use `ffprobe` to confirm that each clip has an audio stream.
  A clip without audio needs a silent baseline or the mix may fail.
- Exact exotic-vehicle interiors are unreliable without a reference image.
  Prefer a tighter composition when cabin geometry matters.
