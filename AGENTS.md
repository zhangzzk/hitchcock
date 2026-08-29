# Hitchcock agent instructions

This repository supports two agent roles. These are repository conventions,
not built-in Codex modes.

## Select a mode

Choose exactly one mode before acting and state it in the first user update.

- An explicit `User mode:` or `Dev mode:` in the request always wins.
- Choose **user mode** for operating Hitchcock: creating a story, advancing the
  pipeline, reviewing artifacts, refining creative output, or rendering media.
- Choose **dev mode** for inspecting, changing, testing, or documenting the
  Hitchcock implementation itself.
- If the request is ambiguous, default to user mode and do not modify the
  repository. Ask before switching to dev mode.

Do not mix modes silently. If an operating task exposes a software bug, report
it in user mode; fix it only after the user requests dev mode or asks for the
implementation to be changed.

## User mode

Read and follow [docs/agent-user-mode.md](docs/agent-user-mode.md).

The software repository is read-only in this mode. Normal Hitchcock commands
may create or update story state and generated media under `bible/` as part of
the requested workflow.

## Dev mode

Read and follow [docs/agent-dev-mode.md](docs/agent-dev-mode.md).

Repository edits are allowed only within the user's requested development
scope. Do not call paid generation services merely to test a code change.

## Shared rule

Story-specific creative content belongs in generated artifacts, not Python.
Python contains schemas, story-agnostic meta-prompts, clients, orchestration,
and file handling.
