# Hitchcock — Agent-Driver Contract

This document is the CLI contract for driving the Hitchcock pipeline. The
intended reader is an **AI agent** acting on behalf of a human creative
director — or the human themselves. Both drive the same CLI.

The core invariant: **you never hand-edit the generated content (scripts,
storyboards, shot prompts, etc.)**. You iterate via `refine` commands that
carry natural-language feedback. Python code holds only the pipeline
plumbing + the meta-prompts (prompts-to-generate-prompts) — never story-
specific content.

---

## 1. Pipeline shape

Four layers, each gated by explicit `approve`:

```
    source text  →  [ ScriptAgent + MIMO ]   →  script.json          (Layer 1)
                                                     ↓ approve
                                               [ StoryboardAgent + MIMO ] →  storyboard.json  (Layer 2)
                                                     ↓ approve
                                               [ SceneArtAgent + gpt-image-2 ] →  scene_arts/sXX.png + art.json  (Layer 3)
                                                     ↓ approve
                                               [ ShotGenAgent + Seedance (or Jimeng packages) ]
                                                   + [ TTSAgent + BGMAgent ]
                                                   + [ PostAgent + ffmpeg ]     →  reel.mp4   (Layer 4)
```

Each layer has the same 4-verb lifecycle:

| verb       | effect                                                                 |
|------------|------------------------------------------------------------------------|
| `generate` | MIMO (or image/video model) reads **upstream approved** + source, writes `pending/` |
| `show`     | Prints current `pending/` (or `approved/` if no pending) — supports `--json` |
| `refine`   | MIMO reads `pending/` + `--feedback "<text>"`, writes new `pending/`   |
| `approve`  | Promotes `pending/` → `approved/`, unlocks next layer                  |

Refine is always **against the current pending**. If you want to refine
against the approved baseline, call `generate` first (overwriting pending
with a fresh run from approved upstream).

---

## 2. Data model on disk

All pipeline state for one story lives under `bible/stories/<story-id>/`:

```
bible/stories/<sid>/
├── script/
│   ├── approved.json            ← canonical Script (Story)
│   ├── pending.json             ← last unapproved draft (optional)
│   └── history/                 ← every approved version, timestamped
├── storyboard/
│   ├── approved.json            ← canonical Storyboard
│   ├── pending.json
│   └── history/
├── art/
│   ├── approved.json            ← canonical scene-art picks (which candidate per scene)
│   ├── pending.json
│   ├── candidates/<scene_id>/cand_NN.png    ← all candidates generated
│   └── scene_arts/<scene_id>.png            ← approved-pick copy
├── render/
│   ├── approved.json            ← render state per scene (seedance_task_id / jimeng_package_path / clip_path)
│   ├── clips/<scene_id>.mp4
│   └── reel.mp4
├── tts/                          ← (Layer 4b) VO audio per line
│   └── <scene_id>/<line_idx>.wav
├── bgm/                          ← (Layer 4b) BGM stem per scene
│   └── <scene_id>.mp3
└── feedback.log                  ← chronological record of every refine feedback + author
```

Each `refine` appends an entry to `feedback.log` with timestamp, stage,
scene_id (if any), feedback text. Full provenance so another AI reading
the story later can see why it is the way it is.

---

## 3. CLI commands

All commands exit with `0` on success, `≥2` on error. All support
`--story-id <sid>` (`-s`) and `--json` (machine-readable output).

### 3.1 Bootstrap

```
hitchcock init --story-id <sid> --source <path-to-source.txt>
                [--character <id>]...   # characters already in bible
                [--location  <id>]...   # locations already in bible
```

Creates `bible/stories/<sid>/` skeleton. Registers source text as the
immutable input. Lists required characters/locations; errors if any aren't
in `bible/characters/` or `bible/locations/`.

### 3.2 Layer 1 — Script

```
hitchcock script generate  -s <sid>
hitchcock script show      -s <sid> [--json]
hitchcock script refine    -s <sid> --feedback "<natural language>"
hitchcock script approve   -s <sid>
```

Script data model (one `Story` JSON):
- `id`, `title`, `logline`, `synopsis`, `style_guide`
- `characters: [id, ...]`
- `scenes: [{ id, title, location_id, time_of_day, emotional_register,
            characters_in_scene, summary, beats, duration_sec_target }]`

`refine` feedback examples: *"scene 3 is too quiet, add a moment of confrontation"*, *"make the whole story 50% shorter"*, *"s06 should end on tears not laughter"*.

### 3.3 Layer 2 — Storyboard

```
hitchcock storyboard generate  -s <sid>                    # all scenes
hitchcock storyboard show      -s <sid> [--scene <id>] [--json]
hitchcock storyboard refine    -s <sid> [--scene <id>] --feedback "<text>"
hitchcock storyboard approve   -s <sid> [--scene <id>]
```

Storyboard per scene (expands each Script scene):
- `shots: [{ id, duration_sec, shot_type, camera_movement,
           characters_in_shot, action, dialogue }]`
- `scene_art_prompt` — a v6-recipe-compliant T2I prompt for Shot 1
  first-frame art (consumed by gpt-image-2 by default; backend is
  swappable). Auto-authored by MIMO from the scene's shots + VO +
  location. Never hand-edited.
- `seedance_prompt` — the final schema-format Shot 1/2/3 + VO + BGM text
  that the video model will consume. Auto-assembled.

`refine --scene s06 --feedback "..."` tells MIMO to revise that single
scene's shots/prompts while keeping others untouched.

### 3.4 Layer 3 — Scene Art

```
hitchcock art generate  -s <sid> [--scene <id>] [--candidates N]   # default N=1
hitchcock art show      -s <sid> [--scene <id>] [--json]    # list candidates w/ paths
hitchcock art pick      -s <sid>  --scene <id>  --candidate <N>
hitchcock art refine    -s <sid>  --scene <id>  --feedback "<text>"
hitchcock art approve   -s <sid>
```

`generate` calls the configured T2I backend (default **gpt-image-2** via
OpenAI; Nano Banana Pro available as fallback) with the storyboard's
`scene_art_prompt` to produce N candidates →
`candidates/<scene_id>/cand_NN.png`. Default `N=1`; pass `--candidates 2-3`
when a shot needs variety. When `N=1`, `pick` is auto-selected; when `N>1`
the user/agent must `pick` explicitly.

`refine --scene <id> --feedback "<text>"` does the important thing:
MIMO reads the current `scene_art_prompt` + the user's feedback,
produces a **new** `scene_art_prompt`, writes to pending, then re-runs
`generate` for that scene. This is the mechanism that replaces me
hand-writing `regen_s12_v8.py`.

`pick` marks one candidate canonical (copies to `scene_arts/<id>.png`).
`approve` promotes all picks into the approved manifest.

### 3.5 Layer 4 — Render

```
hitchcock render package   -s <sid> [--scene <id>]                   # build Jimeng upload bundles
hitchcock render seedance  -s <sid> --scene <csv> [--use-package-prompt]  # run Seedance 2.0 via Ark
hitchcock render tts       -s <sid> [--scene <id>]                   # generate VO audio
hitchcock render bgm       -s <sid> [--scene <id>]                   # generate/select BGM
hitchcock render post      -s <sid>                                  # ffmpeg mix + concat → reel.mp4
hitchcock render status    -s <sid>
```

Two backends, both live:

- `render package` produces `render/packages/<scene_id>/{01_first_frame.png,
  02_<char>_ref.png, ..., prompt.txt}` bundles for manual upload to
  jimeng.jianying.com. Useful as a fallback when Ark is slow/overloaded,
  or when a human wants to hand-edit `prompt.txt` before sending.

- `render seedance` runs Seedance 2.0 directly via Volcengine Ark with
  @tag grammar (shot keyframes + char refs + optional style refs). Writes
  `render/clips/<scene_id>.mp4`. `--scene` is required (each call costs
  Ark tokens). Default `--resolution 480p`; bump to `720p` for a final
  pass.

  **`--use-package-prompt`** reads the (possibly hand-edited)
  `render/packages/<scene>/prompt.txt` verbatim as the prompt_text, and
  skips style refs so the uploaded ref order matches the package's
  `@image1..N` numbering (keyframe + char refs only). Workflow:
  `render package` → hand-edit `prompt.txt` → `render seedance
  --use-package-prompt`. Without the flag, the prompt is re-assembled
  fresh from storyboard data and package edits are ignored.

### 3.6 Status & navigation

```
hitchcock status -s <sid> [--json]
```

Returns per-stage state:
```json
{
  "story_id": "mountain-arc-full",
  "stages": {
    "script":     { "state": "approved",  "last_refined": "2026-04-20T12:01Z" },
    "storyboard": { "state": "pending",   "last_refined": "2026-04-20T12:45Z" },
    "art":        { "state": "not_started" },
    "render":     { "state": "not_started" }
  },
  "next_action": "hitchcock storyboard show -s mountain-arc-full"
}
```

`next_action` is the suggested command for an agent to run next — the CLI
guides the agent forward.

---

## 4. Refine semantics (the heart of the contract)

Every `refine` command follows this MIMO call pattern:

```
meta-prompt (hard-coded in agent code)
  + current_artifact (from pending.json or approved.json)
  + user_feedback   (the --feedback string)
  ↓
new_artifact → pending.json
```

The meta-prompt lives in code (e.g. `agents/script.py::_REFINE_SYSTEM`).
It is the only story-agnostic content in the repo. Everything else is
generated by MIMO or the user.

Multiple refines stack: each refine reads the latest pending and emits a
new pending. The user may refine 5 times before approving.

### Good feedback examples
- *"scene s06 wants more tension — add a moment where one of them considers turning back"* (script)
- *"shot 2 of s11 should pull back rather than push in"* (storyboard)
- *"s12 first-frame: switch to over-shoulder POV looking up at the sky"* (art)
- *"reduce total runtime to ~3 min; cut what feels most cuttable"* (script)

### Discouraged feedback
- Feedback that tries to directly edit the artifact by quoting exact
  phrases to replace. This defeats the point of MIMO-authored content.
  Instead, describe the desired change in intent terms.

---

## 5. Error contract for AI drivers

Every non-zero exit prints a machine-parseable error line to stderr:
```
hitchcock-error: <code>: <msg>
```

Key codes:
- `NO_UPSTREAM_APPROVED` — tried to generate a stage without approving the previous
- `NO_PENDING` — tried to refine/approve with no pending artifact
- `UNKNOWN_SCENE` — `--scene <id>` doesn't match
- `MIMO_PARSE_FAIL` — LLM returned malformed JSON (retry safe)
- `IMAGE_GEN_FAIL` — T2I backend error (check key / quota; default backend gpt-image-2)
- `OVER_BUDGET` — generated cost exceeds soft cap; pass `--allow-cost <N>` to override

---

## 6. Example driver flow (what an AI agent looks like)

```bash
# ① Initialize
hitchcock init -s dragon-raja-act6 \
  --source path/to/source.txt \
  --character lu-mingfei --character chen-mo-tong \
  --location mountain-road-night-bugatti --location mountain-top-lake

# ② Layer 1: Script. Generate draft, review, iterate, approve.
hitchcock script generate  -s dragon-raja-act6
hitchcock script show      -s dragon-raja-act6 --json | jq '.scenes | length'
hitchcock script refine    -s dragon-raja-act6 \
  --feedback "12 scenes is too many — merge the closely-related ones and trim to 8; preserve all dialogue"
hitchcock script show      -s dragon-raja-act6
hitchcock script approve   -s dragon-raja-act6

# ③ Layer 2: Storyboard
hitchcock storyboard generate  -s dragon-raja-act6
hitchcock storyboard refine    -s dragon-raja-act6 --scene s06 \
  --feedback "both figures must be barefoot, same rock, over-shoulder framing"
hitchcock storyboard approve   -s dragon-raja-act6

# ④ Layer 3: Scene Art (iterate freely per scene)
hitchcock art generate -s dragon-raja-act6                          # 3 cand per scene
hitchcock art show     -s dragon-raja-act6 --scene s12 --json       # inspect candidates
hitchcock art refine   -s dragon-raja-act6 --scene s12 \
  --feedback "over-shoulder POV looking up; firework text should not look like a font — should look like actual pyrotechnic sparks with broken strokes and heat shimmer"
hitchcock art pick     -s dragon-raja-act6 --scene s12 --candidate 3
hitchcock art approve  -s dragon-raja-act6

# ⑤ Layer 4: Render (Jimeng packages for now)
hitchcock render package  -s dragon-raja-act6
# → prints paths to 12 upload bundles; user uploads, downloads mp4s
hitchcock render tts      -s dragon-raja-act6
hitchcock render bgm      -s dragon-raja-act6
hitchcock render post     -s dragon-raja-act6  # ffmpeg concat + mix → reel.mp4
```

An agent can drive this loop by running `hitchcock status --json`,
reading `next_action`, and iterating through the user (*"show me s12";
*"try over-shoulder POV"*).

---

## 7. Story bibles (characters + locations)

Characters and locations are independent resources shared across stories.

```
hitchcock design <description-path>              # → bible/characters/<id>/
hitchcock location <description-path>            # → bible/locations/<id>/
```

These two commands remain one-shot (no gate/refine loop) because each
character or location is typically iterated visually via re-running
`design` / `location` with an adjusted description, not via per-story
feedback. (Can be gated later if there's a clear use case.)

---

## 8. What's NOT in code — and must never be

- Story titles
- Scene titles / summaries / beats / VO / emotional_register
- Shot-level camera + framing + action
- Scene art prompts
- Seedance prompts
- BGM style descriptions
- TTS voice selections

All of these are MIMO-authored from the user's source text + feedback.
Python code holds: meta-prompts, JSON schemas, API clients, file IO,
ffmpeg orchestration. Period.

The test for whether something belongs in code: *"would changing the
story break it?"* — if yes, it belongs in MIMO-generated artifacts, not
Python.

---

## 9. Cost transparency

Each generating command prints the estimated cost up front and exits
without calling an API if stdout is a TTY and `--yes` wasn't passed.
Costs (approx):
- `script generate` / `refine`: ¥0.01–0.05 per call (MIMO only)
- `storyboard generate`: ¥0.05–0.20 per story
- `storyboard refine --scene`: ¥0.01–0.05 per scene
- `art generate` (3 cand): ¥1.2 per scene
- `art refine` (includes regenerate): ¥1.2 per scene
- `render seedance`: ¥15–30 per 15s clip
- `render tts`: ¥0.01 per VO line
- `render bgm` (Suno): ¥0.5–2 per scene
- `render post`: free (ffmpeg local)

With `--json` output, cost estimates appear as `{"cost_estimate_rmb": N}`
in the pre-flight response and can be inspected before committing.

---

## 10. Versioning

`AGENTS.md` is source-of-truth for the contract. Any agent reading this
can assume the CLI follows it. If the CLI diverges, AGENTS.md is the
bug. Run `hitchcock --version` to see the contract version the binary
implements.

---

## 11. Known issues / Phase 2 TODOs

Bugs + design gaps discovered during real-story testing. An agent driving
the pipeline should be aware of these and work around them until fixed.

### 11.1 Scene ID reassignment leaves stale downstream artifacts

**What breaks:** When `script refine` changes the number or IDs of scenes
(e.g. 4 scenes → 15 scenes), already-built downstream artifacts keyed on
old IDs stay on disk:
- `bible/stories/<sid>/art/candidates/<old_scene_id>/` orphan dirs
- `bible/stories/<sid>/art/scene_arts/<old_scene_id>.png` orphan files
- `bible/stories/<sid>/render/packages/<old_scene_id>/` orphan bundles

**Workaround:** Manually `rm -rf` orphan subdirectories after `script
refine` if scene structure changed substantially.

**Fix plan:** `script approve` (or `storyboard generate`) should diff the
new scene id set against prior stages' artifacts and prune/flag
orphans. Add a `hitchcock clean -s <sid>` command to wipe orphans.

### 11.2 Style change does not invalidate downstream art

**What breaks:** `style refine` + `style approve` changes the canonical
art direction for the story. All downstream artifacts (character refs,
location establishing art, scene arts) built under the OLD style are now
style-mismatched but still marked `approved`. There's no automatic
staleness tracking.

**Workaround:**
- After style change, always run `cast build --force --only <ids>` for
  every character that needs restyling (no automated trigger).
- Remember to `storyboard generate` + `art generate` again — they'll
  pull the new style but existing approved art won't auto-update.

**Fix plan:** Every produced artifact should record the `style_guide`
hash it was built under. Any stage with a hash ≠ current approved style
should be automatically flagged `stale`, surfaced in `hitchcock status`,
and downstream operations should prompt to rebuild.

### 11.3 `render package` lacks `--clean` flag

**What breaks:** `render package` merges new output into an existing
`render/packages/<sid>/` directory without clearing stale scene subdirs.
Combined with 11.1, you can end up with packages for scenes that no
longer exist in the current script.

**Workaround:** `rm -rf bible/stories/<sid>/render/packages/` before
calling `render package` if you've changed scene structure.

**Fix plan:** Add `--clean` flag to `render package` that wipes the
packages directory before writing. Or default-clean: always wipe
unselected scenes' package dirs.

### 11.4 Nano Banana cannot reliably render exotic vehicle interiors

**What breaks:** Prompts specifying a Bugatti Veyron (2-seat coupe)
interior consistently render as sedan-like cabins (3+ seats visible, back
row, rear doors) no matter how precisely the `scene_art_prompt`
constrains geometry.

**Workaround:** For shots requiring specific exotic-vehicle interiors,
`storyboard refine --scene <id> --feedback "..."` and replace the
whole-cabin composition with an extreme close-up (rearview mirror only,
driver's eyes only, dashboard insert) that hides the cabin geometry
Nano Banana cannot render. Or change the location to a more common car.

**Fix plan:** Upload a Bugatti Veyron interior reference image as a 4th
ref to Nano Banana (currently we pass only location + character refs).
Adding `scene.extra_refs: list[str]` to the schema would enable this.

### 11.5 Compression is best-effort, not guaranteed-under-budget

**What breaks:** `render package --max-chars N` runs MIMO to compress each
scene's `seedance_prompt` into the budget AND applies a CJK-leak scrub
pass. Neither step is 100% reliable:

- MIMO often returns output 10-30% over the target char cap when the
  input is content-rich (detailed SCENE + dense VO + transitions + many
  shots). We do NOT hard-truncate — that would risk dropping VO mid-line.
- The CJK scrub pass catches most Chinese-in-English tokens but usually
  leaves 1-2 stragglers per scene (`小跑`, `涌动人群`, etc.).
- Raising the budget cuts Seedance's ability to follow the brief
  (prompt-distraction); lowering it drops environmental density.

**Workaround:** Treat `render package` output as ~90% production-ready.
Before uploading to Jimeng Web UI (2000-char limit), skim each
`prompt.txt` for (a) remaining CJK leaks in English sentences — replace
with English equivalents — and (b) char count over 2000 — trim
redundant style prose or shot descriptions.

**Fix plan:** Later — add a deterministic code-side fallback that trims
style anchor prose first, then character summaries, then shot action
prose's second sentence, in that order, until under budget. Keeps VO +
SCENE + shot headers untouched.

### 11.6 BRIEF regeneration ripples unpredictably into downstream

**What breaks:** Running `brief approve` late in a story's lifecycle —
after `script approve`, `storyboard approve`, `art approve` — and then
re-generating `storyboard` / `art` to pick up the new brief inputs
creates a **different film** than the first pass:

- Storyboard MIMO writes different shot breakdowns (different camera
  choices, durations, transitions).
- Scene art compositions change since the scene_art_prompt is MIMO-
  authored each run.
- If the first-pass video (already uploaded to Jimeng + downloaded mp4s)
  was acceptable, the regenerated second pass is *not guaranteed* to be
  as good — MIMO is non-deterministic and may make worse choices.

**Workaround:**
- Front-load BRIEF approval BEFORE script generate when possible.
- If retrofitting BRIEF mid-pipeline, budget for at least one full
  downstream regen + re-upload cycle, and expect the new reel to
  require fresh review (not just a touch-up of the prior one).
- Keep the prior approved storyboard as a reference (it's in
  `bible/stories/<sid>/storyboard/history/`) — you can diff shot
  structures to see what changed.

**Fix plan:** Add `hitchcock status --diff-upstream` that shows which
downstream stages have stale outputs (generated under an older version
of an upstream stage) so the user knows the regen blast radius before
acting.

### 11.7 `art approve --allow-unpicked` is a partial-approve workaround

**What works:** Introduced to enable incremental 2-scene validation
without spending on all scenes. `art approve --allow-unpicked` approves
an art manifest even when some scenes have no picked candidate; `render
package` silently skips those scenes.

**What to watch:** If you later `art generate` the skipped scenes, pick,
and want them included, you must run `art approve` again (the manifest
state transitions pending → approved → pending → approved). This is
functional but not obviously documented.

**Fix plan:** Add `hitchcock art status -s <sid>` that shows unpicked /
unapproved / missing scenes explicitly, and a `hitchcock art complete -s
<sid>` that generates+picks all missing scenes in one call.

### 11.8 `render post` may drop Seedance's native audio when mixing TTS

**What breaks:** When `--with-tts` is used, the filter graph mixes
`[0:a]volume=0.35[orig]` + `[vo]` via amix. If the `[0:a]` audio stream
is missing (some Seedance/Jimeng output mp4s have no audio track at all),
the filter crashes or produces silent output.

**Workaround:** Check `ffprobe -show_streams` on the clip before running
`render post --with-tts`. If no audio stream, either skip TTS overlay or
add a silent audio track first via
`ffmpeg -i in.mp4 -f lavfi -i anullsrc=cl=stereo -shortest out.mp4`.

**Fix plan:** Detect missing audio track in `_mix_scene_audio` and
substitute silent baseline before mixing.

### 11.10 Canonical role labels — one Chinese label per character, ALL surfaces

**What breaks:** Seedance 2.0 position-teleports characters when a prompt
refers to the same person with multiple labels (observed 2026-04-21 on
s02: prompt mixed `楚子航` / typo `朱子航` / `生父` / `父亲` / `司机` /
`young man` / `middle-aged man` — six labels for two characters. Output
showed the son and father swapping between front-seat and back-seat
between shots). Seedance binds identity by exact token match; divergent
labels = diverged identities.

**Contract (enforced in storyboard pipeline):**

- `_canonical_role_cn(c: Character)` derives one Chinese label per
  character from gender+age-bucket (`少年`, `中年男人`, `少女`, ...).
  Same helper used EVERYWHERE a character is referenced in the Seedance
  prompt: @tag binding line, `角色 ↔ 参考图` block, VO speaker tag,
  shot action prose (via MIMO), and scene 概要 (via name substitution).
- `_SHOTS_SYSTEM` / `_REFINE_SHOTS_SYSTEM` now include a HARD RULE
  telling MIMO to use ONLY the canonical_label in shot action. Names
  (楚子航), story-role words (父亲/生父/司机), and English labels
  (young man) are forbidden in action prose. Dialogue TEXT is exempt
  (names inside quoted speech stay verbatim).
- `_scene_to_user_prompt` surfaces each character's canonical_label as
  a distinct field in the Cast block + writes an explicit label_rule
  preamble so MIMO can't miss it.
- The assembler substitutes `c.name → canonical_label` in scene.summary
  before emitting. Story-role words in summary (父亲/生父) remain — user
  tightens via `script refine` if needed.
- Canonical labels are Chinese because Seedance is Chinese-native. T2I
  paths (design.py, scene_art.py) still use English age-bucketed labels
  via `_vo_role_tag`.

**What to still watch for:**

- When MIMO occasionally drifts (uses `父亲` instead of `中年男人`), a
  one-shot `storyboard refine` citing the canonical_label rule pulls
  it back. Happens ~10-20% of refine runs pre-training.
- If a story has two characters with identical gender+age-bucket (two
  middle-aged men), both get label `中年男人` → collision. Fix: extend
  `_canonical_role_cn` to disambiguate via first visual feature
  (`中年男人（蓄胡）` vs `中年男人（眼镜）`). Not yet implemented —
  Dragon Raja v2-prologue doesn't hit this case.

**Fix plan:** The current enforcement is the fix. If MIMO drift
persists after training, add a deterministic post-scan: assembler strips
any known character-name from shot action and replaces with canonical.

### 11.11 Canonical-label collisions — 甲/乙 suffix

**What breaks:** When two characters in the same scene share a
canonical_role_cn (both "少年", both "中年男人", ...), the @image
binding block has duplicate labels (`@image2: 少年`, `@image4: 少年`).
Seedance can't distinguish which face a `少年:` VO line corresponds
to → the second character's lines get rendered onto the first
character's face. Observed 2026-04-21 on v2-prologue s01
(chu-zi-hang + lu-mingfei both young male teenagers).

**Contract (enforced in `_disambiguate_labels`):** if a base canonical
label appears ≥2 times in `characters_in_scene`, suffix each occurrence
with 甲/乙/丙/丁/戊/己/庚/辛 by characters_in_scene order. Single-
occurrence labels stay bare. Applied in three surfaces that must stay
consistent: MIMO user prompt (so shot action uses disambiguated labels),
assembler @tag line, assembler VO speaker tag.

### 11.12 Script self-constraint validator — speakers ↔ characters_in_scene

**What breaks:** ScriptAgent's _GENERATE_SYSTEM says every dialogue
speaker must appear in that scene's `characters_in_scene`, but MIMO
sometimes emits dialogue for a speaker not in the list (observed on
v2-prologue s01: dialogue had a lu-mingfei line while chars_in_scene
was [chu-zi-hang, liu-miao-miao]). Downstream, the assembler skips the
missing speaker's @image binding — their VO lines collapse onto an
existing character's face.

**Contract (enforced in `_validate_scene_speakers` in script.py):**
after `Story.model_validate()` in both generate + refine, scan every
scene's dialogue. If a speaker_id is not in characters_in_scene,
auto-APPEND it + `log.warning()` with scene id and a hint to run
`script refine` to drop the cameo instead. Auto-include is the safer
default — MIMO decided the speaker needed a line for a reason, and
dropping silently would lose that source-text cameo.

### 11.9 VO pacing / BGM / duration — enforced deterministically, not via prompt

**What breaks:** Seedance 2.0 (and Jimeng pulling the same brief) have
three reliable failure modes when fed a schema-format production brief:

1. **All VO crammed into the final 2–3s.** Even when shots have dialogue
   assigned, the model voices them back-to-back at the scene tail with
   no emotional beat. Manual fix per-scene is not an option — we need
   pipeline-level pacing.
2. **BGM / background score bleed.** Model inserts generic emotive
   music regardless of what the brief says, poisoning the later
   `render post` mix because the "Seedance native audio" is no longer
   just dialogue+SFX.
3. **Scene runtime drift.** MIMO's shot breakdown targets ~15s but
   often returns 18–22s totals, and ignores soft density caps (≤4
   lines/shot, ≤8 lines/scene) ~30% of the time.

**Contract (enforced in `storyboard._assemble_seedance_prompt`):**

- `DialogueLine.delivery` (schema) — each line carries a short
  emotion/pacing note ("flat, dismissive", "warm, paternal, pushing").
  Required field, authored by MIMO in `_SHOTS_SYSTEM`.
- **Per-line timing windows.** Each VO line is emitted as
  `[t=5.5-6.9s] middle-aged man: 插在车门上… (casual, instructional)`.
  Windows are computed from shot start + line-index within shot, with
  a 0.5s lead-in so lines don't collide with the shot's opening action.
  Floats (`.1f`), not integer division — prevents zero-length slots on
  short shots with many lines.
- **Deterministic shot-duration clamp.** `_clamp_shot_durations(shots,
  target_sec=15, min_shot_sec=3)` rescales to ≤15s total, preserving
  proportions. Rounding overshoots are shaved from the longest shot.
- **Deterministic VO density clamp.** ≤4 lines per shot, ≤8 lines per
  scene — loose defaults; raise per-scene only when a scene is
  genuinely driven by back-and-forth dialogue (and inspect Seedance
  output for "final-2s cram" regression). Dropped lines are logged
  (`log.info("clamped N VO line(s)...")`) so the director knows. MIMO
  is also told this cap in `_SHOTS_SYSTEM` and `_REFINE_SHOTS_SYSTEM`,
  but the assembler is the safety net.
- **BGM + subtitle negation trailer.** Always appended verbatim (not
  conditionally) to every assembled prompt:
  `NO background music, NO BGM, NO soundtrack — audio is dialogue +
  ambient diegetic sound only; music is added in post-production. NO
  on-screen subtitles, NO captions, ...` Music is a post step
  (`render post --with-music` if ever added); subtitles come from the
  TTS overlay path, not burned into the video model output.
- `_COMPRESS_SYSTEM` preserves `[t=X-Ys]` timing brackets and
  `(delivery)` parentheticals verbatim when compressing for Jimeng's
  2000-char limit. If the LLM drops them, compress falls back to the
  uncompressed prompt.

**What to still watch for:**

- Even with clamps, `render package --max-chars 1800` can return
  outputs ~2500–3600 chars when scene content is dense. Treat this as
  11.5 territory — manual trim if needed before Jimeng upload.
- `_vo_role_tag` is anonymized on gender+age-bucket only (never relational
  roles like "father" / "son"). Prior keyword-based tagging false-matched
  when a character's `role_in_story` sentence mentioned the OTHER
  character's relation (e.g. son's role saying "defined by his father's
  sacrifice" → son falsely tagged as "(father)").

**Fix plan:** The clamps are already the fix. If scenes routinely exceed
the char cap after compression, extend the deterministic trimmer in 11.5
to run after compress and strip redundant style prose while preserving
the VO + shot headers + negation trailer.
