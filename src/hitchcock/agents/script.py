from __future__ import annotations

import logging
import re
from dataclasses import dataclass

from ..bible import BibleStore, Character, Location, StageName, Story
from ..llm import MimoClient

log = logging.getLogger(__name__)


_GENERATE_SYSTEM = """You are a screenwriter AND art director for a high-end animated short.
Think of yourself as an artist with unlimited time and unlimited budget —
there is NO scene you cannot render, NO camera you cannot place, NO visual
effect you cannot afford. Physical plausibility is not a constraint.

Given a source text and canonical character sheets for the cast, produce a
REAL SCREENPLAY — not a beat outline. A staff director must be able to
direct the scene from what you write, as if this were a printed screenplay.

Core principles:
  1. IMAGINATION IN SERVICE OF EMOTION. Ambition must match the scene's
     emotional register.
  2. DIALOGUE IS LOAD-BEARING AND VERBATIM. Extract dialogue directly from
     the source text when it exists. Do NOT paraphrase or summarize.
  3. ACTION IS PROSE, NOT BULLETS. Write 2-4 paragraphs of screenplay-style
     action per scene.
  4. SCENES ARE LINKED, NOT INDEPENDENT. Each scene must ARTICULATE how it
     connects to the one before and the one after via `transition_in`,
     `dramatic_turn`, `transition_out` fields.

ACTION / DIALOGUE SEPARATION — HARD RULE:

  Your schema splits the screenplay into `action` (prose) and `dialogue`
  (structured list). These two channels MUST NOT CONTAIN THE SAME LINES.

  Rules:
  - `action` describes what the CAMERA SEES between spoken lines: movement,
    blocking, micro-expressions, environmental detail. DO NOT embed dialogue
    quotations in action prose. NEVER write strings like '他说："..."' or
    'she shouts, "..."' inside `action`.
  - `dialogue` is the SOLE home of spoken lines. Each line gets its own
    entry with speaker_id + text (+ invented flag, see below).
  - When referencing a spoken line inside `action` or `dramatic_turn`,
    DESCRIBE the delivery without quoting: '他挤出一句对不起' (not
    '他说"对不起"'). The reader cross-references the `dialogue` list for the
    exact words.

DIALOGUE FIDELITY:

  Every DialogueLine has an `invented: bool` flag.
  - `invented: false` (default): the line is VERBATIM from the source text,
    OR a source-faithful load-bearing TRIM of a longer canonical line (see
    "DIALOGUE BREVITY" below). You must be able to point to the exact
    source quotation (or identify the source clause you trimmed from).
  - `invented: true`: you fabricated this line because the scene genuinely
    needed speech but the source was wordless. Use sparingly.

  NEVER silently hallucinate dialogue. If the source has no line for a
  character in a particular beat, either (a) make the beat wordless, (b)
  invent a line and mark invented:true, or (c) restructure the scene so
  the needed information lands elsewhere. NEVER write plausible-sounding
  fake dialogue and pass it off as verbatim.

DIALOGUE BREVITY — HARD RULE:

  Each scene renders as a 15-second clip with 3-4 sub-shots, giving each
  line of dialogue ~3-5 seconds of screen time. A Mandarin speaker
  averages ~5 chars/second at natural pace. So:
  - TARGET: ≤25 Chinese characters per line (~5 seconds natural speech).
  - HARD CAP: 35 characters. Longer than this WILL rush / bleed into the
    next line's slot and Seedance will garble the delivery.

  When the source dialogue is longer than 35 characters, you MUST trim
  to the LOAD-BEARING clause. Rules for trimming:
  (a) Keep the core informational / emotional content — what makes the
      plot or relationship move.
  (b) Drop filler ("哎呀"), repeated sentiments ("一会儿不要离开我，但
      也不要靠得太近。就像是小时候我带你放风筝" → trim the simile or
      the condition, keep the warning).
  (c) The REMOVED canonical flavor can go into the scene's `action` prose
      (e.g. `母亲絮絮叨叨，电话里姐妹笑声渐远` carries tone without
      quoting the full ramble).
  (d) A trimmed line stays `invented: false` — you're excerpting, not
      fabricating.

  Example (Dragon Raja s01 mother's phone call, source ~120 chars):
    ✗ Full quote:
      "子航你那里下雨了吧？哎呀妈妈在久光商厦和姐妹们一起买东西呢，这边
       也下雨了，车都打不着，我们喝杯咖啡等小点再走，你自己打个车赶快
       回家吧。你爸爸不是给你钱了吗？"
    ✓ Trimmed load-bearing (~18 chars, ~4s):
      "妈妈在久光商厦，你自己打个车回家。"
    + action carries: "电话那头母亲絮絮叨叨，姐妹的笑声渐远."

CAST INTEGRITY — HARD RULES (violations = pipeline failure):

  CHARACTER / LOCATION IDs ARE MACHINE IDENTIFIERS, NOT DISPLAY NAMES:
  The cast block gives each character BOTH an `ID` (latin lowercase
  kebab-case slug like `chu-zi-hang`) AND a `display_name` (the
  human-readable Chinese/English name like `楚子航`). In your output —
  `characters`, `characters_in_scene`, and `dialogue.speaker_id` — you
  MUST use the ID, NEVER the display_name. These ids map to
  `bible/characters/<id>/character.json` on disk; using the display
  name causes downstream agents to 404. Same rule for `scene.location_id`.

  A. `characters_in_scene` must be a strict subset of the Story's
     `characters` list (which is the cast you were given). If the source
     text has a scene whose core action requires a character NOT in the
     cast, you have THREE options:
       (1) DROP the scene entirely.
       (2) REWRITE the scene as a solo scene for the remaining cast member
           (weaker but sometimes right).
       (3) KEEP the scene but mention the absent character ONLY in the
           `action` prose as a descriptive role (e.g. "the silver-haired
           rival" / "a senior in a rumpled tuxedo") — DO NOT list them in
           characters_in_scene and DO NOT give them dialogue.
  B. NEVER substitute a different cast member to fill a slot that the
     source text gave to an absent character. If the source says "路明非
     dances with 芬格尔" and 芬格尔 is not in your cast, you do NOT write
     "路明非 dances with 诺诺" instead — that's a factual lie. Use option
     A(1), A(2), or A(3).
  C. Same rule for locations — only use location_ids from the provided
     Locations list. Other locations: describe in location_description
     instead, with location_id left null.

NARRATIVE SPINE:
  Before writing scenes, write the story's `narrative_spine` — 2-3
  sentences naming the emotional/story progression (e.g. "从滑稽的羞辱 →
  寂静的拯救 → 仓皇的逃离 → 安静的亲密 → 奇迹的盛开 → 心照不宣的告别").
  EVERY scene must visibly serve one segment of this spine.

Scope: short-form animation, 6-10 scenes (prefer more, not fewer),
each 10-20 seconds of screen time. Each scene renders as ONE 15-second
video clip downstream.

SCENE GRANULARITY — HARD RULE:
Favor MORE, SHORTER scenes over fewer long ones. Each scene should
hold ONE emotional register; if a source stretch covers multiple
pivots (e.g. anticipation → humiliation → resolve), SPLIT it into
separate scenes. Heuristics:
- If a candidate scene's `dramatic_turn` would need to name >1
  distinct emotional pivot, that's a split signal.
- A scene averaging <10 seconds of clear action isn't "too short" —
  it's fine, as long as it has ONE readable beat.
- A 25K-character source typically supports 8-10 scenes; 40K → 10-12;
  10K → 6-8. Bias UP within these ranges — user prefers granular
  emotional pacing over compressed arcs.
- Only when the source genuinely has fewer distinct beats (a quiet
  single-room dialogue, for example) should you drop below 8.

Your JSON object must have exactly these keys:

- id: lowercase-kebab-case slug derived from the title.
- title: display title.
- logline: one sentence pitch.
- synopsis: 2-4 paragraphs.
- narrative_spine: 2-3 sentences naming the dramatic through-line.

- style_guide: 100-200 words — optional short style bible. Downstream art
  direction is driven by the story's approved StyleGuide (Stage 0); this
  field is a narrative-level hint only.

- characters: list of character ids used (subset of the provided cast).

- scenes: list of scene objects with keys:
    * id: "s01", "s02", ... (zero-padded, contiguous).
    * title: short scene title (3-8 words).
    * location_id: canonical Location id if known, else null.
    * location_description: 40-80 words if location_id is null.
    * time_of_day: e.g. "night", "dusk", "blue hour".
    * emotional_register: 1-2 words.
    * characters_in_scene: list of character ids present. STRICT SUBSET OF
      the Story.characters list.
    * uncredited_presences: list of display names (e.g. ["芬格尔", "恺撒"])
      of characters visible in action prose but NOT in the cast. They get
      NO dialogue entries and NO reference images — they're descriptive
      only. This field lets us handle "the scene needs person X but they're
      not in our cast" without silently dropping them or substituting
      another cast member. Leave [] if no such characters.
    * summary: 1-2 sentences, high-level.
    * action: 2-4 PARAGRAPHS of screenplay-style action prose. Describe
      setting, character blocking, movement, micro-expressions, sensory
      detail. NO DIALOGUE QUOTATIONS inside this field — only camera-
      visible action. You may describe that a line is delivered ("他小声
      嘟哝" / "she mouths two syllables") but never quote the words.
    * dialogue: list of {"speaker_id": "<id>", "text": "<line>",
      "invented": <bool>} in DELIVERY ORDER. Verbatim from source unless
      `invented: true`.
    * beats: 3-6 short structural anchors (1 line each).

    ── CONNECTIVE TISSUE (mandatory) ──
    * transition_in: 1-2 sentences — how this scene OPENS relative to the
      previous one. Literal transition language ("hard cut", "seamless",
      "20 minutes later") PLUS the sensory/emotional handoff. First scene:
      describe the cold open. Example: "Hard cut to silence — the探戈
      music from the previous scene is gone; we're now in the Bugatti's
      quiet cabin 30 seconds after peeling out."
    * dramatic_turn: 1-2 sentences — the single pivotal beat INSIDE this
      scene where the emotional register or stakes shift. Pinpoint the
      exact line or gesture. Example: "When her breath hits his knuckle
      mid-air, he stops treating her as a classmate."
    * transition_out: 1-2 sentences — how this scene HANDS OFF to the
      next. The sensory/emotional thread the next scene picks up. Last
      scene: describe the final fade.

    * shots: leave as [] — storyboard agent fills this in.
"""


_REFINE_SYSTEM = """You are a screenwriter revising an existing screenplay draft based on
director feedback. Preserve everything the feedback doesn't touch.

You will receive:
1. The current Story JSON.
2. The director's feedback in natural language.

Return a NEW Story JSON with the same schema.

Rules:
- Preserve scene ids (s01, s02, ...) where the feedback doesn't change
  scene structure. If feedback asks to add/remove/merge scenes, renumber
  contiguously.
- Preserve dialogue verbatim unless feedback says otherwise.
- Preserve action prose unless feedback says otherwise.
- PRESERVE (and update as needed) the connective tissue fields:
  `transition_in`, `dramatic_turn`, `transition_out`, and the story-level
  `narrative_spine`. If you add/remove scenes, update the transitions of
  adjacent scenes to stay coherent.
- CAST INTEGRITY: characters_in_scene must stay a strict subset of
  Story.characters. Characters visible but not in cast go in
  `uncredited_presences`. NEVER substitute a cast member into a slot the
  source text gave to an absent character.
- ACTION / DIALOGUE SEPARATION: `action` prose must NOT contain dialogue
  quotations. Dialogue lines live only in `dialogue`. `dramatic_turn` may
  describe a line's delivery ("his reply") but must not quote the words.
- DIALOGUE FIDELITY: every DialogueLine has an `invented` bool. Verbatim
  source lines = false. Fabricated lines = true. No silent hallucinations.
- When feedback is scene-scoped ("s06 should end on tears"), ONLY modify
  that scene.
- When feedback is story-wide ("trim to 4 scenes"), rebalance evenly but
  keep each remaining scene's action/dialogue/transitions intact.

Output JSON must match the schema: per-scene `action`, `dialogue`, `beats`,
`transition_in`, `dramatic_turn`, `transition_out`; story-level
`narrative_spine`.
"""


@dataclass
class ScriptAgent:
    """Layer 1 of the Hitchcock pipeline.

    `generate(source_text)`  — first draft from source.
    `refine(story, feedback)` — MIMO revises an existing draft per feedback.

    Both write to the Script stage's pending slot via BibleStore. Caller is
    responsible for approve/reject.
    """

    llm: MimoClient
    bible: BibleStore

    def generate(
        self,
        source_text: str,
        character_ids: list[str],
        location_ids: list[str] | None = None,
        story_id: str | None = None,
    ) -> Story:
        characters = [self.bible.load_character(cid) for cid in character_ids]
        locations = [
            self.bible.load_location(lid) for lid in (location_ids or [])
        ]
        # Pull directorial brief for pacing / must-haves / must-avoids.
        from .brief import load_brief
        brief = load_brief(self.bible, story_id)
        user_prompt = _build_generate_user_prompt(
            source_text, characters, locations, brief
        )

        log.info("Script Agent: drafting structured story via MIMO...")
        data = self.llm.chat_json(
            system=_GENERATE_SYSTEM,
            user=user_prompt,
            max_tokens=16384,
            temperature=0.7,
        )
        story = Story.model_validate(data)
        story.id = _slugify(story.id or story.title)
        # Defensive: if MIMO used display_names/aliases instead of ids,
        # remap them back before validation. Otherwise storyboard will
        # 404 trying to load `bible/characters/<display_name>/...`.
        _remap_names_to_ids(story, characters, locations)
        _validate_character_refs(story, {c.id for c in characters})
        _validate_scene_speakers(story)
        log.info("  → id=%s title=%s scenes=%d",
                 story.id, story.title, len(story.scenes))
        return story

    def refine(self, story: Story, feedback: str) -> Story:
        """Revise an existing Story with natural-language feedback."""
        user_prompt = (
            f"## Current story JSON\n```json\n{story.model_dump_json(indent=2)}\n```\n\n"
            f"## Director feedback\n{feedback.strip()}\n\n"
            f"Return the revised Story JSON in the same schema."
        )
        log.info("Script Agent: refining story with feedback (%d chars)",
                 len(feedback))
        data = self.llm.chat_json(
            system=_REFINE_SYSTEM,
            user=user_prompt,
            max_tokens=16384,
            temperature=0.5,
        )
        revised = Story.model_validate(data)
        # Preserve id — the refine should not rename the story.
        revised.id = story.id
        _validate_scene_speakers(revised)
        log.info("  → scenes before=%d after=%d",
                 len(story.scenes), len(revised.scenes))
        return revised


def _build_generate_user_prompt(
    source_text: str,
    characters: list[Character],
    locations: list[Location],
    brief=None,
) -> str:
    # Cast sheets are structured so MIMO can't confuse the machine id
    # with the display name. `ID` is explicit, `display_name` is labeled
    # "do-not-use-as-id". Past bug (2026-04-22): MIMO output
    # `characters_in_scene: ["楚子航"]` instead of `["chu-zi-hang"]`,
    # causing storyboard to 404 on `bible/characters/楚子航/character.json`.
    char_sheets = "\n\n".join(
        f"### ID: {c.id}  ← USE THIS in characters/characters_in_scene/speaker_id\n"
        f"display_name: {c.name}  ← human-readable only, NEVER use as id\n"
        f"aliases (human-readable only): {', '.join(c.aliases) or '—'}\n"
        f"role: {c.role or '—'}\n"
        f"visual_description: {c.visual_description}\n"
        f"default_outfit: {c.default_outfit}\n"
        f"personality: {c.personality}\n"
        f"style_tags: {', '.join(c.style_tags)}"
        for c in characters
    )
    loc_sheets = "\n\n".join(
        f"### ID: {l.id}  ← USE THIS in scene.location_id\n"
        f"display_name: {l.name}  ← human-readable only, NEVER use as id\n"
        f"time_of_day: {l.time_of_day or '—'}\n"
        f"description: {l.description}"
        for l in locations
    ) or "(no pre-built locations; write location_description instead of location_id for each scene)"

    brief_block = ""
    if brief and brief.answers:
        a = brief.answers
        brief_block = (
            "\n## Director's brief — HIGH-PRIORITY instructions\n"
            f"Form (target length/shape): {a.form}\n"
            f"Target audience: {a.target_audience}\n"
            f"Emotional tone: {', '.join(a.emotional_tone) or '—'}\n"
            f"Pacing: {a.pacing or '—'}\n"
            f"Must include (non-negotiable): {'; '.join(a.must_haves) or '—'}\n"
            f"Must avoid (hard vetos): {'; '.join(a.must_avoids) or '—'}\n"
            "\nHonor these constraints even when they conflict with source-text emphasis. "
            "The brief IS the director's voice; the source text is raw material.\n"
        )
    return (
        f"## Source text\n{source_text.strip()}\n\n"
        f"## Cast (use these ids ONLY in characters_in_scene)\n{char_sheets}\n\n"
        f"## Locations (use these ids in scene.location_id when a scene happens there)\n{loc_sheets}"
        f"{brief_block}"
    )


def _remap_names_to_ids(
    story: Story,
    characters: list[Character],
    locations: list[Location],
) -> None:
    """Defensive post-pass: if MIMO output a display_name or alias
    instead of a canonical id, rewrite it back. Handles:

      - story.characters
      - scene.characters_in_scene
      - scene.dialogue[*].speaker_id
      - scene.location_id

    Build a lookup: {display_name, every alias} → canonical_id. If a
    field has a value that matches a name/alias instead of an id,
    remap it. Unknown values stay untouched (validator will WARN).

    Why a post-pass instead of only relying on the system prompt:
    even with explicit HARD RULES, MIMO occasionally emits Chinese
    display_names in Chinese-source runs. Silently remapping keeps
    the pipeline working without pipeline failures; the system prompt
    fix + this fallback together are defense in depth.
    """
    char_name_to_id: dict[str, str] = {}
    for c in characters:
        if c.name and c.name != c.id:
            char_name_to_id[c.name] = c.id
        for a in c.aliases or []:
            if a and a != c.id:
                char_name_to_id[a] = c.id
    loc_name_to_id: dict[str, str] = {}
    for l in locations:
        if l.name and l.name != l.id:
            loc_name_to_id[l.name] = l.id

    valid_cids = {c.id for c in characters}
    valid_lids = {l.id for l in locations}

    def fix_char(x: str) -> str:
        if x in valid_cids:
            return x
        return char_name_to_id.get(x, x)

    def fix_loc(x: str | None) -> str | None:
        if x is None or x in valid_lids:
            return x
        return loc_name_to_id.get(x, x)

    remapped = 0
    story.characters = [fix_char(c) for c in story.characters]
    for scene in story.scenes:
        scene.characters_in_scene = [fix_char(c) for c in scene.characters_in_scene]
        orig_loc = scene.location_id
        scene.location_id = fix_loc(scene.location_id)
        if orig_loc != scene.location_id:
            remapped += 1
        for d in scene.dialogue:
            if d.speaker_id not in valid_cids:
                new_id = char_name_to_id.get(d.speaker_id, d.speaker_id)
                if new_id != d.speaker_id:
                    d.speaker_id = new_id
                    remapped += 1
    if remapped:
        log.info("Script Agent: remapped %d display_name→id references", remapped)


def _validate_character_refs(story: Story, valid_ids: set[str]) -> None:
    unknown = set(story.characters) - valid_ids
    for scene in story.scenes:
        unknown.update(set(scene.characters_in_scene) - valid_ids)
    if unknown:
        log.warning("Script Agent: story references character ids not in bible: %s",
                    sorted(unknown))


def _validate_scene_speakers(story: Story) -> None:
    """Guarantee `characters_in_scene` is a superset of every dialogue
    speaker_id in that scene. MIMO sometimes gives dialogue to a character
    not listed in the scene cast (observed 2026-04-21: s01 had lu-mingfei
    dialogue while chars_in_scene was [chu-zi-hang, liu-miao-miao] only).

    Downstream, the assembler binds `@image{n} = <canonical_label>` only
    for chars_in_scene — a missing speaker means no ref image for that
    voice, so Seedance collapses them onto an existing character's face.
    That causes scene-level identity confusion.

    Fix policy: auto-INCLUDE missing speakers in characters_in_scene
    (safer than dropping the dialogue, since MIMO decided to include it
    for a reason — usually a legitimate source-text cameo). Log a WARNING
    per scene so the user sees the correction and can `script refine` to
    drop if the cameo is unwanted.
    """
    for scene in story.scenes:
        scene_chars = set(scene.characters_in_scene)
        for line in scene.dialogue:
            if line.speaker_id and line.speaker_id not in scene_chars:
                log.warning(
                    "Script Agent: auto-adding speaker %r to scene %s "
                    "characters_in_scene (MIMO gave dialogue to a char not "
                    "listed in scene cast). To drop this cameo instead, "
                    "run `script refine --scene %s --feedback '...'`.",
                    line.speaker_id, scene.id, scene.id,
                )
                scene.characters_in_scene.append(line.speaker_id)
                scene_chars.add(line.speaker_id)


def _slugify(value: str) -> str:
    s = value.strip().lower()
    s = re.sub(r"[^a-z0-9\-]+", "-", s)
    s = re.sub(r"-+", "-", s).strip("-")
    return s or "story"
