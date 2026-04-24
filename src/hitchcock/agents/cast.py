"""Cast Agent — pre-Script discovery layer (Phase 1.5).

Reads source text via MIMO, extracts characters + locations, matches against
bible (deterministic alias match done in code, not LLM), proposes new ones
with MIMO-inferred visual descriptions.

The `build` step materializes new proposals into the bible by:
  - Characters: CastAgent calls DesignAgent.build_from_character (Nano Banana
    generates 2 refs: front + three_quarter).
  - Locations: CastAgent calls LocationAgent.build_from_location (Nano Banana
    generates establishing.png).

Refine is natural-language feedback: "merge 诺诺 into chen-mo-tong", "drop 恺
撒, he doesn't have enough screen time for an identity build", "give 芬格尔
a more specific outfit".
"""
from __future__ import annotations

import logging
import re
from dataclasses import dataclass

from ..bible import (
    BibleStore,
    CastManifest,
    Character,
    CharacterProposal,
    Location,
    LocationProposal,
    MatchStatus,
    ReferenceView,
)
from ..image import NanoBananaClient
from ..llm import MimoClient
from .design import DEFAULT_VIEWS, DesignAgent
from .location import LocationAgent

log = logging.getLogger(__name__)


# ── Meta-prompts ────────────────────────────────────────────────────────

_DISCOVER_SYSTEM = """You are a dramaturg reading a source text to identify the CAST
(characters + locations) needed to stage it as an animated short.

You will receive: (1) the source text, (2) a list of bible entries already
available (id, name, aliases for characters; id + name for locations).

For each distinct NAMED character and each distinct NAMED location referenced
in the source, produce a proposal. Match against the existing bible list
where possible — if the source calls someone '诺诺' and the bible has a
character with name '诺诺' or that alias, it's a match.

Rules:
- NAMED characters only. Background crowd or uncredited minors: skip. If in
  doubt whether a name-dropped figure needs an identity build, prefer YES
  only if they speak or act on screen more than twice.
- Canonical_id is a lowercase-kebab-case slug. Chinese names → pinyin.
  Examples: "诺诺" → "chen-mo-tong" (if that's her full name) or
  "nuonuo" (if nickname is canonical). PREFER matching an existing bible id
  over inventing a new slug.
- aliases list every form the source uses: nicknames, full names,
  honorifics, roles ("师兄XX", "队长").
- For match_status: you propose best-guess ("new" vs "in_bible"). The system
  will verify against the bible list — trust code-side matching more than
  yours. Always populate the visual fields because code may reclassify you.

- THREE IDENTITY FIELDS ARE MANDATORY for every character (never leave blank):
  * age — numeric or bucket ("14", "~40", "mid-teens"). Downstream T2I
    buckets into child/teen/young/middle-aged/elderly; missing age pulls
    the image toward a generic ~25 adult.
  * gender — "male" / "female" / "nonbinary". Missing gender makes body
    shape ambiguous and the model flips silhouette.
  * ethnicity — REQUIRED free-text phrase like "East Asian (Han Chinese)",
    "African American", "Nordic white European", "Southeast Asian
    (Vietnamese)". State it as a LABEL. Missing ethnicity defaults the
    T2I model to Western Caucasian priors — a Chinese character is then
    rendered as a generic Western actor. Always derive this from source
    text context or brief canon; if genuinely ambiguous, pick the most
    plausible and note the uncertainty in visual_description.

- HARD BAN on stereotyped phenotype enumeration. The ethnicity label
  ("East Asian (Han Chinese)") carries identity on its own — do NOT
  ALSO list reductive racial features to "reinforce" it. Banned phrases
  (these produce caricatures, not individuals):
    * "flatter midface" / "broad flat midface"
    * "monolid eyes" / "almond-shaped eyes" as an ethnic marker
      (fine if describing ONE specific character's actual eye shape
      distinct from same-ethnicity peers — but never as a generic
      "Chinese people have X" filler)
    * "high cheekbones" as a racial descriptor
    * "unmistakably Han Chinese face" / "clearly Asian face"
    * any "NOT Caucasian / NOT Western / NOT mixed-race / NOT Latino"
      style negation lists — the positive ethnicity label suffices
    * generic "ivory / warm-ivory skin tone" as an Asian marker
  Describe what makes THIS SPECIFIC character visually distinct (hair
  color/style, eye color, scars, posture, expression, build). If the
  character looks like an ordinary person of their stated ethnicity,
  say so ("ordinary Chinese teenager") and move on — no feature list.

- visual_description: 60-100 words, LITERAL, costume-designer style.
  Hair / eyes (color only, no shape-as-ethnic-marker) / build /
  distinguishing marks / posture. No metaphor. The FIRST sentence
  must restate age + gender + ethnicity LABEL (e.g. "A 14-year-old
  Chinese schoolgirl...") so the identity clause survives even if
  downstream drops the subject_kind prefix — but keep it to ONE short
  clause, don't pile on phenotype descriptors.

- voice_description: 2-3 sentences on voice timbre + delivery cadence +
  personality flavor. Used by the storyboard assembler to inject a
  "speaker cues" block into the Seedance video prompt so the character's
  voice stays consistent across scenes. Without it, Seedance reinvents
  timbre per clip. Rules:
  * Include: approximate register (低沉 baritone / 清脆 / 沙哑 / 略尖),
    cadence (语速偏快 / 稳 / 拖长), distinctive speech habit
    (啰嗦自嘲 / 克制少言 / 话带哄劝), gender-age voice cue
    (未变声 / 刚变声 / 成熟男低音 / 温软女声 / 苍老).
  * 中文 for Chinese characters (matches Seedance's Mandarin-native
    speaker modeling), English for English-source stories.
  * Examples:
    - 中年男人（父亲）: "中年男性声线，低沉 warm baritone 略带沙哑。
      语速不快但字句清楚，讲话带明显的啰嗦和自嘲感——一句话能
      绕两个弯；情急时转为干脆利落的命令式，声音压低不拔高。"
    - 少年（16岁）: "少年男性声线，已完成变声但还带一丝未稳的
      青涩。平时话少克制、语速偏慢、几乎不带情绪；真正紧张时
      会收到几乎压住的急促短句，不喊不嘶吼。"
    - 少女（14岁初中女）: "少女声线，清脆略带怯弱，尾音偏软。
      语速中等，紧张时变快；对话里常带'嗯'、'哦'这样的小迟疑。"
- default_outfit: 40-80 words, LITERAL. Garments + named colors + 1-2
  accessories. No mood words.
- role: 1 line describing narrative role.
- style_tags: 2-4 short rendering words (e.g. ["cinematic", "painterly"]).

For LOCATIONS: same logic — only named distinct places that host an on-
screen scene. description: 50-100 words, material/light/architecture,
literal.

Your JSON must have exactly these keys:
{
  "characters": [CharacterProposal, ...],
  "locations":  [LocationProposal, ...]
}

CharacterProposal keys:
  canonical_id, display_name, aliases (list), role, age, gender, ethnicity,
  visual_description, voice_description, default_outfit, personality,
  style_tags, match_status ("in_bible" | "new"), matched_bible_id (id or null)

LocationProposal keys:
  canonical_id, display_name, aliases, description, time_of_day,
  match_status, matched_bible_id
"""


_REFINE_SYSTEM = """You are revising a cast manifest (characters + locations)
based on director feedback. Preserve entries the feedback doesn't touch.

Typical feedback: "merge A into B", "drop X", "make Y's outfit more
specific", "split the ballroom location into two (entrance hall + dance
floor)". Apply surgically.

HARD RULES for every CharacterProposal returned (including untouched ones):
- age, gender, ethnicity, voice_description are MANDATORY. If a
  pre-existing entry is missing any of them, fill from context (source
  text, canon, or the feedback) rather than passing through empty.
- ethnicity is free text LABEL form — "East Asian (Han Chinese)" /
  "African American" / "Nordic white European". State it as a LABEL,
  not as a feature list.
- voice_description is 2-3 sentences covering timbre + cadence +
  speech habit (see generate-path rules); match Chinese sources with
  Chinese descriptions.
- The first sentence of visual_description must restate age + gender +
  ethnicity LABEL in ONE short clause (e.g. "A 14-year-old Chinese
  schoolgirl..."). Don't pad with phenotype descriptors.
- HARD BAN on stereotyped phenotype enumeration: do NOT emit
  "flatter midface", "monolid eyes", "high cheekbones", "unmistakably
  Han Chinese face", "NOT Caucasian/Western/mixed-race" negation lists,
  "ivory skin as Asian marker", or any list of racial features dressed
  up as description. These produce caricatures, not individuals. The
  positive ethnicity label carries identity; describe what makes THIS
  specific character distinct (hair, eye color, build, scars, posture).
- If you're revising an existing entry and the old visual_description
  contains banned phrases (from a prior stereotyped draft), REWRITE it
  to drop the phenotype enumeration while preserving genuine character-
  specific details (hair color/style, eye color, build, outfit).

Return the same JSON schema as the input — keys: characters, locations.
"""


# ── Agent ───────────────────────────────────────────────────────────────

@dataclass
class CastAgent:
    llm: MimoClient
    images: NanoBananaClient
    bible: BibleStore

    # ── MIMO: discover from source ──────────────────────────────────────
    def discover(self, source_text: str, story_id: str) -> CastManifest:
        """Extract cast from source text. Matches against bible afterward."""
        bible_chars = self.bible.list_characters()
        bible_locs = self.bible.list_locations()

        char_catalog = "\n".join(
            f"- {c.id} (name: {c.name}, aliases: {c.aliases})"
            for c in bible_chars
        ) or "(empty)"
        loc_catalog = "\n".join(
            f"- {loc.id} (name: {loc.name})" for loc in bible_locs
        ) or "(empty)"

        # Pull canon facts from the approved brief (if any) so MIMO gets the
        # wider source context, not just this chapter. E.g. Chapter 6 of
        # Dragon Raja never says 诺诺's hair is red — but canon says so,
        # and the brief research captured that.
        from .brief import load_brief
        brief = load_brief(self.bible, story_id)
        canon_context = ""
        if brief and brief.canon_facts:
            canon_context = (
                "\n## Canon facts (from wider source / fan wiki — use these "
                "verbatim for named characters even if not in this chapter's text)\n"
            )
            for f in brief.canon_facts:
                canon_context += (
                    f"- {f.character_alias}:\n"
                    f"    appearance: {f.canonical_appearance}\n"
                    f"    personality: {f.canonical_personality}\n"
                    f"    role: {f.canonical_role}\n"
                )
            if brief.world_canon:
                canon_context += f"\n## World canon\n{brief.world_canon}\n"

        user_prompt = (
            "## Source text\n" + source_text.strip()[:8000] + "\n\n"
            "## Bible characters (already exist)\n" + char_catalog + "\n\n"
            "## Bible locations (already exist)\n" + loc_catalog
            + canon_context
        )

        log.info("Cast Agent: discovering characters + locations via MIMO%s...",
                 " (with canon)" if canon_context else "")
        data = self.llm.chat_json(
            system=_DISCOVER_SYSTEM, user=user_prompt,
            max_tokens=12288, temperature=0.4,
        )

        # Validate + reclassify match_status using deterministic bible lookup.
        chars = [CharacterProposal.model_validate(c) for c in data.get("characters", [])]
        locs = [LocationProposal.model_validate(l) for l in data.get("locations", [])]
        for cp in chars:
            self._reclassify_character(cp)
        for lp in locs:
            self._reclassify_location(lp)

        # Dedupe by canonical_id (MIMO occasionally lists the same person twice)
        chars = _dedupe_by_id(chars)
        locs = _dedupe_by_id(locs)

        manifest = CastManifest(story_id=story_id, characters=chars, locations=locs)
        log.info("Cast Agent: %d characters (%d new), %d locations (%d new)",
                 len(chars), sum(1 for c in chars if c.match_status == MatchStatus.NEW),
                 len(locs), sum(1 for l in locs if l.match_status == MatchStatus.NEW))
        return manifest

    # ── MIMO: refine with feedback ──────────────────────────────────────
    def refine(self, manifest: CastManifest, feedback: str) -> CastManifest:
        import json as _json
        user_prompt = (
            "## Current cast manifest\n```json\n"
            + _json.dumps(
                {
                    "characters": [c.model_dump(mode="json") for c in manifest.characters],
                    "locations":  [l.model_dump(mode="json") for l in manifest.locations],
                },
                ensure_ascii=False, indent=2,
            )
            + "\n```\n\n## Director feedback\n" + feedback.strip()
        )
        data = self.llm.chat_json(
            system=_REFINE_SYSTEM, user=user_prompt,
            max_tokens=12288, temperature=0.4,
        )
        chars = [CharacterProposal.model_validate(c) for c in data.get("characters", [])]
        locs = [LocationProposal.model_validate(l) for l in data.get("locations", [])]
        for cp in chars:
            self._reclassify_character(cp)
        for lp in locs:
            self._reclassify_location(lp)
        chars = _dedupe_by_id(chars)
        locs = _dedupe_by_id(locs)
        return CastManifest(story_id=manifest.story_id, characters=chars, locations=locs)

    # ── Deterministic: build NEW proposals into bible ───────────────────
    def build(
        self,
        manifest: CastManifest,
        *,
        only: list[str] | None = None,
        views: list[ReferenceView] | None = None,
        skip_refs: bool = False,
        dry_run: bool = False,
        story_id: str | None = None,
        force: bool = False,
    ) -> dict:
        """Materialize proposals into bible. Returns a summary dict.

        - only: optional list of canonical_ids to build (all NEW if None;
          when --force is set, all if None).
        - views: reference views to generate per character (default 2).
        - skip_refs: if True, only write character.json/location.json (no images).
        - dry_run: print plan, don't call APIs.
        - force: rebuild even if match_status is in_bible/in_bible_by_alias.
          Used to restyle existing characters under a new approved StyleGuide
          (without --force, in_bible proposals are skipped as a no-op).
        """
        design_agent = DesignAgent(llm=self.llm, images=self.images, bible=self.bible)
        location_agent = LocationAgent(llm=self.llm, images=self.images, bible=self.bible)
        views = views or DEFAULT_VIEWS

        plan = {"characters_to_build": [], "locations_to_build": [], "skipped": []}
        for cp in manifest.characters:
            if not force and cp.match_status != MatchStatus.NEW:
                plan["skipped"].append(f"character:{cp.canonical_id} ({cp.match_status.value})")
                continue
            if only and cp.canonical_id not in only \
                    and (cp.matched_bible_id or "") not in only:
                plan["skipped"].append(f"character:{cp.canonical_id} (not in --only)")
                continue
            plan["characters_to_build"].append(cp.canonical_id)
        for lp in manifest.locations:
            if not force and lp.match_status != MatchStatus.NEW:
                plan["skipped"].append(f"location:{lp.canonical_id} ({lp.match_status.value})")
                continue
            if only and lp.canonical_id not in only \
                    and (lp.matched_bible_id or "") not in only:
                plan["skipped"].append(f"location:{lp.canonical_id} (not in --only)")
                continue
            plan["locations_to_build"].append(lp.canonical_id)

        img_cost = 0.4  # rough RMB per image
        n_char_imgs = 0 if skip_refs else len(plan["characters_to_build"]) * len(views)
        n_loc_imgs = 0 if skip_refs else len(plan["locations_to_build"])  # 1 establishing each
        plan["estimated_cost_rmb"] = round((n_char_imgs + n_loc_imgs) * img_cost, 2)
        plan["images_to_generate"] = n_char_imgs + n_loc_imgs

        if dry_run:
            return plan

        # Load brief canon for alias → CharacterCanon lookup (if brief approved).
        # A force-rebuild pulls canon into the Character's visual_description
        # so the new refs reflect source-wider canon (e.g. 诺诺 red hair) that
        # this chapter may not mention.
        from .brief import load_brief
        brief = load_brief(self.bible, story_id)
        canon_by_alias: dict = {}
        if brief and brief.canon_facts:
            for f in brief.canon_facts:
                canon_by_alias[f.character_alias] = f

        # Build characters
        built_chars: list[str] = []
        for cp in manifest.characters:
            if cp.canonical_id not in plan["characters_to_build"]:
                continue

            # Build a Character for T2I. Layering (innermost wins):
            #   1. bible character.json  — identity (id, aliases, voice_id, backstory)
            #   2. brief canon_facts     — initial appearance seed from Gemini research
            #   3. cast manifest (cp)    — user-refined appearance (via `cast refine`)
            #
            # Cast-manifest edits ALWAYS win for visual_description /
            # default_outfit, because that's the surface `cast refine`
            # writes to. Canon is only a first-boot seed: applied when
            # the cast proposal has no meaningful appearance yet (e.g.
            # just after `cast discover` populated a stub).
            if force and cp.match_status != MatchStatus.NEW and cp.matched_bible_id:
                character = self.bible.load_character(cp.matched_bible_id)
                # Cast manifest wins when it has a real description.
                cp_has_desc = bool(cp.visual_description and len(cp.visual_description.strip()) >= 50)
                if cp_has_desc:
                    character.visual_description = cp.visual_description
                    character.default_outfit = cp.default_outfit or character.default_outfit
                    if cp.voice_description:
                        character.voice_description = cp.voice_description
                    log.info("Cast: %s using cast-manifest appearance (%d chars, user-refined)",
                             character.id, len(cp.visual_description))
                else:
                    # No user-refined description → fall back to brief canon seed.
                    canon = _match_canon(canon_by_alias, character)
                    if canon and canon.canonical_appearance:
                        character.visual_description = canon.canonical_appearance
                        # Canon text usually includes wardrobe ("tuxedo", "gown"),
                        # so clear default_outfit to avoid contradiction in T2I.
                        character.default_outfit = ""
                        log.info("Cast: %s got canon overlay (%d chars, first build seed)",
                                 character.id, len(canon.canonical_appearance))
            else:
                character = _proposal_to_character(cp)

            if skip_refs:
                self.bible.save_character(character)
            else:
                design_agent.build_from_character(
                    character, views=views, story_id=story_id,
                )
            built_chars.append(character.id)
            log.info("Cast: built character %s", character.id)

        # Build locations
        built_locs: list[str] = []
        for lp in manifest.locations:
            if lp.canonical_id not in plan["locations_to_build"]:
                continue
            location = _proposal_to_location(lp)
            if skip_refs:
                self.bible.save_location(location)
            else:
                location_agent.build_from_location(location, story_id=story_id)
            built_locs.append(lp.canonical_id)
            log.info("Cast: built location %s", lp.canonical_id)

        plan["built_characters"] = built_chars
        plan["built_locations"] = built_locs
        return plan

    # ── Deterministic match against bible (NOT via MIMO) ────────────────
    def _reclassify_character(self, cp: CharacterProposal) -> None:
        # Try id first, then display_name, then each alias.
        for probe in [cp.canonical_id, cp.display_name] + list(cp.aliases):
            match = self.bible.find_character_match(probe)
            if match is not None:
                matched_by_alias = (match.id.lower() != cp.canonical_id.lower())
                cp.match_status = (
                    MatchStatus.IN_BIBLE_BY_ALIAS if matched_by_alias
                    else MatchStatus.IN_BIBLE
                )
                cp.matched_bible_id = match.id
                return
        # No match.
        cp.match_status = MatchStatus.NEW
        cp.matched_bible_id = None
        # Ensure canonical_id is valid slug.
        cp.canonical_id = _slugify(cp.canonical_id or cp.display_name)

    def _reclassify_location(self, lp: LocationProposal) -> None:
        for probe in [lp.canonical_id, lp.display_name] + list(lp.aliases):
            match = self.bible.find_location_match(probe)
            if match is not None:
                matched_by_alias = (match.id.lower() != lp.canonical_id.lower())
                lp.match_status = (
                    MatchStatus.IN_BIBLE_BY_ALIAS if matched_by_alias
                    else MatchStatus.IN_BIBLE
                )
                lp.matched_bible_id = match.id
                return
        lp.match_status = MatchStatus.NEW
        lp.matched_bible_id = None
        lp.canonical_id = _slugify(lp.canonical_id or lp.display_name)


# ── helpers ─────────────────────────────────────────────────────────────

def _proposal_to_character(cp: CharacterProposal) -> Character:
    return Character(
        id=cp.canonical_id,
        name=cp.display_name,
        aliases=cp.aliases,
        age=cp.age,
        gender=cp.gender,
        ethnicity=cp.ethnicity or None,  # cp uses "" default; Character uses None
        role=cp.role,
        visual_description=cp.visual_description,
        voice_description=cp.voice_description or "",
        default_outfit=cp.default_outfit,
        personality=cp.personality,
        style_tags=cp.style_tags,
    )


def _proposal_to_location(lp: LocationProposal) -> Location:
    return Location(
        id=lp.canonical_id,
        name=lp.display_name,
        description=lp.description,
        time_of_day=lp.time_of_day,
    )


def _match_canon(canon_by_alias: dict, character) -> object | None:
    """Best-effort lookup of a CharacterCanon for a bible Character.
    Tries id, name, aliases (case-insensitive)."""
    probes = [character.id, character.name] + list(character.aliases)
    for p in probes:
        if not p:
            continue
        for alias, canon in canon_by_alias.items():
            if p.lower() == alias.lower():
                return canon
    return None


def _dedupe_by_id(items: list) -> list:
    seen: set[str] = set()
    out = []
    for x in items:
        if x.canonical_id in seen:
            continue
        seen.add(x.canonical_id)
        out.append(x)
    return out


def _slugify(value: str) -> str:
    s = value.strip().lower()
    s = re.sub(r"[^a-z0-9\-]+", "-", s)
    s = re.sub(r"-+", "-", s).strip("-")
    return s or "entry"
