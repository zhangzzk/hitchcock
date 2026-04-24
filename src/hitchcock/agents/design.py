from __future__ import annotations

import logging
import re
from dataclasses import dataclass

from ..bible import BibleStore, Character, ReferenceImage, ReferenceView
from ..image import NanoBananaClient
from ..llm import MimoClient
from .style import load_style_prompt

log = logging.getLogger(__name__)


# ── prompts ─────────────────────────────────────────────────────────────

_STRUCTURE_SYSTEM = """You are writing the canonical character sheet for an
animated short. Two readers will use this sheet:

  * Downstream AGENTS (script / storyboard / narrative) — they use the
    personality, backstory, and narrative fields.
  * A text-to-image MODEL — it uses ONLY `visual_description` and
    `default_outfit`, and takes every word LITERALLY.

Because the image model takes words literally, visual fields MUST be
literal. No metaphors, no narrative framing, no "tells a story of", no
"refracts moonlight into slits", no "records a life lived in shadowed halls",
no "material/texture signature". If you write "her jacket is frayed from
rope abrasion," the model will draw literal flames on her shoulders. If you
write "salt crystals in her hair," it will draw literal crystals. Treat
these fields like a police-report description of what someone is wearing
right now.

Save the poetic and narrative richness for `personality` and `backstory`,
which the image model never sees.

Your JSON object must have exactly these keys:

- id: lowercase-kebab-case slug. Chinese names → pinyin, e.g. "云汐" → "yun-xi".
- name: display name (original script preserved).
- aliases: list of alternate names (may be empty).
- age: string (e.g. "teen", "early twenties", "16").
- gender: string.
- role: narrative role (e.g. "protagonist", "mentor", "rival").

- visual_description: 70-110 words. LITERAL only — list what is visibly on
  the body as a costume designer would. Cover in this order, terse phrases,
  no metaphor:
    hair (color, length, style, one adjective for texture),
    eyes (color, shape — e.g. "almond", "round"),
    face shape + any skin note (pale, tan, freckled),
    build (height relative term + body type),
    distinguishing marks (scars, moles — plain sight only).
  Example good: "Long waist-length pale silver hair, straight, one streak
  of pale blue from left temple to tip. Pale gray-blue eyes, almond-shaped.
  Oval face, porcelain skin, subtle freckles across the nose. Slim, 160cm.
  Thin pale scar above left eyebrow."

- default_outfit: 50-90 words. LITERAL only. Name garments, colors (use
  specific tone words, not adjectives about mood), and ONE or TWO accessories.
  No wear-pattern narratives, no "tells a story", no fiber chemistry.
  Example good: "Navy sailor-collar wool blazer with brass buttons, white
  cotton shirt underneath, knee-length charcoal pleated skirt, black
  knee-high socks, black leather ankle boots. Thin silver chain with a small
  silver fish pendant at the throat."

- other_outfits: list of alternate outfits (may be empty), same literal style.

- personality: 2-4 sentences. THIS is where the narrative richness goes.

- backstory: 2-4 sentences. Rich.

- style_tags: list of 2-4 short technical words/phrases describing the
  RENDERING STYLE, not world elements. Examples of GOOD tags:
    ["cel-shaded", "painterly", "restrained palette", "cinematic lighting"]
  Examples of BAD tags (cause literal pollution):
    ["Makoto Shinkai skies" → model paints literal skies everywhere],
    ["salt spray in hair" → literal crystals],
    ["oxidized silver highlights on metalwork" → literal metal objects appear].
  When in doubt, use generic technique words, not named references.
"""

# ── image prompt construction ──────────────────────────────────────────
# MIMO is no longer used to write per-view T2I prompts: it tends to drift
# into poetic padding and re-introduce anime-adjacent style cues. Prompts
# are built in Python from the Character sheet + these constants, so the
# style and background are locked across every character.

_BG_NEGATION = (
    "PLAIN SOFT WARM-GRAY PAINTED STUDIO BACKGROUND with subtle oil-paint "
    "brushwork texture on the wall. No sky, no moon, no stars, no clouds, "
    "no nebula, no landscape, no outdoor elements, no decorative sparkles, "
    "no crystal effects, no glowing particles in the air, no dramatic "
    "weather, no magical lighting."
)

# For mythological / fantasy / supernatural characters, the mortal portrait
# template above strips away the character's defining visual elements
# (cloak, weapon, mount, divine aura, stormy environment). Use a scenic
# alternative that PRESERVES context while still keeping the figure as
# the clear focus. Applied when `_is_fantasy_character(c)` returns True.
_BG_FANTASY = (
    "Cinematic full-body character-design portrait painted against an "
    "atmospheric scenic backdrop appropriate to the character's mythology "
    "(stormy sky, mist, or otherworldly negative space — whatever the "
    "visual_description implies). The figure is the single clear subject, "
    "centrally placed, with negative space around them so identity reads "
    "cleanly. Preserve all props, weapons, mounts, cloaks, and divine-aura "
    "elements described in the visual_description — these are identity-"
    "defining, not decorative. No text, no logos, no HUD elements, no "
    "floating particles UNLESS explicitly described as a mythological "
    "effect the character emits."
)

_VIEW_COMPOSITIONS_FANTASY: dict[ReferenceView, str] = {
    ReferenceView.FRONT: (
        "Full-body front-facing character-design portrait. The mythological "
        "figure is centered (or slightly off-center), posed in a dignified "
        "divine / heroic stance — standing or mounted as the "
        "visual_description specifies, facing camera, clearly displaying "
        "all identity-defining props (weapons, cloaks, mounts, aura). "
        "Dramatic directional light appropriate to the figure's mythology "
        "rims the silhouette for readability."
    ),
    ReferenceView.THREE_QUARTER: (
        "Full-body three-quarter character-design portrait. Figure turned "
        "roughly 30 degrees, showing profile of cloak / weapon / mount if "
        "present. Dramatic rim-light for silhouette readability."
    ),
    ReferenceView.SIDE: (
        "Full-body side profile character-design portrait. Silhouette "
        "emphasized — cloaks, capes, and weapons cut bold shapes against "
        "the atmospheric scenic backdrop."
    ),
    ReferenceView.EXPRESSIONS: (
        "A 2x2 expression sheet: four head-and-shoulders painted portraits "
        "of the same mythological figure on a single canvas, identical "
        "props and collar, showing four expressions appropriate to the "
        "character (e.g. solemn, wrathful, pitying, ancient) — the emotion "
        "lives in the face, not in the palette."
    ),
}


def _is_fantasy_character(c: Character) -> bool:
    """Is this a mythological / divine / supernatural character who should
    be rendered with props + atmospheric backdrop instead of the mortal-
    portrait studio template? Detected via ethnicity field (most
    reliable — MIMO cast discover emits 'mythological / Norse-divine'
    for gods) or visual_description keywords (cloak + weapon + mount +
    aura pattern)."""
    eth = (getattr(c, "ethnicity", None) or "").lower()
    if any(k in eth for k in (
        "mythological", "divine", "immortal", "supernatural", "deity",
        "god-figure", "demon", "spirit"
    )):
        return True
    desc = (c.visual_description or "").lower()
    # Cloak + (weapon OR mount OR aura) together signals mythological.
    has_cloak = "cloak" in desc or "robe" in desc or "cape" in desc
    has_weapon = any(k in desc for k in (
        "spear", "sword", "staff", "halberd", "scythe", "gungnir",
        "mjolnir", "bow and arrow"
    ))
    has_mount = any(k in desc for k in (
        "mounted on", "riding", "steed", "stallion", "sleipnir", "dragon"
    ))
    has_aura = any(k in desc for k in (
        "divine aura", "divine light", "glowing aura", "glowing with",
        "halo", "radiant", "emits a soft"
    ))
    return has_cloak and (has_weapon or has_mount or has_aura)

# NOTE: art direction is no longer hardcoded here. The StyleAgent produces a
# per-story StyleGuide whose `global_style_prompt` is appended by every
# prompt builder via `load_style_prompt(bible, story_id)`. For standalone
# calls (bible-level `hitchcock design`), the DEFAULT_STYLE_FALLBACK from
# agents/style.py is used.

_VIEW_COMPOSITIONS: dict[ReferenceView, str] = {
    ReferenceView.FRONT: (
        "Full-body front view portrait. Figure standing relaxed, placed just "
        "right of center, looking at camera. Warm directional window light "
        "from the upper-left casts a soft shaft of pale gold across the wall "
        "behind her and a gentle rim on her left side. Generous negative "
        "space on the left of the frame."
    ),
    ReferenceView.THREE_QUARTER: (
        "Full-body three-quarter angle portrait. Figure turned roughly 30 "
        "degrees to her left, weight on one foot, arms relaxed. Placed in "
        "the right third of the frame with empty painted wall filling the "
        "left. Same warm directional window light from the upper-left."
    ),
    ReferenceView.SIDE: (
        "Full-body side profile portrait. Figure facing right, placed in the "
        "left third of the frame, the rest of the frame an empty wall "
        "catching a soft warm glow. Back-light rims the silhouette from "
        "behind; the face in partial shadow, dignified and still."
    ),
    ReferenceView.EXPRESSIONS: (
        "A 2x2 expression sheet: four head-and-shoulders painted portraits "
        "of the same character on a single canvas, identical hair and "
        "clothing collar, showing four expressions — top-left neutral, "
        "top-right a soft genuine smile, bottom-left controlled anger, "
        "bottom-right quiet grief. Each portrait is a tight study painted "
        "with real human facial anatomy; the emotion lives in the face, "
        "not in the palette."
    ),
}


def _build_prompt(
    view: ReferenceView, character: Character, style_prompt: str,
) -> str:
    # Gender + age must go first in the Subject block — without them the
    # image model defaults to its own priors and e.g. draws a young man in
    # a tuxedo as a woman because the body-shape words alone are ambiguous.
    subject_kind = _subject_kind(character)
    # Pick the composition + background template. Mortal characters use
    # the mortal-portrait studio template (clean studio, no props). Gods
    # / mythological figures need scenic context preserving cloak /
    # weapon / mount / aura — the mortal template strips those and
    # renders e.g. Odin as a modern young adult in black casuals.
    if _is_fantasy_character(character):
        comp = _VIEW_COMPOSITIONS_FANTASY[view]
        bg = _BG_FANTASY
    else:
        comp = _VIEW_COMPOSITIONS[view]
        bg = _BG_NEGATION
    return (
        f"{comp} "
        f"{bg} "
        f"Subject: {subject_kind}. {character.visual_description} "
        f"Wearing: {character.default_outfit} "
        f"{style_prompt}"
    )


def _subject_kind(c: Character) -> str:
    """Convert the character's ethnicity/gender/age fields into an explicit
    identity clause the T2I model will respect — e.g.
    'An East Asian (Han Chinese) teenage girl (14 years old)'.

    All THREE fields matter:
      - age → without it the model's prior sets the apparent age arbitrarily
        (a 40-year-old father got rendered as a 30ish face when only "male"
        was passed; a 14-year-old got drawn as a small child when only
        'delicate and petite' was in the visual description).
      - gender → without it body-shape words are ambiguous (a young man in
        a tuxedo was rendered as a woman).
      - ethnicity → T2I models default to Caucasian priors; without an
        explicit ethnicity the Chinese characters in this story were
        rendered as Western actors. See
        `feedback_cast_force_canon_override.md` for the prior debugging pass.

    Bucket age into: child / teen / young / middle-aged / elderly. Ethnicity
    is free text (e.g. 'East Asian (Han Chinese)', 'African American',
    'Southeast Asian', 'Nordic European') and goes up-front as an adjective
    so the model reads it first."""
    gender = (c.gender or "").strip().lower()
    is_male = gender in {"male", "m", "man", "boy"}
    is_female = gender in {"female", "f", "woman", "girl"}
    # Parse age. MIMO cast discover emits fuzzy formats — "~40", "40s",
    # "mid-40s", "late forties", "early-20s" — so extract the FIRST
    # integer in the string instead of requiring a clean int. Without
    # this, "~40" → int() raises ValueError → age fallback 25 → noun
    # bucket becomes "young man" → a 40yo father gets prompted as a
    # young man (see `feedback_cast_force_canon_override.md` round 4).
    import re as _re_age
    age_raw = (c.age or "").strip()
    m = _re_age.search(r"\d{1,3}", age_raw)
    if m:
        age_int = int(m.group(0))
    else:
        # Fall back to semantic keywords.
        lower = age_raw.lower()
        # Mythological / divine / immortal beings → elderly bucket (60+)
        # so T2I prompts them as "elderly man / woman" instead of falling
        # through to the young-man default. Odin's age="ancient" was
        # silently rendered as "young man (ancient years old)", producing
        # a modern-looking young-adult portrait instead of a Norse god.
        mythic_keywords = (
            "ancient", "timeless", "immortal", "eternal",
            "god", "divine", "deity", "mythological", "primordial",
        )
        if any(k in lower for k in mythic_keywords):
            age_int = 70  # treat as elderly / bearded sage archetype
        else:
            word_decades = {
                "teens": 15, "twenties": 25, "thirties": 35,
                "forties": 45, "fifties": 55, "sixties": 65,
                "seventies": 75, "eighties": 85, "nineties": 95,
            }
            age_int = 25
            for word, val in word_decades.items():
                if word in lower:
                    age_int = val
                    break
    if age_int < 13:
        noun = "boy" if is_male else ("girl" if is_female else "child")
    elif age_int < 20:
        noun = "teenage boy" if is_male else ("teenage girl" if is_female else "teenager")
    elif age_int < 35:
        noun = "young man" if is_male else ("young woman" if is_female else "young person")
    elif age_int < 60:
        noun = "middle-aged man" if is_male else ("middle-aged woman" if is_female else "middle-aged person")
    else:
        noun = "elderly man" if is_male else ("elderly woman" if is_female else "elderly person")
    age = (c.age or "").strip()
    ethnicity = (getattr(c, "ethnicity", None) or "").strip()

    # Compose: "A [ethnicity] [noun] ([age] years old)"
    # e.g. "An East Asian (Han Chinese) teenage girl (14 years old)"
    article = "An" if ethnicity and ethnicity[0].lower() in "aeiou" else "A"
    parts = [article]
    if ethnicity:
        parts.append(ethnicity)
    parts.append(noun)
    head = " ".join(parts)
    return f"{head} ({age} years old)" if age else head


# ── agent ───────────────────────────────────────────────────────────────

# Seedream 4.5 requires ≥ 3,686,400 pixels per image. All sizes below clear that.
# Landscape refs align with the video target aspect ratio downstream.
_VIEW_SIZES: dict[ReferenceView, tuple[int, int]] = {
    ReferenceView.FRONT: (2304, 1728),          # 4:3 landscape
    ReferenceView.THREE_QUARTER: (2304, 1728),  # 4:3 landscape
    ReferenceView.SIDE: (2304, 1728),           # 4:3 landscape
    ReferenceView.EXPRESSIONS: (2048, 2048),    # 1:1 square, 2x2 grid
}


# Default reference views for cost efficiency. Front-only is sufficient
# for downstream scene-art identity anchoring — Nano Banana composes the
# character into new shots from the front ref alone, and generating a
# second view doubled the cost without measurably improving consistency.
# Full set [front, three_quarter, side, expressions] still available via
# `views=[...]` passed explicitly.
DEFAULT_VIEWS: list[ReferenceView] = [
    ReferenceView.FRONT,
]


def _build_prompts_for_views(
    character: Character, views: list[ReferenceView], style_prompt: str,
) -> dict[str, str]:
    return {v.value: _build_prompt(v, character, style_prompt) for v in views}


@dataclass
class DesignAgent:
    llm: MimoClient
    images: NanoBananaClient
    bible: BibleStore

    def design(
        self,
        description: str,
        views: list[ReferenceView] | None = None,
        story_id: str | None = None,
    ) -> Character:
        """Run the full Design flow: description → Character + ref images.

        `views` controls which reference images are generated. Default is
        [front, three_quarter] (2 images ≈ ¥0.8). Pass `list(ReferenceView)`
        for the full 4-view set.

        `story_id`: if provided, DesignAgent pulls the story's approved
        StyleGuide to style the refs. Without it, falls back to the default
        painterly style.
        """
        log.info("Design Agent: structuring character sheet via MIMO...")
        sheet = self.llm.chat_json(
            system=_STRUCTURE_SYSTEM,
            user=description.strip(),
            max_tokens=4096,
        )
        character = Character.model_validate(sheet)
        character.id = _slugify(character.id or character.name)
        log.info("  → id=%s name=%s", character.id, character.name)

        self._build_refs(character, views or DEFAULT_VIEWS, story_id)
        self.bible.save_character(character)
        log.info("Design Agent: saved %s", self.bible.character_json(character.id))
        return character

    def build_from_character(
        self,
        character: Character,
        views: list[ReferenceView] | None = None,
        story_id: str | None = None,
    ) -> Character:
        """Build reference images for a Character that's already been
        structured (e.g. by CastAgent). Skips MIMO; only runs Nano Banana."""
        self._build_refs(character, views or DEFAULT_VIEWS, story_id)
        self.bible.save_character(character)
        return character

    def _build_refs(
        self,
        character: Character,
        views: list[ReferenceView],
        story_id: str | None,
    ) -> None:
        style_prompt = load_style_prompt(self.bible, story_id)
        prompts = _build_prompts_for_views(character, views, style_prompt)
        refs_dir = self.bible.refs_dir(character.id)
        refs_dir.mkdir(parents=True, exist_ok=True)

        # Portraits are generated TEXT-ONLY — no style-anchor reference
        # image is passed. Previously we piped `style/anchor_character.png`
        # in as `reference_images[0]` to lock the painted-animation style,
        # but empirically gpt-image-2 blended the anchor's hairstyle /
        # face / wardrobe into every character portrait, collapsing cast
        # identity. Carrying style through the text `style_prompt` is
        # sufficient for consistency across views (see
        # `feedback_design_agent_consistency`).
        references: list[ReferenceImage] = []
        for view in views:
            prompt = prompts[view.value]
            w, h = _VIEW_SIZES[view]
            log.info("  → generating %s (%dx%d)...", view.value, w, h)
            img = self.images.generate(
                prompt, width=w, height=h,
                reference_images=None,
            )
            out = refs_dir / f"{view.value}.png"
            img.save(out)
            references.append(
                ReferenceImage(
                    view=view,
                    path=str(out.relative_to(self.bible.root)),
                    prompt=img.prompt_used,
                    width=w,
                    height=h,
                )
            )
        character.references = references


# ── helpers ─────────────────────────────────────────────────────────────

def _slugify(value: str) -> str:
    s = value.strip().lower()
    s = re.sub(r"[^a-z0-9\-]+", "-", s)
    s = re.sub(r"-+", "-", s).strip("-")
    return s or "character"
