"""Style Agent — Phase 1.6. Produces the canonical StyleGuide for a story.

One story, one StyleGuide. Every downstream image/video call appends the
guide's `global_style_prompt` verbatim — replacing the four hardcoded
_ARCANE_STYLE constants that used to live in design.py, location.py,
scene_art.py, and storyboard.py.

Workflow:
  hitchcock style generate -s <sid>                  # MIMO proposes
  hitchcock style show     -s <sid>                  # review
  hitchcock style refine   -s <sid> --feedback "..." # iterate
  hitchcock style approve  -s <sid>                  # lock
"""
from __future__ import annotations

import logging
from dataclasses import dataclass

from ..bible import BibleStore, StyleGuide
from ..llm import MimoClient

log = logging.getLogger(__name__)


_GENERATE_SYSTEM = """You are an art director picking the visual language for an
animated short. You receive the source text. Propose a cohesive style that
serves the source's emotional register — not a generic "pretty" style.

# HARD RULE — DO NOT drift toward Japanese anime / manga

Unless the `art_direction_anchor` is EXPLICITLY anime (Studio Ghibli,
Makoto Shinkai, Kyoto Animation, mainstream TV anime feature, etc.),
every field you write (`global_style_prompt`, `chinese_anchor`,
`character_anchor_prompt`, `environment_anchor_prompt`) MUST avoid
anime/manga tropes — T2I models have a strong anime prior and any of
these phrases will pull the output into manga territory:

FORBIDDEN phrases for non-anime targets (Arcane / Spider-Verse / Moebius
/ Lasaine / comic-book / painted Western animation / ink landscape):
- "large expressive eyes" / "big eyes" / "sparkling eyes" / "starry eyes"
- "slim" / "slender" / "androgynous" (unless the style genuinely needs it)
- "smooth gradient skin" / "soft skin shading" / "rosy cheeks"
- "flowing hair" / "detailed hair strands" / individual hair strokes
- "kawaii" / "cute" / "idol" / "bishounen" / "bishoujo" / "pretty boy"

For Arcane / Fortiche-style targets specifically, PREFER:
- eyes: "medium-to-small eyes with flat iris color, wide-set, no sparkle
  highlights" (Arcane characters do NOT have anime-large eyes)
- proportions: "heavy-boned adult build with angular bone structure
  (prominent cheekbones, jaw, brow), stocky or chunky silhouette — NOT
  slim anime proportions"
- skin: "block-shaded skin planes with hard shadow/light transitions,
  no smooth gradients, no sub-surface-scattering softness"
- hair: "hair as block-shaded solid shapes, simplified into large bold
  chunks — NOT individual detailed strands"

Default assumption for Western painted-animation targets is REALIST
ADULT with angular bone-structure, NOT anime-teen. Only relax these
restrictions if the author explicitly picks an anime anchor.

Your JSON has exactly these keys:

- art_direction_anchor: 1 named reference (artist, studio, or specific film)
  whose look is the target. Pick something with a clear, unambiguous
  painterly identity. Examples:
    "Arcane by Fortiche Studio"   (painted brushwork over 3D, teal+amber)
    "Studio Ghibli feature"       (Kazuo Oga backgrounds, soft watercolor edge)
    "Makoto Shinkai"              (hyper-real lighting, saturated skies)
    "Paul Lasaine matte painting" (cinematic wide landscape tradition)
    "Moebius / Jean Giraud"       (line art + flat color + dry brush)
    "Song-dynasty shan shui 宋代山水" (water-ink landscape with restraint)
  Pick what serves the STORY. A quiet grief-story does not want Arcane's
  high-contrast chiaroscuro; a battle epic does not want Ghibli's soft edges.

- palette: 3-5 named tones with LAYERED SATURATION. Specify which tones
  anchor the DEEP SHADOWS (cool, low-saturation) vs which pop as
  SATURATED HIGHLIGHTS (warm, high-saturation). This is how Arcane /
  Spider-Verse / Blade Runner 2049 look painterly-cinematic instead of
  washed-out: deep shadows + bright saturated highlights, almost no
  mid-grey.

  CRITICAL: palette describes ONLY how colors RELATE (cool shadow /
  warm highlight, deep anchor / saturated pop, cool desaturated /
  warm saturated). It does NOT name scene objects or light fixtures.
  Example phrasings:
    ✅ "deep cool-desaturated shadow anchor; warm high-saturation
       highlight pop; cool/warm chromatic juxtaposition"
    ❌ "hot amber STREETLAMP hotspots, magenta NEON accents, cold
       cobalt MOONLIGHT" — these are SCENE OBJECTS; Nano Banana will
       then paint streetlamps, neon signs, and moons in every frame
       regardless of what the shot actually contains. Style is HOW,
       not WHAT.
    ❌ "muted teal, bone white, oxidized silver" — all mid-value, no
       contrast anchor; T2I reads as 'make everything mid-grey'.
  FORBIDDEN palette adjectives: `muted`, `soft`, `subdued`, `pastel`,
  `washed`, `faded`, `dusty`. These pull T2I toward desaturated grey.
  FORBIDDEN palette content: any scene object (streetlamps, neon
  signs, moonlight, headlights, firelight, windows, etc.). Those go
  in the scene description, not the style palette.

- lighting_model: describes the VALUE RANGE RELATIONSHIP and cool/warm
  TEMPERATURE RELATIONSHIP only — not specific light sources. State
  whether the target look is HIGH-contrast (deep shadows + bright
  highlights, no mid-grey) or LOW-contrast (overcast, flat).
  Example phrasings:
    ✅ "high-contrast cinematic value range — deep cool shadows, warm
       saturated highlights, no mid-grey wash; strong warm/cool
       juxtaposition; rim-lit silhouettes where subjects meet light"
    ❌ "hot amber sodium-lamp hotspots against near-black sky" — again
       scene-object content leaks into style.
    ❌ "overcast flat diffuse" / "soft even ambient light" — kill
       Arcane's signature punch.
  FORBIDDEN lighting content: named light fixtures (streetlamp, neon,
  candle, moon). Lighting_model is a RELATIONSHIP spec, not a scene
  description. Scene-specific light sources come from the shot's
  action prose + the scene-level location description.
  FORBIDDEN lighting adjectives: `overcast flat diffuse`, `soft even`,
  `gentle wash`, `ambient evenly-lit`.

- texture_materials: how surfaces render — brushwork density, grain, cloth
  physics, metal patina, water behavior. What the eye sees AT the surface.

- recurring_motifs: list of 2-3 visual motifs unique to THIS film that
  should appear across scenes (e.g. ["small drifting silver particles",
  "one window always catching warm light no matter the time of day"]).

- avoid: list of 2-4 things the film must NOT look like. Be specific
  (e.g. ["anime", "gacha cute", "photorealistic", "smooth digital illustration"]).
  (Note: these `avoid` items are for human reference / QA — they do NOT
  go into the T2I prompt as "NOT X" negations, which Nano Banana / Gemini
  ignore. The `global_style_prompt` below should phrase everything
  POSITIVELY.)

- global_style_prompt: 3-5 sentences, ENGLISH, ready-to-concatenate before
  every downstream T2I prompt. FRAME this as ANIMATION FILM first, not
  fine-art painting. T2I models have an extremely strong "oil painting"
  prior — if `oil-brush`, `impasto`, `oil painting`, `brushwork` are the
  LEADING words, the model produces 19th-century realist oil portraits,
  NOT Fortiche/Arcane animation. Keep "painted animation" / "painted
  texture on 3D" in front; mention brushwork as secondary texture
  quality, not as the primary medium.
  Must include (in this order):
  (a) PRIMARY FRAMING: "stylized cel-shaded painted animation"
      + studio/style anchor (e.g. "Fortiche Studio / Arcane look"),
  (b) FORM LANGUAGE: geometric 3D character models, angular stylized
      features, non-photorealistic proportions, block-out silhouettes,
      hard crisp edges (counteract the "round portrait-realist face" drift),
  (c) SHADING: crisp cel-shaded hard-edge shadows, flat dark planes
      abutting flat light planes, NO smooth gradients,
  (d) TEXTURE (secondary): hand-painted texture overlays ON the 3D
      form (not an oil-on-canvas surface), visible brushstroke quality,
  (e) palette with LAYERED SATURATION spec (dark anchor + saturated
      hotspots, cool shadows / warm highlights),
  (f) lighting: high-contrast cinematic chiaroscuro (no mid-grey wash).
  Do NOT include `NOT X` / `not photorealistic` / `no anime` lists —
  T2I models ignore negation. Phrase everything positively.
  FORBIDDEN words (these pull the model toward fine-art oil painting):
    `oil-brush`, `impasto`, `oil painting`, `oil on canvas`,
    `painterly realist`, `fine-art painting`, `Sargent`, `Bastien-Lepage`.
    Use `painted animation` / `painted texture` / `animation brushwork`
    / `hand-painted stylized` instead.
  Example (Arcane target — STYLE ONLY, NO SCENE OBJECTS):
    "Stylized cel-shaded painted animation in the Fortiche Studio /
    Arcane feature-film style. Characters are geometric 3D models with
    angular stylized bone-structure (prominent cheekbones, jaw, brow),
    non-photorealistic chunky heavy-boned adult proportions,
    medium-to-small eyes with flat iris color (NOT anime-large), and
    block-out silhouettes with hard crisp edges. Shading is cel-shaded with flat dark planes
    abutting flat light planes — hard edges, no smooth gradient
    transitions. Hand-painted texture overlays sit ON the 3D form
    (not an oil-on-canvas surface), with visible animation-brushstroke
    quality as a secondary texture pass. BOLD HIGH-CONTRAST cinematic
    value range: deep cool near-black shadows alongside warm near-
    white saturated highlights, no mid-grey wash. Selective saturation:
    shadows cool and low-saturation, light regions warm and high-
    saturation — cinematic warm/cool juxtaposition. Dramatic
    chiaroscuro lighting with strong silhouette readability."
  Notice: NO streetlamps / neon / moonlight / headlights / window
  mentions — the prompt is scene-agnostic. Those go in shot prose.

- chinese_anchor: dense MANDARIN style descriptor (~200-280 chars) for
  Seedance VIDEO prompts. Same rules as global_style_prompt but in
  Chinese, and slightly longer — the extra length pays off because
  Seedance's style anchor text carries more weight when there's only
  one reference keyframe per clip (see memory
  `feedback_video_single_ref.md`). Must include:
  (a) art_direction_anchor in Chinese (`Arcane 动画风格（Fortiche Studio）`),
  (b) palette with **明暗对比** (keep pigment names like `phthalo teal`
      as-is — Seedance reads them),
  (c) **明确高对比 value range** — "近纯黑 + 近纯白，禁中灰雾、没有
      中灰过渡",
  (d) texture 关键词 — 厚涂 impasto / 笔触方向与肌理 / 材质由笔触塑造而
      非平滑过渡,
  (e) FORM 语言 — 块面化造型 / 硬边锐利轮廓 / 几何化设计强化体积感
      （对抗 T2V 把角色画成光滑圆润卡通的倾向）,
  (f) 构图 + 光影戏剧性 — 电影级构图 + 戏剧性光影,
  (g) 禁止词反向防御 — 禁平涂数字渐变 / 禁柔光过渡.
  禁止 `muted` / `柔和` / `漫射` / `均匀` 等让模型拉灰的词。
  Example (STYLE ONLY, NO SCENE OBJECTS):
    "Arcane 动画风格（Fortiche Studio），厚涂油画 impasto 笔触在 3D
    体积上清晰可见，笔触方向与肌理明确，材质由笔触塑造而非平滑过
    渡；硬边 block shading + 干脆色阶（禁平涂数字渐变、禁柔光过
    渡），所有表面有手绘颗粒质感。块面化造型，硬边锐利轮廓，几何
    化设计强化体积感。强明暗对比：近纯黑阴影 + 近纯白高光，没有
    中灰过渡、没有中灰雾。选择性饱和：暗部冷静低饱和，亮部暖烈高
    饱和；冷暖对撞，阴影偏冷、光源偏暖。电影级构图 + 戏剧性光影。
    画意不写实、非光滑数字喷绘、非 3D 渲染光泽。"
  禁止出现 scene 对象：街灯 / 霓虹 / 月光 / 窗户 / 车灯 / 火光 等。
  这些交给 shot prose 写，style 只描述"怎么画"不描述"画什么"。
  This is separate from global_style_prompt (which targets T2I in
  English); both must coexist and describe the SAME high-contrast look.

- character_anchor_prompt: 80-120 words, ENGLISH. A concise, self-
  contained T2I prompt that — when rendered by GPT Image alone — will
  produce ONE clean CHARACTER STYLE ANCHOR image. Downstream
  `DesignAgent` renders this prompt once at `style approve` time, saves
  the output to `bible/stories/<sid>/style/anchor_character.png`, and
  passes it as `reference_images[0]` to every `cast build` portrait
  call. Pure style anchor — no specific identity, no narrative.
  Must include:
  (a) Opening frame: "Style reference card for <art_direction_anchor>
      painted-animation style.",
  (b) Subject: a single anonymous character bust (young adult, neutral
      age/gender/ethnicity — the downstream prompt replaces identity),
      three-quarter view, neutral expression, plain painted background,
  (c) FORM language: "geometric 3D head with angular stylized
      bone-structure (prominent cheekbones, jaw, brow), medium-to-small
      eyes with flat iris color (wide-set, no sparkle highlights — NOT
      anime-large), hand-painted texture overlays on the 3D form",
  (d) SHADING: "crisp cel-shaded block shading with hard edges — flat
      dark planes abutting flat light planes, no smooth gradients",
  (e) PALETTE: "cool teal/slate shadows, warm amber/gold highlights,
      high-contrast cinematic lighting",
  (f) EXPLICIT ANTI-OFFICIAL-ART: "Pure style anchor: no specific
      identity, no text, no logos, no narrative".
  Example (for an Arcane target):
    "Style reference card for Fortiche Studio / Arcane painted-
    animation style. A single stylized anonymous young-adult character
    bust, three-quarter view, neutral expression, on a plain warm-gray
    painted background. Geometric 3D head with angular stylized
    bone-structure (prominent cheekbones, jaw, brow), medium-to-small
    eyes with flat iris color (wide-set, no sparkle highlights — NOT
    anime-large), hand-painted texture overlays on the 3D form. Crisp cel-shaded block shading with hard edges — flat
    dark planes abutting flat light planes, no smooth gradients. Cool
    teal/slate shadows, warm amber/gold highlights, high-contrast
    cinematic lighting. Pure style anchor: no specific identity, no
    text, no logos, no narrative."

- environment_anchor_prompt: 80-120 words, ENGLISH. Parallel purpose
  for ENVIRONMENT STYLE ANCHOR. Saved as
  `bible/stories/<sid>/style/anchor_environment.png`, passed as
  `reference_images[0]` to every `art generate` scene-keyframe call
  BEFORE character refs.
  Must include:
  (a) Opening frame: "Style reference card for
      <art_direction_anchor> painted-animation style environment art.",
  (b) Subject: an anonymous abstract architectural fragment
      (stairwell / corner / archway / alley opening) — NO specific
      landmark, NO identifiable location, NO signage,
  (c) MULTI-MATERIAL surface study (stone, brick, metal, wood — each
      with distinct brushwork) so style transfer is robust across
      downstream scenes,
  (d) SHADING + PALETTE parallel to character_anchor_prompt,
  (e) LIGHTING: "single off-frame warm light source, deep cool
      shadows, hard-edged light/shadow boundaries (no atmospheric
      softening)",
  (f) EXPLICIT ANTI: "Pure style anchor: no specific location, no
      characters, no signage, no text, no narrative".
  Example (for an Arcane target):
    "Style reference card for Fortiche Studio / Arcane painted-
    animation style environment art. An anonymous abstract
    architectural fragment (stairwell / corner / archway / alley
    opening) with geometric block-out forms, hand-built angular
    stylization, and hand-painted texture overlays on 3D surfaces
    (stone, brick, metal, wood — each with distinct brushwork). Crisp
    cel-shaded block shading with hard edges — flat dark planes
    abutting flat light planes, no smooth gradients, no atmospheric
    haze softening the shadow/light boundary. Deep cool teal/slate
    shadows, warm amber/gold light hot-spots from a single off-frame
    light source. High-contrast cinematic moody lighting, silhouette
    readability. Pure style anchor: no specific location, no
    characters, no signage, no text, no narrative."

  Both anchor prompts must use the SAME `art_direction_anchor` + palette
  + shading language as `global_style_prompt` and `chinese_anchor` —
  they're all facets of the SAME look, rendered as text prompts for
  different downstream consumers (T2I scene prose vs T2V Chinese vs
  pure-style T2I anchor image).
"""


_REFINE_SYSTEM = """You are revising a StyleGuide based on director feedback.
Preserve fields the feedback doesn't touch; revise what it does.

Typical feedback: "switch from Arcane to水墨 shan-shui", "less saturation,
more bone-white negative space", "drop the amber — this story wants cold-only
palette", "make the brushwork rougher".

Apply the SAME HARD RULES as the generate path (above):
- palette must specify LAYERED SATURATION (dark anchor + saturated
  hotspots), not a flat list of mid-value descriptors.
- lighting_model must state EXPLICIT VALUE RANGE (high-contrast vs
  flat) and name PRACTICAL HOTSPOT light sources.
- FORBIDDEN adjectives in palette/lighting/global_style_prompt/
  chinese_anchor: `muted`, `soft`, `subdued`, `pastel`, `washed`,
  `faded`, `dusty`, `overcast flat diffuse`, `soft even`, `gentle
  wash`, `ambient evenly-lit`, `柔和`, `漫射`, `均匀` — these words
  pull Nano Banana / Seedance toward desaturated mid-grey output.
  Phrase what SATURATES, not what's subdued.
- global_style_prompt and chinese_anchor BOTH must include the full
  rule set: anchor + palette spec + value-range + TEXTURE with
  directional brushwork + BLOCK-OUT GEOMETRIC FORM language + hard
  silhouette edges + CINEMATIC COMPOSITION + dramatic chiaroscuro.
  When revising, preserve these categories — don't drop form or
  composition language even if the feedback only mentions palette.
- global_style_prompt and chinese_anchor are BOTH output (English for
  T2I, Chinese for Seedance). Never let them drift — they describe
  the SAME high-contrast look in two languages.
- chinese_anchor target length ~200-280 chars (style weight matters
  more under single-ref video architecture — see
  `feedback_video_single_ref.md`).
- character_anchor_prompt and environment_anchor_prompt MUST exist
  (80-120 words each, English). When revising other style fields,
  keep these two anchor prompts in sync — they describe the same
  look in concise form for anchor-image rendering.
- No NOT-lists in global_style_prompt (Gemini ignores negation).

If the input StyleGuide uses forbidden muted descriptors, REWRITE them
proactively (not just the user-requested change). Muted → saturated
hotspot layering is a defect-fix, not preference.

Return JSON with the same keys as input.
"""


@dataclass
class StyleAgent:
    """Stateless. BibleStore is unused internally but kept in signature for
    symmetry with other agents."""

    llm: MimoClient
    bible: BibleStore

    def generate(self, source_text: str, story_id: str) -> StyleGuide:
        # If a DirectorialBrief has been approved, use its style_references,
        # emotional_tone, and must_avoids to bias the art direction.
        from .brief import load_brief
        brief = load_brief(self.bible, story_id)
        brief_context = ""
        if brief and brief.answers:
            a = brief.answers
            brief_context = (
                "\n## Director's brief (HIGH-PRIORITY user intent — align to this)\n"
                f"Target form: {a.form}\n"
                f"Style references (user picks these — anchor closely): {', '.join(a.style_references) or '—'}\n"
                f"Emotional tone: {', '.join(a.emotional_tone) or '—'}\n"
                f"Must avoid: {', '.join(a.must_avoids) or '—'}\n"
                f"Music direction hint: {a.music_direction or '—'}\n"
            )
        user_prompt = brief_context + "\n## Source text\n" + source_text.strip()[:10000]
        log.info("Style Agent: proposing art direction via MIMO%s...",
                 " (with brief)" if brief_context else "")
        data = self.llm.chat_json(
            system=_GENERATE_SYSTEM, user=user_prompt,
            max_tokens=3072, temperature=0.5,
        )
        data["story_id"] = story_id
        guide = StyleGuide.model_validate(data)
        log.info("  → anchor=%s motifs=%d avoid=%d",
                 guide.art_direction_anchor,
                 len(guide.recurring_motifs), len(guide.avoid))
        return guide

    def refine(self, guide: StyleGuide, feedback: str) -> StyleGuide:
        user_prompt = (
            "## Current StyleGuide JSON\n```json\n"
            + guide.model_dump_json(indent=2)
            + "\n```\n\n## Director feedback\n" + feedback.strip()
        )
        data = self.llm.chat_json(
            system=_REFINE_SYSTEM, user=user_prompt,
            max_tokens=3072, temperature=0.4,
        )
        data["story_id"] = guide.story_id
        revised = StyleGuide.model_validate(data)
        return revised


# ── Helpers for other agents ────────────────────────────────────────────

DEFAULT_STYLE_FALLBACK = (
    "Painted animation frame in a cinematic painterly style — block-shaded "
    "forms with visible hand-painted brushwork on 3D volumes, adult "
    "realistic human proportions, restrained muted palette, chiaroscuro "
    "lighting. NOT anime, NOT photorealistic, NOT idol-cute, NOT gacha, "
    "NOT mobile-game illustration."
)

# Appended to every T2I call's style prompt (portraits, locations, scene
# keyframes). Keeps output Seedance-animable: fewer fine details per
# pixel = less per-frame drift when the keyframe is fed to I2V. The
# `preserve the story's art style` + `simplification applies to SHAPE,
# not quality` clauses are load-bearing — without them the model
# flattens toward cheap cartoon / degrades aesthetic quality.
# STYLE-AGNOSTIC: no brand/anchor names here so this rule is portable
# across stories that use different art directions (Arcane, watercolor,
# vector-flat, oil, anime, etc.). 27 words. Do NOT expand; crowds the
# style prompt budget.
_GRAPHIC_SIMPLIFICATION_EN = (
    " Graphic simplification: bigger flat color planes, simplified "
    "forms, fewer fine details, bolder silhouettes — preserve the "
    "story's art style (palette, brushwork, materials). "
    "Simplification applies to SHAPE, not quality."
)

# Chinese parallel for the Seedance 画风 line — kept under ~50 CJK chars
# for the compact assembler shape.
_GRAPHIC_SIMPLIFICATION_CN = (
    "画面简化优先：放大色块、简化形体、减少细节密度、强化剪影；"
    "保留本片既定的风格质感（色调、笔触、材质）。"
)


def load_style_prompt(bible: BibleStore, story_id: str | None) -> str:
    """Resolve the style prompt to concatenate in an image-prompt build.

    Priority:
      1. If `story_id` is given and that story has an approved StyleGuide,
         return its `global_style_prompt`.
      2. Otherwise return `DEFAULT_STYLE_FALLBACK` (used for standalone
         `hitchcock design` / `hitchcock location` calls that aren't
         scoped to a story).
    """
    if story_id is None:
        return DEFAULT_STYLE_FALLBACK + _GRAPHIC_SIMPLIFICATION_EN
    from ..bible import StageName
    try:
        approved = bible.load_approved(story_id, StageName.STYLE)
    except (FileNotFoundError, ValueError, OSError) as e:
        log.warning(
            "load_style: %s has no approved StyleGuide (%s: %s) — "
            "falling back to DEFAULT_STYLE_FALLBACK", story_id, type(e).__name__, e,
        )
        approved = None
    if approved is None:
        return DEFAULT_STYLE_FALLBACK + _GRAPHIC_SIMPLIFICATION_EN
    assert isinstance(approved, StyleGuide)
    return approved.global_style_prompt + _GRAPHIC_SIMPLIFICATION_EN


def load_style_anchor_compact(bible: BibleStore, story_id: str | None) -> str:
    """Compact Chinese style line for Seedance video prompts.

    Replaces the older long `chinese_anchor` paragraph (palette / lighting /
    impasto / negations) with a short line that trusts the keyframe
    reference image to carry actual style. Shape:

        `{art_direction_anchor}. 艺术风格保持与参考图一致。专业构图、运镜、动作和画面质量。`

    Rationale (hand-edit test 2026-04-23): Seedance receives the scene
    keyframe (and optional per-shot keyframes) as `reference_image`
    inputs that already encode the painted-animation look. A verbose
    Chinese style recitation was redundant at best and burned prompt
    budget at worst; the compact form preserves the style-anchor NAME
    (so Seedance knows WHICH style) while offloading the specifics onto
    the refs themselves."""
    generic = (
        "艺术风格保持与参考图一致。专业构图、运镜、动作和画面质量。"
        + _GRAPHIC_SIMPLIFICATION_CN
    )
    if story_id is None:
        return generic
    from ..bible import StageName
    try:
        approved = bible.load_approved(story_id, StageName.STYLE)
    except (FileNotFoundError, ValueError, OSError) as e:
        log.warning(
            "load_style: %s has no approved StyleGuide (%s: %s) — "
            "falling back to generic compact anchor",
            story_id, type(e).__name__, e,
        )
        approved = None
    if approved is None:
        return generic
    assert isinstance(approved, StyleGuide)
    anchor = (approved.art_direction_anchor or "").strip()
    if anchor:
        return (
            f"{anchor}。艺术风格保持与参考图一致。"
            f"专业构图、运镜、动作和画面质量。"
            + _GRAPHIC_SIMPLIFICATION_CN
        )
    return generic


