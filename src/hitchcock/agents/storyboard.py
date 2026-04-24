"""Storyboard Agent — Layer 2 of the Hitchcock pipeline.

Responsibilities:
1. Break each Script scene into SHOTS (MIMO, structured list).
2. Author a v6-recipe-compliant scene_art_prompt per scene (MIMO).
3. Deterministically assemble a schema-format seedance_prompt per scene.

Refines are per-scene: `refine_scene(storyboard, scene_id, feedback)` only
re-runs MIMO for that scene's shots + scene_art_prompt + rebuilds seedance
prompt. Other scenes are untouched.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass

from ..bible import (
    BibleStore,
    CameraMovement,
    Character,
    Scene,
    Shot,
    ShotType,
    Story,
    Storyboard,
    StoryboardScene,
)
from ..llm import MimoClient
from .style import load_style_prompt


_COMPRESS_SYSTEM = """You are compressing a Seedance / Jimeng video production brief to
fit under a character cap, AND scrubbing it for IP-safety.

You receive (1) the current brief, (2) the target character cap. THE OUTPUT
MUST BE STRICTLY <= target_chars. Count characters. If in doubt, err small.

DIALOGUE FIDELITY — HARD RULE (do not violate):
- If the input brief HAS a VO block, preserve every line verbatim.
- If the input brief has NO VO block (no "VO (...)" section), the output
  MUST ALSO have no VO block. **DO NOT FABRICATE** dialogue, even if the
  scene "feels like it needs something said." Silent scenes stay silent.
- Never invent VO from action prose. Never infer unstated lines.

PRIORITY ORDER (when budget is tight, drop from the BOTTOM of this list):
  1. [MUST KEEP] @image/@video/@audio tag binding line.
  2. [MUST KEEP] VO block IF AND ONLY IF input has one; otherwise OMIT entirely.
  3. [MUST KEEP] Shot headers: "Shot N (Xs, <type>, hard cut at Ys)".
  4. [KEEP, TIGHT] Environmental SCENE — compress to 30-45 words of
     concrete visual facts (architecture, lighting, weather, signature props).
     NOT "a building at night" but NOT full original paragraph either.
  5. [KEEP, TIGHT] SCENE OPENING / SCENE HANDOFF — each one 1 short
     clause max (e.g. "hard cut from silent ballroom to chaotic exterior").
  6. [TIGHT] Shot action prose — 1-2 short sentences per shot. One sentence
     if possible. Focus: camera placement + ONE visible character action.
  7. [TIGHT] Character summaries — one 10-15 word role phrase per char.
  8. [DROP IF NEEDED] Style prose — can be "Style: <anchor-only>".

LANGUAGE UNITY:
The output action/scene/character prose must be written in ONE language
(pick English). Do NOT mix Chinese tokens into English sentences (no
"Lu Mingfei's 茫然的脸" or "burning 校门" — translate or drop). VO
dialogue IS exempt (those stay in original language, verbatim).

IP-SAFETY ANONYMIZATION:
Replace character NAMES in prose with role tags:
- "路明非" / "Lu Mingfei" → "the driver" / "the young man"
- "诺诺" / "陈墨瞳" → "the passenger" / "the young woman"
- "路鸣泽" → "the mysterious boy"
- Other named chars → descriptive roles.
Proper-noun brands (Bugatti Veyron) → descriptive ("silver two-seat supercar").

OUTPUT STRUCTURE (in this order):
  1. @tag line
  2. Title + duration header
  3. Style line (short anchor only)
  4. SCENE: 40-60 words of environmental density
  5. SCENE OPENING: 1 sentence (from input's transition_in if present)
  6. CHARACTERS: short role summaries
  7. Shot 1 (...): action
  8. Shot 2 (...): action
  9. ... (more shots)
 10. SCENE HANDOFF: 1 sentence (from input's transition_out if present)
 11. VO block IFF input had VO (otherwise omit this line entirely)

IP-SAFETY ANONYMIZATION (hard rule):
- Remove ALL specific character NAMES from prose (action descriptions,
  character summaries, scene descriptions). Substitute role tags:
    - "路明非" / "Lu Mingfei" → "the driver" / "the young man"
    - "诺诺" / "陈墨瞳" / "Chen Mo-tong" → "the passenger" / "the young woman"
    - "路鸣泽" / "Lu Mingze" → "the mysterious boy" / "the junior-high boy"
    - Any other named character → a descriptive role
- VO speaker tags stay as "young man (driver)" / "young woman (passenger)"
  — never name-tagged.
- VO text itself is verbatim — do NOT scrub Chinese dialogue content even
  if names appear inside (e.g. "今天是我生日" stays as-is).
- If a proper-noun brand is mentioned (e.g. "Bugatti Veyron"), replace
  with a descriptive equivalent ("a silver two-seat supercar").

DROP these sections entirely:
- BGM / music / soundtrack / mood cue lines. BGM is added in a later
  post-production stage, NEVER by the video model. The video model MUST
  NOT generate background music.

VO FORMAT FIDELITY — HARD RULE:
If the input VO block uses per-line timing markers ("[t=3-5s] ...") or
per-line delivery notes ("(cold, flat)"), you MUST preserve them verbatim.
These markers are what prevent the video model from dumping all dialogue
in the final 2 seconds. Do NOT collapse "[t=3-5s] man (father): 这车…
(warm paternal boast)" into "man (father): 这车…". The timing and the
delivery parenthetical are both load-bearing.

MANDATORY NEGATIVE INSTRUCTION (append exactly once, near the end of the
compressed output, before any trailing newline):
"NO background music, NO BGM, NO soundtrack — audio is dialogue +
ambient diegetic sound only; music is added in post-production.
NO on-screen subtitles, NO captions, NO burned-in dialogue overlays,
NO text graphics, NO on-video dialogue transcription. Dialogue is audio-
only (natively voiced by the video model). Subtitles will be added as a
separate post-processing step."

The video model (Seedance / Jimeng) sometimes burns dialogue text onto
the frame when it sees VO blocks in the prompt, and generates stock BGM
when it sees no explicit negation. The above negation suppresses both.

Your JSON output has exactly one key:
  compressed: the rewritten brief as a single string, <= target_chars,
              fully anonymized, no BGM line.
No code fences, no other keys.
"""


# Dictionary of common Chinese tokens that leak into Nano Banana
# keyframe_prompts. Deterministic substitution pre-pass eliminates the
# vast majority of leaks without any MIMO call (observed 2026-04-21:
# rapid MIMO scrub calls rate-limit → 5min hangs per call; dict sub
# catches 80%+ of tokens, MIMO only needed for stragglers).
_CJK_KEYFRAME_DICT: dict[str, str] = {
    # Canonical labels
    "少年甲": "the first teenager",
    "少年乙": "the second teenager",
    "少年丙": "the third teenager",
    "少年": "the teenager",
    "中年男人": "the middle-aged man",
    "少女甲": "the first young girl",
    "少女乙": "the second young girl",
    "少女": "the young girl",
    "男孩": "the boy",
    "女孩": "the girl",
    "人物甲": "the first character",
    "人物乙": "the second character",
    "人物丙": "the third character",
    # Clothing
    "罩衫": "jacket",
    "格子围巾": "checkered scarf",
    "围巾": "scarf",
    "校服": "school uniform",
    "制服": "uniform",
    "西装": "suit",
    "风衣": "trench coat",
    "外套": "coat",
    "T恤": "T-shirt",
    "衬衫": "shirt",
    # Props / objects
    "珐琅吊灯": "enamel pendant lamp",
    "吊灯": "pendant lamp",
    "操场": "schoolyard",
    "教室": "classroom",
    "黑板": "blackboard",
    "讲台": "lectern",
    "课桌": "desk",
    "手提箱": "briefcase",
    "雨伞": "umbrella",
    "雨刷器": "windshield wipers",
    "仪表盘": "dashboard",
    "后视镜": "rearview mirror",
    "方向盘": "steering wheel",
    "长枪": "spear",
    "宝剑": "sword",
    "日本刀": "katana",
    "刀": "blade",
    "剑": "sword",
    "枪": "gun",
    "盾": "shield",
    "弓": "bow",
    "箭": "arrow",
    # Body parts that occasionally leak as single chars
    "发梢": "hair tip",
    "梢": "tip",
    "发": "hair",
    "眼": "eyes",
    "手": "hand",
    "脸": "face",
    # Colors / atmosphere
    "苍白": "pale",
    "白茫茫": "white hazy",
    "漆黑": "pitch-black",
    "冷光": "cold light",
    "暖光": "warm light",
    "水银色": "mercury-silver",
    "琥珀色": "amber",
    "金色": "gold",
    # Verbs / motion
    "小跑": "trotting",
    "挥刀": "swinging a blade",
    "斩杀": "cutting down",
    "屹立": "standing firm",
    "发动": "unleashing",
    "闪烁": "flickering",
    "飘扬": "fluttering",
    "摆动": "swaying",
    # Places / settings
    "高架路": "elevated highway",
    "高架桥": "elevated bridge",
    "城市": "city",
    "夜空": "night sky",
    "雨幕": "sheet of rain",
    "月光": "moonlight",
    "街灯": "streetlamp",
    "路灯": "streetlamp",
    "火车站": "train station",
    "穹顶": "dome",
    "天花板": "ceiling",
    # Emotions (rare but seen)
    "敬畏": "awe",
    "恐惧": "fear",
    "悲壮": "solemn heroism",
    "冷漠": "indifference",
    "茫然": "dazed",
    # Proper nouns that sometimes leak Chinese form
    "雷神之锤": "Mjolnir",
    "死侍": "Death Servants",
}


def _deterministic_cjk_sub(text: str) -> str:
    """Fast regex substitution of known Chinese tokens → English.
    Eliminates MIMO calls for the 80% common case. Longest-match-first
    so '少年甲' matches before '少年'. Returns text with known tokens
    replaced; any unknown CJK runs remain for the MIMO scrub pass."""
    # Sort by length descending so multi-char phrases match before substrings.
    for cn in sorted(_CJK_KEYFRAME_DICT.keys(), key=len, reverse=True):
        if cn in text:
            text = text.replace(cn, _CJK_KEYFRAME_DICT[cn])
    return text


_KEYFRAME_CJK_SCRUB_SYSTEM = """You are a translator. The input is a Nano
Banana Pro (T2I) prompt in English, but a few Chinese tokens have leaked
inline mid-sentence. Translate EVERY Chinese token to its natural English
equivalent. Return the corrected prompt only — no wrappers, no headers,
no code fences, no commentary.

Common leak patterns to translate:
  - Canonical labels: 少年 → "the teenager" / "the young man"; 中年男人 →
    "the middle-aged man"; 少女 → "the young girl"; 男孩 → "the boy"
  - Clothing: 罩衫 → "jacket" / "hoodie" (context dependent); 格子围巾 →
    "checkered scarf"; 围巾 → "scarf"; 校服 → "school uniform"
  - Props: 珐琅吊灯 → "enamel pendant lamp"; 操场 → "schoolyard / playground";
    手提箱 → "briefcase"; 雨伞 → "umbrella"; 雨刷器 → "windshield wipers"
  - Colors / atmosphere: 苍白 → "pale"; 白茫茫 → "white hazy expanse";
    漆黑 → "pitch-black"; 冷光 → "cold light"; 水银色 → "mercury-silver"
  - Verbs: 小跑 → "trot"; 挥刀 → "swing the blade"; 斩杀 → "cut down";
    屹立 → "stand firm"; 发动 → "unleash / invoke"
  - Placeholders like 人物甲/乙 → translate using the surrounding context's
    canonical label (e.g. "the teenager" for 少年, "the middle-aged man"
    for 中年男人 — match what the prompt says elsewhere)

HARD RULES:
  - Preserve line breaks, section headers (SCENE:, CHARACTER IDENTITY:,
    KEY SPECIFICS:, TONE:), punctuation, and ALL English text byte-identical.
  - Keep proper nouns as-is: Maybach, Burberry, Diesel, Hermès, Gungnir,
    Sleipnir, Odin, Arcane, Fortiche.
  - Do NOT add markdown, code fences, or prefix like "## Brief" / "Here's
    the scrubbed version:". Just emit the corrected prompt as plain text.
  - Do NOT paraphrase untouched English sentences — only swap Chinese
    tokens for English equivalents.

Output JSON with exactly one key:
  scrubbed: the prompt with all Chinese tokens translated to English.
"""


_CJK_SCRUB_SYSTEM = """You are a light copy-editor. A Seedance / Jimeng video
production brief has been compressed, but some Chinese tokens have leaked
into otherwise-English action/scene prose. Translate ONLY those leaked
CJK tokens into English while keeping everything else byte-identical.

HARD RULES — do NOT violate:
- VO lines (dialogue) stay VERBATIM in their original language. Identify
  VO by: lines starting with 'young woman (...)', 'young man (...)', or
  falling under a 'VO (...)' header. Do not touch them.
- Chinese scene titles (e.g. '警报响起', '诺诺召唤') at the top of the
  brief stay as-is — they identify the scene, not narrative content.
- Do NOT re-compose sentences. Only swap individual Chinese tokens
  mid-sentence (e.g. '身影', '小跑', '冷光', '红灯', '茫然的脸',
  '流线型车身', '泛着冷光') for equivalent English phrases.
- Preserve ALL line breaks, section headers, punctuation.

Output JSON with one key:
  scrubbed: the brief with in-sentence CJK tokens translated to English.
No code fences, no other keys.
"""


import re as _re
_CJK_INLINE_PATTERN = _re.compile(r"[\u4e00-\u9fff]+")


def _has_cjk_leak(text: str) -> bool:
    """Heuristic: ANY CJK char(s) in lines that are OTHERWISE English
    suggest leaks. Pure-Chinese lines (VO dialogue, scene titles) are skipped.

    Previously required runs of 2-6 CJK chars, but single-char leaks
    (e.g. 'the 刀 blade', 'at the 梢') were slipping through. Any CJK in
    a predominantly-English line is treated as a leak — dict pre-pass
    handles the common cases cheaply; MIMO fallback catches unknown tokens."""
    for line in text.splitlines():
        # Skip known safe sections.
        if line.startswith("VO ") or line.startswith("VO:"):
            continue
        if _re.match(r"^\s*young (woman|man) \(", line):
            continue
        stripped = line.strip()
        if not stripped:
            continue
        # Count english letters + chinese chars.
        english_chars = sum(1 for c in stripped if c.isascii() and c.isalpha())
        chinese_runs = _CJK_INLINE_PATTERN.findall(stripped)
        if chinese_runs and english_chars > len(stripped) * 0.3:
            # Predominantly English line with Chinese tokens → leak.
            return True
    return False


def compress_seedance_prompt(llm, prompt: str, max_chars: int) -> str:
    """Compress a Seedance/Jimeng brief to <= max_chars via MIMO.

    Two-pass approach:
    1. Structural compression (_COMPRESS_SYSTEM) — hits char budget + IP scrub.
    2. CJK-leak scrub (_CJK_SCRUB_SYSTEM) — translates any leftover
       Chinese tokens in English action prose to English (but preserves
       VO dialogue in original language).

    No hard truncation (that would risk dropping VO). If still over budget
    after both passes, we log a warning and return as-is.
    """
    if len(prompt) <= max_chars:
        # Still scrub CJK leaks even if under budget.
        if _has_cjk_leak(prompt):
            prompt = _cjk_scrub(llm, prompt)
        return prompt

    # Pass 1: structural compression
    user = (
        f"## Target char cap\n{max_chars}\n\n"
        f"## Current brief ({len(prompt)} chars)\n```\n{prompt}\n```"
    )
    data = llm.chat_json(
        system=_COMPRESS_SYSTEM, user=user, max_tokens=4096, temperature=0.2,
    )
    out = data.get("compressed", "").strip() or prompt

    # Pass 2: CJK-leak scrub (only fires if actual leaks detected)
    if _has_cjk_leak(out):
        out = _cjk_scrub(llm, out)

    if len(out) > max_chars:
        log.warning(
            "compress_seedance_prompt: output %d chars exceeds target %d; "
            "returning as-is (no hard truncation to protect VO/shots).",
            len(out), max_chars,
        )
    return out


def _cjk_scrub(llm, prompt: str) -> str:
    """Second-pass MIMO call to translate leaked CJK tokens in English prose."""
    log.info("compress_seedance_prompt: CJK leaks detected, running scrub pass")
    data = llm.chat_json(
        system=_CJK_SCRUB_SYSTEM,
        user=f"## Brief\n```\n{prompt}\n```",
        max_tokens=4096, temperature=0.1,
    )
    scrubbed = data.get("scrubbed", "").strip()
    return scrubbed if scrubbed else prompt

log = logging.getLogger(__name__)


# ─── Meta-prompt 1: Scene → Shots ────────────────────────────────────────

_SHOTS_SYSTEM = """You are a storyboard artist for an animated short. You receive ONE
scene that already has: (a) full action prose, (b) verbatim dialogue lines
in delivery order, (c) beats, (d) transition_in / dramatic_turn /
transition_out connective tissue. Your job is to BREAK the scene into
2-4 SHOTS that together fill ~15 seconds of screen time.

LANGUAGE — HARD RULE (the downstream video model is Seedance, Chinese-
native; Chinese action prose gives it cleaner guidance):
- `action` field: written entirely in **natural MANDARIN CHINESE**.
  Short, spoken, direct — like a storyboard caption, not a screenplay.
- No mixing: no English words inside the Chinese action sentence. (A
  brand name like "Maybach" is fine; inline English phrases are not.)
- `dialogue` lines stay verbatim in source language.
- `delivery` notes stay English (they're TTS voice direction).

CHARACTER NAMING — HARD RULE:
The user's prompt lists a `canonical_label` for each character (Chinese
role tag like `少年`, `中年男人`). In `action` prose you MUST use ONLY
those canonical labels. FORBIDDEN:
- Character names (楚子航, 朱子航, Chu Zihang, etc.) — NEVER in action.
- Story-role variants (父亲, 生父, 司机, the driver) — NEVER invent these.
- English labels (young man, middle-aged man) — NEVER in Chinese prose.
Use the SAME canonical_label for the same character across EVERY shot.
Rationale: Seedance treats divergent labels as different people and
teleports them between positions; labels must match the @image binding
that the downstream assembler emits. Dialogue text (quoted) is exempt —
names inside dialogue stay verbatim.

INTENT vs MICRO-DIRECTION — HARD RULE:
You are writing INTENT, not a shot list. Seedance does the choreography
and camera work itself; prescribing micro-steps breaks physics (e.g.
teleport-through-wall) and wastes the model's lookahead.
- ≤40 Chinese characters per action, one clause.
- Subject + ONE key action at intent level.
- FORBIDDEN: camera-movement directives (`镜头推进`, `缓缓平移`,
  `pan`, `dolly-in`) — these live in the `camera_movement` enum field,
  not prose.
- FORBIDDEN: animation chains (`先整领带，然后披外套，然后钻入后座`)
  — pick the ONE moment that defines the shot.
- FORBIDDEN: facial microexpression scripting (`表情从期待转为失望`,
  `cheeks puffed eyes wide`) — Seedance writes expressions from the
  dialogue delivery + reference images.
- FORBIDDEN: prop-level choreography (`雨刷器刮去前挡风玻璃的雨水`,
  `推开Burberry伞直接钻入后座`) — say `少年坐进后座`, nothing more.

LOGIC & PHYSICAL COHERENCE — HARD RULE:
Every action must be PHYSICALLY POSSIBLE and have an UNAMBIGUOUS SUBJECT.
Seedance will try to execute literally what you write; impossible or
ambiguous action produces broken clips.
- Every subject must be named explicitly — NO floating verbs. Bad:
  `指向后备箱黑箱` (who points?). Good: `少年打开后备箱，露出黑色手提箱`.
- NO impossible simultaneity. Bad: `父子并肩从车中走出` (two people can
  not exit the same car through the same door simultaneously). Good:
  `中年男人下车，少年跟着下车，两人并肩站在前大灯光柱中` (sequenced).
- NO teleport-through-wall / prop-passthrough / instant-position-change.
  If a character needs to move from point A to B, either (a) show the
  move explicitly as the shot's action, or (b) have them already at B
  at shot start (use a hard-cut at the shot boundary to justify the
  jump).
- NO referring to events that haven't happened yet or props not yet
  established. Each shot's action stands on what's been shown so far.

SOURCE FIDELITY — HARD RULE:
Do NOT invent action beats or dialogue situations that aren't in the
source text. The scene's `action` field (script layer) is your canon.
Use ONLY action and dialogue grounded there; do NOT advance the plot
past the scene's source range. Specifically:
- Do NOT compress multiple source chapters / sections into one shot.
  If the source scene covers 3 source paragraphs, your 3 shots should
  track those 3 paragraphs, not jump to events 30+ paragraphs later.
- Do NOT synthesize composite action ("middle-aged man throws black
  box + fights death servants + activates time incantation") when the
  source has each of those beats separated across different scenes.
- When in doubt, TRUST THE SOURCE ACTION PROSE — mirror its cadence
  and its beats.

PROPRIETARY NOVEL TERMS — HARD RULE:
Seedance / T2I models do not know Dragon-Raja-specific (or any
novel-specific) terminology. Translate to GENERIC visual language:
- `言灵` / `言灵发动` / `时间零言灵` → `时间减缓效果` / `慢动作视觉` /
  `默念一段咒语，时间开始凝滞`
- `死侍` → `金色瞳孔的黑色幻影战士` / `雨中浮现的阴影战士`
- `混血` / `S级混血` → `半人半神血统`（描述效果不描述等级）
- `卡塞尔学院` → `欧洲古堡风格的精英学院`
- `世界树` / `Yggdrasil` 铭牌 → `黑色手提箱上银色的树形铭牌`
- 其他组织名（`黑太子集团` etc.）→ describe visually or drop
Rule of thumb: if the term is a proper noun specific to the source IP
that Seedance couldn't visualize from general knowledge, REWRITE as a
generic visual description. Dialogue is exempt — characters can still
say the proper noun out loud (it's audio, not visual).

CINEMATIC MOTION DEFAULT — HARD RULE:
Assume NORMAL real-time cinematic motion. Slow-motion / time-dilation
/ freeze-frame effects are OFF by default.
- If a shot NEEDS slow-motion, write it explicitly in the action:
  `时间减缓 / 慢动作 / slow motion freeze`.
- AVOID words that Seedance commonly misreads as slow-motion:
    `缓慢向前推进` (reads as slow-mo) → `以低速稳稳向前滑行`
    `缓缓` / `徐徐` → `稳稳` / `稳定地`
- A divine / supernatural / climactic moment does NOT automatically
  mean slow-motion. State it explicitly if you want that effect.

SHOT-TO-SHOT CONTINUITY — HARD RULE:
Seedance renders each of the 3-4 sub-shots inside a 15s clip with a
hard cut between them, but VISUAL CONTINUITY across the cut is only
preserved if the action prose ANCHORS it. Default rendering treats
each shot as an independent composition — producing clips that feel
like 3 isolated images rather than a flowing scene.
Rule for shots N > 1 (every shot after the first):
- The action MUST open with a pickup phrase that picks up from
  Shot N-1's final visual. Name the carry-over element explicitly
  (a hand, an object, a camera angle, the character's gaze, an
  environmental element like the rain-curtain).
- Use either `承接：<carry-over element>` explicit prefix OR a
  substitutive opening that references the element.
  Example:
    Shot 1 ending: "少年紧握手机，屏幕暗下"
    ✗ Shot 2 opening (atomic): "少年站在车前，父亲关门"
    ✓ Shot 2 opening (pickup): "少年放下暗下的手机，转身走向迈巴赫"
- The carry-over does not need to be a literal match-cut — just a
  visual thread. A prop, a gaze direction, a lighting tone, a
  spatial orientation — pick ONE and reference it.

SCENE-TO-SCENE CONTINUITY:
Scene-boundary continuity is handled by the script layer's
`transition_in` / `transition_out` fields and surfaced at
assembly time in the "衔接 / CONTINUITY" block. When writing
Shot 1 (scene's first shot), ALIGN its opening with the scene's
`transition_in` beat — the assembler will show both to Seedance,
so Shot 1 should match the transition-in's visual promise, not
a contradiction. Similarly, the LAST shot's final frame should
match `transition_out`.

GOOD examples:
  - `豪车内后座，少年靠窗听父亲说话`
  - `父亲喋喋不休，少年沉默看向窗外雨幕`
  - `特写少年侧脸，眼神冷漠`
  - `司机为少年撑伞开后座门`

BAD examples (do NOT write prose like these):
  - `硬切至迈巴赫车内：雨刷器刮去前挡风玻璃的雨水，短暂露出生父笑脸，
    镜头推进至楚子航钻入后座的动作。` (camera + chain + micro-detail)
  - `The driver pushes open his door, holding a large Burberry umbrella.
    Chu Zihang pushes the umbrella aside and ducks directly into the
    back seat.` (English + chain + props)
  - `父亲表情从期望变困惑` (microexpression scripting)

CONNECTIVE TISSUE RULES:
- Shot 1's opening MUST reflect transition_in. If the scene is "hard cut
  to silence", Shot 1 opens on silence. If it's "seamless continuation",
  Shot 1's visual picks up where the previous scene's final shot left off.
- ONE shot (usually Shot 2) must be the dramatic_turn — the framing must
  visually signal the pivot (e.g. "close-up on her breath landing on his
  knuckle" if that's the turn).
- The LAST shot must set up transition_out — the final frame's composition
  or mood must cue the next scene's opening.

CRITICAL: you are DISTRIBUTING the scene's existing dialogue across shots.
Do NOT invent new dialogue, do NOT paraphrase, do NOT drop lines. Every
dialogue line from the input Scene must appear in exactly one shot's
`dialogue` field, in the original order.

DIALOGUE DENSITY BUDGET (hard rule — the video model crams all lines into
the final 2s if overloaded):
- Total scene is ~15s. Realistic paced Mandarin speech is ~3 chars/sec;
  English ~2 words/sec. A 15s scene carries at most ~40-50 Mandarin
  characters OR ~25-30 English words of dialogue TOTAL across all shots.
- SOFT TARGET: aim for **≤3 dialogue lines per 15s scene**. If a line
  is genuinely load-bearing (defines the pivot or a key source beat),
  going to 4-5 lines is acceptable — but ONLY when scene duration
  affords ≥3s of screen time per line. The assembler enforces time
  density at render time: lines that can't fit ≥3s each are silently
  dropped, so authoring 5 lines in a 15s scene means 2 get trimmed.
  Rule of thumb: per-shot line budget = floor((shot_duration − 0.5) / 3).
  A 5s shot → 1 line max; a 7s shot → 2 lines max.
- Action prose per shot: **≤40 Chinese chars, one clause, intent-level**
  (see LANGUAGE + INTENT rules above). The assembler does NOT truncate
  action any more — MIMO's short natural Chinese is taken as-is.
- If the input scene has more dialogue than that, YOU MUST DROP lines.
  Keep the emotionally load-bearing ones. Drop filler, drop repeated
  information, drop explanations. Prefer silence over rushed delivery.
- Wordless shots (zero dialogue) are valuable — use them for atmosphere.

  EXAMPLE — input scene has 14 dialogue lines, 15s duration:
    input lines (abbreviated):
      father: "plug it into the car door…"
      son: "I know, you've said so."
      father: "clothes are wet? let me turn on seat heating…"
      son: "no need, I'll change at home."
      father: "9-million yuan car, only 3 voices start it…"
      son: "don't care."
      father: "this CD is about fatherly love, good right?"
      son: "it's a girl talking to her father, not right for me."
      father: "you understand English? I heard you won a contest…"
      son: "this is Altan's song, about a father marrying his 14yo daughter…"
      son: "later the husband died, the girl mourned on the grave…"
      father: "what kind of song is this? makes no sense."
      son: "just an Irish folk song, shouldn't play it for me."
      father: "if you don't want to listen, turn it off, let's chat."

    correct output (drop 10 lines, keep 4 load-bearing ones):
      Shot 2, dialogue:
        - father: "900万的车，全世界就三个人的声音能启动。" (boasting, casual)
        - son: "不关心。" (flat, cold)
      Shot 3, dialogue:
        - father: "这碟讲父爱的，不错吧？" (hopeful, pushing)
        - son: "是个女孩和父亲的对话，不合适。" (quiet rebuke)

  That's 4 lines, ~40 Mandarin chars, ~13s of speech with breathing
  room. The emotional arc (father's boast → son's coldness → father's
  hopeful offering → son's pointed rejection) is preserved. The rest
  is redundant.

DELIVERY NOTES (required per line — not optional):
- Every DialogueLine MUST have `delivery` populated. Example:
    {"speaker_id": "chu-zi-hang", "text": "不关心。",
     "delivery": "flat, cold"}

- `delivery` is SHORT (1-2 simple emotion words), NOT literary stage
  directions. Seedance over-performs when given stacked adjectives
  or narrative descriptors — "soft, nostalgic, almost whispered"
  produces a breathy reminiscing tone even when the line content is
  urgent combat instruction. Keep it plain.
  ✓ GOOD (plain, 1-2 words):
      "flat, cold" / "warm" / "calm determined" / "urgent" /
      "shouting" / "quiet firm" / "tender" / "resigned" / "pleading"
  ✗ BAD (literary / too specific / stacked):
      "warm paternal boast, slow, lingering" → drop to "warm"
      "quiet rebuke, almost whispered" → drop to "quiet firm"
      "nostalgic, wistful, distant" → drop to "warm" or "resigned"
  Banned descriptors: "nostalgic", "wistful", "reminiscing",
  "lingering", "almost whispered", "breathy", "sultry", "husky",
  "paternal" (these pull Seedance into dramatic performance).
  Safe descriptors: plain emotion words + plain volume words only.

- CONSISTENCY across adjacent lines in the SAME shot: if line 1 is
  "calm determined" and line 2 follows in the same shot, line 2's
  delivery should be the SAME or a small variant ("calm firm" /
  "calm urgent") — not a jarring flip to "soft nostalgic". Seedance
  smears voice character if tone flips mid-shot. Changes in tone
  should happen at shot boundaries, not mid-shot.

- ANCHOR to scene.emotional_register: all lines in a scene default
  to that register's baseline emotion, with small variants per line.
  A "solemn, tense" scene should not have "cheerful" or "tender"
  lines unless the source text EXPLICITLY supports a tonal pivot.

- When in doubt, go PLAIN. A line with delivery="calm" is better
  than a line with delivery="soft, nostalgic, almost whispered"
  that mismatches its content. Seedance's default natural delivery
  with a simple one-word tone is usually correct; over-specification
  is what creates the mismatch.

- An empty `delivery` string is a bug — the video model will default
  to robotic rapid-fire delivery. Always populate with at least one
  plain word.

Shot structure:
- Shot 1 (5-10s): establishing / inciting framing for the scene.
- Shot 2 (3-5s): the reactive or intimate cut (close-up, OTS, or insert).
- Optional Shot 3 (3-5s): resolution / handoff to next scene.

Match the scene's emotional_register:
- Quiet registers: held shots, simple framings. One impossibility at most.
- Operatic: dynamic camera, scale jumps, emotion-as-environment permitted.

Your JSON must have exactly one key:
  shots: ordered list of shot objects with these keys:
    - id: "sh01", "sh02", "sh03" (zero-padded)
    - duration_sec: float in [3.0, 10.0]
    - shot_type: one of
        ["wide", "full", "medium", "close_up", "extreme_close_up",
         "over_shoulder", "pov", "insert"]
    - camera_movement: one of
        ["static", "pan", "tilt", "dolly_in", "dolly_out", "tracking",
         "handheld", "crane"]
    - characters_in_shot: list of character ids visible (subset of scene cast).
    - action: ≤40 Chinese characters, one clause, intent-level. Subject +
      ONE key action in natural Mandarin. Do NOT write camera movements
      (those go in `camera_movement`), do NOT chain multiple actions,
      do NOT script facial microexpressions. See LANGUAGE + INTENT rules
      above for good/bad examples.
    - dialogue: list of {"speaker_id": "<char_id>", "text": "<line>"}.
      VERBATIM from the Scene's dialogue field, assigned to the shot where
      the line is delivered. Concatenate if two consecutive lines land in
      the same shot. May be empty if the shot is wordless.
    - keyframe_prompt: ENGLISH prompt for Nano Banana Pro (Google Gemini
      image) to paint THIS SHOT's keyframe PNG. THIS SHOT — not the
      scene in general. Each shot's keyframe_prompt MUST depict the
      specific moment described by THIS shot's `action` field, with
      THIS shot's `shot_type` framing. DO NOT copy-paste the same
      prompt across shots — Shot 1's frame ≠ Shot 2's frame ≠ Shot 3's.

      ~400-700 chars, 3-paragraph structure (DO NOT write a STYLE
      paragraph — the pipeline prepends the full approved style guide
      at render time. Writing your own STYLE paragraph just dilutes it
      with MIMO-drift adjectives like "cinematic ethereal light"):

        Paragraph 1 — SCENE / CAMERA. Use the 4-part camera formula:
          angle + lens + height + framing. Concrete specs beat dramatic
          adjectives (documented: Nano Banana ignores "dramatic low
          angle", obeys "24mm lens, camera 20cm above ground, tilted 30°
          up, subject fills 70% of frame, head cropped at top edge").

          ENVIRONMENT CLAUSE — HARD RULE (MIMO drops this 90% of the
          time, don't). MUST include ALL THREE in plain English:
            1. LOCATION TYPE — verbatim from scene's location_description
               or scene's title (e.g. "elevated highway bridge at night",
               "inside a luxury sedan's rear seat", "empty classroom").
               If location_description is in Chinese, translate the
               environment keywords (高架路 → elevated highway,
               迈巴赫 → Maybach sedan, 教室 → classroom).
            2. TIME OF DAY — verbatim from `time_of_day` (night / dawn /
               dusk / overcast day / golden hour).
            3. WEATHER + ATMOSPHERE — from the scene's action or
               location_description (rain-soaked, torrential downpour,
               dry heat-haze, overcast fog, clear winter light, etc.).

          Common MIMO drift to avoid:
            ❌ "SCENE: A man and a boy look on from a halted Maybach." —
               no location, no weather, no time. Nano Banana then paints
               a generic daytime street.
            ✅ "SCENE: Torrential night rain drenches an elevated urban
               highway bridge. A black Maybach halts mid-lane. A
               middle-aged man and a teenage boy inside look on as..."

          NAMED STORY ELEMENTS CLAUSE — HARD RULE (MIMO drops this too).
          Scan `scene.action` and `scene.location_description` for NAMED
          entities: creatures (Sleipnir), deities (Odin), weapons
          (Gungnir), named light sources (白光如神堂 — "white divine-hall
          light"), named crowds (死侍 — "Death Servants"), named objects
          (Hermès backpack, Maybach 62). If any of these are on-screen
          in THIS shot, include them VERBATIM (translated to English if
          source is Chinese) in the SCENE paragraph. Generic descriptors
          ("a horse", "a god", "bright light") are NOT substitutes —
          Nano Banana renders named entities differently from generic ones.

          Common MIMO drift to avoid:
            ❌ "SCENE: A man and a boy look on from a halted Maybach." —
               no location, no weather, no time. Nano Banana then paints
               a generic daytime street.
            ❌ "Odin stands atop a horse, holding a spear." — dropped
               Sleipnir, Gungnir, and the source text's "白光如神堂",
               "甲胄微光", "独目如灯", "死侍涌向手提箱" etc.
            ✅ "SCENE: Torrential night rain drenches an elevated urban
               highway bridge. A black Maybach halts mid-lane. A column
               of white divine-hall light splits the storm; inside it,
               Odin (one-eyed, eye glowing like a lamp, grey cloak,
               armor glinting) atop Sleipnir (his eight-legged horse),
               holding Gungnir. Death Servants flank them. Inside the
               Maybach..."

          SPATIAL POSITIONS CLAUSE — HARD RULE. When characters are in
          a constrained shared space (car, classroom, elevator), state
          WHO is WHERE explicitly. The character positions for a given
          location are ESTABLISHED IN THE EARLIEST SCENE in that
          location. Carry those positions forward to every later scene
          in the same location. The "Prior-scene spatial context"
          section of the user message surfaces these — USE them.
          Example: if s02 establishes "father drives, son rear seat"
          inside a Maybach, s05 in the same Maybach MUST preserve
          that — don't invent new positions.

          Then describe the shot's subjects (what they're doing + how
          they feel) and key iconic items. Visual imagery, not plot facts.

        Paragraph 2 — CHARACTER IDENTITY. Every character in the shot
          MUST be described with ICONIC physical features from the Cast
          block — hair color (exact — "deep blue-black", not "dark"),
          eye color (exact — "light amber", not "brown"), signature
          wardrobe piece, any unusual feature (scar, accessory). Use
          role descriptors (never names). Give each character at least
          one feature UNIQUE to them (scar / braid / eye-patch / specific
          wardrobe) in a DIFFERENT body region, so the model doesn't
          feature-bleed between similar-bucket characters.

        Paragraph 3 — KEY SPECIFICS (only when disambiguation needed).
          Phrase DISAMBIGUATION POSITIVELY, not as "NOT X".
            ❌ "Odin has no wings, not an angel" (Nano Banana ignores)
            ✅ "Odin's cloak falls flat against his back, bare-
                shouldered, no feathered or skeletal appendages"
            ❌ "Sleipnir is not multiple horses"
            ✅ "Sleipnir's body is one continuous horse-torso with 2
                forelegs and 2 hindlegs on each side front-to-back,
                8 visible limbs total on one body"
          NOTE: for truly hard anatomy (8-legged horse, 6-armed deity)
          T2I models cannot be prompted into obedience; provide a
          reference sketch via the scene's ref-image pool instead, or
          frame the shot to avoid showing the full anatomy.

        Optional Paragraph 4 (1 line, OK to skip) — TONE.
          "Solemn, intimate." / "Operatic, vertiginous."

      FORBIDDEN:
        - Style proper-name drops without mechanical decomposition
          ("in the style of Arcane" alone — dilute prior).
        - Chinese text (except proper nouns: Gungnir, Sleipnir).
        - Character names (楚子航 / Chu Zihang).
        - Pure adjective-camera ("dramatic low angle") — use mechanical
          spec instead.
        - Negative lists ("no X, not Y, not Z") — reframe positively.
        - Weight syntax `(word:1.3)` — ignored by Gemini.
        - Hard-coded feature templates across characters ("son = X, son
          = X") — each character uses ONLY their own `visual_description`.

      Goal: mechanical specs + positive iconography + distinct anchors.
      Less adjective fluff, more render-able specifics.

Durations should sum to roughly the scene's implied screen time (default 15s).
"""



_REFINE_SHOTS_SYSTEM = """You are revising a scene's shot breakdown based on director
feedback. You receive:
1. The scene's full context (same format as generate-path user message:
   time_of_day, location_description, action, dialogue, beats,
   connective tissue, Environment triplet callout, Prior-scene spatial
   context if available, canonical labels, Cast).
2. The current list of shots (JSON).
3. The director's natural-language feedback.

Produce a NEW list of shots keeping the same schema. Preserve shots the
feedback doesn't touch; revise those it does. Renumber if shots are
added/removed.

IMPORTANT — refine is NOT a lightweight tweak. When regenerating, you
MUST re-apply ALL the same rules as the generate path (see the scene
context above):
- action: natural MANDARIN, ≤40 chars, intent-level, no micro-direction
- keyframe_prompt: ENGLISH, ~400-700 chars, 3-paragraph (CAMERA +
  CHARACTER + KEY SPECIFICS, then optional TONE); NO STYLE paragraph
  (pipeline prepends full style_prompt at render time)
- keyframe_prompt SCENE must include ENVIRONMENT TRIPLET verbatim:
  (1) location type (translated to English from location_description),
  (2) time_of_day, (3) weather/atmosphere
- keyframe_prompt SCENE must include NAMED STORY ELEMENTS verbatim
  (translated to English): every named character / creature / named
  entity / named object / named light source from scene.action and
  location_description that is on-screen in THIS shot. Don't reduce
  "奥丁持枪屹立" to "a god with a spear" — write "Odin, a one-eyed
  deity, armor glinting, holding Gungnir, atop Sleipnir his eight-
  legged horse, amid a column of white divine-hall light".
- keyframe_prompt SCENE must honor SPATIAL POSITIONS from the
  Prior-scene spatial context (e.g. "father drives, son rear seat" —
  established in s02, must persist in every later Maybach scene).
- Canonical character labels (少年甲/少年乙/中年男人) — use ONLY those,
  never names, never invented role words.

INTENT vs MICRO-DIRECTION — do NOT write camera movements (`镜头推进`,
`pan`), animation chains (`先 X 然后 Y`), facial microexpressions
(`表情从 A 转为 B`), or prop choreography. Say `少年坐进后座`, not
`推开伞直接钻入后座`. Seedance composes the choreography from the
reference images + dialogue; prescribing micro-steps breaks it.

DIALOGUE DENSITY BUDGET: SOFT target ≤3 lines per 15s scene; may go
to 4-5 if load-bearing AND scene duration affords ≥3s per line. The
assembler enforces time density at render: lines that can't fit
≥3s each are dropped. Rule: per-shot budget = floor((dur − 0.5) / 3).
A 5s shot → 1 line; 7s → 2; 10s → 3. Keep the load-bearing ones;
drop filler and repeats. Silence beats rushed delivery.

DELIVERY NOTES: every dialogue line MUST include a `delivery` field.
Rules (same as the generate path):
  - 1-2 simple emotion words, NOT literary stage directions.
  - Banned: "nostalgic", "wistful", "reminiscing", "lingering",
    "almost whispered", "breathy", "paternal", "sultry".
  - Prefer plain: "flat, cold", "warm", "calm determined", "urgent",
    "shouting", "quiet firm", "tender", "resigned".
  - Adjacent lines in same shot should have CONSISTENT tone (same or
    small variant), anchor to scene.emotional_register.
  - When in doubt, go PLAIN. "calm" beats "soft, nostalgic".

LOGIC & PHYSICAL COHERENCE: every action must be physically possible
and have an unambiguous subject.
  - NO floating verbs ("指向后备箱" → specify WHO points).
  - NO impossible simultaneity ("父子并肩从车中走出" → sequence them).
  - NO teleport-through-wall / instant-position-change without a cut.

SOURCE FIDELITY: action beats must be grounded in the script's
action prose for this scene. Do NOT compress multi-chapter events
into one shot. Do NOT synthesize composite combat from different
source sections.

PROPRIETARY NOVEL TERMS: translate source-specific terms to generic
visual language in action prose — "言灵" → "时间减缓效果 / 慢动作";
"死侍" → "金眼黑色幻影战士"; "卡塞尔学院" → "欧洲古堡风格学院".
Dialogue (spoken) is exempt — characters can still say the term.

CINEMATIC MOTION DEFAULT: normal real-time speed unless the action
explicitly says otherwise. Avoid "缓慢/缓缓/徐徐" — Seedance reads
them as slow-motion effects. Use "稳稳/以低速稳稳向前" instead.

Preserve existing delivery notes unless the feedback explicitly
changes them OR they violate the rules above (in which case rewrite
to comply).

Your JSON output has exactly one key:
  shots: the revised list with the same schema as input.
"""


# Global style is resolved per-story at call time via load_style_prompt().
# _ARCANE_GLOBAL_STYLE constant removed — replaced by StyleGuide.


# ─── Agent ──────────────────────────────────────────────────────────────

@dataclass
class StoryboardAgent:
    """Layer 2. Consumes an approved Script (Story), produces a Storyboard.

    Stateless — reads Story via `generate(story)`, reads current Storyboard
    via `refine_scene(storyboard, ...)`. Caller persists via BibleStore.
    """

    llm: MimoClient
    bible: BibleStore

    # ── full storyboard generation ──────────────────────────────────────
    def generate(self, story: Story) -> Storyboard:
        char_cache: dict[str, Character] = {
            cid: self.bible.load_character(cid) for cid in story.characters
        }
        style_prompt = load_style_prompt(self.bible, story.id)
        scenes_out: list[StoryboardScene] = []
        for scene in story.scenes:
            log.info("Storyboard: %s '%s'", scene.id, scene.title)
            sb_scene = self._build_scene(story, scene, char_cache, style_prompt)
            scenes_out.append(sb_scene)
        return Storyboard(story_id=story.id, scenes=scenes_out)

    # ── per-scene refine ────────────────────────────────────────────────
    def refine_scene(
        self,
        storyboard: Storyboard,
        story: Story,
        scene_id: str,
        feedback: str,
    ) -> Storyboard:
        """Apply feedback to ONE scene only; return new Storyboard with that
        scene replaced. Other scenes untouched."""
        char_cache: dict[str, Character] = {
            cid: self.bible.load_character(cid) for cid in story.characters
        }
        style_prompt = load_style_prompt(self.bible, story.id)
        scene = next((s for s in story.scenes if s.id == scene_id), None)
        if scene is None:
            raise ValueError(f"scene_id '{scene_id}' not in story")
        sb_scene = next((s for s in storyboard.scenes if s.scene_id == scene_id), None)
        if sb_scene is None:
            raise ValueError(f"scene_id '{scene_id}' not in current storyboard")

        # Feedback re-runs shots (MIMO authors action + dialogue +
        # per-shot keyframe_prompt in one call). scene_art_prompt is
        # RETIRED (per-shot keyframes replace it). The cached
        # scene_art_prompt on sb_scene is kept as a fallback only for
        # stories generated before this pipeline change.
        new_shots = self._refine_shots(
            sb_scene.shots, feedback,
            scene=scene, story=story, char_cache=char_cache,
        )
        new_seedance = self._assemble_seedance_prompt(
            story, scene, new_shots, char_cache, style_prompt
        )

        new_sb_scene = StoryboardScene(
            scene_id=scene_id,
            shots=new_shots,
            scene_art_prompt=sb_scene.scene_art_prompt,  # passthrough; deprecated
            seedance_prompt=new_seedance,
        )
        # Replace the scene in the list, preserving order.
        new_scenes = [
            new_sb_scene if s.scene_id == scene_id else s
            for s in storyboard.scenes
        ]
        return Storyboard(story_id=storyboard.story_id, scenes=new_scenes)

    # ── internal: build one scene from scratch ──────────────────────────
    def _build_scene(
        self,
        story: Story,
        scene: Scene,
        char_cache: dict[str, Character],
        style_prompt: str,
    ) -> StoryboardScene:
        # `_SHOTS_SYSTEM` now authors `keyframe_prompt` per shot inline
        # (English Nano Banana style). `scene_art_prompt` is RETIRED —
        # per-shot keyframes replace it. The StoryboardScene field stays
        # empty for new stories; legacy stories keep their cached value
        # as a fallback path.
        shots = self._generate_shots(scene, char_cache, story=story)
        seedance_prompt = self._assemble_seedance_prompt(
            story, scene, shots, char_cache, style_prompt
        )
        return StoryboardScene(
            scene_id=scene.id,
            shots=shots,
            scene_art_prompt="",  # deprecated; per-shot keyframe_prompt replaces
            seedance_prompt=seedance_prompt,
        )

    # ── MIMO call 1: shots ──────────────────────────────────────────────
    def _generate_shots(
        self,
        scene: Scene,
        char_cache: dict[str, Character],
        story: Story | None = None,
    ) -> list[Shot]:
        user = _scene_to_user_prompt(scene, char_cache, story=story)
        data = self.llm.chat_json(
            system=_SHOTS_SYSTEM, user=user, max_tokens=4096, temperature=0.6
        )
        shots = [Shot.model_validate(_normalize_shot_dict(s)) for s in data.get("shots", [])]
        return self._scrub_shot_prompts_cjk(shots, scene.id)

    def _scrub_shot_prompts_cjk(
        self, shots: list[Shot], scene_id: str,
    ) -> list[Shot]:
        """Post-process: translate leaked CJK tokens inside each shot's
        keyframe_prompt (English-target T2I prompt). MIMO drifts Chinese
        terms like `少年`, `罩衫`, `珐琅吊灯`, `操场` into otherwise-English
        prose ~70% of runs; a deterministic scrub pass per shot fixes it.

        Uses a keyframe-specific system prompt (not the production-brief
        one) and retries once if the first scrub left CJK behind. Also
        strips any markdown wrapper MIMO sometimes adds.
        Per shot cost: 1-2 extra MIMO calls when leak detected."""
        out: list[Shot] = []
        for sh in shots:
            kfp = sh.keyframe_prompt
            if not kfp or not _has_cjk_leak(kfp):
                out.append(sh)
                continue
            # Pass 1 — deterministic dictionary substitution (zero API cost).
            # Catches the 80% common vocabulary (少年, 罩衫, 珐琅吊灯, etc.).
            scrubbed = _deterministic_cjk_sub(kfp)
            if not _has_cjk_leak(scrubbed):
                log.info(
                    "storyboard: %s/%s CJK scrubbed via dict (no MIMO)",
                    scene_id, sh.id,
                )
                out.append(sh.model_copy(update={"keyframe_prompt": scrubbed}))
                continue
            # Pass 2 — MIMO scrub for whatever dict didn't catch. Rare
            # enough that the 5-min rate-limit hang is acceptable when
            # it hits (vs. the per-shot dict path that handles most cases).
            log.info(
                "storyboard: %s/%s CJK dict pre-pass left residue — "
                "falling back to MIMO scrub", scene_id, sh.id,
            )
            scrubbed = self._scrub_keyframe_prompt_once(scrubbed)
            out.append(sh.model_copy(update={"keyframe_prompt": scrubbed}))
        return out

    def _scrub_keyframe_prompt_once(self, prompt: str) -> str:
        """Single-pass CJK scrub for a keyframe_prompt. Uses the
        keyframe-specific system prompt + strips markdown wrappers MIMO
        sometimes injects ("## Brief", code fences, prefixes)."""
        import re as _re
        data = self.llm.chat_json(
            system=_KEYFRAME_CJK_SCRUB_SYSTEM,
            user=f"Prompt to scrub:\n{prompt}",
            max_tokens=3072, temperature=0.1,
        )
        scrubbed = (data.get("scrubbed") or "").strip()
        if not scrubbed:
            return prompt
        # Strip common wrapper patterns MIMO sometimes adds.
        # - leading "## Brief\n```\n" ... trailing "```"
        # - "Here is the scrubbed prompt:" prefix
        scrubbed = _re.sub(
            r"^##\s*\w+\s*\n```[a-zA-Z]*\s*\n?", "", scrubbed,
        ).strip()
        scrubbed = _re.sub(r"\n```\s*$", "", scrubbed).strip()
        scrubbed = _re.sub(
            r"^(Here('?s| is) the scrubbed.*?:?)\s*\n", "", scrubbed,
            flags=_re.IGNORECASE,
        ).strip()
        return scrubbed or prompt

    def _refine_shots(
        self,
        shots: list[Shot],
        feedback: str,
        scene: Scene | None = None,
        story: Story | None = None,
        char_cache: dict[str, Character] | None = None,
    ) -> list[Shot]:
        import json
        shots_json = [s.model_dump() for s in shots]
        # Serialize enums as their string values so MIMO can parse them back.
        # ALSO: clear keyframe_prompt. MIMO's refine behavior is "preserve
        # unless feedback demands change"; when the existing keyframe_prompt
        # has stale content (e.g. "Beside him" locking father + son into
        # adjacent seats despite scene.action saying father drives), MIMO
        # copies it through on minimal feedback. By passing "" we force
        # MIMO to regenerate from scratch against the up-to-date scene
        # context (environment triplet + spatial positions + canonical
        # labels) rather than anchoring to stale text. Preservation when
        # truly desired comes via explicit feedback, not implicit memory.
        for s in shots_json:
            if hasattr(s.get("shot_type"), "value"):
                s["shot_type"] = s["shot_type"].value
            if hasattr(s.get("camera_movement"), "value"):
                s["camera_movement"] = s["camera_movement"].value
            s["keyframe_prompt"] = ""
        # Include the same scene + story context that _generate_shots does
        # so refines honor ENVIRONMENT TRIPLET, canonical labels, prior-scene
        # character positions, and named story elements — exactly like
        # first-time generation. Without this, refines drop all these rules.
        scene_ctx = ""
        if scene is not None and char_cache is not None:
            scene_ctx = _scene_to_user_prompt(scene, char_cache, story=story)
        user = (
            f"{scene_ctx}\n\n" if scene_ctx else ""
        ) + (
            f"## Current shots\n```json\n{json.dumps(shots_json, ensure_ascii=False, indent=2)}\n```\n\n"
            f"## Director feedback\n{feedback.strip()}\n"
        )
        data = self.llm.chat_json(
            system=_REFINE_SHOTS_SYSTEM, user=user, max_tokens=4096, temperature=0.5
        )
        refined = [Shot.model_validate(_normalize_shot_dict(s)) for s in data.get("shots", [])]
        scene_id = scene.id if scene else "?"
        return self._scrub_shot_prompts_cjk(refined, scene_id)

    # ── deterministic: seedance_prompt assembly (no MIMO) ──────────────
    def _assemble_seedance_prompt(
        self,
        story: Story,
        scene: Scene,
        shots: list[Shot],
        char_cache: dict[str, Character],
        style_prompt: str,
        shots_with_keyframes: list[str] | None = None,
        n_style_refs: int = 0,
    ) -> str:
        """Schema-format Seedance production brief. Also used for Jimeng.

        `shots_with_keyframes`: ordered list of shot ids that have an
        approved per-shot keyframe available. When provided, the prompt
        emits one `@imageN` tag per listed shot followed by character
        refs, and each shot body gets an inline `@imageN: 此构图参考。`
        When None (legacy mode or no keyframes yet): falls back to the
        scene-level `@image1: 场景关键帧` layout.

        Ref budget: Seedance caps at 9 refs. When shots + chars > 9, the
        earliest shot keyframes (usually wide establishing, lowest info
        density) are dropped first; later shots keep their refs.

        Applies deterministic clamps:
          - shot durations scaled so total ≤ 15s (non-mutating)
          - VO time-density gate ≥ 3s/line (drops overflow)
        """
        shots = _clamp_shot_durations(shots, target_sec=15, min_shot_sec=3)
        total_dur = int(sum(s.duration_sec for s in shots) or 15)

        # @tag binding — each ref is described by its speaker role so the
        # video model links dialogue speaker tags to the correct face
        # reference. @image1 is always the scene keyframe; then characters
        # follow in the order they appear in scene.characters_in_scene.
        #
        # In Seedance API multi-ref mode every ref has role=reference_image
        # regardless; the @imageN tags in prompt body are just hints. In
        # Jimeng Web UI mode the first ref goes to the "首帧" slot. Either
        # way the binding is load-bearing for speaker→face matching.
        # Filter scene.characters_in_scene down to those actually USED
        # in this scene — i.e. they speak (dialogue.speaker_id) or appear
        # in at least one shot's characters_in_shot. Characters that sit
        # in characters_in_scene without speaking or appearing are
        # artifacts of upstream MIMO over-inclusion; including them here
        # over-declares @image slots that nothing in the prompt references
        # (Seedance may then try to place the unused character on-screen,
        # causing identity drift).
        speaking_ids = {d.speaker_id for d in getattr(scene, "dialogue", [])}
        used_labels: set[str] = set()
        for sh in shots:
            for lbl in getattr(sh, "characters_in_shot", []) or []:
                used_labels.add(lbl)
        # Canonical labels aren't known yet (they depend on char_ids), so
        # do a two-pass: first compute raw labels for ALL scene chars,
        # then drop chars whose label isn't referenced AND who don't speak.
        _all_ids = [cid for cid in scene.characters_in_scene if cid in char_cache]
        _raw = [_canonical_role_cn(char_cache[cid]) for cid in _all_ids]
        _disambig = _disambiguate_labels(_raw)
        _cid_lbl_pre = dict(zip(_all_ids, _disambig))
        char_ids = [
            cid for cid in _all_ids
            if cid in speaking_ids or _cid_lbl_pre.get(cid) in used_labels
        ]
        # Safety: if the filter would drop ALL characters, fall back to
        # the full list (rare — e.g. a pure-action scene with no VO and
        # no populated characters_in_shot). Better to over-declare than
        # render a ghost scene with no identity refs at all.
        if not char_ids and _all_ids:
            log.warning(
                "assemble_seedance: scene %s has characters_in_scene but "
                "no one speaks or appears in any shot; keeping full list",
                scene.id,
            )
            char_ids = _all_ids
        # Canonical Chinese role labels — used consistently across @tag,
        # char binding, VO speaker. MIMO is told (in _SHOTS_SYSTEM) to use
        # ONLY these labels in shot action prose; the assembler also
        # surfaces them in `_scene_to_user_prompt` so MIMO sees them.
        #
        # Collision disambiguation: two characters with the same gender+age
        # bucket (e.g. chu-zi-hang + lu-mingfei both "少年") would share
        # one label → Seedance maps all "少年" VO lines to one face. When
        # that happens we suffix with 甲/乙/丙/... by characters_in_scene
        # order (stable across regens). Single-occurrence labels stay bare.
        raw_roles = [_canonical_role_cn(char_cache[cid]) for cid in char_ids]
        char_roles = _disambiguate_labels(raw_roles)
        # Map speaker_id → disambiguated label for VO tagging. This mapping
        # covers EVERY character in the scene — voice-only characters
        # (e.g. a mother on the phone, no on-screen appearance, no
        # portrait file) still need their VO line labeled with 中年女人
        # so Seedance knows which voice to synthesize.
        cid_to_label = dict(zip(char_ids, char_roles))

        # Which characters get an @image slot + binding line? Only those
        # with an actual portrait file on disk. Voice-only characters
        # (no `front.png`) would otherwise produce `@image8: 中年女人`
        # in the prompt while the image bundle has no 08_*.png file —
        # Seedance then either errors or silently falls back to a blend
        # of the remaining refs. Keeping the speaker label but dropping
        # the @image slot gives a clean voice-over with no visual mismatch.
        def _has_portrait(cid: str) -> bool:
            try:
                return (self.bible.refs_dir(cid) / "front.png").exists()
            except Exception:
                return False
        visible_char_ids = [cid for cid in char_ids if _has_portrait(cid)]
        visible_char_roles = [cid_to_label[cid] for cid in visible_char_ids]

        # Ref slot allocation: cap at 9 total. Upload order:
        #   (a) shot keyframes (from earliest when over budget, drop earliest first)
        #   (b) style reference images (real-world Arcane frames — brushwork/palette)
        #   (c) character refs (identity lock — must never drop)
        # Characters are load-bearing for identity, so they claim slots first
        # in the budget calculation, then style refs, then shot keyframes are
        # trimmed to fit the remainder. This matches the CLI's upload order.
        # @image budget counts ONLY visible characters (those with
        # portraits). Voice-only characters don't consume a slot.
        N_chars = len(visible_char_ids)
        # Cap style refs at the user-provided count; reduce to fit budget.
        effective_style_refs = max(0, min(n_style_refs, 9 - N_chars))
        shot_idx_to_image_n: dict[int, int] = {}
        tag_parts: list[str] = []
        image_n = 1
        if shots_with_keyframes:
            idxs = [i for i, sh in enumerate(shots) if sh.id in shots_with_keyframes]
            max_shots = max(0, 9 - N_chars - effective_style_refs)
            if len(idxs) > max_shots:
                idxs = idxs[len(idxs) - max_shots:]
            for idx in idxs:
                shot_idx_to_image_n[idx] = image_n
                tag_parts.append(f"@image{image_n}: Shot {shots[idx].id} 构图参考")
                image_n += 1
        else:
            # Legacy: single scene-level keyframe, only if budget permits.
            if 1 + N_chars + effective_style_refs <= 9:
                tag_parts.append(f"@image{image_n}: 场景关键帧")
                image_n += 1
        # Style reference slots. Nano Banana/Seedance need these refs
        # explicitly labeled so the model treats them as STYLE (brushwork,
        # palette, shading), not as subjects or composition.
        style_start_image_n = image_n
        for i in range(effective_style_refs):
            tag_parts.append(
                f"@image{image_n}: style reference (copy brushwork + "
                f"palette + shading hardness ONLY — do NOT copy subjects "
                f"or scene content from this frame)"
            )
            image_n += 1
        # Character refs — only visible characters (with portraits) get
        # slots. Voice-only chars appear in VO speaker tags only, with
        # no @image binding.
        char_start_image_n = image_n
        for role in visible_char_roles:
            tag_parts.append(f"@image{image_n}: {role}")
            image_n += 1
        tag_line = ", ".join(tag_parts) + "."

        binding_lines = []
        for offset, role in enumerate(visible_char_roles):
            binding_lines.append(f"- {role} = @image{char_start_image_n + offset}")
        char_block = "\n".join(binding_lines) or "(无角色出镜)"

        # Shot blocks — MIMO now writes short natural Chinese (≤40 chars
        # intent-level). No truncation: trust the upstream rules. The
        # Shot header stays English for schema regularity (Seedance has
        # been observed to respect multi-shot cuts ONLY when shots are
        # labeled `Shot N (dur, type, hard cut at Xs)` — see memory
        # `feedback_seedance_schema_prompt.md`).
        #
        # NEW (2026-04-22): dialogue is emitted INLINE per shot, not in
        # a separate VO block at the end. Each Shot N block ends with
        # its own `[t=Xs] speaker (tone): line` cues. This binds each
        # line to its visual context (which shot it lives in) and
        # removes the need for elaborate "VERBATIM HARD RULES" prose —
        # Seedance can't hallucinate dialogue into a silence window
        # when the silence is bounded by a shot cut. See user feedback
        # 2026-04-22 "dialogue inline with shot, drop the hard-rules
        # block".
        shot_blocks = _build_shot_blocks_with_inline_vo(
            shots=shots,
            scene=scene,
            char_cache=char_cache,
            cid_to_label=cid_to_label,
            shot_idx_to_image_n=shot_idx_to_image_n,
        )

        # NOTE: VO lines + hallucination-rule block REMOVED 2026-04-22.
        # Dialogue is now emitted INLINE inside each shot block (see
        # `_build_shot_blocks_with_inline_vo`). Rationale:
        # - Seedance has silence-windows between lines that it used to
        #   fill with improvised speech; anchoring dialogue to shot
        #   visual context closes those windows naturally.
        # - The VERBATIM HARD RULES prose block (4 numbered rules) is
        #   no longer needed — dialogue grounded in its shot context
        #   has much lower drift.
        # Time-density clamping (MIN_SEC_PER_LINE) still runs inside
        # `_build_shot_blocks_with_inline_vo`.

        # Style anchor: COMPACT form (2026-04-23). Leans on the keyframe
        # reference image to carry the actual painted-animation look; the
        # text line only names the style anchor so Seedance knows WHICH
        # tradition. Replaced the earlier verbose chinese_anchor paragraph
        # (palette / lighting / impasto / negation list) — that was
        # redundant with the ref image and burned prompt budget.
        from .style import load_style_anchor_compact
        cn_style = load_style_anchor_compact(self.bible, story.id if story else None)

        # Scene summary with canonical-label substitution. The Script-layer
        # `summary` was authored before canonical labels existed and still
        # contains names (楚子航, 朱子航) and story-role phrases (生父, 父亲,
        # 司机). Those collide with the canonical labels in the rest of
        # the prompt and confuse Seedance. We substitute each character's
        # name → its canonical_label so the summary stays consistent.
        summary = (scene.summary or scene.title or "").strip()[:120]
        for cid in char_ids:
            c = char_cache[cid]
            label = _canonical_role_cn(c)
            if c.name and c.name in summary:
                summary = summary.replace(c.name, label)
            # Also substitute parenthetical-stripped name (e.g. "男人（X 生父）" → "男人")
            # is lossy; skip. The user can tighten summary prose via script refine.

        # Audio/subtitle negation, compact form (2026-04-23 hand-edit test):
        # dropped former rule 3 (explicit env-sound allowlist — merged tail
        # into rule 2) and rule 5 (time-effects prohibition — Seedance's
        # default pacing turned out fine without the long warning). Kept
        # numbering gaps 1/2/4/6 intentionally so removals stay legible.
        # Rule 6 shortened to the speaker-label-lock one-liner.
        negation = (
            "音频 + 视频规则（HARD RULES）：\n"
            "1. 禁止任何外挂背景音乐 / BGM。整个 clip 中不得出现任何"
            "不属于剧情内音源的音乐。\n"
            "2. 剧情内音源（例如车载音响、酒吧扬声器、收音机）**仅** "
            "在 shot action 明确提及该音源的那一个 shot 的时间窗口内"
            "播放，不允许贯穿整个 clip。只有对白 + 环境音 + 动作声。\n"
            "4. 禁止字幕。\n"
            "6. 对白说话者严格以每行 [t=Xs] 时间戳后紧跟的 canonical "
            "label 为准。"
        )

        # Speaker cues block — surface each visible character's
        # voice_description so Seedance has a stable voice anchor per
        # speaker across scenes. Voice-only characters (no portrait,
        # not in `visible_char_ids`) ALSO get their cues listed here
        # because their VO still needs a voice (the render package
        # binds them without an @image slot). Added 2026-04-22.
        speaker_cues_lines: list[str] = []
        for cid, label in zip(char_ids, char_roles):
            c = char_cache.get(cid)
            voice = (getattr(c, "voice_description", None) or "").strip()
            if voice:
                speaker_cues_lines.append(f"- {label}: {voice}")
        speaker_cues_block = ""
        if speaker_cues_lines:
            speaker_cues_block = (
                "角色声音特征（speaker cues — 跨 scene 保持声线一致）：\n"
                + "\n".join(speaker_cues_lines)
            )

        # CLIP CONTINUITY (2026-04-23): the `narrative_spine` +
        # `transition_in/dramatic_turn/transition_out` echo that used to
        # live here (per `feedback_scene_shot_continuity.md`) has been
        # removed from the final Seedance prompt. Rationale: MIMO still
        # reads those fields during `_scene_to_user_prompt` and writes
        # pickup phrases into each shot's `action` prose, so Shot N>1
        # naturally opens "承接 Shot N-1…". Echoing the same beats in
        # a scene-level `衔接` block was redundant and crowded the
        # prompt. Upstream continuity support is preserved; only the
        # downstream emission is dropped.

        parts = [
            tag_line,
            "",
            negation,
            "",
            f"{total_dur}s · {scene.title}",
            "",
            f"画风：{cn_style}",
            "",
            f"概要：{summary}",
            "",
            "角色 ↔ 参考图：",
            char_block,
        ]
        if speaker_cues_block:
            parts.extend(["", speaker_cues_block])
        parts.append("")
        parts.extend(shot_blocks)
        return "\n".join(parts)


# ─── helpers ────────────────────────────────────────────────────────────

def _scene_to_user_prompt(
    scene: Scene,
    char_cache: dict[str, Character],
    story: Story | None = None,
) -> str:
    """Build the user-message context for MIMO shot generation.

    `story` (optional): full Story object. When provided, surfaces
    prior scenes sharing this scene's location so MIMO can carry
    established character positions forward (e.g. father drives,
    son in rear seat — established in s02, must persist in s05)."""
    # Canonical Chinese role label per character. MIMO must use ONLY these
    # labels in shot action prose — no names (楚子航/朱子航), no story-roles
    # (父亲/生父/司机), no English (young man). Divergence between labels
    # in action prose and @image binding causes Seedance position-teleport.
    #
    # Collision disambiguation: if two chars in scene share a base label
    # (e.g. two "少年"), we suffix with 甲/乙/... so MIMO writes distinct
    # subjects. Must match what the assembler uses at render time.
    raw_labels: list[str] = []
    cids_with_c: list[tuple[str, Character]] = []
    for cid in scene.characters_in_scene:
        c = char_cache.get(cid)
        if not c:
            continue
        cids_with_c.append((cid, c))
        raw_labels.append(_canonical_role_cn(c))
    canonical_labels = _disambiguate_labels(raw_labels)
    sheet_parts: list[str] = []
    for (cid, c), label in zip(cids_with_c, canonical_labels):
        sheet_parts.append(
            f"### Character: {c.id} ({c.name})\n"
            f"canonical_label (USE THIS in action prose — do NOT use name): {label}\n"
            f"visual_description: {c.visual_description[:200]}\n"
            f"default_outfit: {c.default_outfit[:100]}"
        )
    sheets = "\n\n".join(sheet_parts)
    label_rule = (
        "## Canonical labels — HARD RULE\n"
        "In every shot's `action` field, refer to characters ONLY by the "
        "canonical_label listed in the Cast block below. Do NOT use the "
        "character's name. Do NOT invent new labels (e.g. 父亲, 司机, 生父). "
        "Do NOT use English labels. Keep the same label for the same "
        "character across all shots — Seedance treats divergent labels "
        "as different people and teleports them between positions.\n"
        f"The labels to use in this scene: {', '.join(canonical_labels) or '(none)'}.\n"
    )
    dialogue_block = "\n".join(
        f"- {d.speaker_id}: {d.text}" for d in scene.dialogue
    ) or "(none)"
    transition_block = (
        f"transition_in (shape Shot 1 opening to match this handoff):\n"
        f"  {scene.transition_in or '(scene opens cold; no prior context)'}\n\n"
        f"dramatic_turn (reserve one shot's action for the pivot moment):\n"
        f"  {scene.dramatic_turn or '(no explicit turn — choose your own dominant beat)'}\n\n"
        f"transition_out (final shot must set up this handoff):\n"
        f"  {scene.transition_out or '(scene fades without specific handoff)'}\n"
    )
    # Surface the environment triplet prominently so MIMO can't miss it
    # when writing keyframe_prompt SCENE paragraphs (observed 2026-04-21:
    # MIMO routinely drops location/weather/time_of_day and writes only
    # subject prose, leaving Nano Banana to paint generic daytime streets
    # regardless of the script's "雨夜高架桥").
    env_callout = (
        "## Environment triplet (MUST appear in EVERY shot's keyframe_prompt "
        "SCENE paragraph, in English — MIMO drops this 90% of runs)\n"
        f"- time_of_day: {scene.time_of_day}\n"
        f"- weather/atmosphere: extract from location_description below\n"
        f"- location type: translate to English if the description is Chinese\n"
    )

    # Deterministic spatial-position extraction. Scans scene.action
    # (+ prior same-location scenes' actions if a Story is provided)
    # for Chinese spatial keywords tied to character names, and emits
    # an English structured block. MIMO reads this instead of re-inferring
    # positions from prose — which it does unreliably (observed s02 with
    # "楚天骄在驾驶座" clearly in action, MIMO still wrote "Beside him"
    # placing father in rear seat beside son).
    spatial_records: dict[str, str] = {}  # canonical_label → english_phrase
    # Pass 1: THIS scene (most authoritative, overrides prior).
    for label, phrase in _extract_spatial_positions(scene, char_cache):
        spatial_records[label] = f"{phrase} (from this scene's action)"
    # Pass 2: prior same-location scenes (fill gaps).
    if story is not None and scene.location_id:
        prior = [
            s for s in story.scenes
            if s.id < scene.id and s.location_id == scene.location_id
        ]
        for p in prior:
            for label, phrase in _extract_spatial_positions(p, char_cache):
                if label not in spatial_records:
                    spatial_records[label] = (
                        f"{phrase} (established in prior scene {p.id})"
                    )

    spatial_context = ""
    if spatial_records:
        lines = "\n".join(
            f"- {label}: {phrase}" for label, phrase in spatial_records.items()
        )
        spatial_context = (
            "## Character spatial positions — HARD RULE, MIMO MUST preserve "
            "these in every shot's keyframe_prompt SCENE paragraph\n"
            "(auto-extracted from scene.action + same-location prior scenes; "
            "positions are established by scripted prose, not guessed)\n"
            f"{lines}\n"
        )
    return (
        f"## Scene {scene.id}: {scene.title}\n"
        f"time_of_day: {scene.time_of_day}\n"
        f"emotional_register: {scene.emotional_register}\n"
        f"location_description: {scene.location_description[:300]}\n\n"
        f"summary: {scene.summary}\n\n"
        f"action (full prose):\n{scene.action[:1200]}\n\n"
        f"dialogue (verbatim — assign to shots, do NOT paraphrase):\n{dialogue_block}\n\n"
        f"beats:\n" + "\n".join(f"- {b}" for b in scene.beats) + "\n\n"
        f"## Connective tissue\n{transition_block}\n"
        f"{env_callout}\n"
        f"{spatial_context}"
        f"{label_rule}\n"
        f"## Cast\n{sheets}"
    )


# Chinese spatial keywords → English positional phrases. Used by the
# deterministic spatial-position extractor (observed 2026-04-21: MIMO
# fails to pull "楚天骄在驾驶座" from scene.action into keyframe_prompt
# ~50% of runs, even though scene.action is in context. Pre-parsing
# and surfacing as a structured block bypasses MIMO's attention drift).
_SPATIAL_KEYWORDS_CN_TO_EN: dict[str, str] = {
    "驾驶座": "driver's seat",
    "驾驶位": "driver's seat",
    "驾驶": "driving the vehicle",
    "副驾驶": "front passenger seat",
    "副驾": "front passenger seat",
    "前排": "front seat",
    "后排": "rear passenger seat",
    "后座": "rear seat",
    "靠坐": "seated (reclined)",
    "坐在": "seated",
    "站在": "standing at",
    "躺在": "lying on",
    "靠着": "leaning against",
    "挨着": "beside",
    "趴在": "leaning over",
    "从后视镜": "(visible in the rearview mirror)",
}


def _extract_spatial_positions(
    scene: Scene,
    char_cache: dict[str, Character],
) -> list[tuple[str, str]]:
    """Return [(canonical_label, english_spatial_phrase)] pairs extracted
    from scene.action. For each character in scene.characters_in_scene,
    find sentences where the character's name/aliases appear alongside a
    known Chinese spatial keyword; emit a structured pair so MIMO can't
    miss it. First match per character wins.
    """
    import re as _re
    if not scene.action:
        return []
    sentences = _re.split(r"[。.!?！？\n]", scene.action)
    seen: dict[str, str] = {}  # char_id → english_phrase (first hit wins)
    for sent in sentences:
        s = sent.strip()
        if not s:
            continue
        for cid in scene.characters_in_scene:
            if cid in seen:
                continue
            c = char_cache.get(cid)
            if not c:
                continue
            candidates = [c.name] + list(c.aliases or [])
            if not any(cand and cand in s for cand in candidates if cand):
                continue
            for kw_cn, kw_en in _SPATIAL_KEYWORDS_CN_TO_EN.items():
                if kw_cn in s:
                    seen[cid] = kw_en
                    break
    # Return in canonical role-label form so it matches the labels used
    # elsewhere in the prompt (MIMO writes those labels).
    out: list[tuple[str, str]] = []
    for cid, phrase in seen.items():
        c = char_cache.get(cid)
        if c:
            out.append((_canonical_role_cn(c), phrase))
    return out


def _resolve_location_description(bible: BibleStore, scene: Scene) -> str:
    if scene.location_id:
        try:
            loc = bible.load_location(scene.location_id)
            return loc.description
        except FileNotFoundError:
            pass
    return scene.location_description or "(unspecified)"


def _clamp_shot_durations(
    shots: list[Shot], target_sec: int = 15, min_shot_sec: int = 3
) -> list[Shot]:
    """Scale shot durations so total ≤ target_sec, preserving proportions.

    MIMO often emits scenes that run 18–22s even when told to aim for 15s.
    Seedance charges per second and longer scenes blow the char cap. This
    deterministic clamp rescales (floor, then shave from the longest) so
    the total lands at target_sec — while keeping every shot ≥ min_shot_sec
    so no shot becomes a blink.

    Non-mutating: returns copies with adjusted duration_sec.
    """
    if not shots:
        return shots
    total = sum(float(s.duration_sec) for s in shots)
    if total <= target_sec:
        return shots
    scale = target_sec / total
    new_durs = [max(min_shot_sec, int(round(float(s.duration_sec) * scale))) for s in shots]
    # Post-shave: if rounding pushed us back over, shave whole seconds from
    # the longest shots until we hit target_sec (never below min_shot_sec).
    while sum(new_durs) > target_sec:
        i = max(range(len(new_durs)), key=lambda k: new_durs[k])
        if new_durs[i] <= min_shot_sec:
            break
        new_durs[i] -= 1
    return [s.model_copy(update={"duration_sec": float(d)}) for s, d in zip(shots, new_durs)]


def _build_shot_blocks_with_inline_vo(
    *,
    shots: list[Shot],
    scene: Scene,
    char_cache: dict[str, Character],
    cid_to_label: dict[str, str],
    shot_idx_to_image_n: dict[int, int],
) -> list[str]:
    """Render shot blocks with dialogue CUES INLINE per shot.

    Output per shot:
        Shot N (durs, shot_type[, hard cut at Xs]): action @imageM: 此构图参考。
        [t=X.Xs] speaker (tone): line
        [t=X.Xs] speaker (tone): line
        (blank line)

    Rationale: binding each VO line to its shot visual context
    eliminates the silence-windows Seedance used to fill with
    improvised speech, and removes the need for an elaborate
    VERBATIM HARD RULES prose block at the prompt tail.

    Time-density clamp: each line needs at least MIN_SEC_PER_LINE
    of shot time; overflow is dropped (logged).
    """
    MIN_SEC_PER_LINE = 3.0

    blocks: list[str] = []
    cumulative_t = 0
    total_dropped = 0
    shot_start_f = 0.0
    for i, sh in enumerate(shots, start=1):
        # Header line.
        cut_note = ""
        if i > 1:
            cut_note = f", hard cut at {cumulative_t}s"
        header = (
            f"Shot {i} ({int(sh.duration_sec)}s, {sh.shot_type.value}{cut_note}): "
            f"{sh.action.strip()}"
        )
        if (i - 1) in shot_idx_to_image_n:
            header += f" @image{shot_idx_to_image_n[i - 1]}: 此构图参考。"
        block_lines = [header]

        # Inline VO for this shot.
        sh_dur = float(sh.duration_sec)
        lead_in = min(0.5, sh_dur * 0.2) if sh.dialogue else 0.0
        avail = max(sh_dur - lead_in, 0.1)
        shot_line_budget = max(1, int(avail // MIN_SEC_PER_LINE)) if sh.dialogue else 0
        kept = sh.dialogue[:shot_line_budget]
        total_dropped += len(sh.dialogue) - len(kept)
        n = len(kept)
        slot = avail / max(n, 1)
        for j, d in enumerate(kept):
            role = cid_to_label.get(
                d.speaker_id,
                _canonical_role_cn(char_cache.get(d.speaker_id)),
            )
            t_start = shot_start_f + lead_in + j * slot
            delivery = (d.delivery or "").strip()
            delivery_suffix = f" ({delivery})" if delivery else ""
            block_lines.append(
                f"[t={t_start:.1f}s] {role}{delivery_suffix}: {d.text}"
            )
        shot_start_f += sh_dur
        cumulative_t += int(sh.duration_sec)

        blocks.append("\n".join(block_lines))
        blocks.append("")  # blank line separator between shots

    if total_dropped > 0:
        log.info(
            "assemble_seedance: clamped %d VO line(s) (scene=%s; "
            "time-density gate \u2265%.1fs/line)",
            total_dropped, scene.id, MIN_SEC_PER_LINE,
        )

    # Drop trailing blank line.
    while blocks and blocks[-1] == "":
        blocks.pop()
    return blocks


def _role_noun(c: Character) -> str:
    g = (c.gender or "").lower()
    return {"male": "young man", "female": "young woman"}.get(g, "young person")


def _vo_role_tag(c: Character | None) -> str:
    """Anonymized role tag for VO speaker — never the character's name.

    Strategy:
    1. If the character has a role_in_story phrase mentioning a common
       relational role (father/mother/son/daughter/driver/passenger/
       stranger), use that as the qualifier.
    2. Otherwise fall back to gender+age-bucket ("young man",
       "middle-aged man", "young woman", ...).

    The video model only needs to distinguish speakers and apply correct
    voice timbre. Names in the prose would defeat IP-safety anonymization,
    so we never emit the character's real name.
    """
    if c is None:
        return "speaker"
    # Gender + age bucket only — this disambiguates speakers without
    # leaking names and without the false-match problem that arises
    # when one character's `role` sentence mentions the OTHER character's
    # relational role (e.g. son's role saying "defined by his father's
    # sacrifice" would falsely tag the son as '(father)').
    # The action prose around each VO line already tells the video model
    # who is speaking; the speaker tag only needs to distinguish voice
    # timbre (young male vs middle-aged male vs young female vs ...).
    # Non-mortal archetype — deities / spirits / timeless beings go to a
    # distinct "deity" bucket. Otherwise an `age` field like
    # "age-ambiguous" has no digits and no elderly/middle-aged keyword,
    # so the gender+age fallback below would route a god like Odin to
    # "young man" and Seedance would render him as a teenage boy.
    _deity_src = " ".join((
        (c.role or "").lower(),
        (c.visual_description or "").lower(),
        (c.personality or "").lower(),
        (c.backstory or "").lower(),
        (c.age or "").lower(),
    ))
    _deity_markers = (
        "deity", "goddess", "divine being", "immortal", "mythic",
        "ageless", "age-ambiguous", "神祇", "神明",
    )
    if any(m in _deity_src for m in _deity_markers):
        return "deity"

    noun = _role_noun(c)
    age_str = (c.age or "").lower()
    older = False
    elderly = False
    young_child = False
    import re as _re
    age_match = _re.search(r"\d+", age_str)
    if age_match:
        n = int(age_match.group())
        if n >= 60:
            elderly = True
        elif n >= 35:
            older = True
        elif n <= 12:
            young_child = True
    elif any(k in age_str for k in (
        "ancient", "timeless", "immortal", "eternal", "god", "divine",
        "seventies", "eighties", "nineties", "elderly", "elder"
    )):
        elderly = True
    elif any(k in age_str for k in (
        "forties", "fifties", "sixties", "middle-aged", "older"
    )):
        older = True
    elif any(k in age_str for k in ("child", "kid", "boy", "girl")) \
            and "teen" not in age_str and "young" not in age_str:
        young_child = True
    if elderly:
        if noun == "young man":
            noun = "elderly man"
        elif noun == "young woman":
            noun = "elderly woman"
    elif older:
        if noun == "young man":
            noun = "middle-aged man"
        elif noun == "young woman":
            noun = "middle-aged woman"
    elif young_child:
        if noun == "young man":
            noun = "boy"
        elif noun == "young woman":
            noun = "girl"
    return noun


def _canonical_role_cn(c: Character | None) -> str:
    """Chinese canonical role label for a character in Seedance prompts.

    Uses gender + age bucket ONLY — no names, no story-role phrases.
    The label must be STABLE across the prompt so Seedance can link
    speaker tags ↔ reference images ↔ action subjects. If we mixed
    names ('楚子航', '朱子航'), roles ('父亲', '司机'), and English
    labels ('young man') in one prompt — Seedance would treat them as
    different people and position-teleport between shots.

    Mapping (parallel to _vo_role_tag EN output):
      young man   → 少年（年轻男子）  (<=18: 少年, 19-34: 青年男子)
      middle-aged man → 中年男人
      young woman → 少女（年轻女子）
      middle-aged woman → 中年女人
      boy         → 男孩
      girl        → 女孩
    """
    if c is None:
        return "人物"
    en = _vo_role_tag(c)
    return {
        "young man": "少年",
        "middle-aged man": "中年男人",
        "elderly man": "老者",
        "young woman": "少女",
        "middle-aged woman": "中年女人",
        "elderly woman": "老妇",
        "boy": "男孩",
        "girl": "女孩",
        "young person": "人物",
        "speaker": "人物",
        "deity": "神祇",
    }.get(en, en)


def _disambiguate_labels(labels: list[str]) -> list[str]:
    """Add 甲/乙/丙/丁/戊 suffix to repeated canonical labels so each
    character in a scene has a UNIQUE label. Scene-level only — the
    same character may take the bare label in another scene where they
    don't collide. Order is stable: first occurrence becomes 甲, second
    becomes 乙, etc.

    Single occurrences stay bare (`少年`, not `少年甲`) to keep prompts
    clean when there's no ambiguity.
    """
    SUFFIXES = ["甲", "乙", "丙", "丁", "戊", "己", "庚", "辛"]
    counts: dict[str, int] = {}
    for L in labels:
        counts[L] = counts.get(L, 0) + 1
    seen: dict[str, int] = {}
    out: list[str] = []
    for L in labels:
        if counts[L] <= 1:
            out.append(L)
            continue
        idx = seen.get(L, 0)
        seen[L] = idx + 1
        suffix = SUFFIXES[idx] if idx < len(SUFFIXES) else str(idx + 1)
        out.append(f"{L}{suffix}")
    return out


def _compact_char_line(c: Character, role: str) -> str:
    """Ultra-short role tag — full appearance lives in the @image ref."""
    noun = _vo_role_tag(c)
    # Only the first clause of visual_description, max 50 chars. The
    # reference image carries the full visual fingerprint; prose is
    # only a voice/age anchor so Seedance picks the right speaker.
    vd = c.visual_description[:60].split(".")[0].split(",")[0]
    return f"{noun}: {vd.strip()}"


# ─── compat shim for legacy ShotGenAgent ────────────────────────────────
# Old callers import `build_scene_brief(story, scene, char_cache, loc_desc)` and
# expect a schema-format Seedance prompt. The new pipeline stores this on
# `StoryboardScene.seedance_prompt`; for legacy compatibility we re-assemble
# from the Scene directly.

def reassemble_seedance_prompt(
    bible: BibleStore,
    story: Story,
    sb_scene: StoryboardScene,
    shots_with_keyframes: list[str] | None = None,
    n_style_refs: int = 0,
) -> str:
    """Deterministically rebuild a scene's seedance_prompt from current code.

    `shots_with_keyframes` (list of shot ids): if provided, assembler emits
    per-shot @imageN tags inline and the tag/binding line lists shot
    keyframes first. When None, falls back to legacy scene-level keyframe
    layout.

    `n_style_refs`: number of user-curated style reference images the
    caller will upload alongside the prompt. Assembler allocates @imageN
    slots for them after shot keyframes (or after the scene keyframe in
    legacy mode), before character refs. Caller must upload images in
    exactly that order.
    """
    scene = next((s for s in story.scenes if s.id == sb_scene.scene_id), None)
    if scene is None:
        return sb_scene.seedance_prompt
    char_cache: dict[str, Character] = {
        cid: bible.load_character(cid) for cid in story.characters
    }
    style_prompt = load_style_prompt(bible, story.id)
    agent = StoryboardAgent(llm=None, bible=bible)  # type: ignore[arg-type]
    return agent._assemble_seedance_prompt(
        story, scene, sb_scene.shots, char_cache, style_prompt,
        shots_with_keyframes=shots_with_keyframes,
        n_style_refs=n_style_refs,
    )


def build_scene_brief(
    story: Story,
    scene: Scene,
    char_cache: dict[str, Character],
    location_description: str | None = None,
) -> str:  # noqa: ARG001 — kept for API compat; location_description unused here
    """Legacy adapter — builds a Seedance brief for a single Scene."""
    from dataclasses import dataclass as _dc  # local import to avoid cycle

    # Create an ad-hoc agent (BibleStore is None-safe here — we don't use it
    # in _assemble_seedance_prompt except for location lookup). For legacy
    # paths, fall back to scene.location_description.
    class _Stub:
        pass
    stub = _Stub()
    stub.root = None  # type: ignore[attr-defined]
    agent = StoryboardAgent(llm=None, bible=stub)  # type: ignore[arg-type]
    # Legacy callers get a neutral fallback style since they don't have a story_id.
    from .style import DEFAULT_STYLE_FALLBACK
    return agent._assemble_seedance_prompt(
        story, scene, scene.shots, char_cache, DEFAULT_STYLE_FALLBACK,
    )


_SHOT_TYPES = {v.value for v in ShotType}
_CAMERA_MOVES = {v.value for v in CameraMovement}


def _normalize_shot_dict(d: dict) -> dict:
    """Auto-repair common LLM confusions before pydantic validation."""
    st = d.get("shot_type")
    cm = d.get("camera_movement")
    if st in _CAMERA_MOVES and st not in _SHOT_TYPES:
        d["camera_movement"] = st
        d["shot_type"] = "medium" if cm not in _SHOT_TYPES else cm
    # Map common LLM-invented shot_type values to nearest valid option.
    _SHOT_ALIAS = {
        "high_angle": "wide", "low_angle": "wide",
        "aerial": "wide", "overhead": "wide",
        "bird_eye": "wide", "birds_eye": "wide",
        "extreme_wide": "wide", "long": "wide",
        "establishing": "wide",
        "two_shot": "medium", "2_shot": "medium",
        "detail": "insert", "cutaway": "insert",
        "dutch": "medium", "dutch_angle": "medium",
        "ots": "over_shoulder",
    }
    if d.get("shot_type") in _SHOT_ALIAS:
        d["shot_type"] = _SHOT_ALIAS[d["shot_type"]]
    if d.get("shot_type") not in _SHOT_TYPES:
        d["shot_type"] = "medium"

    # Same for camera_movement — commonly LLM invents values like "high_angle".
    _CAM_ALIAS = {
        "zoom_in": "dolly_in", "zoom": "dolly_in",
        "zoom_out": "dolly_out",
        "truck": "tracking", "follow": "tracking",
        "high_angle": "static", "low_angle": "static",
        "aerial": "crane", "overhead": "crane",
        "push_in": "dolly_in", "pull_out": "dolly_out",
    }
    if d.get("camera_movement") in _CAM_ALIAS:
        d["camera_movement"] = _CAM_ALIAS[d["camera_movement"]]
    if "camera_movement" not in d or d.get("camera_movement") not in _CAMERA_MOVES:
        d["camera_movement"] = "static"

    if isinstance(d.get("duration_sec"), str):
        try:
            d["duration_sec"] = float(d["duration_sec"])
        except ValueError:
            d["duration_sec"] = 5.0
    # Clip duration to allowed range [1,30].
    dur = d.get("duration_sec", 5.0)
    d["duration_sec"] = max(1.0, min(30.0, float(dur)))
    return d
