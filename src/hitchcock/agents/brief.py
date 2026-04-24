"""Brief Agent — Stage 0 of the pipeline.

Captures directorial intent (user Q&A) + external canon research into a
single `DirectorialBrief` artifact that every downstream agent (STYLE,
CAST, SCRIPT) can optionally read for higher-fidelity output.

Three sub-operations:
  1. `list_questions()` → returns the fixed question spec as JSON.
     The AI driver (or human) answers these and writes the result to a
     JSON file that `ingest_answers()` consumes.
  2. `ingest_answers(answers, source_text)` → validates answers, creates
     an initial DirectorialBrief (no canon yet).
  3. `plan_research(brief, source_text)` → produces a research plan:
     suggested web-search queries per named character + world, plus an
     empty canon-facts template. The AI driver (with WebSearch tools)
     fills in the template and feeds it back via `ingest_canon()`.
  4. `refine(brief, feedback)` → MIMO-revises the brief per natural-
     language feedback (drop a canon fact, adjust an answer, etc.)

Design note: the research sub-step is intentionally ASSISTED, not
automated. It outputs search queries + a schema to fill in. This lets
the pipeline work regardless of which search provider the driver uses
(Claude WebSearch, Gemini grounding, Serper API, manual copy-paste).
"""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from typing import Any

from ..bible import (
    BibleStore,
    BriefAnswers,
    CharacterCanon,
    DirectorialBrief,
)
from ..llm import GeminiTextClient, GeminiTextError, MimoClient

log = logging.getLogger(__name__)


# ── Question specification (static) ─────────────────────────────────────

_QUESTIONS: list[dict] = [
    {
        "id": "form",
        "question": "What form is the film? (e.g. '3-minute animated short', '5-minute dramatic scene excerpt', '60-second MV', '2-minute wordless landscape piece')",
        "field": "form",
        "type": "str",
        "required": True,
    },
    {
        "id": "target_audience",
        "question": "Who is the target audience? (e.g. 'fans of the source novel', 'general audience unfamiliar with the IP', 'festival-circuit adult viewers')",
        "field": "target_audience",
        "type": "str",
        "required": True,
    },
    {
        "id": "style_references",
        "question": "Up to 3 concrete art/style reference works or artists the film should evoke (e.g. 'Spider-Verse', 'Arcane', 'Moebius', 'Song-dynasty shan-shui').",
        "field": "style_references",
        "type": "list[str]",
        "required": True,
    },
    {
        "id": "emotional_tone",
        "question": "2-3 tonal descriptors (e.g. ['bittersweet', 'reverent'], ['kinetic', 'comedic', 'operatic']).",
        "field": "emotional_tone",
        "type": "list[str]",
        "required": True,
    },
    {
        "id": "must_haves",
        "question": "Non-negotiable moments/beats/details the final film MUST contain (e.g. 'preserve the fireworks sequence', 'end on the NoNo Happy Birthday spell'). List any number.",
        "field": "must_haves",
        "type": "list[str]",
        "required": False,
    },
    {
        "id": "must_avoids",
        "question": "Hard vetos — what the film MUST NOT do (e.g. 'no ballroom comedy scenes', 'no anime-style chibi faces'). List any number.",
        "field": "must_avoids",
        "type": "list[str]",
        "required": False,
    },
    {
        "id": "pacing",
        "question": "Overall tempo preference. One of: 'dialogue-heavy', 'action-driven', 'balanced', 'contemplative with long holds'.",
        "field": "pacing",
        "type": "str",
        "required": False,
    },
    {
        "id": "music_direction",
        "question": "Initial BGM guidance — even rough. (e.g. 'solo piano + occasional strings', 'electronic percussion with Chinese flute accents', 'wordless choral'.)",
        "field": "music_direction",
        "type": "str",
        "required": False,
    },
]


# ── Meta-prompts ────────────────────────────────────────────────────────

_RESEARCH_PLAN_SYSTEM = """You are a research planner. Given a source text,
produce a RESEARCH PLAN that tells an external search agent what to look
up about each CHARACTER and about the world/setting, so downstream agents
can enrich their proposals with canon facts not present in this source
text.

Input:
- The source text (current chapter/scene)

Output JSON with exactly these keys:
  character_queries: list of {"character_alias": "<name>", "search_queries": ["q1", "q2"]}
  world_queries:     list of strings (queries about setting, institutions, objects)

HARD RULE — WHO COUNTS AS A CHARACTER (read carefully):

Do NOT limit yourself to proper-noun-named characters. Include EVERY
substantive agent in the source, including those referenced only by
role / kinship / occupation terms. An agent is "substantive" if they
(a) speak at least one line, OR (b) drive an action beat visible to the
protagonist. Examples:
  ✓ 楚子航 (proper name)              → include
  ✓ 柳淼淼 (proper name)              → include
  ✓ 父亲 / 爸爸 / 司机 (role-only)    → include — pick the role term the
                                       source uses most often as alias,
                                       and add a search query guessing
                                       the canonical fan-wiki name
                                       ("龙族 楚子航 父亲 canonical
                                       appearance" / "Dragon Raja Chu
                                       Zihang father canon name")
  ✓ 女人 / 老者 / 师傅 (role-only)    → include — same treatment
  ✗ background crowd, unnamed passersby → skip

For adaptations (source is a scene from a well-known work), role-only
characters almost always have canonical fan-wiki names. Your search
queries should try BOTH the role term AND a guessed canonical full name,
so the grounded-search step has a chance of surfacing canonical visual
/ personality info even when the source text is sparse on physical
description.

For characters, emit 1-3 queries per character, favoring queries that
will surface CANONICAL VISUAL + PERSONALITY details (hair color, eye
color, BUILD / body type / physique, wardrobe signature, iconic
mannerisms) and role in the broader source (protagonist / rival /
mentor / etc.). Include "build" / "体型" / "身材" as explicit keywords
in at least one query per character — downstream T2I defaults to "lean
average build" when this info isn't in canon, which is often wrong for
fighter/warrior/athlete characters.

Prefer Chinese queries if the source is Chinese; English for English
sources.

Example output:
{
  "character_queries": [
    {"character_alias": "诺诺",
     "search_queries": ["龙族 诺诺 陈墨瞳 外貌 红发 身材", "Dragon Raja Chen Mo-tong canonical appearance build"]},
    {"character_alias": "路明非",
     "search_queries": ["龙族 路明非 外貌 身材 性格 canonical"]},
    {"character_alias": "父亲",
     "search_queries": ["龙族 楚子航 父亲 楚天骄 外貌 身材 canonical", "Dragon Raja Chu Zihang father canon build physique"]}
  ],
  "world_queries": [
    "龙族 卡塞尔学院 setting visual",
    "Dragon Raja academy canonical look"
  ]
}
"""


_PARSE_INTENT_SYSTEM = """You are parsing a director's free-form intent paragraph
into a structured BriefAnswers JSON. The director dropped a single paragraph
(or a couple of sentences) describing what they want — tone, style, must-haves,
must-avoids, etc. Your job is to extract each concrete directive into the
right field. When the paragraph is silent on a field, INFER a sensible
default from the source text context — don't leave load-bearing fields empty.

Output JSON with exactly these keys (the BriefAnswers schema):

  form: str
    What kind of film. Infer from source length if user didn't say (default:
    "3-minute animated short" for ~20k-char source, "5-minute" for ~40k,
    "10-minute" for feature-length).
  target_audience: str
    Default: "fans of the source material + viewers open to stylized
    animation" if user didn't specify.
  style_references: list[str]
    1-3 concrete art/style works the film should evoke. Extract any the
    user named. If none, default to ["Arcane by Fortiche Studio"] (our
    pipeline's default painterly anchor).
  emotional_tone: list[str]
    2-3 tonal descriptors extracted from the paragraph (e.g.
    ["somber", "alienated"], ["kinetic", "operatic"]). If absent, infer
    from the source text's register.
  must_haves: list[str]
    Non-negotiable beats / visual elements the user called out. Extract
    verbatim-ish phrasing. OK to be empty if user didn't say.
  must_avoids: list[str]
    Things the user said to avoid. Also fold in pipeline defaults the
    user's paragraph doesn't contradict: "anime-style chibi proportions",
    "photorealistic rendering", "glossy commercial animation",
    "mobile-game gacha aesthetic", "external BGM / added soundtrack
    (diegetic in-world music is OK)".
  pacing: str
    "contemplative", "dialogue-heavy", "action-driven", or "balanced".
    Infer from source text's register if user didn't say.
  music_direction: str
    Default "minimal — diegetic sources only" unless user specified.

Do NOT invent user intent that isn't implied. Do NOT drop an explicit
user constraint (e.g. "all characters are Asian" — put that verbatim into
`must_haves`). If the user's paragraph is in Chinese, translate
requirements to English for downstream agents but preserve proper nouns
and brand names (Arcane, Ghibli, 楚子航).
"""


_REFINE_SYSTEM = """You are revising a DirectorialBrief based on director feedback.
Preserve fields the feedback doesn't touch; revise what it does.

Typical feedback:
- "drop canon fact about X"
- "change form to 5-min drama"
- "add 'Ghibli' to style_references"
- "tighten the music direction to 'solo piano only'"

Return JSON with the same schema: story_id, answers (BriefAnswers),
canon_facts (list of CharacterCanon), world_canon (str), research_sources (list).

Do NOT fabricate canon facts — if feedback asks to add canon info, it must
be provided in the feedback itself with sources. If feedback asks to
research something new, set canon_facts to empty and instruct the user
to re-run `brief research`.
"""


# ── Agent ───────────────────────────────────────────────────────────────

@dataclass
class BriefAgent:
    llm: MimoClient
    bible: BibleStore

    # ── Q&A ─────────────────────────────────────────────────────────────

    @staticmethod
    def list_questions() -> list[dict]:
        """Returns the single free-form intent prompt. Replaces the old
        8-question Q&A flow — one paragraph from the user covers
        everything, MIMO parses structured fields via `parse_intent`."""
        return [
            {
                "id": "director_intent",
                "question": (
                    "用一段话（中文或英文）描述你想要什么样的片子。可以"
                    "包括任何你在意的：时长、情绪、视觉风格参考、必须保留"
                    "的元素、要避免的东西、配乐偏好……都说或只说一条都行。"
                    "留空的部分 pipeline 会按默认值（Arcane 画风、3 分钟短片、"
                    "无外挂 BGM）处理。"
                ),
                "field": "director_intent",
                "type": "str",
                "required": True,
            }
        ]

    def parse_intent(
        self, story_id: str, director_intent: str, source_text: str = "",
    ) -> DirectorialBrief:
        """Parse a director's free-form intent paragraph into structured
        BriefAnswers via MIMO. Source text is optional context for
        defaults (form inferred from length, tone from register)."""
        user = (
            f"## Director's free-form intent (a paragraph in user's words)\n"
            f"{director_intent.strip()}\n\n"
            f"## Source text (for inferring defaults — first 4000 chars)\n"
            f"{source_text.strip()[:4000]}"
        )
        log.info("Brief Agent: parsing free-form intent via MIMO (%d chars)",
                 len(director_intent))
        data = self.llm.chat_json(
            system=_PARSE_INTENT_SYSTEM, user=user,
            max_tokens=2048, temperature=0.3,
        )
        validated = BriefAnswers.model_validate(data)
        return DirectorialBrief(story_id=story_id, answers=validated)

    def ingest_answers(
        self, story_id: str, answers: dict[str, Any],
    ) -> DirectorialBrief:
        """Legacy structured path (still supported for direct JSON input).
        New free-form path: see `parse_intent()`."""
        validated = BriefAnswers.model_validate(answers)
        return DirectorialBrief(story_id=story_id, answers=validated)

    # ── Research planning ───────────────────────────────────────────────

    def plan_research(
        self, brief: DirectorialBrief, source_text: str,
    ) -> dict:
        """Produce a research plan: search queries + empty canon template.
        The caller (AI driver) runs the searches and fills the template,
        then feeds it back via `ingest_canon()`."""
        user_prompt = (
            "## Source text (first 6000 chars)\n"
            + source_text.strip()[:6000]
        )
        log.info("Brief Agent: planning research via MIMO...")
        data = self.llm.chat_json(
            system=_RESEARCH_PLAN_SYSTEM, user=user_prompt,
            max_tokens=2048, temperature=0.3,
        )

        # Build an empty canon template the caller can fill.
        template: list[dict] = []
        for cq in data.get("character_queries", []):
            template.append({
                "character_alias": cq.get("character_alias", ""),
                "search_queries": cq.get("search_queries", []),
                "canonical_appearance": "",     # to be filled by caller
                "canonical_personality": "",
                "canonical_role": "",
                "sources": [],
            })
        return {
            "story_id": brief.story_id,
            "plan": {
                "character_queries": data.get("character_queries", []),
                "world_queries": data.get("world_queries", []),
            },
            "canon_template": template,
            "instructions": (
                "1. Run the search queries using your preferred search tool "
                "(WebSearch / Gemini grounding / manual). 2. Fill each "
                "canon_template entry's `canonical_*` fields with findings. "
                "3. Populate `sources` with URLs consulted. 4. Optionally "
                "add a `world_canon` string summarizing setting context. "
                "5. Write the filled template to a JSON file and call "
                "`hitchcock brief ingest-canon -s <sid> --file <path>`."
            ),
        }

    def research_canon(
        self,
        brief: DirectorialBrief,
        source_text: str,
        gemini: GeminiTextClient,
    ) -> DirectorialBrief:
        """Automate the canon-research step that was previously manual.

        Flow:
          1. Call `plan_research()` to generate per-character search queries.
          2. For each character, ask Gemini (with Google Search grounding)
             for appearance + personality + role in ONE grounded call.
          3. Parse the structured answer into CharacterCanon.
          4. Run world_queries similarly; aggregate into `world_canon`.
          5. Return updated DirectorialBrief with canon_facts + sources.

        Cheaper than MIMO-only because Gemini grounding scopes queries
        to authoritative web sources (fan wikis, novel summaries)
        instead of asking the LLM to hallucinate canon details.
        """
        plan = self.plan_research(brief, source_text)

        canon_facts: list[CharacterCanon] = []
        all_sources: list[str] = []

        # Per-character: one structured grounded query pulls appearance +
        # personality + role together. Parsing by section header keeps
        # the extractor deterministic.
        for cq in plan["plan"]["character_queries"]:
            alias = cq.get("character_alias", "").strip()
            if not alias:
                continue
            query_hints = " ".join(cq.get("search_queries", [])[:3])
            prompt = (
                f"Research the character `{alias}` in their canonical source "
                f"material (novels, fan wiki, official materials). "
                f"Search hints: {query_hints}\n\n"
                f"Answer with EXACTLY these three labeled lines, concise "
                f"English (1-2 sentences each). If a field is unknown, "
                f"leave it blank after the colon.\n\n"
                f"APPEARANCE: hair color + eye color + build + signature "
                f"wardrobe (iconic pieces only, not generic).\n"
                f"PERSONALITY: 2-3 defining traits.\n"
                f"ROLE: position in the source's plot and world."
            )
            log.info("Brief research: grounded query for %r", alias)
            try:
                ans = gemini.generate_grounded(prompt, max_output_tokens=1024)
            except GeminiTextError as e:
                log.warning("Brief research: %s failed (%s) — skipping", alias, e)
                continue
            appearance, personality, role = _parse_apr(ans.text)
            canon_facts.append(CharacterCanon(
                character_alias=alias,
                canonical_appearance=appearance,
                canonical_personality=personality,
                canonical_role=role,
                sources=ans.sources,
            ))
            all_sources.extend(ans.sources)

        # World: ask one grounded query per world-level prompt.
        world_parts: list[str] = []
        for wq in plan["plan"]["world_queries"]:
            log.info("Brief research: world query %r", wq)
            try:
                ans = gemini.generate_grounded(
                    f"{wq}\n\nAnswer in 2-3 concise English sentences.",
                    max_output_tokens=512,
                )
            except GeminiTextError as e:
                log.warning("Brief research: world query failed (%s) — skipping", e)
                continue
            if ans.text:
                world_parts.append(ans.text.strip())
            all_sources.extend(ans.sources)

        world_canon = "\n\n".join(world_parts) if world_parts else brief.world_canon
        # Dedupe sources, preserve order.
        seen: set[str] = set()
        dedup_sources = []
        for s in all_sources:
            if s and s not in seen:
                seen.add(s)
                dedup_sources.append(s)

        return DirectorialBrief(
            story_id=brief.story_id,
            answers=brief.answers,
            canon_facts=canon_facts,
            world_canon=world_canon,
            research_sources=dedup_sources,
        )

    def ingest_canon(
        self,
        brief: DirectorialBrief,
        canon_file: dict,
    ) -> DirectorialBrief:
        """Ingest a filled canon template (from the caller's research work).

        canon_file format:
          {
            "canon_facts": [
              {"character_alias": "诺诺", "canonical_appearance": "...",
               "canonical_personality": "...", "canonical_role": "...",
               "sources": ["https://..."]},
              ...
            ],
            "world_canon": "...",      # optional
            "research_sources": [...]  # aggregated URLs (optional)
          }
        """
        facts = [
            CharacterCanon.model_validate(f)
            for f in canon_file.get("canon_facts", [])
        ]
        return DirectorialBrief(
            story_id=brief.story_id,
            answers=brief.answers,
            canon_facts=facts,
            world_canon=canon_file.get("world_canon", brief.world_canon),
            research_sources=canon_file.get(
                "research_sources", brief.research_sources,
            ),
        )

    # ── Refine ──────────────────────────────────────────────────────────

    def refine(self, brief: DirectorialBrief, feedback: str) -> DirectorialBrief:
        user_prompt = (
            "## Current DirectorialBrief JSON\n```json\n"
            + brief.model_dump_json(indent=2)
            + "\n```\n\n## Director feedback\n" + feedback.strip()
        )
        # 16384 (was 4096): after `brief research` runs, canon_facts +
        # research_sources carry 30-50 verbatim URLs that MIMO has to
        # echo back in the refined JSON. 4096 truncates mid-URL.
        data = self.llm.chat_json(
            system=_REFINE_SYSTEM, user=user_prompt,
            max_tokens=16384, temperature=0.3,
        )
        data["story_id"] = brief.story_id
        return DirectorialBrief.model_validate(data)


# ── Helper: parse structured APPEARANCE/PERSONALITY/ROLE answer ────────

def _parse_apr(text: str) -> tuple[str, str, str]:
    """Extract APPEARANCE / PERSONALITY / ROLE sections from Gemini's answer.

    Section-based parse (not line-based) — Gemini may wrap long answers
    onto multiple lines, and we want the full content per section. We
    split the text on the next section header and take everything between.
    Tolerant of casing, markdown bold wrappers (**APPEARANCE:**), inline
    citation markers ([cite: N, M]).
    """
    import re as _re
    # Find each header + its start index. Order in input may vary.
    keys = ["APPEARANCE", "PERSONALITY", "ROLE"]
    header_re = _re.compile(
        r"\*?\*?(APPEARANCE|PERSONALITY|ROLE)\*?\*?\s*:",
        _re.IGNORECASE,
    )
    hits: list[tuple[str, int, int]] = []  # (key_upper, header_start, content_start)
    for m in header_re.finditer(text):
        key = m.group(1).upper()
        hits.append((key, m.start(), m.end()))
    if not hits:
        return "", "", ""
    # Each section's content runs from content_start to the next header's header_start.
    result = {k: "" for k in keys}
    for i, (key, _hs, cs) in enumerate(hits):
        end = hits[i + 1][1] if i + 1 < len(hits) else len(text)
        content = text[cs:end].strip()
        # Strip trailing punctuation, inline citation markers.
        content = _re.sub(r"\s*\[cite:[^\]]*\]\s*", " ", content).strip()
        content = content.rstrip(".").strip()
        if key in result and content:
            result[key] = content
    return result["APPEARANCE"], result["PERSONALITY"], result["ROLE"]


# ── Helper: resolve approved brief for downstream agents ────────────────

def load_brief(bible: BibleStore, story_id: str | None) -> DirectorialBrief | None:
    """Downstream helper: returns the approved DirectorialBrief if present,
    else None. Used by StyleAgent, CastAgent, ScriptAgent to enrich prompts
    with user intent + canon facts when available."""
    if story_id is None:
        return None
    from ..bible import StageName
    try:
        b = bible.load_approved(story_id, StageName.BRIEF)
    except (FileNotFoundError, ValueError, OSError) as e:
        log.debug(
            "load_brief: %s has no approved brief (%s: %s) — downstream "
            "agents will run without brief context",
            story_id, type(e).__name__, e,
        )
        return None
    if b is None:
        return None
    assert isinstance(b, DirectorialBrief)
    return b
