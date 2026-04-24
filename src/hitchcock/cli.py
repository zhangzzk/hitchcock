"""Unified Hitchcock CLI.

Contract: see AGENTS.md. Single binary `hitchcock <subcommand> <verb> [args]`
drives the four-stage gate-based pipeline. AI agents and humans both talk to
this CLI — never to hand-edited Python content.

Subcommands implemented:
  init           — bootstrap a story
  script         — Layer 1 (source → structured story)
  storyboard     — Layer 2 (story → per-scene shot breakdowns + prompts)   [stub]
  art            — Layer 3 (storyboard → scene-art candidates)             [stub]
  render         — Layer 4 (storyboard + art → video clips + reel)         [stub]
  status         — per-stage state + next_action for this story
  design         — one-shot character creation (bypasses the gate loop)
  location       — one-shot location creation (bypasses the gate loop)
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any

import hashlib
import shutil

from .agents import (
    BriefAgent,
    CastAgent,
    DesignAgent,
    LocationAgent,
    ScriptAgent,
    StoryboardAgent,
    StyleAgent,
    TTSAgent,
)
from .bible import (
    ArtCandidate,
    ArtManifest,
    ArtScene,
    BibleStore,
    CastManifest,
    DirectorialBrief,
    MatchStatus,
    RenderBackend,
    RenderManifest,
    RenderScene,
    StageName,
    StageState,
    Story,
    Storyboard,
    StoryInit,
    StyleGuide,
)
from .config import Settings, load_settings
from .image import NanoBananaClient
from .image.gpt_image import GPTImageClient
from .llm import MimoClient


CONTRACT_VERSION = "0.2.0"

log = logging.getLogger("hitchcock.cli")


# ─── Error handling ──────────────────────────────────────────────────────

class CliError(Exception):
    """Raised for user-visible CLI failures. Maps to an error code in stderr."""

    def __init__(self, code: str, msg: str, exit_code: int = 2):
        super().__init__(msg)
        self.code = code
        self.msg = msg
        self.exit_code = exit_code


def _emit_error(err: CliError) -> None:
    print(f"hitchcock-error: {err.code}: {err.msg}", file=sys.stderr)


# ─── Shared argparse helpers ─────────────────────────────────────────────

def _add_common_flags(p: argparse.ArgumentParser) -> None:
    p.add_argument("--bible-dir", default=None,
                   help="Override bible root (default: from env HITCHCOCK_BIBLE_DIR or ./bible).")
    p.add_argument("--json", dest="json_out", action="store_true",
                   help="Emit machine-readable JSON to stdout.")
    p.add_argument("-v", "--verbose", action="store_true")


def _add_image_quality_flag(p: argparse.ArgumentParser) -> None:
    """Add `--quality {low,medium,high,auto}` override for T2I cost/fidelity.
    Medium is ~4× cheaper than high (≈¥0.4 vs ¥1.8 per 1536×1024); use it
    for draft iteration, switch to high for final production."""
    p.add_argument(
        "--quality", default=None,
        choices=["low", "medium", "high", "auto"],
        help=(
            "GPT Image quality override (falls back to HITCHCOCK_OPENAI_"
            "IMAGE_QUALITY env var, default 'high'). 'medium' ≈ 4× cheaper "
            "for draft iteration, 'high' for final."
        ),
    )


def _gpt_image_client(settings: Settings, args: argparse.Namespace) -> "GPTImageClient":
    """Factory that respects an optional `--quality` CLI override."""
    quality = getattr(args, "quality", None)
    return GPTImageClient(settings.openai, quality=quality)


def _bible_from(settings: Settings, override: str | None) -> BibleStore:
    root = Path(override).resolve() if override else settings.bible_dir
    return BibleStore(root)


def _setup_logging(verbose: bool) -> None:
    logging.basicConfig(
        level=logging.INFO if verbose else logging.WARNING,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    logging.getLogger("hitchcock").setLevel(logging.INFO)


def _read_input(arg: str) -> str:
    if arg == "-":
        return sys.stdin.read()
    return Path(arg).read_text(encoding="utf-8")


def _emit(payload: Any, *, as_json: bool) -> None:
    """Emit a dict either as JSON (machine) or pretty-printed (human)."""
    if as_json:
        print(json.dumps(payload, ensure_ascii=False, indent=2))
    else:
        _pretty(payload)


def _pretty(obj: Any, indent: int = 0) -> None:
    pad = "  " * indent
    if isinstance(obj, dict):
        for k, v in obj.items():
            if isinstance(v, (dict, list)):
                print(f"{pad}{k}:")
                _pretty(v, indent + 1)
            else:
                print(f"{pad}{k}: {v}")
    elif isinstance(obj, list):
        for item in obj:
            if isinstance(item, (dict, list)):
                print(f"{pad}-")
                _pretty(item, indent + 1)
            else:
                print(f"{pad}- {item}")
    else:
        print(f"{pad}{obj}")


# ─── init ────────────────────────────────────────────────────────────────

def _cmd_init(args: argparse.Namespace) -> int:
    settings = load_settings()
    bible = _bible_from(settings, args.bible_dir)

    # Legacy: if --character/--location given, validate. Primary flow expects
    # these to be empty and uses `hitchcock cast discover` instead.
    for cid in (args.character or []):
        if not bible.character_json(cid).exists():
            raise CliError(
                "UNKNOWN_CHARACTER",
                f"character '{cid}' not found at {bible.character_json(cid)}. "
                f"Either drop --character (recommended — `hitchcock cast discover` "
                f"will extract + build automatically), or run `hitchcock design` first.",
            )
    for lid in (args.location or []):
        if not bible.location_json(lid).exists():
            raise CliError(
                "UNKNOWN_LOCATION",
                f"location '{lid}' not found at {bible.location_json(lid)}.",
            )

    source_text = _read_input(args.source)
    if not source_text.strip():
        raise CliError("EMPTY_SOURCE", "source text is empty")

    bible.init_story(args.story_id)
    src_path = bible.save_source_text(args.story_id, source_text)
    init = StoryInit(
        story_id=args.story_id,
        character_ids=list(args.character or []),
        location_ids=list(args.location or []),
        source_text_path=str(src_path.relative_to(bible.root)),
    )
    bible.save_story_init(init)

    _emit(
        {
            "story_id": args.story_id,
            "story_dir": str(bible.story_dir(args.story_id)),
            "source_chars": len(source_text),
            "prepopulated_characters": init.character_ids,
            "prepopulated_locations": init.location_ids,
            "next_action": f"hitchcock brief questions -s {args.story_id}",
        },
        as_json=args.json_out,
    )
    return 0


# ─── brief (Stage 0) ─────────────────────────────────────────────────────

def _cmd_brief_questions(args: argparse.Namespace) -> int:
    """Emit the fixed list of directorial questions (machine-readable)."""
    qs = BriefAgent.list_questions()
    if args.json_out:
        print(json.dumps(
            {"story_id": args.story_id, "questions": qs,
             "answer_with": f"hitchcock brief answer -s {args.story_id} --responses <path.json>"},
            ensure_ascii=False, indent=2,
        ))
    else:
        print(f"=== Brief questions for '{args.story_id}' ===\n")
        for q in qs:
            req = " (required)" if q["required"] else " (optional)"
            print(f"[{q['id']}]{req}  (field: {q['field']}, type: {q['type']})")
            print(f"  {q['question']}\n")
        print("Answer by writing a JSON file with { field_name: value, ... }")
        print(f"Then: hitchcock brief answer -s {args.story_id} --responses <path.json>")
    return 0


def _cmd_brief_answer(args: argparse.Namespace) -> int:
    settings = load_settings()
    bible = _bible_from(settings, args.bible_dir)
    _require_init(bible, args.story_id)

    agent = BriefAgent(llm=MimoClient(settings.mimo), bible=bible)

    # Two input modes:
    #   --intent "free-form paragraph"  → MIMO parses to structured
    #   --responses file.json           → legacy structured JSON
    intent = getattr(args, "intent", None)
    if intent:
        # Read source text for defaults inference (optional but helpful).
        try:
            init = bible.load_init(args.story_id)
            source_text = (bible.root / init.source_text_path).read_text(
                encoding="utf-8"
            )
        except (FileNotFoundError, OSError, AttributeError):
            source_text = ""
        try:
            brief = agent.parse_intent(args.story_id, intent, source_text)
        except Exception as e:
            raise CliError(
                "INTENT_PARSE_FAILED",
                f"MIMO failed to parse intent into BriefAnswers: {e}",
            )
    else:
        if not args.responses:
            raise CliError(
                "MISSING_INPUT",
                "brief answer requires either --intent 'paragraph' (free-form, "
                "MIMO parses) or --responses path.json (structured).",
            )
        raw = Path(args.responses).read_text(encoding="utf-8")
        answers = json.loads(raw)
        try:
            brief = agent.ingest_answers(args.story_id, answers)
        except Exception as e:
            raise CliError("INVALID_ANSWERS", f"answer JSON didn't match BriefAnswers schema: {e}")

    # Merge onto existing pending if any (preserves canon facts from prior research).
    existing = bible.load_current(args.story_id, StageName.BRIEF)
    if isinstance(existing, DirectorialBrief):
        brief.canon_facts = existing.canon_facts
        brief.world_canon = existing.world_canon
        brief.research_sources = existing.research_sources
    bible.save_pending(args.story_id, StageName.BRIEF, brief)

    _emit(
        {
            "story_id": args.story_id,
            "stage": "brief",
            "state": "pending",
            "answers_captured": True,
            "has_canon": len(brief.canon_facts) > 0,
            "next_action": (
                f"hitchcock brief approve -s {args.story_id}"
                if brief.canon_facts
                else f"hitchcock brief plan-research -s {args.story_id}  # recommended, or skip to approve"
            ),
        },
        as_json=args.json_out,
    )
    return 0


def _cmd_brief_plan_research(args: argparse.Namespace) -> int:
    settings = load_settings()
    bible = _bible_from(settings, args.bible_dir)
    _require_init(bible, args.story_id)
    current = bible.load_current(args.story_id, StageName.BRIEF)
    if not isinstance(current, DirectorialBrief):
        raise CliError(
            "NO_BRIEF",
            "answer the questions first: `hitchcock brief answer -s ... --responses ...`",
        )
    source_text = bible.load_source_text(args.story_id)
    agent = BriefAgent(llm=MimoClient(settings.mimo), bible=bible)
    plan = agent.plan_research(current, source_text)

    # Dump the plan to a file the driver fills in.
    plan_path = bible.story_dir(args.story_id) / "brief" / "research_plan.json"
    plan_path.parent.mkdir(parents=True, exist_ok=True)
    plan_path.write_text(json.dumps(plan, ensure_ascii=False, indent=2), encoding="utf-8")

    _emit(
        {
            "story_id": args.story_id,
            "stage": "brief",
            "plan_path": str(plan_path),
            "char_query_count": len(plan["plan"]["character_queries"]),
            "world_query_count": len(plan["plan"]["world_queries"]),
            "next_action": (
                f"run the search queries in {plan_path}, fill each canon_template "
                f"entry's canonical_* fields, then "
                f"`hitchcock brief ingest-canon -s {args.story_id} --file <filled.json>`"
            ),
        },
        as_json=args.json_out,
    )
    return 0


def _cmd_brief_research(args: argparse.Namespace) -> int:
    """One-shot automated canon research via Gemini + Google Search grounding.

    Replaces the old 2-step flow (plan-research + manual searches +
    ingest-canon). MIMO generates queries, Gemini runs them with web
    grounding, canon_facts + world_canon + research_sources are merged
    into the pending brief."""
    settings = load_settings()
    bible = _bible_from(settings, args.bible_dir)
    _require_init(bible, args.story_id)
    current = bible.load_current(args.story_id, StageName.BRIEF)
    if not isinstance(current, DirectorialBrief):
        raise CliError(
            "NO_BRIEF",
            "answer intent first: `hitchcock brief answer -s ... --intent '...'`",
        )
    source_text = bible.load_source_text(args.story_id)

    from .llm import GeminiTextClient
    mimo = MimoClient(settings.mimo)
    gemini = GeminiTextClient(settings.gemini)
    agent = BriefAgent(llm=mimo, bible=bible)

    log.info("Brief research: running automated canon research via Gemini grounding…")
    enriched = agent.research_canon(current, source_text, gemini)
    bible.save_pending(args.story_id, StageName.BRIEF, enriched)

    _emit(
        {
            "story_id": args.story_id,
            "stage": "brief",
            "state": "pending",
            "canon_facts_count": len(enriched.canon_facts),
            "source_count": len(enriched.research_sources),
            "has_world_canon": bool(enriched.world_canon.strip()),
            "next_action": f"hitchcock brief approve -s {args.story_id}",
        },
        as_json=args.json_out,
    )
    return 0


def _cmd_brief_ingest_canon(args: argparse.Namespace) -> int:
    settings = load_settings()
    bible = _bible_from(settings, args.bible_dir)
    _require_init(bible, args.story_id)
    current = bible.load_current(args.story_id, StageName.BRIEF)
    if not isinstance(current, DirectorialBrief):
        raise CliError("NO_BRIEF", "answer the questions first.")

    canon_data = json.loads(Path(args.file).read_text(encoding="utf-8"))
    agent = BriefAgent(llm=MimoClient(settings.mimo), bible=bible)
    try:
        updated = agent.ingest_canon(current, canon_data)
    except Exception as e:
        raise CliError("INVALID_CANON", f"canon file didn't match schema: {e}")

    bible.save_pending(args.story_id, StageName.BRIEF, updated)
    _emit(
        {
            "story_id": args.story_id,
            "stage": "brief",
            "canon_facts_ingested": len(updated.canon_facts),
            "world_canon_chars": len(updated.world_canon),
            "next_action": f"hitchcock brief show -s {args.story_id}",
        },
        as_json=args.json_out,
    )
    return 0


def _cmd_brief_show(args: argparse.Namespace) -> int:
    settings = load_settings()
    bible = _bible_from(settings, args.bible_dir)
    _require_init(bible, args.story_id)
    current = bible.load_current(args.story_id, StageName.BRIEF)
    if not isinstance(current, DirectorialBrief):
        raise CliError("NO_ARTIFACT", "no brief yet — run `brief questions` first.")
    state = bible.stage_state(args.story_id, StageName.BRIEF)
    if args.json_out:
        print(json.dumps(
            {"story_id": current.story_id, "state": state.value,
             "brief": json.loads(current.model_dump_json())},
            ensure_ascii=False, indent=2,
        ))
    else:
        print(f"=== Brief ({state.value}) ===\n")
        a = current.answers
        if a:
            print(f"Form:            {a.form}")
            print(f"Target audience: {a.target_audience}")
            print(f"Style refs:      {', '.join(a.style_references) or '—'}")
            print(f"Tone:            {', '.join(a.emotional_tone) or '—'}")
            print(f"Must-haves:      {', '.join(a.must_haves) or '—'}")
            print(f"Must-avoids:     {', '.join(a.must_avoids) or '—'}")
            print(f"Pacing:          {a.pacing or '—'}")
            print(f"Music:           {a.music_direction or '—'}")
        print(f"\nCanon facts: {len(current.canon_facts)}")
        for f in current.canon_facts:
            print(f"  {f.character_alias}:")
            print(f"    appearance: {f.canonical_appearance[:120]}…")
            print(f"    sources:    {len(f.sources)}")
        if current.world_canon:
            print(f"\nWorld canon ({len(current.world_canon)} chars): {current.world_canon[:180]}…")
    return 0


def _cmd_brief_refine(args: argparse.Namespace) -> int:
    settings = load_settings()
    bible = _bible_from(settings, args.bible_dir)
    _require_init(bible, args.story_id)
    current = bible.load_current(args.story_id, StageName.BRIEF)
    if not isinstance(current, DirectorialBrief):
        raise CliError("NO_ARTIFACT", "no brief yet.")
    if not args.feedback.strip():
        raise CliError("EMPTY_FEEDBACK", "--feedback cannot be empty")
    agent = BriefAgent(llm=MimoClient(settings.mimo), bible=bible)
    revised = agent.refine(current, args.feedback)
    bible.save_pending(args.story_id, StageName.BRIEF, revised)
    bible.append_feedback(args.story_id, StageName.BRIEF, None, args.feedback)
    _emit(
        {
            "story_id": args.story_id, "stage": "brief", "state": "pending",
            "next_action": f"hitchcock brief show -s {args.story_id}",
        },
        as_json=args.json_out,
    )
    return 0


def _cmd_brief_approve(args: argparse.Namespace) -> int:
    settings = load_settings()
    bible = _bible_from(settings, args.bible_dir)
    _require_init(bible, args.story_id)
    state = bible.stage_state(args.story_id, StageName.BRIEF)
    if state != StageState.PENDING:
        raise CliError("NO_PENDING", f"no pending brief (state={state.value}).")
    approved_path = bible.approve_pending(args.story_id, StageName.BRIEF)
    _emit(
        {
            "story_id": args.story_id, "stage": "brief", "state": "approved",
            "approved_path": str(approved_path),
            "next_action": f"hitchcock style generate -s {args.story_id}",
        },
        as_json=args.json_out,
    )
    return 0


# ─── style (Phase 1.6) ───────────────────────────────────────────────────

def _cmd_style_generate(args: argparse.Namespace) -> int:
    settings = load_settings()
    bible = _bible_from(settings, args.bible_dir)
    _require_init(bible, args.story_id)
    source_text = bible.load_source_text(args.story_id)

    agent = StyleAgent(llm=MimoClient(settings.mimo), bible=bible)
    guide = agent.generate(source_text, args.story_id)
    bible.save_pending(args.story_id, StageName.STYLE, guide)

    _emit(
        {
            "story_id": args.story_id,
            "stage": "style",
            "state": "pending",
            "art_direction_anchor": guide.art_direction_anchor,
            "motifs": guide.recurring_motifs,
            "avoid": guide.avoid,
            "next_action": f"hitchcock style show -s {args.story_id}",
        },
        as_json=args.json_out,
    )
    return 0


def _cmd_style_show(args: argparse.Namespace) -> int:
    settings = load_settings()
    bible = _bible_from(settings, args.bible_dir)
    _require_init(bible, args.story_id)
    current = bible.load_current(args.story_id, StageName.STYLE)
    if current is None:
        raise CliError("NO_ARTIFACT", "no style artifact — run `style generate` first.")
    assert isinstance(current, StyleGuide)
    state = bible.stage_state(args.story_id, StageName.STYLE)
    if args.json_out:
        print(json.dumps(
            {
                "story_id": current.story_id,
                "state": state.value,
                "style": json.loads(current.model_dump_json()),
            },
            ensure_ascii=False, indent=2,
        ))
    else:
        print(f"=== Style ({state.value}) — {current.art_direction_anchor} ===\n")
        print(f"PALETTE:   {current.palette}\n")
        print(f"LIGHTING:  {current.lighting_model}\n")
        print(f"TEXTURES:  {current.texture_materials}\n")
        print(f"MOTIFS:    {', '.join(current.recurring_motifs) or '—'}")
        print(f"AVOID:     {', '.join(current.avoid) or '—'}\n")
        print("GLOBAL_STYLE_PROMPT (appended to every image call):")
        print(current.global_style_prompt)
    return 0


def _cmd_style_refine(args: argparse.Namespace) -> int:
    settings = load_settings()
    bible = _bible_from(settings, args.bible_dir)
    _require_init(bible, args.story_id)
    current = bible.load_current(args.story_id, StageName.STYLE)
    if current is None:
        raise CliError("NO_ARTIFACT", "generate style first.")
    assert isinstance(current, StyleGuide)
    if not args.feedback.strip():
        raise CliError("EMPTY_FEEDBACK", "--feedback cannot be empty")
    agent = StyleAgent(llm=MimoClient(settings.mimo), bible=bible)
    revised = agent.refine(current, args.feedback)
    bible.save_pending(args.story_id, StageName.STYLE, revised)
    bible.append_feedback(args.story_id, StageName.STYLE, None, args.feedback)
    _emit(
        {
            "story_id": args.story_id,
            "stage": "style",
            "state": "pending",
            "new_anchor": revised.art_direction_anchor,
            "next_action": f"hitchcock style show -s {args.story_id}",
        },
        as_json=args.json_out,
    )
    return 0


def _cmd_style_approve(args: argparse.Namespace) -> int:
    settings = load_settings()
    bible = _bible_from(settings, args.bible_dir)
    _require_init(bible, args.story_id)
    state = bible.stage_state(args.story_id, StageName.STYLE)
    force_anchors = getattr(args, "force_anchors", False)
    # If already approved: allow --force-anchors to re-render the anchor
    # images without re-approving (the approved style JSON already has
    # the anchor prompts). Otherwise require pending state.
    if state == StageState.APPROVED and force_anchors:
        approved_path = bible.stage_approved_path(args.story_id, StageName.STYLE)
    elif state != StageState.PENDING:
        raise CliError(
            "NO_PENDING",
            f"no pending style (state={state.value}). "
            f"Use `--force-anchors` to re-render anchors on an "
            f"already-approved style.",
        )
    else:
        approved_path = bible.approve_pending(args.story_id, StageName.STYLE)

    # After approval, render the two STYLE ANCHOR images from the
    # approved StyleGuide's character_anchor_prompt + environment_anchor_prompt.
    # These images are used as `reference_images[0]` by downstream T2I
    # (cast build portraits + art generate scene keyframes) to lock the
    # painted-animation style visually — text-only prompts drift toward
    # GPT Image's fine-art-oil-painting prior. See feedback_video_single_ref.md
    # for context. Seedance (video) does NOT consume these anchors.
    anchors_built = _build_style_anchors(
        bible, args.story_id, settings,
        force=getattr(args, "force_anchors", False),
        quality=getattr(args, "quality", None),
    )

    _emit(
        {
            "story_id": args.story_id,
            "stage": "style",
            "state": "approved",
            "approved_path": str(approved_path),
            "style_anchors_built": anchors_built,
            "next_action": f"hitchcock cast discover -s {args.story_id}",
        },
        as_json=args.json_out,
    )
    return 0


def _build_style_anchors(
    bible: BibleStore, story_id: str, settings, *,
    force: bool = False, quality: str | None = None,
) -> dict:
    """Render `anchor_character.png` + `anchor_environment.png` via GPT
    Image from the approved StyleGuide's two anchor prompts. Skip (no
    re-render) if the file already exists unless `force=True`."""
    approved = bible.load_approved(story_id, StageName.STYLE)
    if approved is None:
        return {"skipped_reason": "no approved style"}
    char_prompt = getattr(approved, "character_anchor_prompt", "").strip()
    env_prompt = getattr(approved, "environment_anchor_prompt", "").strip()
    style_dir = bible.story_dir(story_id) / "style"
    style_dir.mkdir(parents=True, exist_ok=True)
    targets = [
        ("character_anchor", char_prompt, style_dir / "anchor_character.png"),
        ("environment_anchor", env_prompt, style_dir / "anchor_environment.png"),
    ]
    result: dict = {"rendered": [], "skipped": [], "missing_prompt": []}
    if not any(p for _, p, _ in targets):
        result["skipped_reason"] = (
            "style guide has no anchor prompts — regenerate via `style generate`"
        )
        return result
    gpt = GPTImageClient(settings.openai, quality=quality)
    for name, prompt, out_path in targets:
        if not prompt:
            result["missing_prompt"].append(name)
            log.warning("style approve: %s prompt empty — skipping", name)
            continue
        if out_path.exists() and not force:
            result["skipped"].append(name)
            continue
        log.info("style approve: rendering %s → %s", name, out_path.name)
        img = gpt.generate(prompt, width=1024, height=1024)
        img.save(out_path)
        result["rendered"].append(name)
    return result


# ─── cast (Phase 1.5) ────────────────────────────────────────────────────

def _cmd_cast_discover(args: argparse.Namespace) -> int:
    settings = load_settings()
    bible = _bible_from(settings, args.bible_dir)
    _require_init(bible, args.story_id)
    source_text = bible.load_source_text(args.story_id)

    agent = CastAgent(
        llm=MimoClient(settings.mimo),
        images=GPTImageClient(settings.openai),
        bible=bible,
    )
    manifest = agent.discover(source_text, args.story_id)
    bible.save_pending(args.story_id, StageName.CAST, manifest)

    _emit(
        {
            "story_id": args.story_id,
            "stage": "cast",
            "state": "pending",
            "characters": [
                {"id": c.canonical_id, "name": c.display_name,
                 "status": c.match_status.value,
                 "matched_bible_id": c.matched_bible_id}
                for c in manifest.characters
            ],
            "locations": [
                {"id": l.canonical_id, "name": l.display_name,
                 "status": l.match_status.value,
                 "matched_bible_id": l.matched_bible_id}
                for l in manifest.locations
            ],
            "new_characters_count": sum(
                1 for c in manifest.characters if c.match_status == MatchStatus.NEW
            ),
            "new_locations_count": sum(
                1 for l in manifest.locations if l.match_status == MatchStatus.NEW
            ),
            "next_action": f"hitchcock cast show -s {args.story_id}",
        },
        as_json=args.json_out,
    )
    return 0


def _cmd_cast_show(args: argparse.Namespace) -> int:
    settings = load_settings()
    bible = _bible_from(settings, args.bible_dir)
    _require_init(bible, args.story_id)
    current = bible.load_current(args.story_id, StageName.CAST)
    if current is None:
        raise CliError("NO_ARTIFACT", "no cast artifact yet — run `cast discover` first.")
    assert isinstance(current, CastManifest)
    state = bible.stage_state(args.story_id, StageName.CAST)

    if args.json_out:
        payload = {
            "story_id": current.story_id,
            "state": state.value,
            "cast": json.loads(current.model_dump_json()),
        }
        print(json.dumps(payload, ensure_ascii=False, indent=2))
    else:
        print(f"=== Cast ({state.value}) ===\n")
        print(f"Characters: {len(current.characters)}")
        for c in current.characters:
            status_marker = {
                "in_bible": "✓",
                "in_bible_by_alias": "≈",
                "new": "✱ new",
            }[c.match_status.value]
            match_info = f" → {c.matched_bible_id}" if c.matched_bible_id else ""
            print(f"  [{status_marker}] {c.canonical_id}  ({c.display_name}){match_info}")
            if c.match_status.value == "new":
                print(f"         role: {c.role}")
                print(f"         visual: {c.visual_description[:120]}…")
        print(f"\nLocations: {len(current.locations)}")
        for l in current.locations:
            status_marker = {
                "in_bible": "✓",
                "in_bible_by_alias": "≈",
                "new": "✱ new",
            }[l.match_status.value]
            match_info = f" → {l.matched_bible_id}" if l.matched_bible_id else ""
            print(f"  [{status_marker}] {l.canonical_id}  ({l.display_name}){match_info}")
            if l.match_status.value == "new":
                print(f"         {l.description[:120]}…")
    return 0


def _cmd_cast_refine(args: argparse.Namespace) -> int:
    settings = load_settings()
    bible = _bible_from(settings, args.bible_dir)
    _require_init(bible, args.story_id)
    current = bible.load_current(args.story_id, StageName.CAST)
    if current is None:
        raise CliError("NO_ARTIFACT", "discover cast first before refining.")
    if not args.feedback.strip():
        raise CliError("EMPTY_FEEDBACK", "--feedback cannot be empty")
    assert isinstance(current, CastManifest)

    agent = CastAgent(
        llm=MimoClient(settings.mimo),
        images=GPTImageClient(settings.openai),
        bible=bible,
    )
    revised = agent.refine(current, args.feedback)
    bible.save_pending(args.story_id, StageName.CAST, revised)
    bible.append_feedback(args.story_id, StageName.CAST, None, args.feedback)
    _emit(
        {
            "story_id": args.story_id,
            "stage": "cast",
            "state": "pending",
            "characters_after": len(revised.characters),
            "locations_after": len(revised.locations),
            "next_action": f"hitchcock cast show -s {args.story_id}",
        },
        as_json=args.json_out,
    )
    return 0


def _cmd_cast_build(args: argparse.Namespace) -> int:
    settings = load_settings()
    bible = _bible_from(settings, args.bible_dir)
    _require_init(bible, args.story_id)
    current = bible.load_current(args.story_id, StageName.CAST)
    if current is None:
        raise CliError("NO_ARTIFACT", "discover cast first.")
    assert isinstance(current, CastManifest)

    agent = CastAgent(
        llm=MimoClient(settings.mimo),
        images=_gpt_image_client(settings, args),
        bible=bible,
    )
    only = [x.strip() for x in (args.only or "").split(",") if x.strip()] or None
    result = agent.build(
        current,
        only=only,
        skip_refs=args.skip_refs,
        dry_run=args.dry_run,
        story_id=args.story_id,
        force=getattr(args, "force", False),
    )
    # If build actually ran, reclassify matched entries (they're now in_bible)
    # and save the updated manifest back to pending.
    # Reclassify match_status after build (regardless of skip_refs):
    # --skip-refs still writes the bible character.json, so the NEW entry
    # is now in_bible and the manifest must reflect that — otherwise
    # `cast approve` will reject it with UNBUILT_CAST. Previously this
    # block was gated on `not args.skip_refs`, which silently trapped
    # voice-only / non-portrait characters in the NEW limbo.
    if not args.dry_run:
        for cp in current.characters:
            agent._reclassify_character(cp)
        for lp in current.locations:
            agent._reclassify_location(lp)
        bible.save_pending(args.story_id, StageName.CAST, current)

    next_action = (
        f"hitchcock cast approve -s {args.story_id}"
        if not args.dry_run else
        f"hitchcock cast build -s {args.story_id}  # drop --dry-run to run"
    )
    _emit(
        {
            "story_id": args.story_id,
            "stage": "cast",
            "state": "pending",
            "result": result,
            "next_action": next_action,
        },
        as_json=args.json_out,
    )
    return 0


def _cmd_cast_approve(args: argparse.Namespace) -> int:
    settings = load_settings()
    bible = _bible_from(settings, args.bible_dir)
    _require_init(bible, args.story_id)
    state = bible.stage_state(args.story_id, StageName.CAST)
    if state != StageState.PENDING:
        raise CliError("NO_PENDING", f"no pending cast (state={state.value}).")
    # Guard: every character + location must be in_bible* before approve.
    current = bible.load_pending(args.story_id, StageName.CAST)
    assert isinstance(current, CastManifest)
    unresolved = [
        c.canonical_id for c in current.characters
        if c.match_status == MatchStatus.NEW
    ] + [
        l.canonical_id for l in current.locations
        if l.match_status == MatchStatus.NEW
    ]
    if unresolved:
        raise CliError(
            "UNBUILT_CAST",
            f"these are still NEW (not in bible): {unresolved}. "
            f"Run `hitchcock cast build -s {args.story_id}` first.",
        )
    approved_path = bible.approve_pending(args.story_id, StageName.CAST)

    # Update StoryInit to point at the approved character + location ids.
    init = bible.load_story_init(args.story_id)
    init.character_ids = [c.matched_bible_id or c.canonical_id for c in current.characters]
    init.location_ids = [l.matched_bible_id or l.canonical_id for l in current.locations]
    bible.save_story_init(init)

    _emit(
        {
            "story_id": args.story_id,
            "stage": "cast",
            "state": "approved",
            "approved_path": str(approved_path),
            "character_ids": init.character_ids,
            "location_ids": init.location_ids,
            "next_action": f"hitchcock script generate -s {args.story_id}",
        },
        as_json=args.json_out,
    )
    return 0


# ─── script ──────────────────────────────────────────────────────────────

def _require_init(bible: BibleStore, sid: str) -> StoryInit:
    if not bible.story_init_path(sid).exists():
        raise CliError(
            "UNKNOWN_STORY",
            f"story '{sid}' not initialized. Run `hitchcock init -s {sid} --source ...` first.",
        )
    return bible.load_story_init(sid)


def _cmd_script_generate(args: argparse.Namespace) -> int:
    settings = load_settings()
    bible = _bible_from(settings, args.bible_dir)
    init = _require_init(bible, args.story_id)
    source_text = bible.load_source_text(args.story_id)

    agent = ScriptAgent(llm=MimoClient(settings.mimo), bible=bible)
    story = agent.generate(
        source_text, init.character_ids, init.location_ids,
        story_id=args.story_id,
    )
    # Preserve the story_id assigned at init (MIMO may slug from title).
    story.id = args.story_id
    bible.save_pending(args.story_id, StageName.SCRIPT, story)

    _emit(
        {
            "story_id": story.id,
            "stage": "script",
            "state": "pending",
            "title": story.title,
            "scenes": len(story.scenes),
            "scene_ids": [s.id for s in story.scenes],
            "pending_path": str(bible.stage_pending_path(story.id, StageName.SCRIPT)),
            "next_action": f"hitchcock script show -s {story.id}",
        },
        as_json=args.json_out,
    )
    return 0


def _cmd_script_show(args: argparse.Namespace) -> int:
    settings = load_settings()
    bible = _bible_from(settings, args.bible_dir)
    _require_init(bible, args.story_id)

    current = bible.load_current(args.story_id, StageName.SCRIPT)
    if current is None:
        raise CliError(
            "NO_ARTIFACT",
            f"no script artifact yet for '{args.story_id}'. "
            f"Run `hitchcock script generate -s {args.story_id}`.",
        )
    state = bible.stage_state(args.story_id, StageName.SCRIPT)
    assert isinstance(current, Story)

    if args.json_out:
        payload = {
            "story_id": current.id,
            "stage": "script",
            "state": state.value,
            "story": json.loads(current.model_dump_json()),
        }
        print(json.dumps(payload, ensure_ascii=False, indent=2))
    else:
        # Check for --scene filter (attribute may not exist on older arg sets)
        scene_filter = getattr(args, "scene", None)
        print(f"=== Script ({state.value}) — {current.title} ({current.id}) ===\n")
        if not scene_filter:
            print(f"Logline: {current.logline}\n")
            print("Synopsis:")
            print(current.synopsis)
            if current.narrative_spine:
                print(f"\nNarrative spine: {current.narrative_spine}")
            print(f"\nCharacters: {', '.join(current.characters)}")
            print(f"\nScenes: {len(current.scenes)}")
            for sc in current.scenes:
                n_dialogue = len(sc.dialogue)
                act_len = len(sc.action)
                print(
                    f"  {sc.id} — {sc.title}  [{sc.emotional_register or '—'}]  "
                    f"dialogue={n_dialogue} action={act_len}ch"
                )
                print(f"       chars: {', '.join(sc.characters_in_scene) or '—'}")
                if sc.transition_in:
                    print(f"       in →  {sc.transition_in[:100]}")
                if sc.dramatic_turn:
                    print(f"       turn: {sc.dramatic_turn[:100]}")
            print("\n(use `--scene sXX` to see full screenplay for a single scene)")
        else:
            # Full screenplay view for one scene
            sc = next((s for s in current.scenes if s.id == scene_filter), None)
            if sc is None:
                raise CliError("UNKNOWN_SCENE", f"scene '{scene_filter}' not in script")
            loc = sc.location_id or sc.location_description or "—"
            print(f"{sc.id.upper()} — {sc.title}")
            print(f"SETTING: {loc}")
            print(f"TIME:    {sc.time_of_day or '—'}")
            print(f"REGISTER: {sc.emotional_register or '—'}")
            print(f"CHARACTERS (cast):      {', '.join(sc.characters_in_scene) or '—'}")
            if sc.uncredited_presences:
                print(f"UNCREDITED (mentioned): {', '.join(sc.uncredited_presences)}")
            print()
            if sc.transition_in:
                print("TRANSITION IN:")
                print(f"  {sc.transition_in}\n")
            if sc.action:
                print("ACTION (camera-visible, no dialogue quotations):")
                print(sc.action)
                print()
            if sc.dialogue:
                print("DIALOGUE (delivery order):")
                for d in sc.dialogue:
                    tag = "  [INVENTED] " if getattr(d, "invented", False) else "  "
                    print(f"{tag}{d.speaker_id}: {d.text}")
                print()
            if sc.dramatic_turn:
                print("DRAMATIC TURN:")
                print(f"  {sc.dramatic_turn}\n")
            if sc.transition_out:
                print("TRANSITION OUT:")
                print(f"  {sc.transition_out}\n")
            if sc.beats:
                print("BEATS (for storyboard):")
                for b in sc.beats:
                    print(f"  - {b}")
    return 0


def _cmd_script_refine(args: argparse.Namespace) -> int:
    settings = load_settings()
    bible = _bible_from(settings, args.bible_dir)
    _require_init(bible, args.story_id)

    current = bible.load_current(args.story_id, StageName.SCRIPT)
    if current is None:
        raise CliError(
            "NO_ARTIFACT",
            f"no script artifact yet — generate first before refining.",
        )
    assert isinstance(current, Story)

    if not args.feedback.strip():
        raise CliError("EMPTY_FEEDBACK", "--feedback cannot be empty")

    agent = ScriptAgent(llm=MimoClient(settings.mimo), bible=bible)
    revised = agent.refine(current, args.feedback)
    bible.save_pending(args.story_id, StageName.SCRIPT, revised)
    bible.append_feedback(args.story_id, StageName.SCRIPT, None, args.feedback)

    _emit(
        {
            "story_id": revised.id,
            "stage": "script",
            "state": "pending",
            "scenes_before": len(current.scenes),
            "scenes_after": len(revised.scenes),
            "scene_ids": [s.id for s in revised.scenes],
            "pending_path": str(bible.stage_pending_path(revised.id, StageName.SCRIPT)),
            "next_action": f"hitchcock script show -s {revised.id}",
        },
        as_json=args.json_out,
    )
    return 0


def _cmd_script_approve(args: argparse.Namespace) -> int:
    settings = load_settings()
    bible = _bible_from(settings, args.bible_dir)
    _require_init(bible, args.story_id)

    state = bible.stage_state(args.story_id, StageName.SCRIPT)
    if state != StageState.PENDING:
        raise CliError(
            "NO_PENDING",
            f"no pending script to approve (state={state.value}).",
        )
    approved_path = bible.approve_pending(args.story_id, StageName.SCRIPT)

    _emit(
        {
            "story_id": args.story_id,
            "stage": "script",
            "state": "approved",
            "approved_path": str(approved_path),
            "next_action": f"hitchcock storyboard generate -s {args.story_id}",
        },
        as_json=args.json_out,
    )
    return 0


# ─── status ──────────────────────────────────────────────────────────────

def _cmd_status(args: argparse.Namespace) -> int:
    settings = load_settings()
    bible = _bible_from(settings, args.bible_dir)
    _require_init(bible, args.story_id)

    stages: dict[str, dict[str, Any]] = {}
    for stage in StageName:
        state = bible.stage_state(args.story_id, stage)
        stages[stage.value] = {"state": state.value}
        p_path = bible.stage_pending_path(args.story_id, stage)
        a_path = bible.stage_approved_path(args.story_id, stage)
        if p_path.exists():
            stages[stage.value]["pending_path"] = str(p_path)
        if a_path.exists():
            stages[stage.value]["approved_path"] = str(a_path)

    # Work out next_action from stage states.
    next_action = _suggest_next_action(args.story_id, stages)

    _emit(
        {
            "story_id": args.story_id,
            "contract_version": CONTRACT_VERSION,
            "stages": stages,
            "next_action": next_action,
        },
        as_json=args.json_out,
    )
    return 0


def _suggest_next_action(sid: str, stages: dict[str, dict]) -> str:
    order = ["brief", "style", "cast", "script", "storyboard", "art", "render"]
    prev_approved = True
    for stage in order:
        st = stages[stage]["state"]
        if st == "pending":
            if stage == "render":
                return (
                    f"upload packages/<scene>/ to https://jimeng.jianying.com/ai-tool/generate, "
                    f"drop mp4s into render/clips/, then `hitchcock render post -s {sid}`"
                )
            return f"hitchcock {stage} show -s {sid}"
        if st == "not_started":
            if prev_approved:
                if stage == "brief":
                    return f"hitchcock brief questions -s {sid}"
                if stage == "cast":
                    return f"hitchcock cast discover -s {sid}"
                if stage == "render":
                    return f"hitchcock render package -s {sid}"
                return f"hitchcock {stage} generate -s {sid}"
            return f"hitchcock {order[order.index(stage)-1]} approve -s {sid}"
        prev_approved = (st == "approved")
    return "pipeline complete"


# ─── storyboard ──────────────────────────────────────────────────────────

def _load_approved_script(bible: BibleStore, sid: str) -> Story:
    story = bible.load_approved(sid, StageName.SCRIPT)
    if story is None:
        raise CliError(
            "NO_UPSTREAM_APPROVED",
            f"approve script first: `hitchcock script approve -s {sid}`",
        )
    assert isinstance(story, Story)
    return story


def _cmd_storyboard_generate(args: argparse.Namespace) -> int:
    settings = load_settings()
    bible = _bible_from(settings, args.bible_dir)
    _require_init(bible, args.story_id)
    story = _load_approved_script(bible, args.story_id)

    agent = StoryboardAgent(llm=MimoClient(settings.mimo), bible=bible)
    storyboard = agent.generate(story)
    bible.save_pending(args.story_id, StageName.STORYBOARD, storyboard)

    _emit(
        {
            "story_id": args.story_id,
            "stage": "storyboard",
            "state": "pending",
            "scenes": [
                {
                    "scene_id": s.scene_id,
                    "shots": len(s.shots),
                    "has_scene_art_prompt": bool(s.scene_art_prompt),
                    "has_seedance_prompt": bool(s.seedance_prompt),
                }
                for s in storyboard.scenes
            ],
            "next_action": f"hitchcock storyboard show -s {args.story_id}",
        },
        as_json=args.json_out,
    )
    return 0


def _cmd_storyboard_show(args: argparse.Namespace) -> int:
    settings = load_settings()
    bible = _bible_from(settings, args.bible_dir)
    _require_init(bible, args.story_id)
    current = bible.load_current(args.story_id, StageName.STORYBOARD)
    if current is None:
        raise CliError(
            "NO_ARTIFACT",
            f"no storyboard artifact yet — run `hitchcock storyboard generate -s {args.story_id}`.",
        )
    assert isinstance(current, Storyboard)
    state = bible.stage_state(args.story_id, StageName.STORYBOARD)

    if args.scene:
        sb_scene = next((s for s in current.scenes if s.scene_id == args.scene), None)
        if sb_scene is None:
            raise CliError("UNKNOWN_SCENE", f"scene '{args.scene}' not in storyboard")
        payload = {
            "story_id": current.story_id,
            "state": state.value,
            "scene": json.loads(sb_scene.model_dump_json()),
        }
        if args.json_out:
            print(json.dumps(payload, ensure_ascii=False, indent=2))
        else:
            print(f"=== Storyboard Scene {sb_scene.scene_id} ({state.value}) ===\n")
            for i, sh in enumerate(sb_scene.shots, 1):
                print(f"Shot {i}: {int(sh.duration_sec)}s {sh.shot_type.value} / "
                      f"{sh.camera_movement.value}")
                print(f"  action: {sh.action}")
                for d in sh.dialogue:
                    print(f"  VO [{d.speaker_id}]: {d.text}")
                print()
            print("--- scene_art_prompt (first 400 chars) ---")
            print(sb_scene.scene_art_prompt[:400] + ("…" if len(sb_scene.scene_art_prompt) > 400 else ""))
            print()
            print(f"--- seedance_prompt length: {len(sb_scene.seedance_prompt)} chars ---")
        return 0

    if args.json_out:
        print(json.dumps(
            {
                "story_id": current.story_id,
                "state": state.value,
                "storyboard": json.loads(current.model_dump_json()),
            },
            ensure_ascii=False, indent=2,
        ))
    else:
        print(f"=== Storyboard ({state.value}) — {len(current.scenes)} scenes ===\n")
        for s in current.scenes:
            total = sum(sh.duration_sec for sh in s.shots)
            print(f"  {s.scene_id}  shots={len(s.shots)}  total={int(total)}s  "
                  f"art_prompt={len(s.scene_art_prompt)}ch  "
                  f"seedance={len(s.seedance_prompt)}ch")
    return 0


def _cmd_storyboard_refine(args: argparse.Namespace) -> int:
    settings = load_settings()
    bible = _bible_from(settings, args.bible_dir)
    _require_init(bible, args.story_id)
    current = bible.load_current(args.story_id, StageName.STORYBOARD)
    if current is None:
        raise CliError("NO_ARTIFACT", "generate storyboard first before refining.")
    assert isinstance(current, Storyboard)
    if not args.feedback.strip():
        raise CliError("EMPTY_FEEDBACK", "--feedback cannot be empty")
    if not args.scene:
        raise CliError(
            "MISSING_SCENE",
            "storyboard refine requires --scene <id> (per-scene granularity).",
        )
    story = _load_approved_script(bible, args.story_id)
    agent = StoryboardAgent(llm=MimoClient(settings.mimo), bible=bible)
    revised = agent.refine_scene(current, story, args.scene, args.feedback)
    bible.save_pending(args.story_id, StageName.STORYBOARD, revised)
    bible.append_feedback(args.story_id, StageName.STORYBOARD, args.scene, args.feedback)

    _emit(
        {
            "story_id": args.story_id,
            "stage": "storyboard",
            "state": "pending",
            "refined_scene": args.scene,
            "next_action": f"hitchcock storyboard show -s {args.story_id} --scene {args.scene}",
        },
        as_json=args.json_out,
    )
    return 0


def _cmd_storyboard_approve(args: argparse.Namespace) -> int:
    settings = load_settings()
    bible = _bible_from(settings, args.bible_dir)
    _require_init(bible, args.story_id)
    state = bible.stage_state(args.story_id, StageName.STORYBOARD)
    if state != StageState.PENDING:
        raise CliError("NO_PENDING", f"no pending storyboard (state={state.value}).")
    approved_path = bible.approve_pending(args.story_id, StageName.STORYBOARD)
    _emit(
        {
            "story_id": args.story_id,
            "stage": "storyboard",
            "state": "approved",
            "approved_path": str(approved_path),
            "next_action": f"hitchcock art generate -s {args.story_id}",
        },
        as_json=args.json_out,
    )
    return 0


# ─── art ─────────────────────────────────────────────────────────────────

def _load_approved_storyboard(bible: BibleStore, sid: str) -> Storyboard:
    sb = bible.load_approved(sid, StageName.STORYBOARD)
    if sb is None:
        raise CliError(
            "NO_UPSTREAM_APPROVED",
            f"approve storyboard first: `hitchcock storyboard approve -s {sid}`",
        )
    assert isinstance(sb, Storyboard)
    return sb


def _prompt_hash(prompt: str) -> str:
    return hashlib.sha1(prompt.encode("utf-8")).hexdigest()[:10]


def _art_candidates_dir(bible: BibleStore, sid: str, scene_id: str) -> Path:
    return bible.story_dir(sid) / "art" / "candidates" / scene_id


def _art_scene_art_path(bible: BibleStore, sid: str, scene_id: str) -> Path:
    return bible.story_dir(sid) / "art" / "scene_arts" / f"{scene_id}.png"


def _generate_keyframes_for_scene(
    bible: BibleStore,
    nb: NanoBananaClient,
    sid: str,
    sb_scene,  # StoryboardScene
    story: Story,
    n_per_shot: int,
    shot_filter: Optional[str] = None,
) -> list:  # list[ArtShot]
    """Generate one keyframe per SHOT in the scene via Nano Banana Pro.

    `shot_filter`: if set, ONLY generate for that shot id (saves tokens
    when iterating on a single problematic shot). All other shots keep
    their existing candidates untouched.

    Per-shot architecture (2026-04-21) replaces the old per-scene keyframe.
    Each shot's `keyframe_prompt` is authored by MIMO in `_SHOTS_SYSTEM`
    (English, Nano Banana–tuned style). Refs: user-curated style frames
    + location establishing + each character front.png."""
    from .bible import ArtShot  # local import to avoid cycle

    out_dir = _art_candidates_dir(bible, sid, sb_scene.scene_id)
    out_dir.mkdir(parents=True, exist_ok=True)
    # When filtering to a single shot, only clear THAT shot's files to
    # preserve other shots' existing candidates.
    glob = f"{shot_filter}_*.png" if shot_filter else "*.png"
    for p in out_dir.glob(glob):
        p.unlink()

    # Refs order (most → least authoritative for STYLE; all before identity):
    #   1. ENVIRONMENT STYLE ANCHOR (generated at `style approve` time) —
    #      a pre-rendered anonymous architectural-fragment painting in the
    #      target painted-animation style. Passed FIRST so GPT Image
    #      locks the look before the per-shot prompt drifts toward fine-
    #      art oil-painting priors. See `feedback_video_single_ref.md`.
    #   2. Location establishing art (if available).
    #   3. Character portrait refs (identity lock — per-shot filtered
    #      via `shot.characters_in_shot`).
    # Seedance (video) does NOT consume style anchors — they're T2I only.
    #
    # Character ref selection is PER-SHOT (not per-scene). Passing every
    # scene character's portrait into every shot causes Nano Banana to
    # BLEND faces when the prompt only mentions one of them — e.g. s01
    # mentions only "a Chinese Han teenage boy" but the scene has both
    # 楚子航 and 父亲; passing both portraits produced a blended figure
    # that looked like neither character. Fix: resolve each shot's
    # canonical_label list (少年/中年男人/少年甲/…) back to character IDs
    # via the same _canonical_role_cn + _disambiguate_labels used by
    # storyboard, then pass only those characters' front.png.
    scene = next((s for s in story.scenes if s.id == sb_scene.scene_id), None)
    from .agents.storyboard import _canonical_role_cn, _disambiguate_labels

    # Shared refs (env anchor + location) — identical across all shots.
    shared_refs: list[bytes] = []
    env_anchor_path = bible.story_dir(sid) / "style" / "anchor_environment.png"
    if env_anchor_path.exists():
        shared_refs.append(env_anchor_path.read_bytes())
        log.info("art: using env style anchor: %s", env_anchor_path.name)
    if scene and scene.location_id:
        loc_art = bible.establishing_art(scene.location_id)
        if loc_art.exists():
            shared_refs.append(loc_art.read_bytes())

    # Per-character ref cache + canonical_label → cid mapping for this scene.
    char_ref_by_cid: dict[str, bytes] = {}
    label_to_cid: dict[str, str] = {}
    if scene:
        chars_in_scene = []
        for cid in scene.characters_in_scene:
            try:
                chars_in_scene.append(bible.load_character(cid))
            except Exception:
                chars_in_scene.append(None)
            front = bible.refs_dir(cid) / "front.png"
            if front.exists():
                char_ref_by_cid[cid] = front.read_bytes()
        raw_labels = [_canonical_role_cn(c) for c in chars_in_scene]
        canonical_labels = _disambiguate_labels(raw_labels)
        for cid, lbl in zip(scene.characters_in_scene, canonical_labels):
            label_to_cid[lbl] = cid

    # Deterministic style prompt prepend. MIMO's MIMO-authored STYLE
    # paragraph in sh.keyframe_prompt is systematically weak ("Cinematic
    # wide shot with ethereal light" — no Arcane mechanical features).
    # We inject the full approved StyleGuide.global_style_prompt before
    # the MIMO prompt so Nano Banana sees the Arcane mechanical features
    # up front regardless of MIMO drift. This is the text-side complement
    # to the style reference images: text + image both anchoring style.
    from .agents.style import load_style_prompt
    style_prompt = load_style_prompt(bible, sid)

    # 2026-04-22: under single-ref video architecture (see
    # `feedback_video_single_ref.md`), only the FIRST shot's keyframe
    # reaches Seedance — the render packager drops the rest. Previously
    # we still rendered all per-shot keyframes for "user review", but
    # that meant 24 GPT Image calls per full run with 16 unused outputs
    # (¥10 → ~¥3 savings, and 24 images → 8 images to review).
    # Align art gen with packaging: generate ONE keyframe per scene,
    # using the first shot's keyframe_prompt as the representative
    # composition for the clip. `--shot sh02` etc. still targets a
    # specific shot if the user wants to iterate.
    target_shots = sb_scene.shots
    if not shot_filter:
        target_shots = target_shots[:1]
    art_shots = []
    for sh in target_shots:
        if shot_filter and sh.id != shot_filter:
            continue
        if not sh.keyframe_prompt.strip():
            log.warning(
                "art: %s/%s has empty keyframe_prompt — skipping (storyboard "
                "regen needed for this shot).",
                sb_scene.scene_id, sh.id,
            )
            continue
        # Full prompt sent to Nano Banana: style prepend + MIMO keyframe.
        # The prepend is the ground-truth art direction (mechanical
        # features MIMO often fails to carry verbatim). Hash includes it
        # so changing the style guide invalidates cached candidates.
        full_prompt = f"STYLE ANCHOR (ground truth — follow exactly):\n{style_prompt}\n\n{sh.keyframe_prompt}"
        phash = _prompt_hash(full_prompt)

        # Shot-level ref list: shared (style + location) + ONLY characters
        # in this shot. Fall back to all scene characters if the shot
        # didn't declare characters_in_shot (legacy storyboard data).
        shot_char_labels = getattr(sh, "characters_in_shot", None) or []
        if shot_char_labels:
            shot_cids = [label_to_cid[lbl] for lbl in shot_char_labels if lbl in label_to_cid]
            if not shot_cids and label_to_cid:
                # Labels didn't resolve (storyboard used labels that don't
                # match what _disambiguate_labels produced). Fall back to
                # all scene characters rather than none.
                log.warning(
                    "art: %s/%s characters_in_shot=%r did not resolve via "
                    "canonical labels %r — using all scene chars",
                    sb_scene.scene_id, sh.id, shot_char_labels, list(label_to_cid.keys()),
                )
                shot_cids = list(char_ref_by_cid.keys())
        else:
            shot_cids = list(char_ref_by_cid.keys())
        refs = list(shared_refs) + [char_ref_by_cid[c] for c in shot_cids if c in char_ref_by_cid]
        log.info(
            "art: %s/%s refs: %d style+loc + chars %s",
            sb_scene.scene_id, sh.id,
            len(shared_refs), shot_cids,
        )

        cands = []
        for i in range(1, n_per_shot + 1):
            log.info(
                "art: %s/%s cand %d/%d (prompt=%dch)",
                sb_scene.scene_id, sh.id, i, n_per_shot, len(full_prompt),
            )
            img = nb.generate(
                full_prompt,
                width=2688, height=1512,
                reference_images=refs,
            )
            out = out_dir / f"{sh.id}_cand_{i:02d}.png"
            img.save(out)
            cands.append(ArtCandidate(
                index=i,
                path=str(out.relative_to(bible.root)),
                prompt_hash=phash,
            ))
        # Auto-pick candidate 1 when n_per_shot==1 (saves a separate
        # `art pick` step; user can still switch via --candidate).
        picked = 1 if len(cands) == 1 else None
        art_shots.append(ArtShot(
            shot_id=sh.id,
            candidates=cands,
            picked_index=picked,
        ))
    return art_shots


def _cmd_art_generate(args: argparse.Namespace) -> int:
    settings = load_settings()
    bible = _bible_from(settings, args.bible_dir)
    _require_init(bible, args.story_id)
    storyboard = _load_approved_storyboard(bible, args.story_id)
    story = _load_approved_script(bible, args.story_id)
    nb = _gpt_image_client(settings, args)

    # Start from current pending ArtManifest if exists, else fresh.
    current = bible.load_current(args.story_id, StageName.ART)
    manifest = current if isinstance(current, ArtManifest) else ArtManifest(story_id=args.story_id)
    by_id = {s.scene_id: s for s in manifest.scenes}

    target_scenes = (
        [s for s in storyboard.scenes if s.scene_id == args.scene]
        if args.scene else storyboard.scenes
    )
    if args.scene and not target_scenes:
        raise CliError("UNKNOWN_SCENE", f"scene '{args.scene}' not in storyboard")

    n = args.candidates
    shot_filter = getattr(args, "shot", None)
    total_shots = 0
    for sb_scene in target_scenes:
        new_shots = _generate_keyframes_for_scene(
            bible, nb, args.story_id, sb_scene, story, n,
            shot_filter=shot_filter,
        )
        total_shots += len(new_shots)
        # Merge — preserve other shots' existing candidates when
        # `--shot` targets a single shot.
        existing = by_id.get(sb_scene.scene_id)
        if existing and shot_filter:
            new_by_id = {s.shot_id: s for s in new_shots}
            merged = [new_by_id.get(s.shot_id, s) for s in existing.shots]
            # Include any brand-new shots not in existing.
            for ns in new_shots:
                if ns.shot_id not in {s.shot_id for s in merged}:
                    merged.append(ns)
            by_id[sb_scene.scene_id] = ArtScene(
                scene_id=sb_scene.scene_id,
                shots=merged,
            )
        else:
            by_id[sb_scene.scene_id] = ArtScene(
                scene_id=sb_scene.scene_id,
                shots=new_shots,
            )

    manifest.scenes = list(by_id.values())
    bible.save_pending(args.story_id, StageName.ART, manifest)

    _emit(
        {
            "story_id": args.story_id,
            "stage": "art",
            "state": "pending",
            "scenes_updated": [s.scene_id for s in target_scenes],
            "shots_rendered": total_shots,
            "candidates_per_shot": n,
            "next_action": (
                f"hitchcock art show -s {args.story_id} --scene {target_scenes[0].scene_id}"
                if target_scenes else f"hitchcock art show -s {args.story_id}"
            ),
        },
        as_json=args.json_out,
    )
    return 0


def _cmd_art_show(args: argparse.Namespace) -> int:
    settings = load_settings()
    bible = _bible_from(settings, args.bible_dir)
    current = bible.load_current(args.story_id, StageName.ART)
    if current is None:
        raise CliError("NO_ARTIFACT", "no art artifact yet.")
    assert isinstance(current, ArtManifest)
    state = bible.stage_state(args.story_id, StageName.ART)

    scenes = current.scenes
    if args.scene:
        scenes = [s for s in scenes if s.scene_id == args.scene]
        if not scenes:
            raise CliError("UNKNOWN_SCENE", f"scene '{args.scene}' not in art")

    if args.json_out:
        payload = {
            "story_id": current.story_id,
            "state": state.value,
            "scenes": [json.loads(s.model_dump_json()) for s in scenes],
        }
        print(json.dumps(payload, ensure_ascii=False, indent=2))
    else:
        print(f"=== Art ({state.value}) ===\n")
        for s in scenes:
            if s.shots:
                total_cands = sum(len(sh.candidates) for sh in s.shots)
                picked = sum(1 for sh in s.shots if sh.picked_index)
                print(f"  {s.scene_id}  {len(s.shots)} shots, "
                      f"{total_cands} candidates total, {picked}/{len(s.shots)} picked")
                for sh in s.shots:
                    mark = f"pick=cand_{sh.picked_index:02d}" if sh.picked_index else "pick=<none>"
                    print(f"    {sh.shot_id}  {len(sh.candidates)} candidates  {mark}")
                    for c in sh.candidates:
                        print(f"      cand_{c.index:02d}  {c.path}")
            else:
                # Legacy scene-level
                pick = f"pick=cand_{s.picked_index:02d}" if s.picked_index else "pick=<none>"
                print(f"  {s.scene_id}  (legacy)  {len(s.candidates)} candidates  {pick}")
                for c in s.candidates:
                    print(f"    cand_{c.index:02d}  {c.path}")
    return 0


def _cmd_art_pick(args: argparse.Namespace) -> int:
    """Pick a candidate for a specific SHOT. Requires --scene, --shot, --candidate.

    Only needed when `art generate --candidates N` was run with N>1
    (default 1 auto-picks candidate 1). Legacy scene-level picks (for
    stories generated pre-2026-04-21) still work via --scene + --candidate
    without --shot."""
    settings = load_settings()
    bible = _bible_from(settings, args.bible_dir)
    current = bible.load_current(args.story_id, StageName.ART)
    if current is None:
        raise CliError("NO_ARTIFACT", "no art artifact yet.")
    assert isinstance(current, ArtManifest)
    if not args.scene:
        raise CliError("MISSING_SCENE", "art pick requires --scene <id>")
    if args.candidate is None:
        raise CliError("MISSING_CANDIDATE", "art pick requires --candidate <N>")

    sc = next((s for s in current.scenes if s.scene_id == args.scene), None)
    if sc is None:
        raise CliError("UNKNOWN_SCENE", f"scene '{args.scene}' not in art")

    # Per-shot mode: require --shot
    if sc.shots:
        if not getattr(args, "shot", None):
            raise CliError(
                "MISSING_SHOT",
                f"scene {args.scene} uses per-shot keyframes; also pass --shot <id> "
                f"(shots: {[s.shot_id for s in sc.shots]})",
            )
        art_shot = next((s for s in sc.shots if s.shot_id == args.shot), None)
        if art_shot is None:
            raise CliError("UNKNOWN_SHOT", f"shot '{args.shot}' not in scene {args.scene}")
        cand = next((c for c in art_shot.candidates if c.index == args.candidate), None)
        if cand is None:
            raise CliError(
                "UNKNOWN_CANDIDATE",
                f"candidate {args.candidate} not in shot {args.shot} "
                f"(available: {[c.index for c in art_shot.candidates]})",
            )
        art_shot.picked_index = args.candidate
        picked_path = cand.path
    else:
        # Legacy scene-level pick
        cand = next((c for c in sc.candidates if c.index == args.candidate), None)
        if cand is None:
            raise CliError(
                "UNKNOWN_CANDIDATE",
                f"candidate {args.candidate} not in scene {args.scene} "
                f"(available: {[c.index for c in sc.candidates]})",
            )
        sc.picked_index = args.candidate
        src = bible.root / cand.path
        dst = _art_scene_art_path(bible, args.story_id, args.scene)
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy(src, dst)
        picked_path = str(dst)

    bible.save_pending(args.story_id, StageName.ART, current)

    _emit(
        {
            "story_id": args.story_id,
            "stage": "art",
            "state": "pending",
            "scene": args.scene,
            "shot": getattr(args, "shot", None),
            "picked": args.candidate,
            "picked_path": picked_path,
            "next_action": f"hitchcock art show -s {args.story_id}",
        },
        as_json=args.json_out,
    )
    return 0


def _cmd_art_refine(args: argparse.Namespace) -> int:
    """Regenerate keyframes for an entire scene with director feedback.

    Appends the feedback to the storyboard refine log so MIMO sees it at
    the NEXT `storyboard refine --scene X`. For targeted keyframe-only
    regen without touching shots, just rerun `art generate --scene X`
    after editing the keyframe_prompt inside storyboard/pending.json."""
    settings = load_settings()
    bible = _bible_from(settings, args.bible_dir)
    _require_init(bible, args.story_id)
    if not args.scene:
        raise CliError("MISSING_SCENE", "art refine requires --scene <id>")
    if not args.feedback.strip():
        raise CliError("EMPTY_FEEDBACK", "--feedback cannot be empty")

    sb_current = bible.load_current(args.story_id, StageName.STORYBOARD)
    if sb_current is None:
        raise CliError("NO_UPSTREAM_APPROVED", "approve storyboard first.")
    assert isinstance(sb_current, Storyboard)
    sb_scene = next((s for s in sb_current.scenes if s.scene_id == args.scene), None)
    if sb_scene is None:
        raise CliError("UNKNOWN_SCENE", f"scene '{args.scene}' not in storyboard")

    story = _load_approved_script(bible, args.story_id)
    # Refine the storyboard scene (shots + keyframe_prompt are MIMO-authored
    # together in _SHOTS_SYSTEM).
    sb_agent = StoryboardAgent(llm=MimoClient(settings.mimo), bible=bible)
    new_storyboard = sb_agent.refine_scene(
        sb_current, story, args.scene, args.feedback
    )
    bible.save_pending(args.story_id, StageName.STORYBOARD, new_storyboard)
    bible.append_feedback(args.story_id, StageName.ART, args.scene, args.feedback)

    new_sb_scene = next(s for s in new_storyboard.scenes if s.scene_id == args.scene)
    nb = GPTImageClient(settings.openai)
    art_shots = _generate_keyframes_for_scene(
        bible, nb, args.story_id, new_sb_scene, story, args.candidates
    )

    art_current = bible.load_current(args.story_id, StageName.ART)
    manifest = art_current if isinstance(art_current, ArtManifest) else ArtManifest(story_id=args.story_id)
    by_id = {s.scene_id: s for s in manifest.scenes}
    by_id[args.scene] = ArtScene(scene_id=args.scene, shots=art_shots)
    manifest.scenes = list(by_id.values())
    bible.save_pending(args.story_id, StageName.ART, manifest)

    _emit(
        {
            "story_id": args.story_id,
            "stage": "art",
            "state": "pending",
            "scene": args.scene,
            "regenerated_shots": len(art_shots),
            "note": "storyboard was refined (pending); approve both before render.",
            "next_action": f"hitchcock art show -s {args.story_id} --scene {args.scene}",
        },
        as_json=args.json_out,
    )
    return 0


def _cmd_art_approve(args: argparse.Namespace) -> int:
    settings = load_settings()
    bible = _bible_from(settings, args.bible_dir)
    _require_init(bible, args.story_id)
    state = bible.stage_state(args.story_id, StageName.ART)
    if state != StageState.PENDING:
        raise CliError("NO_PENDING", f"no pending art (state={state.value}).")
    current = bible.load_pending(args.story_id, StageName.ART)
    assert isinstance(current, ArtManifest)

    # Per-shot mode (new): every shot in every scene must have picked_index.
    # Legacy mode (old stories): scene-level picked_index. Accept either per
    # scene.
    unpicked_shots: list[str] = []
    unpicked_scenes: list[str] = []
    for sc in current.scenes:
        if sc.shots:
            for ash in sc.shots:
                if ash.picked_index is None:
                    unpicked_shots.append(f"{sc.scene_id}/{ash.shot_id}")
        else:
            if sc.picked_index is None:
                unpicked_scenes.append(sc.scene_id)

    allow = getattr(args, "allow_unpicked", False)
    if (unpicked_shots or unpicked_scenes) and not allow:
        msg = "pick canonical candidate first: "
        if unpicked_shots:
            msg += f"shots={unpicked_shots}; "
        if unpicked_scenes:
            msg += f"scenes={unpicked_scenes}; "
        msg += (
            "Pass --allow-unpicked to approve partially (render package "
            "will skip unpicked items)."
        )
        raise CliError("UNPICKED", msg)
    if unpicked_shots or unpicked_scenes:
        log.warning(
            "art approve: unpicked (will be skipped downstream) — shots=%s scenes=%s",
            unpicked_shots, unpicked_scenes,
        )
    approved_path = bible.approve_pending(args.story_id, StageName.ART)
    _emit(
        {
            "story_id": args.story_id,
            "stage": "art",
            "state": "approved",
            "approved_path": str(approved_path),
            "next_action": f"hitchcock render package -s {args.story_id}",
        },
        as_json=args.json_out,
    )
    return 0


# ─── render ──────────────────────────────────────────────────────────────

def _load_approved_art(bible: BibleStore, sid: str) -> ArtManifest:
    art = bible.load_approved(sid, StageName.ART)
    if art is None:
        raise CliError(
            "NO_UPSTREAM_APPROVED",
            f"approve art first: `hitchcock art approve -s {sid}`",
        )
    assert isinstance(art, ArtManifest)
    return art


def _cmd_render_package(args: argparse.Namespace) -> int:
    """Build Jimeng upload bundles per scene. Produces directories under
    `bible/stories/<sid>/render/packages/<scene_id>/` with first_frame,
    character refs, and prompt.txt. User uploads these to Jimeng Web UI."""
    settings = load_settings()
    bible = _bible_from(settings, args.bible_dir)
    _require_init(bible, args.story_id)
    sb = _load_approved_storyboard(bible, args.story_id)
    art = _load_approved_art(bible, args.story_id)
    story = _load_approved_script(bible, args.story_id)

    max_chars = getattr(args, "max_chars", None)
    compressor = None
    if max_chars:
        from .agents.storyboard import compress_seedance_prompt as _cz
        compressor = (MimoClient(settings.mimo), _cz)

    pkg_root = bible.story_dir(args.story_id) / "render" / "packages"
    pkg_root.mkdir(parents=True, exist_ok=True)

    # Render manifest (pending slot until `render approve` — though render
    # stage doesn't really need approve; the reel is the final artifact).
    manifest = RenderManifest(story_id=args.story_id)

    target_ids = (
        {x.strip() for x in args.scene.split(",") if x.strip()}
        if args.scene else None
    )
    by_sb = {s.scene_id: s for s in sb.scenes}
    by_art = {s.scene_id: s for s in art.scenes}

    for sb_scene in sb.scenes:
        if target_ids and sb_scene.scene_id not in target_ids:
            continue
        pkg_dir = pkg_root / sb_scene.scene_id
        # Clean pkg_dir to avoid stale files from prior runs when the
        # scene's character set changed (e.g. storyboard regen moved a
        # character out of this scene — old NN_<id>_ref.png would be kept).
        if pkg_dir.exists():
            for old in pkg_dir.iterdir():
                if old.is_file():
                    old.unlink()
        pkg_dir.mkdir(parents=True, exist_ok=True)
        art_scene = by_art.get(sb_scene.scene_id)
        scene = next((s for s in story.scenes if s.id == sb_scene.scene_id), None)

        # Ref budget: 9 total. Resolve which refs will be uploaded and in
        # what order — assembler's @image tag numbering must match this.
        # Per-shot keyframes first (trimmed from earliest on overflow),
        # then character refs. Legacy scene-level keyframe is used only
        # when the scene has no per-shot keyframes.
        shot_keyframe_paths: list[tuple[str, Path]] = []  # [(shot_id, path)]
        if art_scene and art_scene.shots:
            for art_shot in art_scene.shots:
                if art_shot.picked_index:
                    cand = next(
                        (c for c in art_shot.candidates if c.index == art_shot.picked_index),
                        None,
                    )
                    if cand:
                        shot_keyframe_paths.append(
                            (art_shot.shot_id, bible.root / cand.path)
                        )
        legacy_scene_keyframe: Path | None = None
        if not shot_keyframe_paths and art_scene and art_scene.picked_index:
            cand = next(
                (c for c in art_scene.candidates if c.index == art_scene.picked_index),
                None,
            )
            if cand:
                legacy_scene_keyframe = bible.root / cand.path

        # Character refs (one per character actually present in the scene).
        # Filter characters_in_scene down to those who either (a) speak in
        # this scene or (b) appear in at least one shot's characters_in_shot.
        # This mirrors the filter inside `_assemble_seedance_prompt` so
        # files on disk match the `@image N: 中年男人` bindings in prompt.txt
        # (previous versions uploaded unused character refs that Seedance
        # then tried to place on-screen, causing identity drift).
        from .agents.storyboard import _canonical_role_cn, _disambiguate_labels
        char_refs: list[tuple[str, Path]] = []  # [(cid, front_path)]
        if scene:
            speaking_ids = {d.speaker_id for d in getattr(scene, "dialogue", [])}
            used_labels: set[str] = set()
            for sh in sb_scene.shots:
                for lbl in getattr(sh, "characters_in_shot", []) or []:
                    used_labels.add(lbl)
            all_ids = scene.characters_in_scene
            raw = [
                _canonical_role_cn(
                    bible.load_character(cid) if cid else None
                ) for cid in all_ids
            ]
            disambig = _disambiguate_labels(raw)
            cid_lbl = dict(zip(all_ids, disambig))
            filtered = [
                cid for cid in all_ids
                if cid in speaking_ids or cid_lbl.get(cid) in used_labels
            ] or all_ids  # safety: never end up with zero refs
            for cid in filtered:
                front = bible.refs_dir(cid) / "front.png"
                if front.exists():
                    char_refs.append((cid, front))

        # 2026-04-22 architecture: only 1 shot keyframe (establishing) +
        # character refs. Style references are OFF — previous multi-ref
        # mode (3 shot keyframes + 2 style refs + N chars) produced
        # fragmented clips (Seedance treated each keyframe as a hard
        # anchor, interpolation between them was choppy). Single-keyframe
        # gives Seedance motion room; text prompt carries style via
        # the beefed-up `画风：` line (see feedback_video_single_ref.md).
        if shot_keyframe_paths:
            shot_keyframe_paths = shot_keyframe_paths[:1]  # first shot only

        # Emit files: (1) shot keyframe → (2) char refs. No style refs.
        img_idx = 1
        shots_with_keyframes_ids: list[str] = []
        if shot_keyframe_paths:
            for shot_id, src in shot_keyframe_paths:
                shutil.copy(src, pkg_dir / f"{img_idx:02d}_shot_{shot_id}.png")
                shots_with_keyframes_ids.append(shot_id)
                img_idx += 1
        elif legacy_scene_keyframe:
            shutil.copy(legacy_scene_keyframe, pkg_dir / f"{img_idx:02d}_first_frame.png")
            img_idx += 1
        for cid, front in char_refs:
            shutil.copy(front, pkg_dir / f"{img_idx:02d}_{cid}_ref.png")
            img_idx += 1

        from .agents.storyboard import reassemble_seedance_prompt as _reassemble
        prompt_txt = _reassemble(
            bible, story, sb_scene,
            shots_with_keyframes=shots_with_keyframes_ids or None,
            n_style_refs=0,
        )
        if compressor and len(prompt_txt) > max_chars:
            log.info("render: compressing %s prompt %d → ≤%d",
                     sb_scene.scene_id, len(prompt_txt), max_chars)
            llm, fn = compressor
            prompt_txt = fn(llm, prompt_txt, max_chars)
        (pkg_dir / "prompt.txt").write_text(prompt_txt, encoding="utf-8")

        manifest.scenes.append(RenderScene(
            scene_id=sb_scene.scene_id,
            backend=RenderBackend.JIMENG_PACKAGE,
            package_dir=str(pkg_dir.relative_to(bible.root)),
        ))

    bible.save_pending(args.story_id, StageName.RENDER, manifest)

    _emit(
        {
            "story_id": args.story_id,
            "stage": "render",
            "backend": "jimeng_package",
            "package_root": str(pkg_root),
            "scenes_packaged": [s.scene_id for s in manifest.scenes],
            "next_action": (
                f"upload each packages/<scene_id>/ folder to https://jimeng.jianying.com/ai-tool/generate, "
                f"then download mp4s back and run `hitchcock render post -s {args.story_id}`"
            ),
        },
        as_json=args.json_out,
    )
    return 0


def _cmd_render_post(args: argparse.Namespace) -> int:
    """Concat scene clips into the final reel.

    Default: pure ffmpeg concat — Seedance/Jimeng's native audio (including
    dialogue voice-over) is preserved as-is. No TTS overlay.

    Flags:
      --with-tts    : overlay edge-tts VO on top of native audio (old behavior).
      --with-bgm    : mix a BGM track under everything (reads render/bgm/<id>.mp3).

    Either flag triggers per-scene mixing via _mix_scene_audio; without flags,
    we just stream-copy-concat the uploaded clips.
    """
    import subprocess
    settings = load_settings()
    bible = _bible_from(settings, args.bible_dir)
    _require_init(bible, args.story_id)
    sb = _load_approved_storyboard(bible, args.story_id)

    target_ids = (
        {x.strip() for x in args.scene.split(",") if x.strip()}
        if args.scene else None
    )
    target_scenes = [
        s for s in sb.scenes if not target_ids or s.scene_id in target_ids
    ]

    story_root = bible.story_dir(args.story_id)
    clips_dir = story_root / "render" / "clips"
    tts_root = story_root / "tts"
    bgm_root = story_root / "bgm"
    mixed_dir = story_root / "render" / "mixed"
    reel_path = story_root / "render" / "reel.mp4"

    missing = [s.scene_id for s in target_scenes
               if not (clips_dir / f"{s.scene_id}.mp4").exists()]
    if missing:
        raise CliError(
            "MISSING_CLIPS",
            f"missing scene clips: {missing}. Drop mp4s into {clips_dir}/.",
        )

    with_tts = getattr(args, "with_tts", False)
    with_bgm = getattr(args, "with_bgm", False)
    needs_mix = with_tts or with_bgm

    if needs_mix:
        mixed_dir.mkdir(parents=True, exist_ok=True)
        clips_to_concat: list[Path] = []
        for s in target_scenes:
            video_in = clips_dir / f"{s.scene_id}.mp4"
            video_out = mixed_dir / f"{s.scene_id}.mp4"
            _mix_scene_audio(
                video_in=video_in, video_out=video_out,
                tts_dir=(tts_root / s.scene_id) if with_tts else None,
                bgm_file=(bgm_root / f"{s.scene_id}.mp3") if with_bgm else None,
                scene_duration=sum(sh.duration_sec for sh in s.shots) or 15.0,
            )
            clips_to_concat.append(video_out)
        list_dir = mixed_dir
    else:
        # No mixing — concat uploaded clips directly.
        clips_to_concat = [clips_dir / f"{s.scene_id}.mp4" for s in target_scenes]
        list_dir = story_root / "render"

    list_file = list_dir / "_concat.txt"
    list_file.write_text(
        "\n".join(f"file '{p.resolve()}'" for p in clips_to_concat), encoding="utf-8"
    )
    cmd = [
        "ffmpeg", "-y", "-f", "concat", "-safe", "0",
        "-i", str(list_file), "-c", "copy", str(reel_path),
    ]
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        cmd = [
            "ffmpeg", "-y", "-f", "concat", "-safe", "0",
            "-i", str(list_file),
            "-c:v", "libx264", "-c:a", "aac", str(reel_path),
        ]
        r = subprocess.run(cmd, capture_output=True, text=True)
        if r.returncode != 0:
            raise CliError("FFMPEG_FAIL", r.stderr[-500:])

    _emit(
        {
            "story_id": args.story_id,
            "stage": "render",
            "reel_path": str(reel_path),
            "clips_concatenated": len(clips_to_concat),
            "scenes": [s.scene_id for s in target_scenes],
            "audio_mode": "native_seedance" if not needs_mix
                          else f"mixed(tts={with_tts},bgm={with_bgm})",
            "next_action": "pipeline complete (or `hitchcock render subtitles -s <sid>` for SRT)",
        },
        as_json=args.json_out,
    )
    return 0


def _mix_scene_audio(
    *, video_in: Path, video_out: Path,
    tts_dir: Path, bgm_file: Path,
    scene_duration: float,
) -> None:
    """Per-scene audio mixing. Builds an ffmpeg filter graph:
      video → keep video stream
      audio → TTS concat (with silence pad) + optional BGM underneath.
    If no TTS manifest: just keep original video audio (or silence if none).
    """
    import subprocess
    import json as _json

    manifest_path = tts_dir / "manifest.json"
    has_tts = manifest_path.exists()
    has_bgm = bgm_file.exists()

    # Shortcut: no audio overlays requested → copy as-is.
    if not has_tts and not has_bgm:
        subprocess.run(
            ["ffmpeg", "-y", "-i", str(video_in), "-c", "copy", str(video_out)],
            capture_output=True, check=False,
        )
        return

    lines: list[dict] = []
    if has_tts:
        try:
            lines = _json.loads(manifest_path.read_text())["lines"]
        except (json.JSONDecodeError, KeyError, OSError) as e:
            log.warning("render post: TTS manifest %s unreadable (%s): %s — "
                        "proceeding with no VO overlay", manifest_path, type(e).__name__, e)
            lines = []

    # Build a VO track by concatenating TTS lines. Distribute lines across
    # the scene_duration with equal gaps so speech is paced, not crammed.
    # If lines' combined audio > scene_duration, we just play them back-to-back.
    inputs: list[str] = ["-i", str(video_in)]
    tts_paths = [tts_dir / line["path"].split("/")[-1] for line in lines]
    for p in tts_paths:
        inputs.extend(["-i", str(p)])
    if has_bgm:
        inputs.extend(["-i", str(bgm_file)])

    # Compose filter graph.
    #
    # Layer model (bottom → top):
    #   1. Video's original audio (Jimeng SFX: engine, rain, ambient) @ 0.35
    #   2. BGM track, looped @ 0.15 (only if bgm file exists)
    #   3. TTS voiceover @ 1.0 (with silence padding between lines)
    #
    # TTS is OVERLAID on top of the original audio, not a replacement —
    # so Jimeng's ambient sound stays as bed while dialogue rides above it.
    filter_parts: list[str] = []
    audio_sources: list[str] = []  # the labels that will feed the final amix

    # Layer 1: original video audio (ducked)
    filter_parts.append("[0:a]aresample=44100,volume=0.35[orig]")
    audio_sources.append("[orig]")

    # Layer 2: BGM (optional)
    if has_bgm:
        bgm_idx = 1 + len(tts_paths)
        filter_parts.append(
            f"[{bgm_idx}:a]aresample=44100,volume=0.15,"
            f"aloop=loop=-1:size=2e9[bgm]"
        )
        audio_sources.append("[bgm]")

    # Layer 3: TTS concat with silence gaps
    if tts_paths:
        total_tts = sum(line.get("duration_s", 0) for line in lines)
        gap_s = max(0.2, (scene_duration - total_tts) / max(1, len(lines) + 1))
        concat_chain = f"aevalsrc=0:d={gap_s},aresample=44100[s0];"
        labels = ["[s0]"]
        for i in range(len(tts_paths)):
            concat_chain += f"[{i+1}:a]aresample=44100[ln{i}];"
            labels.append(f"[ln{i}]")
            if i < len(tts_paths) - 1:
                concat_chain += f"aevalsrc=0:d={gap_s},aresample=44100[s{i+1}];"
                labels.append(f"[s{i+1}]")
        concat_chain += f"{''.join(labels)}concat=n={len(labels)}:v=0:a=1[vo_raw];"
        concat_chain += "[vo_raw]volume=1.0[vo]"
        filter_parts.append(concat_chain)
        audio_sources.append("[vo]")

    # Final mix — always run amix even for single source, for consistent output.
    if len(audio_sources) == 1:
        audio_out = audio_sources[0]
    else:
        filter_parts.append(
            f"{''.join(audio_sources)}amix=inputs={len(audio_sources)}:"
            f"duration=longest:dropout_transition=0[a]"
        )
        audio_out = "[a]"

    cmd = ["ffmpeg", "-y", *inputs]
    if filter_parts:
        cmd.extend(["-filter_complex", ";".join(filter_parts)])
    cmd.extend(["-map", "0:v"])
    if audio_out:
        cmd.extend(["-map", audio_out])
    else:
        cmd.extend(["-map", "0:a?"])
    cmd.extend([
        "-c:v", "copy",
        "-c:a", "aac", "-b:a", "192k",
        "-shortest",
        str(video_out),
    ])
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        log.warning("_mix_scene_audio ffmpeg failed for %s: %s",
                    video_in.name, r.stderr[-300:])
        # Fallback: copy video as-is (no mix).
        subprocess.run(
            ["ffmpeg", "-y", "-i", str(video_in), "-c", "copy", str(video_out)],
            capture_output=True, check=False,
        )


def _cmd_render_stub(kind: str):
    def _run(args: argparse.Namespace) -> int:  # noqa: ARG001
        raise CliError(
            "NOT_IMPLEMENTED",
            f"`hitchcock render {kind}` is planned but not wired up yet (Phase 2 — TTS/BGM/Seedance API).",
        )
    return _run


def _cmd_render_seedance(args: argparse.Namespace) -> int:
    """Generate scene clips via Seedance 2.0 API directly.

    Skips Jimeng Web UI entirely. Uses omni-reference (全能参考):
      - first_frame = picked scene art keyframe
      - subject refs = character front.png per character in scene
      - environment ref = location establishing art (if distinct from first_frame)

    Prompt is re-assembled from current code (reassemble_seedance_prompt) —
    NO compression needed, the API has no 2000-char cap. Default 480p to
    save tokens; override with --resolution 720p if quality matters more.
    """
    settings = load_settings()
    bible = _bible_from(settings, args.bible_dir)
    _require_init(bible, args.story_id)
    sb = _load_approved_storyboard(bible, args.story_id)
    art = _load_approved_art(bible, args.story_id)
    story = _load_approved_script(bible, args.story_id)

    target_ids = (
        {x.strip() for x in args.scene.split(",") if x.strip()}
        if args.scene else None
    )
    if not target_ids:
        raise CliError(
            "SCENE_REQUIRED",
            "`render seedance` requires --scene (each call costs Ark tokens; "
            "refuse to fan out silently across every scene).",
        )

    from .video import SeedanceClient, SeedanceError
    from .agents.storyboard import reassemble_seedance_prompt as _reassemble
    client = SeedanceClient(settings.ark)

    # `render post` reads clips from `render/clips/`; keep seedance output
    # in the same directory so the two stages agree without a manual move.
    clips_dir = bible.story_dir(args.story_id) / "render" / "clips"
    clips_dir.mkdir(parents=True, exist_ok=True)

    # Merge into existing pending manifest so other scenes' entries
    # (e.g. Jimeng-packaged siblings) aren't clobbered.
    existing = bible.load_pending(args.story_id, StageName.RENDER)
    if not isinstance(existing, RenderManifest):
        existing = RenderManifest(story_id=args.story_id)
    by_manifest = {s.scene_id: s for s in existing.scenes}

    by_art = {s.scene_id: s for s in art.scenes}
    by_sb = {s.scene_id: s for s in sb.scenes}

    results: list[dict] = []
    for scene_id in sorted(target_ids):
        sb_scene = by_sb.get(scene_id)
        if sb_scene is None:
            raise CliError("SCENE_NOT_FOUND", f"scene {scene_id} not in approved storyboard.")
        scene = next((s for s in story.scenes if s.id == scene_id), None)
        if scene is None:
            raise CliError("SCENE_NOT_FOUND", f"scene {scene_id} not in approved script.")

        # Resolve shot keyframes (new per-shot mode) or fall back to
        # scene-level keyframe (legacy). Order MUST match assembler's
        # @image tag numbering: shot keyframes first, chars next.
        art_scene = by_art.get(scene_id)
        shot_keyframe_paths: list[tuple[str, Path]] = []
        legacy_scene_keyframe: Path | None = None
        if art_scene and art_scene.shots:
            for art_shot in art_scene.shots:
                if art_shot.picked_index:
                    cand = next(
                        (c for c in art_shot.candidates if c.index == art_shot.picked_index),
                        None,
                    )
                    if cand:
                        shot_keyframe_paths.append(
                            (art_shot.shot_id, bible.root / cand.path)
                        )
        if not shot_keyframe_paths and art_scene and art_scene.picked_index:
            cand = next(
                (c for c in art_scene.candidates if c.index == art_scene.picked_index),
                None,
            )
            if cand:
                legacy_scene_keyframe = bible.root / cand.path
        if not shot_keyframe_paths and not legacy_scene_keyframe:
            raise CliError(
                "MISSING_KEYFRAME",
                f"scene {scene_id} has no picked/available keyframe. "
                f"Run `art generate -s {args.story_id} --scene {scene_id}` + "
                f"`art approve` first.",
            )

        # Character refs (front.png per char_in_scene).
        char_refs: list[Path] = []
        for cid in scene.characters_in_scene:
            front = bible.refs_dir(cid) / "front.png"
            if front.exists():
                char_refs.append(front)

        # Style refs — skipped in --use-package-prompt mode so the uploaded
        # refs match the Jimeng bundle's @imageN numbering (keyframe + chars).
        style_refs: list[Path] = (
            [] if getattr(args, "use_package_prompt", False)
            else bible.list_style_refs(args.story_id)[:2]
        )

        # Budget: 9 total. Trim shot keyframes from earliest on overflow.
        max_shot_refs = max(0, 9 - len(char_refs) - len(style_refs))
        if len(shot_keyframe_paths) > max_shot_refs:
            drop = len(shot_keyframe_paths) - max_shot_refs
            log.warning(
                "seedance: scene %s has %d shot keyframes + %d style + %d chars > 9; "
                "dropping %d earliest keyframes",
                scene_id, len(shot_keyframe_paths), len(style_refs),
                len(char_refs), drop,
            )
            shot_keyframe_paths = shot_keyframe_paths[drop:]

        # Assemble upload ref list in order matching assembler's @imageN:
        # shot keyframes → style refs → char refs.
        reference_images: list[Path] = []
        shots_with_keyframes_ids: list[str] = []
        if shot_keyframe_paths:
            for shot_id, src in shot_keyframe_paths:
                reference_images.append(src)
                shots_with_keyframes_ids.append(shot_id)
        elif legacy_scene_keyframe:
            reference_images.append(legacy_scene_keyframe)
        reference_images.extend(style_refs)
        reference_images.extend(char_refs)

        if getattr(args, "use_package_prompt", False):
            pkg_prompt = (
                bible.story_dir(args.story_id) / "render" / "packages"
                / scene_id / "prompt.txt"
            )
            if not pkg_prompt.exists():
                raise CliError(
                    "MISSING_PACKAGE_PROMPT",
                    f"--use-package-prompt: {pkg_prompt} not found. "
                    f"Run `render package -s {args.story_id}` first, then edit prompt.txt.",
                )
            prompt_txt = pkg_prompt.read_text(encoding="utf-8")
            log.info("Seedance: scene %s using hand-edited package prompt (%d ch)",
                     scene_id, len(prompt_txt))
        else:
            prompt_txt = _reassemble(
                bible, story, sb_scene,
                shots_with_keyframes=shots_with_keyframes_ids or None,
                n_style_refs=len(style_refs),
            )

        # Duration: Seedance only accepts 5/10/12/15. Our assembler clamps
        # scenes to ≤15s, so 15 is the default. Respect --duration-sec if given.
        duration = args.duration_sec or 15
        if duration not in {5, 10, 12, 15}:
            raise CliError(
                "BAD_DURATION",
                f"Seedance 2.0 only supports duration in {{5, 10, 12, 15}}s; got {duration}.",
            )

        log.info(
            "Seedance: scene %s (%s) | dur=%ds res=%s | refs=%d/9 | prompt=%d ch",
            scene_id, scene.title, duration, args.resolution,
            len(reference_images), len(prompt_txt),
        )

        if args.dry_run:
            results.append({
                "scene_id": scene_id,
                "dry_run": True,
                "prompt_chars": len(prompt_txt),
                "reference_images": [str(p) for p in reference_images],
                "duration_sec": duration,
                "resolution": args.resolution,
            })
            continue

        try:
            clip = client.generate(
                prompt_txt,
                reference_images=reference_images,
                duration_sec=duration,
                resolution=args.resolution,
            )
        except SeedanceError as e:
            raise CliError("SEEDANCE_FAILED", f"scene {scene_id}: {e}") from e

        dst = clips_dir / f"{scene_id}.mp4"
        shutil.move(str(clip.path), str(dst))
        size_kb = dst.stat().st_size // 1024
        log.info("Seedance: saved %s (%d KB)", dst, size_kb)

        rs = RenderScene(
            scene_id=scene_id,
            backend=RenderBackend.SEEDANCE,
            clip_path=str(dst.relative_to(bible.root)),
        )
        by_manifest[scene_id] = rs
        results.append({
            "scene_id": scene_id,
            "clip_path": rs.clip_path,
            "size_kb": size_kb,
            "duration_sec": duration,
            "resolution": args.resolution,
        })

    if not args.dry_run:
        existing.scenes = sorted(by_manifest.values(), key=lambda s: s.scene_id)
        bible.save_pending(args.story_id, StageName.RENDER, existing)

    _emit(
        {
            "story_id": args.story_id,
            "stage": "render",
            "backend": "seedance",
            "resolution": args.resolution,
            "dry_run": args.dry_run,
            "scenes": results,
            "next_action": (
                "inspect clips in bible/stories/<sid>/clips/; if good, run "
                f"`hitchcock render post -s {args.story_id} --scene <id>` to "
                "concat into reel.mp4"
            ),
        },
        as_json=args.json_out,
    )
    return 0


def _cmd_render_tts(args: argparse.Namespace) -> int:
    """Generate edge-tts MP3 per dialogue line + manifest per scene."""
    settings = load_settings()
    bible = _bible_from(settings, args.bible_dir)
    _require_init(bible, args.story_id)
    story = _load_approved_script(bible, args.story_id)

    target_ids = (
        [x.strip() for x in args.scene.split(",") if x.strip()]
        if args.scene else None
    )
    agent = TTSAgent(bible=bible)
    summary = agent.generate(story, scene_ids=target_ids, rate=args.rate)

    total_lines = sum(len(s["lines"]) for s in summary["scenes"])
    total_audio_s = round(
        sum(line["duration_s"] for s in summary["scenes"] for line in s["lines"]),
        1,
    )
    _emit(
        {
            "story_id": args.story_id,
            "stage": "render",
            "task": "tts",
            "scenes_processed": [s["scene_id"] for s in summary["scenes"]],
            "total_lines": total_lines,
            "total_audio_seconds": total_audio_s,
            "next_action": f"hitchcock render bgm -s {args.story_id}  # then `render post`",
        },
        as_json=args.json_out,
    )
    return 0


# ─── design / location (legacy one-shots) ────────────────────────────────

def _cmd_design(args: argparse.Namespace) -> int:
    settings = load_settings()
    bible = _bible_from(settings, args.bible_dir)
    description = _read_input(args.input)
    if not description.strip():
        raise CliError("EMPTY_INPUT", "empty character description")
    agent = DesignAgent(
        llm=MimoClient(settings.mimo),
        images=GPTImageClient(settings.openai),
        bible=bible,
    )
    character = agent.design(description)
    _emit(
        {
            "character_id": character.id,
            "character_json": str(bible.character_json(character.id)),
            "refs_dir": str(bible.refs_dir(character.id)),
        },
        as_json=args.json_out,
    )
    return 0


def _cmd_location(args: argparse.Namespace) -> int:
    settings = load_settings()
    bible = _bible_from(settings, args.bible_dir)
    description = _read_input(args.input)
    if not description.strip():
        raise CliError("EMPTY_INPUT", "empty location description")
    agent = LocationAgent(
        llm=MimoClient(settings.mimo),
        images=GPTImageClient(settings.openai),
        bible=bible,
    )
    loc = agent.create(description)
    _emit(
        {
            "location_id": loc.id,
            "location_json": str(bible.location_json(loc.id)),
            "establishing_art": str(bible.establishing_art(loc.id)),
        },
        as_json=args.json_out,
    )
    return 0


# ─── main dispatcher ─────────────────────────────────────────────────────

def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="hitchcock",
        description=(
            f"Hitchcock pipeline CLI (contract v{CONTRACT_VERSION}). "
            f"See AGENTS.md for the full contract."
        ),
    )
    parser.add_argument("--version", action="version", version=CONTRACT_VERSION)
    sub = parser.add_subparsers(dest="cmd", required=True)

    # init --------------------------------------------------------------
    p_init = sub.add_parser("init", help="Initialize a new story.")
    p_init.add_argument("-s", "--story-id", required=True)
    p_init.add_argument("--source", required=True,
                        help="Path to source text, or '-' for stdin.")
    # Characters + locations are now auto-discovered via `hitchcock cast
    # discover`; the flags below are legacy opt-in for pre-populating.
    p_init.add_argument("--character", action="append", default=[],
                        help="(legacy) pre-existing character id; optional.")
    p_init.add_argument("--location", action="append", default=[],
                        help="(legacy) pre-existing location id; optional.")
    _add_common_flags(p_init)
    p_init.set_defaults(func=_cmd_init)

    # brief -------------------------------------------------------------
    p_brief = sub.add_parser("brief", help="Stage 0 — directorial intent + canon research.")
    brief_sub = p_brief.add_subparsers(dest="verb", required=True)

    bp = brief_sub.add_parser("questions", help="Emit the question list.")
    bp.add_argument("-s", "--story-id", required=True)
    _add_common_flags(bp)
    bp.set_defaults(func=_cmd_brief_questions)

    bp = brief_sub.add_parser(
        "answer",
        help="Capture director intent. Use --intent 'paragraph' (free-form, "
             "MIMO parses) OR --responses path.json (legacy structured).",
    )
    bp.add_argument("-s", "--story-id", required=True)
    bp.add_argument("--intent", default=None,
                    help="Free-form paragraph describing what you want. "
                         "MIMO extracts style/tone/must-haves/etc. into "
                         "structured BriefAnswers.")
    bp.add_argument("--responses", default=None,
                    help="(legacy) Path to structured answers JSON.")
    _add_common_flags(bp)
    bp.set_defaults(func=_cmd_brief_answer)

    bp = brief_sub.add_parser(
        "research",
        help="Automated canon research via Gemini + Google Search grounding.",
    )
    bp.add_argument("-s", "--story-id", required=True)
    _add_common_flags(bp)
    bp.set_defaults(func=_cmd_brief_research)

    bp = brief_sub.add_parser("plan-research", help="Generate search queries + canon template (legacy, manual).")
    bp.add_argument("-s", "--story-id", required=True)
    _add_common_flags(bp)
    bp.set_defaults(func=_cmd_brief_plan_research)

    bp = brief_sub.add_parser("ingest-canon", help="Import filled canon template.")
    bp.add_argument("-s", "--story-id", required=True)
    bp.add_argument("--file", required=True, help="Path to filled canon JSON.")
    _add_common_flags(bp)
    bp.set_defaults(func=_cmd_brief_ingest_canon)

    bp = brief_sub.add_parser("show")
    bp.add_argument("-s", "--story-id", required=True)
    _add_common_flags(bp)
    bp.set_defaults(func=_cmd_brief_show)

    bp = brief_sub.add_parser("refine")
    bp.add_argument("-s", "--story-id", required=True)
    bp.add_argument("--feedback", required=True)
    _add_common_flags(bp)
    bp.set_defaults(func=_cmd_brief_refine)

    bp = brief_sub.add_parser("approve")
    bp.add_argument("-s", "--story-id", required=True)
    _add_common_flags(bp)
    bp.set_defaults(func=_cmd_brief_approve)

    # style -------------------------------------------------------------
    p_style = sub.add_parser("style", help="Stage 0 — art direction (palette/lighting/motifs/avoid).")
    style_sub = p_style.add_subparsers(dest="verb", required=True)
    for verb, fn, needs_feedback in [
        ("generate", _cmd_style_generate, False),
        ("show", _cmd_style_show, False),
        ("refine", _cmd_style_refine, True),
        ("approve", _cmd_style_approve, False),
    ]:
        sp = style_sub.add_parser(verb)
        sp.add_argument("-s", "--story-id", required=True)
        if needs_feedback:
            sp.add_argument("--feedback", required=True)
        if verb == "approve":
            sp.add_argument(
                "--force-anchors", action="store_true",
                help="Re-render style anchors even if anchor_*.png already exists.",
            )
            _add_image_quality_flag(sp)
        _add_common_flags(sp)
        sp.set_defaults(func=fn)

    # cast --------------------------------------------------------------
    p_cast = sub.add_parser("cast", help="Stage 0.5 — discover cast from source.")
    cast_sub = p_cast.add_subparsers(dest="verb", required=True)

    cp = cast_sub.add_parser("discover")
    cp.add_argument("-s", "--story-id", required=True)
    _add_common_flags(cp)
    cp.set_defaults(func=_cmd_cast_discover)

    cp = cast_sub.add_parser("show")
    cp.add_argument("-s", "--story-id", required=True)
    _add_common_flags(cp)
    cp.set_defaults(func=_cmd_cast_show)

    cp = cast_sub.add_parser("refine")
    cp.add_argument("-s", "--story-id", required=True)
    cp.add_argument("--feedback", required=True)
    _add_common_flags(cp)
    cp.set_defaults(func=_cmd_cast_refine)

    cp = cast_sub.add_parser("build",
                             help="Materialize NEW proposals into bible (2 refs per char ≈ ¥0.8).")
    cp.add_argument("-s", "--story-id", required=True)
    cp.add_argument("--only", default=None,
                    help="Comma-separated ids (canonical or matched bible) to build.")
    cp.add_argument("--skip-refs", action="store_true",
                    help="Write character.json/location.json only, skip image gen.")
    cp.add_argument("--dry-run", action="store_true",
                    help="Print plan + cost estimate, don't call APIs.")
    cp.add_argument("--force", action="store_true",
                    help="Rebuild even if already in bible (useful for restyling under a new StyleGuide).")
    _add_image_quality_flag(cp)
    _add_common_flags(cp)
    cp.set_defaults(func=_cmd_cast_build)

    cp = cast_sub.add_parser("approve")
    cp.add_argument("-s", "--story-id", required=True)
    _add_common_flags(cp)
    cp.set_defaults(func=_cmd_cast_approve)

    # script ------------------------------------------------------------
    p_script = sub.add_parser("script", help="Layer 1 — structured story.")
    script_sub = p_script.add_subparsers(dest="verb", required=True)
    for verb, fn, needs_feedback, supports_scene in [
        ("generate", _cmd_script_generate, False, False),
        ("show", _cmd_script_show, False, True),
        ("refine", _cmd_script_refine, True, False),
        ("approve", _cmd_script_approve, False, False),
    ]:
        sp = script_sub.add_parser(verb)
        sp.add_argument("-s", "--story-id", required=True)
        if needs_feedback:
            sp.add_argument("--feedback", required=True)
        if supports_scene:
            sp.add_argument("--scene", default=None, help="Only this scene (full screenplay view).")
        _add_common_flags(sp)
        sp.set_defaults(func=fn)

    # storyboard --------------------------------------------------------
    p_sb = sub.add_parser("storyboard", help="Layer 2 — shots + prompts per scene.")
    sb_sub = p_sb.add_subparsers(dest="verb", required=True)
    for verb, fn, flags in [
        ("generate", _cmd_storyboard_generate, set()),
        ("show", _cmd_storyboard_show, {"scene"}),
        ("refine", _cmd_storyboard_refine, {"scene", "feedback"}),
        ("approve", _cmd_storyboard_approve, set()),
    ]:
        sp = sb_sub.add_parser(verb)
        sp.add_argument("-s", "--story-id", required=True)
        if "scene" in flags:
            sp.add_argument("--scene", default=None, help="Scene id (e.g. s03).")
        if "feedback" in flags:
            sp.add_argument("--feedback", required=True)
        _add_common_flags(sp)
        sp.set_defaults(func=fn)

    # art ---------------------------------------------------------------
    p_art = sub.add_parser("art", help="Layer 3 — scene-art candidates via Nano Banana.")
    art_sub = p_art.add_subparsers(dest="verb", required=True)
    # generate
    ap = art_sub.add_parser("generate")
    ap.add_argument("-s", "--story-id", required=True)
    ap.add_argument("--scene", default=None, help="Only this scene (default: all).")
    ap.add_argument("--shot", default=None,
                    help="Only this shot id within --scene (default: all shots). "
                         "Preserves other shots' existing candidates untouched.")
    ap.add_argument("--candidates", type=int, default=1,
                    help="Candidates per shot (default 1 to save T2I cost; "
                         "use --candidates 2-3 when a shot needs variety to pick from).")
    _add_image_quality_flag(ap)
    _add_common_flags(ap)
    ap.set_defaults(func=_cmd_art_generate)
    # show
    ap = art_sub.add_parser("show")
    ap.add_argument("-s", "--story-id", required=True)
    ap.add_argument("--scene", default=None)
    _add_common_flags(ap)
    ap.set_defaults(func=_cmd_art_show)
    # pick
    ap = art_sub.add_parser("pick")
    ap.add_argument("-s", "--story-id", required=True)
    ap.add_argument("--scene", required=True)
    ap.add_argument("--shot", default=None,
                    help="Shot id (required when scene uses per-shot keyframes; "
                         "optional for legacy scene-level picks).")
    ap.add_argument("--candidate", type=int, required=True)
    _add_common_flags(ap)
    ap.set_defaults(func=_cmd_art_pick)
    # refine
    ap = art_sub.add_parser("refine")
    ap.add_argument("-s", "--story-id", required=True)
    ap.add_argument("--scene", required=True)
    ap.add_argument("--feedback", required=True)
    ap.add_argument("--candidates", type=int, default=1,
                    help="Candidates per refine (default 1; raise when you want options).")
    _add_common_flags(ap)
    ap.set_defaults(func=_cmd_art_refine)
    # approve
    ap = art_sub.add_parser("approve")
    ap.add_argument("-s", "--story-id", required=True)
    ap.add_argument("--allow-unpicked", action="store_true",
                    help="Approve even if some scenes have no pick (render package will skip them).")
    _add_common_flags(ap)
    ap.set_defaults(func=_cmd_art_approve)

    # render ------------------------------------------------------------
    p_rn = sub.add_parser("render", help="Layer 4 — video output (Jimeng packages + ffmpeg).")
    rn_sub = p_rn.add_subparsers(dest="verb", required=True)
    # package
    rp = rn_sub.add_parser("package", help="Build Jimeng upload bundles per scene.")
    rp.add_argument("-s", "--story-id", required=True)
    rp.add_argument("--scene", default=None)
    rp.add_argument("--max-chars", type=int, default=None,
                    help="Compress each prompt.txt to <= this many chars via MIMO (for Jimeng Web UI's 2000-char limit, typical value 1500).")
    _add_common_flags(rp)
    rp.set_defaults(func=_cmd_render_package)
    # post
    rp = rn_sub.add_parser("post", help="Concat clips → reel.mp4 (Seedance native audio preserved).")
    rp.add_argument("-s", "--story-id", required=True)
    rp.add_argument("--scene", default=None,
                    help="Only these scenes (comma-separated). Reel will contain only these.")
    rp.add_argument("--with-tts", action="store_true",
                    help="Overlay edge-tts VO on top of native audio (default: off — native Seedance VO only).")
    rp.add_argument("--with-bgm", action="store_true",
                    help="Mix BGM from render/bgm/<scene>.mp3 (default: off).")
    _add_common_flags(rp)
    rp.set_defaults(func=_cmd_render_post)
    # tts (real)
    rp = rn_sub.add_parser("tts", help="Generate VO audio per dialogue line via edge-tts (free, offline).")
    rp.add_argument("-s", "--story-id", required=True)
    rp.add_argument("--scene", default=None,
                    help="Only these scenes (comma-separated). Default: all.")
    rp.add_argument("--rate", default="+0%",
                    help="edge-tts rate spec, e.g. '-10%' for slower, '+20%' for faster.")
    _add_common_flags(rp)
    rp.set_defaults(func=_cmd_render_tts)

    # seedance (real API)
    rp = rn_sub.add_parser(
        "seedance",
        help="Generate scene clips directly via Seedance 2.0 API (omni-reference, 480p default).",
    )
    rp.add_argument("-s", "--story-id", required=True)
    rp.add_argument(
        "--scene", required=True,
        help="Comma-separated scene ids to render (REQUIRED — each call costs Ark tokens).",
    )
    rp.add_argument(
        "--resolution", default="480p", choices=["480p", "720p", "1080p"],
        help="Seedance output resolution (default 480p to save tokens; use 720p for final pass).",
    )
    rp.add_argument(
        "--duration-sec", type=int, default=None,
        help="Override clip duration (5/10/12/15 only). Default: 15s (our scene clamp target).",
    )
    rp.add_argument(
        "--dry-run", action="store_true",
        help="Print what would be submitted (prompt size, ref files, duration) without hitting the API.",
    )
    rp.add_argument(
        "--use-package-prompt", action="store_true",
        help="Use the (possibly hand-edited) render/packages/<scene>/prompt.txt as the Seedance prompt "
             "instead of re-assembling from storyboard. Also switches refs to the matching Jimeng bundle "
             "(keyframe + character refs only — no style refs) so @imageN bindings stay consistent.",
    )
    _add_common_flags(rp)
    rp.set_defaults(func=_cmd_render_seedance)

    # bgm stub (Phase 2b)
    rp = rn_sub.add_parser("bgm", help="(not implemented — Phase 2b) bgm.")
    rp.add_argument("-s", "--story-id", required=True)
    rp.add_argument("--scene", default=None)
    _add_common_flags(rp)
    rp.set_defaults(func=_cmd_render_stub("bgm"))

    # status ------------------------------------------------------------
    p_status = sub.add_parser("status", help="Per-stage state + next_action.")
    p_status.add_argument("-s", "--story-id", required=True)
    _add_common_flags(p_status)
    p_status.set_defaults(func=_cmd_status)

    # design / location (one-shots for bible) ---------------------------
    p_design = sub.add_parser("design", help="Create a Character in the bible.")
    p_design.add_argument("input", help="Path to description text, or '-' for stdin.")
    _add_common_flags(p_design)
    p_design.set_defaults(func=_cmd_design)

    p_loc = sub.add_parser("location", help="Create a Location in the bible.")
    p_loc.add_argument("input", help="Path to description text, or '-' for stdin.")
    _add_common_flags(p_loc)
    p_loc.set_defaults(func=_cmd_location)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    _setup_logging(getattr(args, "verbose", False))
    try:
        return args.func(args)
    except CliError as e:
        _emit_error(e)
        return e.exit_code
    except FileNotFoundError as e:
        _emit_error(CliError("FILE_NOT_FOUND", str(e)))
        return 2


# ─── legacy entrypoints (kept for pyproject scripts) ─────────────────────

def design_main(argv: list[str] | None = None) -> int:
    """Backward-compat: `hitchcock-design <input>` → `hitchcock design <input>`."""
    return main(["design"] + (argv or sys.argv[1:]))


def location_main(argv: list[str] | None = None) -> int:
    return main(["location"] + (argv or sys.argv[1:]))


def produce_main(argv: list[str] | None = None) -> int:  # noqa: ARG001
    """Deprecated: replaced by the gate-based script/storyboard/art/render flow.
    Prints a migration hint and exits with code 3."""
    print(
        "hitchcock-produce is deprecated. The pipeline is now gate-based:\n"
        "  hitchcock init       -s <sid> --source <path> --character <id>...\n"
        "  hitchcock script     generate|show|refine|approve -s <sid>\n"
        "  hitchcock storyboard generate|show|refine|approve -s <sid>\n"
        "  hitchcock art        generate|show|refine|pick|approve -s <sid>\n"
        "  hitchcock render     package|post -s <sid>\n"
        "See AGENTS.md for the full contract.",
        file=sys.stderr,
    )
    return 3


if __name__ == "__main__":
    sys.exit(main())
