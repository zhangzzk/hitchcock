"""TTS Agent — Phase 2a. Generates voiceover audio for every dialogue line
in the approved script using edge-tts (Microsoft Edge's free neural TTS).

Output layout:
    bible/stories/<sid>/tts/
        <scene_id>/
            line_01.mp3      ← one MP3 per dialogue line
            line_02.mp3
            ...
            manifest.json    ← ordered list of line_id + speaker_id + text + duration_s

Character voice selection priority:
    1. Character.voice_id (if set in bible)
    2. Default by gender + age (female young → Xiaoxiao, male young → Yunxi,
       child male → Yunxia)

Consumed downstream by `hitchcock render post` to mix VO onto video clips.
"""
from __future__ import annotations

import asyncio
import json
import logging
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from ..bible import BibleStore, Character, Scene, Story

log = logging.getLogger(__name__)


# ── Voice selection ─────────────────────────────────────────────────────

_DEFAULT_VOICES = {
    "female_adult": "zh-CN-XiaoxiaoNeural",      # young female, expressive
    "female_warm": "zh-CN-XiaochenNeural",       # warmer female
    "female_child": "zh-CN-XiaoyiNeural",        # younger female
    "male_adult": "zh-CN-YunxiNeural",           # young male
    "male_mature": "zh-CN-YunyangNeural",        # mature male (news-style)
    "male_child": "zh-CN-YunxiaNeural",          # child male voice
    "neutral": "zh-CN-XiaoxiaoNeural",
}


def _pick_default_voice(c: Character) -> str:
    g = (c.gender or "").lower()
    age = (c.age or "").lower()
    is_child = any(k in age for k in ("12", "13", "14", "child", "junior"))
    if g in ("male", "m", "man", "boy"):
        return _DEFAULT_VOICES["male_child"] if is_child else _DEFAULT_VOICES["male_adult"]
    if g in ("female", "f", "woman", "girl"):
        return _DEFAULT_VOICES["female_adult"]
    return _DEFAULT_VOICES["neutral"]


# ── Audio helpers ──────────────────────────────────────────────────────

def _probe_duration(path: Path) -> float:
    """Return audio duration in seconds via ffprobe (0.0 on failure,
    with a WARNING — silent 0.0s duration can quietly desync TTS tracks)."""
    try:
        r = subprocess.run(
            [
                "ffprobe", "-v", "error", "-show_entries", "format=duration",
                "-of", "default=noprint_wrappers=1:nokey=1", str(path),
            ],
            capture_output=True, text=True, timeout=10,
        )
        return float(r.stdout.strip() or 0.0)
    except (subprocess.SubprocessError, OSError, ValueError) as e:
        log.warning(
            "tts: ffprobe failed on %s (%s: %s) — returning 0.0s duration",
            path, type(e).__name__, e,
        )
        return 0.0


async def _synth_one(voice: str, text: str, out: Path, rate: str = "+0%") -> None:
    import edge_tts  # local import to keep top-of-module fast
    comm = edge_tts.Communicate(text, voice=voice, rate=rate)
    out.parent.mkdir(parents=True, exist_ok=True)
    await comm.save(str(out))


# ── Agent ───────────────────────────────────────────────────────────────

@dataclass
class TTSAgent:
    """Generates per-line MP3s + manifest. No LLM; pure edge-tts + ffprobe."""

    bible: BibleStore

    def generate(
        self,
        story: Story,
        scene_ids: Optional[list[str]] = None,
        rate: str = "+0%",
    ) -> dict:
        """Run TTS for all (or selected) scenes. Returns a per-scene summary."""
        char_cache: dict[str, Character] = {
            cid: self.bible.load_character(cid) for cid in story.characters
        }
        scenes = [s for s in story.scenes if not scene_ids or s.id in scene_ids]
        summary: dict = {"story_id": story.id, "scenes": []}
        for scene in scenes:
            scene_sum = self._do_scene(story, scene, char_cache, rate)
            summary["scenes"].append(scene_sum)
        return summary

    def _do_scene(
        self,
        story: Story,
        scene: Scene,
        char_cache: dict[str, Character],
        rate: str,
    ) -> dict:
        scene_dir = self.bible.story_dir(story.id) / "tts" / scene.id
        scene_dir.mkdir(parents=True, exist_ok=True)
        manifest = {"scene_id": scene.id, "lines": []}

        if not scene.dialogue:
            log.info("TTS: %s has no dialogue, writing empty manifest", scene.id)
            (scene_dir / "manifest.json").write_text(
                json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
            )
            return manifest

        loop = asyncio.new_event_loop()
        try:
            for idx, d in enumerate(scene.dialogue, start=1):
                c = char_cache.get(d.speaker_id)
                voice = (c.voice_id if c and c.voice_id else _pick_default_voice(c)) \
                    if c else _DEFAULT_VOICES["neutral"]
                out = scene_dir / f"line_{idx:02d}.mp3"
                log.info("TTS: %s line %02d (%s) via %s — %s",
                         scene.id, idx, d.speaker_id, voice, d.text[:40])
                loop.run_until_complete(_synth_one(voice, d.text, out, rate))
                dur = _probe_duration(out)
                manifest["lines"].append({
                    "idx": idx,
                    "speaker_id": d.speaker_id,
                    "voice": voice,
                    "text": d.text,
                    "invented": getattr(d, "invented", False),
                    "path": str(out.relative_to(self.bible.root)),
                    "duration_s": round(dur, 3),
                })
        finally:
            loop.close()

        (scene_dir / "manifest.json").write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        total = round(sum(line["duration_s"] for line in manifest["lines"]), 3)
        log.info("TTS: %s done — %d lines, %.1fs total audio",
                 scene.id, len(manifest["lines"]), total)
        return manifest
