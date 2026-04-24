from __future__ import annotations

import logging
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from ..bible import BibleStore, Character, Story
from ..video import SeedanceClient, SeedanceError, VideoClip
from .storyboard import build_scene_brief

log = logging.getLogger(__name__)


@dataclass
class ShotGenAgent:
    """Scene → single Seedance 15s video clip, anchored by Location art.

    Cost note: ONE Seedance call per scene. Seedance is the expensive
    layer; design keeps it to the minimum.
    """

    seedance: SeedanceClient
    bible: BibleStore

    def generate_scene(self, story: Story, scene_idx: int) -> Optional[Path]:
        scene = story.scenes[scene_idx]

        # Anchor: per-scene composed art preferred; falls back to location.
        anchor = self._anchor_frame(story, scene)
        if anchor is None:
            log.warning("Scene %s has no anchor frame; skipping", scene.id)
            return None

        char_cache: dict[str, Character] = {
            cid: self.bible.load_character(cid) for cid in story.characters
        }
        location_desc = None
        # Ark Seedance 2.0 (2026-04-21 API) uses a single `reference_images`
        # pool — cannot mix first_frame with refs. We feed the composed
        # anchor + char fronts + location establishing art as refs and let
        # Seedance compose the opening from the prompt.
        reference_images: list[Path] = [anchor]
        if scene.location_id:
            try:
                loc = self.bible.load_location(scene.location_id)
                location_desc = loc.description
                est = self.bible.establishing_art(loc.id)
                if est.exists() and est != anchor:
                    reference_images.append(est)
            except FileNotFoundError:
                pass
        for cid in scene.characters_in_scene:
            front = self.bible.refs_dir(cid) / "front.png"
            if front.exists():
                reference_images.append(front)

        brief = build_scene_brief(story, scene, char_cache, location_desc)
        duration = min(15, max(5, int(sum(s.duration_sec for s in scene.shots))))
        if duration not in {5, 10, 12, 15}:
            duration = 15

        log.info(
            "ShotGen: scene %s (%s), duration %ds, refs=%d (anchor=%s)",
            scene.id, scene.title, duration,
            len(reference_images), anchor.name,
        )
        log.debug("--- brief ---\n%s\n--- /brief ---", brief)

        try:
            clip = self.seedance.generate(
                brief,
                reference_images=reference_images,
                duration_sec=duration,
            )
        except SeedanceError as e:
            log.error("ShotGen: Seedance failed for scene %s: %s", scene.id, e)
            return None

        scene_clips_dir = self.bible.story_dir(story.id) / "clips"
        scene_clips_dir.mkdir(parents=True, exist_ok=True)
        dst = scene_clips_dir / f"{scene.id}.mp4"
        # shutil.move handles cross-filesystem (e.g. /tmp → project dir)
        shutil.move(str(clip.path), str(dst))
        log.info("ShotGen: saved %s (%d KB)", dst, dst.stat().st_size // 1024)
        return dst

    def generate_all(self, story: Story) -> list[Path]:
        paths: list[Path] = []
        for i, scene in enumerate(story.scenes):
            p = self.generate_scene(story, i)
            if p is not None:
                paths.append(p)
        return paths

    # ── anchor frame resolution ─────────────────────────────────────────
    def _anchor_frame(self, story: Story, scene) -> Optional[Path]:
        # Preferred: per-scene composed art (Location + Characters in Arcane).
        # Without this, Seedance renders characters from text alone and
        # drifts to anime. See feedback_seedance_limits.md.
        scene_art = self.bible.scene_art_path(story.id, scene.id)
        if scene_art.exists():
            return scene_art
        # Next: location establishing art (people-free)
        if scene.location_id:
            p = self.bible.establishing_art(scene.location_id)
            if p.exists():
                return p
        # Last resort: any per-shot keyframe
        for shot in scene.shots:
            if shot.keyframe_path:
                p = self.bible.root / shot.keyframe_path
                if p.exists():
                    return p
        return None
