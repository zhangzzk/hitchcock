from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from ..bible import BibleStore, Character, Location, Scene, Story
from ..image import NanoBananaClient
from .style import load_style_prompt

log = logging.getLogger(__name__)


# Style resolved per-story at call time via load_style_prompt().
# style_prompt constant removed — replaced by StyleGuide.


@dataclass
class SceneArtAgent:
    """Compose one scene's canonical keyframe by fusing the Location
    establishing art with the Character reference images.

    This image becomes Seedance's `first_frame` — it anchors not just the
    environment (which the Location art already did) but also the
    characters in the correct Arcane style. Without this step, Seedance
    renders characters from text alone and drifts back toward its anime
    default bias.
    """

    images: NanoBananaClient
    bible: BibleStore

    def compose(
        self,
        story: Story,
        scene: Scene,
        char_cache: Optional[dict[str, Character]] = None,
    ) -> Optional[Path]:
        if not scene.location_id:
            log.warning(
                "scene %s has no location_id; SceneArtAgent cannot compose", scene.id
            )
            return None
        try:
            location = self.bible.load_location(scene.location_id)
        except FileNotFoundError:
            log.warning("location %s not in bible", scene.location_id)
            return None

        loc_ref = self.bible.establishing_art(location.id)
        if not loc_ref.exists():
            log.warning("location %s has no establishing art", location.id)
            return None

        char_cache = char_cache or {
            cid: self.bible.load_character(cid) for cid in scene.characters_in_scene
        }

        # Reference images: location first (gives the environment/style),
        # then each character's canonical front view (gives the face/outfit).
        refs: list[bytes] = [loc_ref.read_bytes()]
        for cid in scene.characters_in_scene:
            front = self.bible.refs_dir(cid) / "front.png"
            if front.exists():
                refs.append(front.read_bytes())

        style_prompt = load_style_prompt(self.bible, story.id)
        prompt = _build_prompt(location, scene, char_cache, style_prompt)
        log.info(
            "SceneArt: composing scene %s (location=%s, refs=%d)",
            scene.id, location.id, len(refs),
        )
        img = self.images.generate(
            prompt, width=2688, height=1512, reference_images=refs,
        )
        out = self.bible.scene_art_path(story.id, scene.id)
        out.parent.mkdir(parents=True, exist_ok=True)
        img.save(out)
        log.info("SceneArt: saved %s", out)
        return out


def _build_prompt(
    location: Location, scene: Scene, char_cache: dict[str, Character],
    style_prompt: str,
) -> str:
    char_lines: list[str] = []
    for cid in scene.characters_in_scene:
        c = char_cache.get(cid)
        if c is None:
            continue
        char_lines.append(
            f"- {c.name}: {_subject_kind(c)}. {c.visual_description} "
            f"Wearing: {c.default_outfit}"
        )
    char_block = "\n".join(char_lines) or "— no characters in frame"
    n_chars = len(char_lines)

    # This frame is a CINEMATIC FILM STILL, not a character sheet.
    # Earlier versions forced "medium 2-shot, every face clearly visible"
    # and produced stiff posed compositions with no artistic voice. The
    # Seedance anchor only needs characters to be RECOGNIZABLE from the
    # references — a half face, a silhouette, an OTS framing all work.
    composition = (
        "This frame is a CINEMATIC FILM STILL — compose it with the boldness "
        "of a feature animation production designer. NOT a character sheet, "
        "NOT a posed portrait.\n"
        "\n"
        "Composition rules:\n"
        "- Characters must be RECOGNIZABLE from the reference images (face "
        "shape, hair, clothing carry over exactly) — but they do NOT have "
        "to be fully visible or centered. Half-face, profile, silhouette, "
        "cropped-at-shoulder, seen-through-window, over-the-shoulder all "
        "work if they serve the emotional register.\n"
        "- Use cinematic asymmetry — rule of thirds or more extreme; let "
        "negative space and empty architecture carry weight.\n"
        "- Let chiaroscuro light do the work. Warm source against cold "
        "environment = sculpting contrast. One character catching the light "
        "while another is a dark silhouette is often MORE powerful than "
        "both lit evenly.\n"
        "- Match the scene's emotional_register:\n"
        "    quiet / tender / intimate → held stillness; one warm light; "
        "much dark space; two figures may be near-silhouettes against a "
        "landscape\n"
        "    operatic / awe → dynamic angle; one figure caught in light in "
        "an otherwise shadowed frame\n"
        "    chaos / turmoil → unbalanced diagonal tension, implied motion\n"
        "    humiliated / restrained → hostile architecture around a small "
        "figure, hard warm-vs-cold divide\n"
        "    forward motion / solitude → bold landscape with the vehicle/"
        "subject as a tiny warm pinprick in vast cold space is GOOD\n"
        "- Characters at 20% of frame height is usually enough anchoring. "
        "Do not force them bigger just to be 'safe'.\n"
        "- Avoid centered medium 2-shots (the 'driver's license photo' "
        "failure mode) unless the scene's emotional register genuinely "
        "calls for symmetrical stillness.\n"
        "- No posed staring-at-camera body language. Catch them mid-breath, "
        "mid-thought, mid-action, off-center, caught in their own light.\n"
        "\n"
        "Think: this frame is the single image a viewer will see on a "
        "poster. It must be BEAUTIFUL first, informative second."
    ) if n_chars >= 1 else (
        "Composition: a cinematic establishing view of the location itself; "
        "no people in the frame. Use bold, painterly framing — don't make "
        "it a flat survey shot. Let the light and atmosphere be the subject."
    )

    return (
        "Compose a single cinematic FILM STILL that will serve as the first "
        "frame of a 15-second animated video. This is the single most "
        "important visual anchor for the whole scene — but it is a piece of "
        "ART first, a reference anchor second. Aim for the artistic ceiling "
        "of a festival-quality animated feature, not a storyboard panel.\n\n"
        "Use the attached reference images as the canonical visual source:\n"
        "  - First image: the LOCATION canvas (environment, architecture, "
        "light, palette). Match its style and lighting exactly.\n"
        "  - Subsequent images: each CHARACTER's canonical appearance. "
        "Match their faces, hair, and clothing EXACTLY — do not reinvent "
        "their features, carry them over verbatim from the references.\n\n"
        # ── character layer ──────────────────────────────────────────
        "CHARACTER DIMENSIONS — compose for all of these simultaneously:\n"
        "  ACTION: characters caught mid-gesture / mid-thought / mid-breath. "
        "Specific body language (hand on wheel, hair pushed behind ear, a "
        "half-turn, weight on one leg). Never posed, never staring at camera.\n"
        "  EXPRESSION: subtle, specific, earned by the scene's emotional "
        "register. A flicker of worry, a suppressed smile, a faraway gaze. "
        "No generic idealized beauty expressions.\n"
        "  INTERACTION: two or more characters in frame should have a "
        "VISIBLE spatial relationship — eyeline direction, physical "
        "distance, who leans toward whom, who doesn't. Their geometry in "
        "the frame says what they are to each other right now.\n\n"
        # ── camera + composition layer ───────────────────────────────
        f"{composition}\n\n"
        "  CAMERA PLACEMENT is itself a directorial choice. Default eye-"
        "level frontal is the failure mode. Consider: low angle looking up, "
        "high angle looking down, through-foreground (a banister / hair / "
        "glass), shoulder-height behind one character, a windshield view, "
        "a reflection on water. Pick the angle that MEANS something about "
        "the scene's emotion.\n\n"
        # ── environment layer ────────────────────────────────────────
        "ENVIRONMENT AESTHETICS: the location is not a backdrop, it is a "
        "CO-SUBJECT. Let architecture, light, materials, and atmospheric "
        "depth carry as much of the frame as the characters. A single warm "
        "lamp in a vast cold room, the geometry of a road vanishing into "
        "fog, a water surface reflecting a distant light — these carry "
        "meaning. Paint materials with weight: cloth, leather, wet stone, "
        "hair, breath on glass.\n\n"
        f"Scene: {scene.title}. Time of day: {scene.time_of_day}. "
        f"Register: {scene.emotional_register or '—'}. "
        f"Summary: {scene.summary or '—'}\n\n"
        f"Location reference (first image): {location.description}\n\n"
        f"Characters in this scene:\n{char_block}\n\n"
        f"{style_prompt}"
    )


def _subject_kind(c: Character) -> str:
    gender = (c.gender or "").strip().lower()
    noun = {
        "male": "young man", "m": "young man", "man": "young man", "boy": "young man",
        "female": "young woman", "f": "young woman", "woman": "young woman", "girl": "young woman",
    }.get(gender, "young person")
    age = (c.age or "").strip()
    return f"A {noun} ({age} years old)" if age else f"A {noun}"
