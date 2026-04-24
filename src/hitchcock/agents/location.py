from __future__ import annotations

import logging
import re
from dataclasses import dataclass

from ..bible import BibleStore, Location
from ..image import NanoBananaClient
from ..llm import MimoClient
from .style import load_style_prompt

log = logging.getLogger(__name__)


_STRUCTURE_SYSTEM = """You are a production designer. Given a free-form location
description, produce a structured JSON entry for the Story Bible. Downstream
agents will reuse this location across many scenes, so the description must
be DENSE and CONCRETE — not poetic.

The image model that renders the establishing art takes every word literally,
so write visual facts only. No metaphor, no mood language unless it renders
as a visible thing (e.g. "volumetric mist between the trees" is visible,
"sense of isolation" is not).

Your JSON object must have exactly these keys:
- id: lowercase-kebab-case slug. Chinese words → pinyin. e.g.
  "bugatti interior at night" → "bugatti-interior-night".
- name: human display name.
- description: 80-160 words, concrete visual facts only. Cover in order:
    architecture / enclosure (what walls, roof, surfaces are visible),
    materials (brass, marble, wood, glass, rubber — specific names),
    lighting (key source, fill, rim — naming the warm/cool temperature),
    atmospheric layers (mist, dust, rain — whatever is visible in frame),
    specific dressing / props that should recur across every shot (a
    particular chandelier, a specific dashboard dial, a single red door).
  Do NOT describe any character here — that's the Character sheet's job.
- time_of_day: e.g. "midnight", "dusk", "dawn", "blue hour".
"""


# Style is resolved at call time via load_style_prompt(bible, story_id).


@dataclass
class LocationAgent:
    llm: MimoClient
    images: NanoBananaClient
    bible: BibleStore

    def create(self, description: str, story_id: str | None = None) -> Location:
        log.info("Location Agent: structuring location via MIMO...")
        data = self.llm.chat_json(
            system=_STRUCTURE_SYSTEM,
            user=description.strip(),
            max_tokens=2048,
            temperature=0.4,
        )
        location = Location.model_validate(data)
        location.id = _slugify(location.id or location.name)
        log.info("  → id=%s name=%s", location.id, location.name)
        self._generate_establishing(location, story_id)
        self.bible.save_location(location)
        log.info("Location Agent: saved %s", self.bible.location_json(location.id))
        return location

    def build_from_location(
        self, location: Location, story_id: str | None = None,
    ) -> Location:
        """Build the establishing art for a Location already structured
        (e.g. by CastAgent). Skips MIMO; only runs Nano Banana."""
        self._generate_establishing(location, story_id)
        self.bible.save_location(location)
        return location

    def _generate_establishing(
        self, location: Location, story_id: str | None,
    ) -> None:
        log.info("Location Agent: generating establishing art for %s", location.id)
        style_prompt = load_style_prompt(self.bible, story_id)
        prompt = _build_establishing_prompt(location, style_prompt)
        img = self.images.generate(prompt, width=2688, height=1512)
        out = self.bible.establishing_art(location.id)
        out.parent.mkdir(parents=True, exist_ok=True)
        img.save(out)
        location.establishing_art_path = str(out.relative_to(self.bible.root))


def _build_establishing_prompt(loc: Location, style_prompt: str) -> str:
    return (
        "Wide cinematic establishing shot of the location described below. "
        "The shot shows the place itself — no characters in frame. Eye-level, "
        "rule-of-thirds composition. "
        f"Location: {loc.description} "
        f"Time of day: {loc.time_of_day}. "
        "No people in the frame, no decorative sparkles, no magical particles. "
        f"{style_prompt}"
    )


def _slugify(value: str) -> str:
    s = value.strip().lower()
    s = re.sub(r"[^a-z0-9\-]+", "-", s)
    s = re.sub(r"-+", "-", s).strip("-")
    return s or "location"
