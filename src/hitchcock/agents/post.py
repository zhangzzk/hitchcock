from __future__ import annotations

import logging
import subprocess
from dataclasses import dataclass
from pathlib import Path

from ..bible import BibleStore, Story

log = logging.getLogger(__name__)


@dataclass
class PostAgent:
    """Concatenate per-scene Seedance clips into a reel. No re-encoding.

    Audio layering / TTS dubbing is a future extension — current Seedance
    clips carry their own ambient audio, which passes through the concat.
    """

    bible: BibleStore

    def make_reel(self, story: Story) -> Path:
        clips_dir = self.bible.story_dir(story.id) / "clips"
        if not clips_dir.exists():
            raise FileNotFoundError(f"no clips dir: {clips_dir}")

        # Ordered by scene id
        clips = [clips_dir / f"{s.id}.mp4" for s in story.scenes]
        clips = [c for c in clips if c.exists()]
        if not clips:
            raise FileNotFoundError(f"no clips found in {clips_dir}")

        concat_txt = clips_dir / "concat.txt"
        concat_txt.write_text(
            "\n".join(f"file '{c.name}'" for c in clips),
            encoding="utf-8",
        )

        reel = self.bible.story_dir(story.id) / "reel.mp4"
        subprocess.run(
            ["ffmpeg", "-y", "-f", "concat", "-safe", "0",
             "-i", str(concat_txt), "-c", "copy", str(reel)],
            check=True, capture_output=True,
        )
        log.info("PostAgent: reel saved %s (%d KB, %d clips)",
                 reel, reel.stat().st_size // 1024, len(clips))
        return reel
