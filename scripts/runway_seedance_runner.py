"""Run a single story's 5 scenes through Runway's Seedance 2 I2V.

Reads the hand-edited package prompts (render/packages/<scene>/prompt.txt)
verbatim as prompt_text. Uploads keyframe + character refs as pure
reference images (no position) — this matches the @image1..N bindings in
the user's prompt. Downloads each clip to render/clips/<scene>.mp4.
"""
from __future__ import annotations

import json
import os
import re
import sys
import time
import urllib.request
from pathlib import Path

from dotenv import load_dotenv
import runwayml

HITCHCOCK_ROOT = Path(__file__).resolve().parent.parent
load_dotenv(HITCHCOCK_ROOT / ".env")

STORY_ID = "dragon-raja-v2-epilogue"
BIBLE = HITCHCOCK_ROOT / "bible"
STORY_DIR = BIBLE / "stories" / STORY_ID
PACKAGES = STORY_DIR / "render" / "packages"
CLIPS_OUT = STORY_DIR / "render" / "clips"
CLIPS_OUT.mkdir(parents=True, exist_ok=True)


def runway_client() -> runwayml.RunwayML:
    key = os.getenv("HITCHCOCK_RUNWAY_API_KEY") or os.getenv("RUNWAYML_API_SECRET")
    if not key:
        sys.exit("HITCHCOCK_RUNWAY_API_KEY not set in .env")
    os.environ["RUNWAYML_API_SECRET"] = key
    return runwayml.RunwayML()


def duration_from_prompt(prompt_text: str) -> int:
    # header line is e.g. "15s · 暮色中的旧楼" or "10s · 最后的躺下"
    m = re.search(r"^(\d+)s\s*·", prompt_text, flags=re.MULTILINE)
    if not m:
        return 15
    n = int(m.group(1))
    # Seedance 2 via Runway accepts 5/10/15 (we've seen on Ark). Snap to nearest.
    return n if n in {5, 10, 15} else 15


def refs_for_scene(scene_id: str) -> list[Path]:
    """Match the @image1..N order the user's prompt expects.

    Jimeng package layout:
      01_shot_sh01.png           ← @image1 (keyframe)
      02_<char>_ref.png          ← @image2
      03_<char>_ref.png          ← @image3 (if present)
    """
    pkg_dir = PACKAGES / scene_id
    ordered: list[Path] = []
    for p in sorted(pkg_dir.glob("*.png")):
        ordered.append(p)
    return ordered


def upload_refs(client: runwayml.RunwayML, paths: list[Path]) -> list[str]:
    uris: list[str] = []
    for p in paths:
        with p.open("rb") as f:
            resp = client.uploads.create_ephemeral(file=f)
        print(f"  uploaded {p.name} → {resp.uri}")
        uris.append(resp.uri)
    return uris


def run_scene(client: runwayml.RunwayML, scene_id: str) -> dict:
    prompt_path = PACKAGES / scene_id / "prompt.txt"
    prompt_text = prompt_path.read_text(encoding="utf-8")
    ref_paths = refs_for_scene(scene_id)
    duration = duration_from_prompt(prompt_text)

    print(f"\n=== {scene_id} ===")
    print(f"  prompt: {len(prompt_text)} chars")
    print(f"  refs:   {len(ref_paths)} images")
    print(f"  dur:    {duration}s")
    print(f"  ratio:  864:496")

    uris = upload_refs(client, ref_paths)
    # Reference mode: all images no position. Cannot mix with first/last.
    prompt_image = [{"uri": u} for u in uris]

    task = client.image_to_video.create(
        model="seedance2",
        prompt_image=prompt_image,
        prompt_text=prompt_text,
        duration=duration,
        ratio="864:496",
        audio=True,
    )
    task_id = task.id
    print(f"  task_id: {task_id}")

    # Poll every 10s
    t0 = time.time()
    while True:
        status = client.tasks.retrieve(task_id)
        elapsed = int(time.time() - t0)
        print(f"  poll {elapsed:>4}s: {status.status}")
        if status.status in ("SUCCEEDED", "FAILED", "CANCELLED"):
            break
        if elapsed > 1800:
            raise TimeoutError(f"{scene_id} exceeded 30-min budget")
        time.sleep(10)

    if status.status != "SUCCEEDED":
        raise RuntimeError(f"{scene_id} {status.status}: {status.failure_code or status.failure}")

    url = status.output[0]
    out = CLIPS_OUT / f"{scene_id}.mp4"
    urllib.request.urlretrieve(url, out)
    size_kb = out.stat().st_size // 1024
    print(f"  saved:   {out} ({size_kb} KB)")
    return {"scene_id": scene_id, "task_id": task_id, "output_url": url,
            "clip_path": str(out), "size_kb": size_kb, "duration_sec": duration,
            "prompt_chars": len(prompt_text), "ref_count": len(ref_paths)}


def main(scene_ids: list[str]) -> None:
    client = runway_client()
    results = []
    for sid in scene_ids:
        results.append(run_scene(client, sid))
    print("\n=== SUMMARY ===")
    print(json.dumps(results, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    ids = sys.argv[1:] or ["s01", "s02", "s03", "s04", "s05"]
    main(ids)
