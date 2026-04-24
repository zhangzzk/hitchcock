from __future__ import annotations

import base64
import io
import logging
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

import httpx
from PIL import Image

from ..config import ArkConfig

log = logging.getLogger(__name__)


class SeedanceError(RuntimeError):
    pass


@dataclass(frozen=True)
class VideoClip:
    """A single generated video clip on disk."""

    path: Path
    duration_sec: int
    prompt_used: str


# Duration values validated against `doubao-seedance-2-0-260128`.
_SUPPORTED_DURATIONS = {5, 10, 12, 15}


class SeedanceClient:
    """Volcengine Ark Seedance 2.0 image-to-video client.

    Submits an async task, polls until the video URL is ready, downloads
    the mp4. First frame and optional last frame are passed as base64
    data URIs inside the `content` array with `role: first_frame` /
    `role: last_frame`.

    Notes on cost:
      - Each submission consumes a non-trivial amount of Ark tokens.
        Keep calls tight and ONLY generate for the final pipeline.
    """

    _SUBMIT_PATH = "/contents/generations/tasks"

    def __init__(self, config: ArkConfig, *, model: str | None = None):
        self._config = config
        self._model = model or config.video_model

    @property
    def model(self) -> str:
        return self._model

    def generate(
        self,
        prompt: str,
        first_frame: Optional[Path] = None,
        *,
        duration_sec: int = 15,
        resolution: str = "720p",
        ratio: str = "16:9",
        last_frame: Optional[Path] = None,
        reference_images: Optional[list[Path]] = None,
        poll_interval_s: float = 12.0,
        poll_timeout_s: int = 1800,
    ) -> VideoClip:
        """Generate a Seedance 2.0 video.

        The Ark API (doubao-seedance-2-0-260128, validated 2026-04-21) has
        TWO mutually-exclusive modes:

          Mode A (I2V): pass `first_frame` (+ optional `last_frame`). No
            `reference_images` allowed. The video starts from that frame.

          Mode B (multi-ref / 多模态参考): pass `reference_images` only.
            No `first_frame`/`last_frame`. Seedance composes the opening
            from prompt + refs. Up to 9 images per request.

        The older `role: subject/environment/motion` taxonomy (documented
        2026-04-19) has been withdrawn — those role values now 400.
        """
        if duration_sec not in _SUPPORTED_DURATIONS:
            raise SeedanceError(
                f"Unsupported duration {duration_sec}s. "
                f"Supported: {sorted(_SUPPORTED_DURATIONS)}."
            )
        refs = reference_images or []
        if refs and (first_frame is not None or last_frame is not None):
            raise SeedanceError(
                "Seedance API mode conflict: `reference_images` cannot be "
                "mixed with `first_frame`/`last_frame`. Pick one mode."
            )

        content: list[dict[str, Any]] = [
            {
                "type": "text",
                "text": (
                    f"{prompt} --resolution {resolution} "
                    f"--duration {duration_sec} --ratio {ratio}"
                ),
            },
        ]

        def _add(role: str, path: Path) -> None:
            content.append({
                "type": "image_url",
                "image_url": {"url": _to_data_uri(path)},
                "role": role,
            })

        if first_frame is not None:
            _add("first_frame", first_frame)
        if last_frame is not None:
            _add("last_frame", last_frame)
        for p in refs:
            _add("reference_image", p)

        # Enforce Seedance's 9-image cap (hard-fail to avoid surprise 400s).
        image_count = sum(1 for c in content if c.get("type") == "image_url")
        if image_count > 9:
            raise SeedanceError(
                f"Too many reference images ({image_count}); "
                f"Seedance caps at 9 images per request."
            )

        task_id = self._submit({"model": self._model, "content": content})
        video_url = self._poll(task_id, interval_s=poll_interval_s, timeout_s=poll_timeout_s)
        tmp = Path(f"/tmp/hitchcock_seedance_{task_id}.mp4")
        self._download(video_url, tmp)
        return VideoClip(path=tmp, duration_sec=duration_sec, prompt_used=prompt)

    # ── HTTP helpers ────────────────────────────────────────────────────
    _MAX_SUBMIT_RETRIES = 3

    def _submit(self, payload: dict[str, Any]) -> str:
        """POST the task to Ark, retry on transient network failures.

        Retry covers httpx transport errors (timeout, connection reset)
        and HTTP 429/5xx — which Ark occasionally returns under load.
        Each retry backs off exponentially (2s, 4s, 8s).
        """
        url = f"{self._config.base_url}{self._SUBMIT_PATH}"
        headers = {
            "Authorization": f"Bearer {self._config.api_key}",
            "Content-Type": "application/json",
        }
        last_err: Exception | None = None
        for attempt in range(self._MAX_SUBMIT_RETRIES + 1):
            try:
                r = httpx.post(
                    url, headers=headers, json=payload,
                    timeout=httpx.Timeout(60.0, write=300.0),
                )
            except httpx.HTTPError as e:
                last_err = e
                if attempt >= self._MAX_SUBMIT_RETRIES:
                    raise SeedanceError(
                        f"Seedance submit transport error (after "
                        f"{attempt} retries): {e}"
                    ) from e
                delay = 2 ** (attempt + 1)
                log.warning(
                    "Seedance submit transport error (%s, attempt %d/%d), "
                    "retrying in %ds",
                    type(e).__name__, attempt + 1, self._MAX_SUBMIT_RETRIES,
                    delay,
                )
                time.sleep(delay)
                continue
            # HTTP status retry (429 + 5xx)
            if r.status_code in {429, 500, 502, 503, 504}:
                last_err = SeedanceError(
                    f"Seedance submit {r.status_code}: {r.text[:200]}"
                )
                if attempt >= self._MAX_SUBMIT_RETRIES:
                    raise last_err
                delay = 2 ** (attempt + 1)
                log.warning(
                    "Seedance submit HTTP %d (attempt %d/%d), retrying in %ds",
                    r.status_code, attempt + 1, self._MAX_SUBMIT_RETRIES, delay,
                )
                time.sleep(delay)
                continue
            if r.status_code != 200:
                raise SeedanceError(f"Seedance submit {r.status_code}: {r.text[:400]}")
            return r.json()["id"]
        raise SeedanceError(f"Seedance submit exhausted retries: {last_err}")

    def _poll(self, task_id: str, *, interval_s: float, timeout_s: int) -> str:
        url = f"{self._config.base_url}{self._SUBMIT_PATH}/{task_id}"
        headers = {"Authorization": f"Bearer {self._config.api_key}"}
        deadline = time.time() + timeout_s
        start = time.time()
        last_status: Optional[str] = None
        polls = 0
        while time.time() < deadline:
            r = httpx.get(url, headers=headers, timeout=30.0)
            if r.status_code != 200:
                raise SeedanceError(f"Seedance poll {r.status_code}: {r.text[:400]}")
            body = r.json()
            status = body.get("status")
            polls += 1
            # Log first poll + every status change + every 5th poll so the
            # user sees progress during the typical 3-8 min wait.
            if status != last_status or polls % 5 == 1:
                elapsed = int(time.time() - start)
                log.info(
                    "Seedance poll: task=%s status=%s elapsed=%ds (poll #%d)",
                    task_id, status, elapsed, polls,
                )
                last_status = status
            if status == "succeeded":
                return body["content"]["video_url"]
            if status == "failed":
                raise SeedanceError(f"Seedance task {task_id} failed: {body}")
            time.sleep(interval_s)
        raise SeedanceError(f"Seedance task {task_id} timed out after {timeout_s}s")

    def _download(self, url: str, dst: Path) -> None:
        with httpx.stream("GET", url, timeout=180.0) as r:
            r.raise_for_status()
            with dst.open("wb") as f:
                for chunk in r.iter_bytes(64 * 1024):
                    f.write(chunk)


def _to_data_uri(path: Path) -> str:
    """Read an image, downscale to 720p-height max, return a base64 data URI."""
    src = Image.open(path).convert("RGB")
    src.thumbnail((1280, 720))
    buf = io.BytesIO()
    src.save(buf, format="PNG")
    b64 = base64.b64encode(buf.getvalue()).decode()
    return f"data:image/png;base64,{b64}"
