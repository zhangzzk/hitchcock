"""OpenAI GPT Image client (default model: gpt-image-2).

Same public interface as `NanoBananaClient.generate(...)` so the two are
interchangeable at the `DesignAgent`/art-gen level — swap the client
instance in the CLI and the rest of the pipeline is unaffected.

- Text-to-image (no refs) → `POST /v1/images/generations` (JSON body)
- Image-conditioned (with refs) → `POST /v1/images/edits` (multipart form)

We auto-route inside `generate(...)` based on whether `reference_images`
is provided. Both endpoints return base64-encoded PNG bytes in
`response.data[0].b64_json`.

Supported sizes (`_SUPPORTED_SIZES`) are the gpt-image-1 set — gpt-image-2
accepts the same sizes and may accept additional larger ones; keep the
conservative list so caller width/height always snaps to something the
server accepts.
"""
from __future__ import annotations

import base64
import io
import logging
import time
from typing import Any

import httpx

from ..config import OpenAIConfig
from .nanobanana import GeneratedImage, ImageGenError

log = logging.getLogger(__name__)


# Allowed sizes (width x height): landscape / square / portrait. These are
# the gpt-image-1 set; gpt-image-2 accepts the same + potentially larger,
# but we snap conservatively so any supported model works.
_SUPPORTED_SIZES: list[tuple[int, int]] = [
    (1024, 1024),
    (1536, 1024),
    (1024, 1536),
]


def _pick_size(width: int, height: int) -> str:
    """Snap a requested (w, h) to the closest supported gpt-image-1 size.
    Matches by aspect ratio first (most visually faithful), size second."""
    target_ratio = width / max(height, 1)

    def score(s: tuple[int, int]) -> tuple[float, int]:
        w, h = s
        ratio_err = abs((w / h) - target_ratio)
        size_err = abs(w * h - width * height)
        return (ratio_err, size_err)

    w, h = min(_SUPPORTED_SIZES, key=score)
    return f"{w}x{h}"


# Retry on transient failures. 429 = rate limit, 5xx = server issues,
# 408/504 = timeouts. 400/401/403 are client errors — don't retry.
_RETRY_STATUS = {408, 429, 500, 502, 503, 504}
_MAX_RETRIES = 3


class GPTImageClient:
    """OpenAI GPT Image generation client (model per `config.image_model`,
    defaults to `gpt-image-2`).

    `quality` controls fidelity vs cost: "high" (best, highest cost),
    "medium" (draft-iteration target), "low", or "auto" (model picks).
    Defaults to config.image_quality, which defaults to
    HITCHCOCK_OPENAI_IMAGE_QUALITY env var or "high"."""

    def __init__(
        self,
        config: OpenAIConfig,
        *,
        model: str | None = None,
        quality: str | None = None,
    ):
        self._config = config
        self._model = model or config.image_model
        self._quality = (quality or config.image_quality or "high").lower()

    @property
    def model(self) -> str:
        return self._model

    @property
    def quality(self) -> str:
        return self._quality

    def generate(
        self,
        prompt: str,
        *,
        width: int = 1536,
        height: int = 1024,
        reference_images: list[bytes] | None = None,
    ) -> GeneratedImage:
        """Generate one image. If `reference_images` is non-empty, use the
        edits endpoint with all refs as conditioning inputs; otherwise
        use plain generations."""
        size = _pick_size(width, height)
        if reference_images:
            return self._generate_edits(prompt, size, reference_images)
        return self._generate_text(prompt, size)

    # ── text-to-image ──────────────────────────────────────────────────
    def _generate_text(self, prompt: str, size: str) -> GeneratedImage:
        url = f"{self._config.base_url}/images/generations"
        headers = {
            "Authorization": f"Bearer {self._config.api_key}",
            "Content-Type": "application/json",
        }
        payload: dict[str, Any] = {
            "model": self._model,
            "prompt": prompt,
            "size": size,
            "n": 1,
            "quality": self._quality,
        }
        body = self._post_with_retry(url, headers=headers, json=payload)
        return _extract_image(body, prompt)

    # ── image-conditioned (reference images) ──────────────────────────
    def _generate_edits(
        self, prompt: str, size: str, refs: list[bytes]
    ) -> GeneratedImage:
        """Use /images/edits with multiple reference images as compositing
        inputs. gpt-image-1 extended edits to accept multiple `image[]`
        entries (April 2025 rollout). Each ref is uploaded as a PNG file."""
        url = f"{self._config.base_url}/images/edits"
        headers = {
            "Authorization": f"Bearer {self._config.api_key}",
            # NOTE: no Content-Type — httpx will set multipart boundary.
        }
        # Form fields (strings) + file parts.
        data = {
            "model": self._model,
            "prompt": prompt,
            "size": size,
            "n": "1",
            "quality": self._quality,
        }
        files = []
        for i, ref in enumerate(refs):
            files.append(("image[]", (f"ref_{i:02d}.png", io.BytesIO(ref), "image/png")))

        body = self._post_with_retry(url, headers=headers, data=data, files=files)
        return _extract_image(body, prompt)

    # ── transport with retry ──────────────────────────────────────────
    def _post_with_retry(
        self,
        url: str,
        *,
        headers: dict[str, str],
        json: dict[str, Any] | None = None,
        data: dict[str, str] | None = None,
        files: list | None = None,
    ) -> dict[str, Any]:
        last_err: Exception | None = None
        for attempt in range(_MAX_RETRIES + 1):
            try:
                resp = httpx.post(
                    url, headers=headers,
                    json=json, data=data, files=files,
                    timeout=180.0,
                )
            except httpx.HTTPError as e:
                last_err = e
                if attempt >= _MAX_RETRIES:
                    raise ImageGenError(
                        f"GPT Image transport error (after {attempt} retries): {e}"
                    ) from e
                delay = 2 ** (attempt + 1)
                log.warning(
                    "GPT Image transport error (%s, attempt %d/%d), retrying in %ds",
                    type(e).__name__, attempt + 1, _MAX_RETRIES, delay,
                )
                time.sleep(delay)
                continue
            if resp.status_code in _RETRY_STATUS:
                last_err = ImageGenError(f"GPT Image {resp.status_code}: {resp.text[:200]}")
                if attempt >= _MAX_RETRIES:
                    raise last_err
                delay = 2 ** (attempt + 1)
                log.warning(
                    "GPT Image HTTP %d (attempt %d/%d), retrying in %ds",
                    resp.status_code, attempt + 1, _MAX_RETRIES, delay,
                )
                time.sleep(delay)
                continue
            if resp.status_code != 200:
                raise ImageGenError(
                    f"GPT Image API {resp.status_code}: {resp.text[:400]}"
                )
            return resp.json()
        raise ImageGenError(f"GPT Image exhausted retries: {last_err}")


def _extract_image(body: dict[str, Any], prompt: str) -> GeneratedImage:
    data_list = body.get("data") or []
    if not data_list:
        raise ImageGenError(f"GPT Image returned no data: {body}")
    b64 = data_list[0].get("b64_json")
    if not b64:
        # Some responses may return a URL instead — handle gracefully.
        url = data_list[0].get("url")
        if url:
            try:
                img_resp = httpx.get(url, timeout=60.0)
                img_resp.raise_for_status()
                return GeneratedImage(data=img_resp.content, prompt_used=prompt)
            except httpx.HTTPError as e:
                raise ImageGenError(f"GPT Image url fetch failed: {e}") from e
        raise ImageGenError(f"GPT Image response missing b64_json and url: {body}")
    return GeneratedImage(data=base64.b64decode(b64), prompt_used=prompt)
