from __future__ import annotations

import base64
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import httpx

from ..config import GeminiConfig


class ImageGenError(RuntimeError):
    pass


@dataclass(frozen=True)
class GeneratedImage:
    data: bytes
    prompt_used: str

    def save(self, path: str | Path) -> Path:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(self.data)
        return p


# Google's Nano Banana / Gemini image models accept aspect ratios, not exact
# pixel sizes. Callers still pass width/height for clarity — we map to the
# closest supported ratio and let the model pick the actual resolution.
_ASPECT_RATIOS: dict[str, float] = {
    "1:1": 1.0,
    "2:3": 2 / 3,
    "3:2": 3 / 2,
    "3:4": 3 / 4,
    "4:3": 4 / 3,
    "4:5": 4 / 5,
    "5:4": 5 / 4,
    "9:16": 9 / 16,
    "16:9": 16 / 9,
    "21:9": 21 / 9,
}


def _pick_aspect(width: int, height: int) -> str:
    target = width / height
    return min(_ASPECT_RATIOS, key=lambda k: abs(_ASPECT_RATIOS[k] - target))


class NanoBananaClient:
    """Google Gemini / Nano Banana image generation client.

    Defaults to `nano-banana-pro-preview` — the painterly high-quality
    preview. For faster/cheaper calls switch to `gemini-2.5-flash-image`.
    """

    _BASE = "https://generativelanguage.googleapis.com/v1beta/models"

    def __init__(self, config: GeminiConfig, *, model: str | None = None):
        self._config = config
        self._model = model or config.image_model

    @property
    def model(self) -> str:
        return self._model

    def generate(
        self,
        prompt: str,
        *,
        width: int = 2304,
        height: int = 1728,
        reference_images: list[bytes] | None = None,
    ) -> GeneratedImage:
        aspect = _pick_aspect(width, height)
        parts: list[dict[str, Any]] = [{"text": prompt}]
        for ref in reference_images or []:
            parts.append({
                "inline_data": {
                    "mime_type": "image/png",
                    "data": base64.b64encode(ref).decode("ascii"),
                }
            })

        payload = {
            "contents": [{"parts": parts}],
            "generationConfig": {
                "responseModalities": ["TEXT", "IMAGE"],
                "imageConfig": {"aspectRatio": aspect},
            },
        }
        # API key in Authorization header (x-goog-api-key), NOT URL query
        # param — key in URL leaks to proxy/CDN/access logs and ends up in
        # shell history, stack traces, curl reproducers.
        url = f"{self._BASE}/{self._model}:generateContent"
        headers = {
            "x-goog-api-key": self._config.api_key,
            "Content-Type": "application/json",
        }
        try:
            resp = httpx.post(url, headers=headers, json=payload, timeout=120.0)
        except httpx.HTTPError as e:
            raise ImageGenError(f"Nano Banana transport error: {e}") from e

        if resp.status_code != 200:
            raise ImageGenError(
                f"Nano Banana API {resp.status_code}: {resp.text[:400]}"
            )

        raw_bytes = _extract_image(resp.json())
        if raw_bytes is None:
            raise ImageGenError(
                f"Nano Banana returned no image. Response: {resp.text[:400]}"
            )
        return GeneratedImage(data=raw_bytes, prompt_used=prompt)


def _extract_image(body: dict[str, Any]) -> bytes | None:
    candidates = body.get("candidates") or []
    if not candidates:
        return None
    for p in candidates[0].get("content", {}).get("parts", []):
        data = p.get("inlineData") or p.get("inline_data")
        if data and data.get("data"):
            return base64.b64decode(data["data"])
    return None
