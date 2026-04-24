from __future__ import annotations

import base64
from dataclasses import dataclass
from pathlib import Path

import httpx
from openai import OpenAI

from ..config import ArkConfig


class ImageGenError(RuntimeError):
    pass


@dataclass(frozen=True)
class GeneratedImage:
    """A single image returned by the Ark image API — stored as raw PNG bytes."""

    data: bytes
    prompt_used: str

    def save(self, path: str | Path) -> Path:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(self.data)
        return p


class ArkImageClient:
    """Volcengine Ark image generation (OpenAI-compatible /images/generations).

    Default model is Seedream 4.5, which requires each image to have ≥ ~3.69M
    pixels (i.e. roughly 2K resolution or larger). Seedream 4.0 accepts smaller
    sizes like 1024x1024 — switch via `HITCHCOCK_ARK_IMAGE_MODEL`.
    """

    def __init__(self, config: ArkConfig, *, model: str | None = None):
        self._config = config
        self._model = model or config.image_model
        self._client = OpenAI(api_key=config.api_key, base_url=config.base_url)

    @property
    def model(self) -> str:
        return self._model

    def generate(
        self,
        prompt: str,
        *,
        width: int = 2048,
        height: int = 2048,
    ) -> GeneratedImage:
        try:
            resp = self._client.images.generate(
                model=self._model,
                prompt=prompt,
                size=f"{width}x{height}",
                response_format="b64_json",
            )
        except Exception as e:
            raise ImageGenError(
                f"Ark image gen failed (model={self._model} size={width}x{height}): {e}"
            ) from e

        item = resp.data[0]
        raw = _decode_image(item)
        return GeneratedImage(data=raw, prompt_used=prompt)


def _decode_image(item) -> bytes:
    b64 = getattr(item, "b64_json", None)
    if b64:
        return base64.b64decode(b64)
    url = getattr(item, "url", None)
    if url:
        r = httpx.get(url, timeout=60.0)
        r.raise_for_status()
        return r.content
    raise ImageGenError(f"Ark response had neither b64_json nor url: {item!r}")
