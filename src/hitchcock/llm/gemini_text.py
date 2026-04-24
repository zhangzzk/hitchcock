"""Gemini text client with optional Google Search grounding.

Used by BriefAgent.research_canon() to automate the canon-research
step without relying on an external AI driver running WebSearch.

Gemini 2.x natively supports `google_search` tool — the model calls
Google Search under the hood and returns grounded text + citation
metadata. We only need the text + sources URLs.
"""
from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from typing import Any

import httpx

from ..config import GeminiConfig

log = logging.getLogger(__name__)


class GeminiTextError(RuntimeError):
    pass


@dataclass(frozen=True)
class GroundedAnswer:
    text: str
    sources: list[str]  # URLs cited by the grounding metadata


# Transient errors worth retrying. Gemini API occasionally 429s under
# burst, or 5xx during region-level hiccups.
_RETRY_STATUS = {429, 500, 502, 503, 504}
_MAX_RETRIES = 3


class GeminiTextClient:
    """Thin wrapper for Gemini generateContent with optional grounding.

    Uses API key in `x-goog-api-key` header (not URL query) so keys
    don't leak to proxy/CDN/shell history. Retries transient failures
    with exponential backoff.
    """

    _BASE = "https://generativelanguage.googleapis.com/v1beta/models"

    def __init__(self, config: GeminiConfig, *, model: str | None = None):
        self._config = config
        self._model = model or config.text_model

    @property
    def model(self) -> str:
        return self._model

    def generate_grounded(
        self,
        prompt: str,
        *,
        max_output_tokens: int = 2048,
        temperature: float = 0.2,
    ) -> GroundedAnswer:
        """Ask Gemini a question with Google Search grounding enabled.
        Returns the answer text plus the list of sources Gemini cited."""
        url = f"{self._BASE}/{self._model}:generateContent"
        headers = {
            "x-goog-api-key": self._config.api_key,
            "Content-Type": "application/json",
        }
        payload: dict[str, Any] = {
            "contents": [{"parts": [{"text": prompt}]}],
            "tools": [{"google_search": {}}],
            "generationConfig": {
                "maxOutputTokens": max_output_tokens,
                "temperature": temperature,
            },
        }

        last_err: Exception | None = None
        for attempt in range(_MAX_RETRIES + 1):
            try:
                resp = httpx.post(url, headers=headers, json=payload, timeout=120.0)
            except httpx.HTTPError as e:
                last_err = e
                if attempt >= _MAX_RETRIES:
                    raise GeminiTextError(
                        f"Gemini transport error (after {attempt} retries): {e}"
                    ) from e
                delay = 2 ** (attempt + 1)
                log.warning(
                    "Gemini transport error (%s, attempt %d/%d), retrying in %ds",
                    type(e).__name__, attempt + 1, _MAX_RETRIES, delay,
                )
                time.sleep(delay)
                continue
            if resp.status_code in _RETRY_STATUS:
                last_err = GeminiTextError(
                    f"Gemini {resp.status_code}: {resp.text[:200]}"
                )
                if attempt >= _MAX_RETRIES:
                    raise last_err
                delay = 2 ** (attempt + 1)
                log.warning(
                    "Gemini HTTP %d (attempt %d/%d), retrying in %ds",
                    resp.status_code, attempt + 1, _MAX_RETRIES, delay,
                )
                time.sleep(delay)
                continue
            if resp.status_code != 200:
                raise GeminiTextError(f"Gemini {resp.status_code}: {resp.text[:400]}")
            return _extract_grounded(resp.json())
        raise GeminiTextError(f"Gemini exhausted retries: {last_err}")


def _extract_grounded(body: dict[str, Any]) -> GroundedAnswer:
    """Pull the answer text + cited source URLs out of Gemini's response."""
    candidates = body.get("candidates") or []
    if not candidates:
        return GroundedAnswer(text="", sources=[])
    cand = candidates[0]
    # Text: concatenate all text parts.
    text_parts: list[str] = []
    for p in cand.get("content", {}).get("parts", []):
        t = p.get("text")
        if t:
            text_parts.append(t)
    text = "\n".join(text_parts).strip()
    # Sources: groundingMetadata.groundingChunks[*].web.uri
    sources: list[str] = []
    gm = cand.get("groundingMetadata") or {}
    for chunk in gm.get("groundingChunks", []) or []:
        web = chunk.get("web") or {}
        uri = web.get("uri")
        if uri:
            sources.append(uri)
    # Deduplicate preserving order.
    seen: set[str] = set()
    dedup_sources = []
    for s in sources:
        if s not in seen:
            seen.add(s)
            dedup_sources.append(s)
    return GroundedAnswer(text=text, sources=dedup_sources)
