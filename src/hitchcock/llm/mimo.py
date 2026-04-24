from __future__ import annotations

import json
import logging
import time
from typing import Any

from openai import APIError, APIStatusError, APITimeoutError, OpenAI

from ..config import MimoConfig


log = logging.getLogger(__name__)


# Retry on transient MIMO failures. MIMO's 小米 infra sometimes returns
# 451 "cross-border isolation policy" mid-run even when the same prompt
# succeeds seconds later (observed 2026-04-21 on a 5-scene storyboard
# generate). Rate limits (429) and server errors (5xx) are also worth
# retrying. We back off exponentially; the user sees each retry in logs.
_RETRY_STATUS_CODES = {429, 451, 500, 502, 503, 504}
_MAX_RETRIES = 3


class MimoClient:
    """Thin wrapper around MIMO's OpenAI-compatible chat completions."""

    def __init__(self, config: MimoConfig):
        self._config = config
        self._client = OpenAI(api_key=config.api_key, base_url=config.base_url)

    def chat(
        self,
        *,
        system: str,
        user: str,
        max_tokens: int = 2048,
        temperature: float = 0.7,
    ) -> str:
        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ]
        last_err: Exception | None = None
        for attempt in range(_MAX_RETRIES + 1):
            try:
                resp = self._client.chat.completions.create(
                    model=self._config.model,
                    messages=messages,
                    max_tokens=max_tokens,
                    temperature=temperature,
                )
                content = resp.choices[0].message.content or ""
                return content.strip()
            except APIStatusError as e:
                last_err = e
                if e.status_code not in _RETRY_STATUS_CODES or attempt >= _MAX_RETRIES:
                    raise
                delay = 2 ** attempt  # 1s, 2s, 4s
                log.warning(
                    "MIMO %d error (attempt %d/%d), retrying in %ds",
                    e.status_code, attempt + 1, _MAX_RETRIES, delay,
                )
                time.sleep(delay)
            except (APITimeoutError, APIError) as e:
                last_err = e
                if attempt >= _MAX_RETRIES:
                    raise
                delay = 2 ** attempt
                log.warning(
                    "MIMO transport error (%s, attempt %d/%d), retrying in %ds",
                    type(e).__name__, attempt + 1, _MAX_RETRIES, delay,
                )
                time.sleep(delay)
        raise RuntimeError(f"MIMO exhausted retries: {last_err}")

    def chat_json(
        self,
        *,
        system: str,
        user: str,
        max_tokens: int = 2048,
        temperature: float = 0.4,
    ) -> dict[str, Any]:
        """Ask for JSON output and parse it.

        MIMO doesn't universally support `response_format={"type":"json_object"}`,
        so we instruct via the system prompt and strip ```json fences if present.
        """
        system_with_json = (
            system
            + "\n\nRespond with a single valid JSON object. No prose, no code fences."
        )
        raw = self.chat(
            system=system_with_json,
            user=user,
            max_tokens=max_tokens,
            temperature=temperature,
        )
        cleaned = _strip_fences(raw)
        try:
            return json.loads(cleaned)
        except json.JSONDecodeError as e:
            raise ValueError(
                f"MIMO returned invalid JSON (likely truncated at {max_tokens} tokens). "
                f"Error: {e}. Raw length: {len(cleaned)} chars. "
                f"Tail: ...{cleaned[-200:]!r}"
            ) from e


def _strip_fences(text: str) -> str:
    t = text.strip()
    if t.startswith("```"):
        lines = t.splitlines()
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].startswith("```"):
            lines = lines[:-1]
        t = "\n".join(lines).strip()
    return t
