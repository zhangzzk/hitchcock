from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv


@dataclass(frozen=True)
class MimoConfig:
    api_key: str
    base_url: str
    model: str


@dataclass(frozen=True)
class ArkConfig:
    api_key: str
    base_url: str
    image_model: str
    video_model: str


@dataclass(frozen=True)
class GeminiConfig:
    api_key: str
    image_model: str
    text_model: str  # for grounded research, distinct from image_model


@dataclass(frozen=True)
class OpenAIConfig:
    api_key: str
    image_model: str  # e.g. "gpt-image-2" (default) or "gpt-image-1"
    base_url: str
    image_quality: str = "high"  # "low" | "medium" | "high" | "auto"


@dataclass(frozen=True)
class Settings:
    mimo: MimoConfig
    ark: ArkConfig
    gemini: GeminiConfig
    openai: OpenAIConfig
    bible_dir: Path


def _require(name: str) -> str:
    value = os.environ.get(name, "").strip()
    if not value:
        raise RuntimeError(f"Missing required env var: {name}")
    return value


def load_settings(env_file: str | os.PathLike[str] | None = None) -> Settings:
    if env_file is None:
        env_file = Path(__file__).resolve().parents[2] / ".env"
    load_dotenv(env_file, override=False)

    return Settings(
        mimo=MimoConfig(
            api_key=_require("HITCHCOCK_MIMO_API_KEY"),
            base_url=os.environ.get(
                "HITCHCOCK_MIMO_BASE_URL", "https://api.xiaomimimo.com/v1"
            ),
            model=os.environ.get("HITCHCOCK_MIMO_MODEL", "mimo-v2-flash"),
        ),
        ark=ArkConfig(
            api_key=_require("HITCHCOCK_ARK_API_KEY"),
            base_url=os.environ.get(
                "HITCHCOCK_ARK_BASE_URL", "https://ark.cn-beijing.volces.com/api/v3"
            ),
            image_model=os.environ.get(
                "HITCHCOCK_ARK_IMAGE_MODEL", "doubao-seedream-4-5-251128"
            ),
            video_model=os.environ.get(
                "HITCHCOCK_ARK_VIDEO_MODEL", "doubao-seedance-2-0-260128"
            ),
        ),
        gemini=GeminiConfig(
            api_key=_require("HITCHCOCK_GEMINI_API_KEY"),
            image_model=os.environ.get(
                "HITCHCOCK_GEMINI_IMAGE_MODEL", "nano-banana-pro-preview"
            ),
            text_model=os.environ.get(
                "HITCHCOCK_GEMINI_TEXT_MODEL", "gemini-2.5-flash"
            ),
        ),
        openai=OpenAIConfig(
            api_key=_require("HITCHCOCK_OPENAI_API_KEY"),
            image_model=os.environ.get(
                # Default `gpt-image-2` — higher-quality successor to
                # gpt-image-1 (what the ChatGPT web UI uses). Requires
                # Organization Verification at
                # platform.openai.com/settings/org. If that's not yet
                # enabled for this key, fall back with
                # HITCHCOCK_OPENAI_IMAGE_MODEL=gpt-image-1 in .env.
                "HITCHCOCK_OPENAI_IMAGE_MODEL", "gpt-image-2"
            ),
            base_url=os.environ.get(
                "HITCHCOCK_OPENAI_BASE_URL", "https://api.openai.com/v1"
            ),
            # "medium" is the DEFAULT (~¥0.4/image at 1536×1024) — 4× cheaper
            # than "high" and empirically produces less-detailed output that
            # animates more stably under Seedance (pairs with the
            # graphic-simplification directive in style.py). "high" (~¥1.8)
            # for final-production passes when detail density is the bottleneck;
            # "low" (~¥0.1) is noticeably lower fidelity. Override via
            # HITCHCOCK_OPENAI_IMAGE_QUALITY env var.
            image_quality=os.environ.get(
                "HITCHCOCK_OPENAI_IMAGE_QUALITY", "medium"
            ),
        ),
        bible_dir=Path(os.environ.get("HITCHCOCK_BIBLE_DIR", "./bible")).resolve(),
    )
