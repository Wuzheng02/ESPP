# -*- coding: utf-8 -*-
"""Shared constants, IO helpers, and the multimodal (image+text) LLM call
wrapper used by every stage of the Social-Weighted Persona Panel pipeline.

Directory layout assumed:
  scoring_pipeline/
    sample_data/personas_1000_structured.json  -- persona pool sample
    sample_data/persona_qa_sample.json          -- per-persona P-Q-A sample
    sample_data/instructions_sample.json        -- UI-generation instructions sample
    sample_data/screenshots/<model>/<id>.png    -- rendered screenshots
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from llm_client import LLMClient, extract_json_block  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parent
DATA_DIR = REPO_ROOT / "sample_data"

PERSONAS_PATH = DATA_DIR / "personas_1000_structured.json"
PERSONA_QA_PATH = DATA_DIR / "persona_qa_sample.json"
INSTRUCTIONS_PATH = DATA_DIR / "instructions_sample.json"
SCREENSHOT_DIR = DATA_DIR / "screenshots"

# Models evaluated in this run (anonymized placeholders here; the actual
# experiments used specific commercial model names).
ALL_MODELS = ["model_A", "model_B", "model_C"]

# The 5 GenUI evaluation dimensions, matching the question bank files and
# the human rating sheet (each human rater scores 1-5 per dimension per image).
DIMENSIONS = [
    "Understanding",
    "Trust_and_Reliance",
    "Usability",
    "Control",
    "Transparency",
]
DIM_TO_QA_KEY = {
    "Understanding": "d1_Understanding",
    "Trust_and_Reliance": "d2_Trust_and_Reliance",
    "Usability": "d3_usability",
    "Control": "d4_control",
    "Transparency": "d5_transparency",
}

# ---------------------------------------------------------------------------
# Basic IO
# ---------------------------------------------------------------------------


def load_json(path: Path) -> Any:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    if not path.exists():
        return out
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                out.append(json.loads(line))
    return out


def append_jsonl(path: Path, record: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


_PERSONAS_CACHE: list[dict[str, Any]] | None = None
_PERSONA_QA_CACHE: dict[str, dict[str, Any]] | None = None
_INSTRUCTIONS_CACHE: list[dict[str, Any]] | None = None


def load_personas() -> list[dict[str, Any]]:
    global _PERSONAS_CACHE
    if _PERSONAS_CACHE is None:
        _PERSONAS_CACHE = load_json(PERSONAS_PATH)
    return _PERSONAS_CACHE


def load_personas_by_id() -> dict[str, dict[str, Any]]:
    return {p["id"]: p for p in load_personas()}


def load_persona_qa() -> dict[str, dict[str, Any]]:
    """persona_id -> {persona, qa: [...]} (10 Q/A per persona, 2 per dimension)."""
    global _PERSONA_QA_CACHE
    if _PERSONA_QA_CACHE is None:
        raw = load_json(PERSONA_QA_PATH)
        _PERSONA_QA_CACHE = {item["persona_id"]: item for item in raw}
    return _PERSONA_QA_CACHE


def load_instructions() -> list[dict[str, Any]]:
    global _INSTRUCTIONS_CACHE
    if _INSTRUCTIONS_CACHE is None:
        _INSTRUCTIONS_CACHE = load_json(INSTRUCTIONS_PATH)
    return _INSTRUCTIONS_CACHE


def instructions_by_id() -> dict[int, dict[str, Any]]:
    return {item["id"]: item for item in load_instructions()}


def png_path(model: str, instruction_id: int) -> Path:
    return SCREENSHOT_DIR / model / f"{instruction_id}.png"


def encode_image_b64(path: Path) -> str:
    import base64

    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode("ascii")


# ---------------------------------------------------------------------------
# Multimodal (image + text) LLM call
# ---------------------------------------------------------------------------

_DEFAULT_CLIENT = LLMClient()


def build_image_user_message(text: str, image_paths: list[Path]) -> list[str]:
    """Base64-encode one or more images for call_multimodal_with_retry."""
    return [encode_image_b64(p) for p in image_paths]


def call_multimodal_with_retry(
    text: str,
    image_paths: list[Path],
    model: str,
    temperature: float = 0.7,
    max_tokens: int = 1024,
    max_retries: int = 5,
) -> str:
    b64_images = build_image_user_message(text, image_paths)
    return _DEFAULT_CLIENT.call_multimodal(
        text,
        b64_images,
        model=model,
        temperature=temperature,
        max_tokens=max_tokens,
        max_retries=max_retries,
    )


__all__ = [
    "ALL_MODELS",
    "DIMENSIONS",
    "DIM_TO_QA_KEY",
    "load_json",
    "load_jsonl",
    "append_jsonl",
    "load_personas",
    "load_personas_by_id",
    "load_persona_qa",
    "load_instructions",
    "instructions_by_id",
    "png_path",
    "encode_image_b64",
    "call_multimodal_with_retry",
    "extract_json_block",
]
