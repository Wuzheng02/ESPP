# -*- coding: utf-8 -*-
"""Retrieve, for a given persona and evaluation dimension, the persona's own
P-Q-A answers that are most relevant as "behavioral evidence" to ground the
initial rating (Stage 1).

Each persona already has exactly 2 answered questions per dimension (1 MC +
1 OE, see pqa_construction/build_prompts.py). Rather than a heavy embedding
index (overkill for 2 candidates per dimension), we simply return both of
that persona's answers for the requested dimension, formatted as short
"evidence" lines. This keeps the retrieval step deterministic, free, and
trivially auditable -- every rating can be traced back to a literal quote
from that persona's own survey responses.

If persona_qa data is unavailable for a given persona (e.g. incomplete
generation), retrieval degrades gracefully to an empty evidence list, and the
prompt builder falls back to using only the structured trait fields.
"""

from __future__ import annotations

from typing import Any

from common import DIM_TO_QA_KEY, load_persona_qa


def get_dimension_evidence(persona_id: str, dimension: str) -> list[dict[str, Any]]:
    """Return this persona's own P-Q-A answers tagged with `dimension`.

    Returns a list of 0-2 dicts: {"question": str, "answer": str, "q_type": str}.
    """
    qa_key = DIM_TO_QA_KEY.get(dimension)
    if qa_key is None:
        return []
    qa_map = load_persona_qa()
    entry = qa_map.get(persona_id)
    if entry is None:
        return []
    evidence = []
    for item in entry.get("qa", []):
        if item.get("dimension") == qa_key:
            evidence.append(
                {
                    "question": item["question"],
                    "answer": item["answer"],
                    "q_type": item["q_type"],
                }
            )
    return evidence


def format_evidence_block(evidence: list[dict[str, Any]]) -> str:
    if not evidence:
        return "(no prior survey evidence available for this dimension)"
    lines = []
    for i, ev in enumerate(evidence, 1):
        lines.append(f"{i}. Q: {ev['question']}\n   Your prior answer: {ev['answer']}")
    return "\n".join(lines)
