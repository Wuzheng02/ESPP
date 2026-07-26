# -*- coding: utf-8 -*-
"""Shared IO helpers for the P-Q-A (Persona-Question-Answer) generation pipeline."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parent
DEFAULT_PROMPTS = REPO_ROOT / "outputs" / "prompts.jsonl"
DEFAULT_ANSWERS = REPO_ROOT / "outputs" / "answers.jsonl"
DEFAULT_ASSEMBLED = REPO_ROOT / "outputs" / "persona_qa.json"
DEFAULT_PERSONAS = REPO_ROOT / "personas_1000_structured.json"


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                out.append(json.loads(line))
    return out


def answer_key(record: dict[str, Any]) -> tuple[str, int]:
    return record["persona_id"], int(record["local_q_idx"])


def existing_keys(path: Path) -> set[tuple[str, int]]:
    """Return the (persona_id, local_q_idx) pairs already answered, for resume support."""
    if not path.exists():
        return set()

    keys: set[tuple[str, int]] = set()
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                keys.add(answer_key(json.loads(line)))
            except (json.JSONDecodeError, KeyError, TypeError, ValueError):
                continue
    return keys


def answer_record(prompt_record: dict[str, Any], answer: str) -> dict[str, Any]:
    return {
        "persona_id": prompt_record["persona_id"],
        "local_q_idx": prompt_record["local_q_idx"],
        "dimension": prompt_record["dimension"],
        "q_type": prompt_record["q_type"],
        "source_qid": prompt_record["source_qid"],
        "question": prompt_record["question"],
        "options": prompt_record.get("options"),
        "answer": answer.strip(),
    }


def assemble_final(answers_path: Path, personas_path: Path, out_path: Path) -> int:
    """Merge per-question answers (answers.jsonl) with the persona pool into
    the final persona_qa.json dataset.

    Each persona keeps its full structured profile plus its 10 questions
    sorted by local_q_idx (5 dimensions x 1 multiple-choice + 1 open-ended
    question each).
    """
    answers = load_jsonl(answers_path)
    with open(personas_path, encoding="utf-8") as f:
        personas = {p["id"]: p for p in json.load(f)}

    by_pid: dict[str, list[dict[str, Any]]] = {}
    for answer in answers:
        by_pid.setdefault(answer["persona_id"], []).append(answer)

    assembled: list[dict[str, Any]] = []
    for pid in sorted(by_pid.keys()):
        items = sorted(by_pid[pid], key=lambda x: int(x["local_q_idx"]))
        qa = []
        for item in items:
            q_idx = int(item["local_q_idx"])
            qa.append(
                {
                    "q_idx": q_idx,
                    "q_label": f"Q{q_idx}",
                    "a_label": f"A{q_idx}",
                    "dimension": item["dimension"],
                    "q_type": item["q_type"],
                    "source_qid": item["source_qid"],
                    "question": item["question"],
                    "options": item.get("options"),
                    "answer": item["answer"],
                }
            )
        assembled.append(
            {
                "persona_id": pid,
                "persona": personas.get(pid),
                "qa": qa,
            }
        )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(assembled, f, ensure_ascii=False, indent=2)
    return len(assembled)
