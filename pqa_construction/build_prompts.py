# -*- coding: utf-8 -*-
"""Build persona + question prompts for role-play answering.

For every persona in personas_1000_structured.json, sample 1
multiple-choice and 1 open-ended question from each of the 5 Q dimensions
(d1..d5), producing exactly 10 questions per persona (Q1..Q10 in the order
d1_MC, d1_OE, d2_MC, d2_OE, ...).

Each (persona, question) pair is turned into a chat message list:
  - system: a persona-in-first-person role-play spec
  - user:   the question (with options appended for MC)

Output: outputs/prompts.jsonl, one JSON object per line:
  {
    "persona_id": "P_0000",
    "local_q_idx": 1,            # 1..10
    "dimension": "d1_Understanding",
    "q_type": "mc" | "oe",
    "source_qid": "Q1",          # id inside the dimension file
    "question": "...",
    "options": [...] | null,
    "messages": [{"role": "system", ...}, {"role": "user", ...}]
  }
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parent
P_FILE = REPO_ROOT / "personas_1000_structured.json"
Q_DIR = REPO_ROOT / "question_bank"

DIMENSION_FILES = [
    ("d1_Understanding", "d1_Understanding_organized.json"),
    ("d2_Trust_and_Reliance", "d2_Trust_and_Reliance_organized.json"),
    ("d3_usability", "d3_usability_organized.json"),
    ("d4_control", "d4_control_organized.json"),
    ("d5_transparency", "d5_transparency_organized.json"),
]


def _fmt_list(v: Any) -> str:
    if isinstance(v, list):
        return ", ".join(str(x) for x in v)
    return str(v)


def persona_to_system_prompt(p: dict) -> str:
    demo = p["demographic"]
    pers = p["personality"]
    cog = p["cognitive_style"]
    exp = p["experience"]

    lines = [
        "You are role-playing as a specific human persona answering a survey about "
        "Generative UI (GenUI) systems. Stay fully in character throughout. Answer "
        "from this persona's viewpoint, using their vocabulary, attitudes, and level "
        "of expertise. Do not mention that you are an AI or that you are role-playing.",
        "",
        f"Persona ID: {p['id']}",
        "",
        "## Demographics",
        f"- Age group: {demo.get('age_group')}",
        f"- Gender: {demo.get('gender')}",
        f"- Ethnicity: {demo.get('ethnicity')}",
        f"- Occupation: {demo.get('occupation')}",
        "",
        "## Big Five personality",
        f"- Openness: {pers.get('openness')}",
        f"- Conscientiousness: {pers.get('conscientiousness')}",
        f"- Extraversion: {pers.get('extraversion')}",
        f"- Agreeableness: {pers.get('agreeableness')}",
        f"- Neuroticism: {pers.get('neuroticism')}",
        "",
        "## Cognitive style",
        f"- Information processing: {cog.get('processing')}",
        f"- Risk tolerance: {cog.get('risk')}",
        f"- Desire for control: {cog.get('control')}",
        "",
        "## Experience with AI / GenUI",
        f"- Tech literacy: {exp.get('tech_literacy')}",
        f"- Target domain: {exp.get('target_domain')}",
        f"- Domain experience: {exp.get('domain_experience')}",
        f"- Usage frequency: {exp.get('usage_frequency')}",
        "",
        "## Motivations and context",
        f"- Primary motivation for using AI tools: {p.get('motivation')}",
        f"- Main source of frustration: {p.get('frustration')}",
        f"- Typical goal: {p.get('goal')}",
        f"- Behavior pattern: {p.get('behavior_pattern')}",
        f"- Usage context: {p.get('context')}",
        f"- UI expectations: {_fmt_list(p.get('ui_expectation'))}",
        "",
        "## Answering rules",
        "- For multiple-choice questions, reply with EXACTLY one line: the chosen "
        "letter (A/B/C/...) followed by ' - ' and a one-sentence justification that "
        "sounds natural for this persona. Do not add extra commentary, bullet points, "
        "or a second paragraph.",
        "- For open-ended questions, reply in 2-5 sentences in first person, "
        "consistent with this persona's tone, vocabulary, and tech literacy. Do not "
        "add lists, headings, or meta commentary about the question.",
    ]
    return "\n".join(lines)


def mc_user_prompt(q: dict) -> str:
    opts = "\n".join(q["options"])
    return (
        "The following is a multiple-choice question. Pick exactly one option and "
        "reply on a single line as '<LETTER> - <one-sentence reason>'.\n\n"
        f"Question: {q['question']}\n\n"
        f"{opts}"
    )


def oe_user_prompt(q: dict) -> str:
    return (
        "The following is an open-ended question. Answer in 2-5 sentences, in first "
        "person, staying in character.\n\n"
        f"Question: {q['question']}"
    )


def load_questions() -> dict:
    out = {}
    for dim_name, fname in DIMENSION_FILES:
        with open(Q_DIR / fname, encoding="utf-8") as f:
            data = json.load(f)
        out[dim_name] = {
            "mc": data["multiple_choice_questions"],
            "oe": data["open_ended_questions"],
        }
    return out


def build_prompts(seed: int, out_path: Path) -> int:
    with open(P_FILE, encoding="utf-8") as f:
        personas = json.load(f)
    questions = load_questions()

    rng = random.Random(seed)
    n_written = 0
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as fout:
        for persona in personas:
            system_prompt = persona_to_system_prompt(persona)
            local_idx = 0
            for dim_name, _ in DIMENSION_FILES:
                mc_q = rng.choice(questions[dim_name]["mc"])
                oe_q = rng.choice(questions[dim_name]["oe"])
                for q_type, q in (("mc", mc_q), ("oe", oe_q)):
                    local_idx += 1
                    user_prompt = mc_user_prompt(q) if q_type == "mc" else oe_user_prompt(q)
                    record = {
                        "persona_id": persona["id"],
                        "local_q_idx": local_idx,
                        "dimension": dim_name,
                        "q_type": q_type,
                        "source_qid": q["id"],
                        "question": q["question"],
                        "options": q.get("options") if q_type == "mc" else None,
                        "messages": [
                            {"role": "system", "content": system_prompt},
                            {"role": "user", "content": user_prompt},
                        ],
                    }
                    fout.write(json.dumps(record, ensure_ascii=False) + "\n")
                    n_written += 1
    return n_written


def main():
    ap = argparse.ArgumentParser(description="Build persona + question prompts for role-play answering.")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument(
        "--out",
        type=Path,
        default=REPO_ROOT / "outputs" / "prompts.jsonl",
    )
    args = ap.parse_args()
    n = build_prompts(args.seed, args.out)
    print(f"Wrote {n} prompts to {args.out}")


if __name__ == "__main__":
    main()
