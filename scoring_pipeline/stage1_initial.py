# -*- coding: utf-8 -*-
"""Stage 1: independent, evidence-grounded initial rating.

For a given (instruction, model-generated UI screenshot, persona), ask the
persona (in-character) to rate the UI on all 5 GenUI dimensions (1-5 each)
and give one short reason per dimension. Each dimension's reasoning prompt is
grounded with that persona's OWN prior P-Q-A survey answers for that
dimension (retrieved verbatim, not paraphrased by us), so the rating has to
be consistent with a documented behavioral profile rather than a free-floating
vibe.

Output schema per (instruction_id, model, persona_id):
{
  "scores": {"Understanding": 3, "Trust_and_Reliance": 4, ...},
  "reasons": {"Understanding": "...", ...}
}
"""

from __future__ import annotations

from typing import Any

from common import (
    DIMENSIONS,
    call_multimodal_with_retry,
    extract_json_block,
    png_path,
)
from pqa_retrieval import format_evidence_block, get_dimension_evidence

DIMENSION_DEFINITIONS = {
    "Understanding": (
        "How quickly and correctly you can grasp what this interface is for, "
        "how it's organized, and what each part does, based only on this screenshot."
    ),
    "Trust_and_Reliance": (
        "How much you would trust the information/functionality shown and feel "
        "comfortable relying on it for a real task, versus feeling suspicious or unsure."
    ),
    "Usability": (
        "How easy and efficient it would feel to actually use this interface to "
        "accomplish a goal -- clarity of actions, layout convenience, effort required."
    ),
    "Control": (
        "How much you feel in control of the interface -- predictable behavior, "
        "visible/undoable actions, no sense of the system deciding things for you unexpectedly."
    ),
    "Transparency": (
        "How clearly the interface explains its own state, options, and the "
        "reasoning/consequences of actions, rather than hiding things as a black box."
    ),
}


def persona_identity_block(p: dict[str, Any]) -> str:
    demo = p["demographic"]
    pers = p["personality"]
    cog = p["cognitive_style"]
    exp = p["experience"]
    ui_exp = p.get("ui_expectation", [])
    return (
        f"Persona ID: {p['id']}\n"
        f"Demographics: {demo.get('age_group')}, {demo.get('gender')}, "
        f"{demo.get('ethnicity')}, {demo.get('occupation')}\n"
        f"Big Five: openness={pers.get('openness')}, conscientiousness={pers.get('conscientiousness')}, "
        f"extraversion={pers.get('extraversion')}, agreeableness={pers.get('agreeableness')}, "
        f"neuroticism={pers.get('neuroticism')}\n"
        f"Cognitive style: processing={cog.get('processing')}, risk={cog.get('risk')}, "
        f"control={cog.get('control')}\n"
        f"Tech literacy={exp.get('tech_literacy')}, domain_experience={exp.get('domain_experience')}, "
        f"target_domain={exp.get('target_domain')}, usage_frequency={exp.get('usage_frequency')}\n"
        f"Motivation={p.get('motivation')}, frustration={p.get('frustration')}, goal={p.get('goal')}, "
        f"behavior_pattern={p.get('behavior_pattern')}, context={p.get('context')}\n"
        f"UI expectations: {', '.join(ui_exp) if ui_exp else 'none stated'}"
    )


def build_stage1_prompt(
    persona: dict[str, Any],
    instruction_text: str,
    use_pqa: bool = True,
) -> str:
    """Build the Stage-1 rating prompt.

    use_pqa=True  (default): grounds each dimension with this persona's own
        prior P-Q-A survey answers, and instructs the model to stay
        consistent with them (see module docstring).
    use_pqa=False: ABLATION condition. Skips retrieval entirely and drops the
        "stay consistent with your prior answers" instruction, so the ONLY
        difference from the default condition is the presence/absence of the
        P-Q-A evidence grounding -- persona identity, dimension definitions,
        image, output schema, temperature, etc. are all identical. This
        isolates the marginal effect of P-Q-A grounding on rating quality.
    """
    dim_blocks = []
    for dim in DIMENSIONS:
        if use_pqa:
            evidence = get_dimension_evidence(persona["id"], dim)
            dim_blocks.append(
                f"### {dim}\n"
                f"Definition: {DIMENSION_DEFINITIONS[dim]}\n"
                f"Your own prior survey answers relevant to this dimension "
                f"(stay consistent with these; do not contradict them):\n"
                f"{format_evidence_block(evidence)}"
            )
        else:
            dim_blocks.append(
                f"### {dim}\n"
                f"Definition: {DIMENSION_DEFINITIONS[dim]}"
            )
    dims_text = "\n\n".join(dim_blocks)

    consistency_line = (
        " Your score AND reason on each dimension must be "
        "consistent with your own prior survey answers shown for that dimension.\n\n"
        if use_pqa
        else "\n\n"
    )

    return (
        "You are role-playing as a specific human persona evaluating a generated "
        "user interface (GenUI). Stay fully in character: use this persona's vocabulary, "
        "tech literacy, and attitudes. Do not mention you are an AI or role-playing.\n\n"
        "## Your persona\n"
        f"{persona_identity_block(persona)}\n\n"
        "## What this interface was asked to do\n"
        f"\"{instruction_text}\"\n\n"
        "## Task\n"
        "Look at the attached screenshot of the interface that was generated for the "
        "instruction above. Rate it on the following 5 dimensions, each on a 1-5 integer "
        "scale (1 = very poor, 5 = excellent), from THIS persona's point of view -- not a "
        "generic 'objective' UX audit." + consistency_line +
        f"{dims_text}\n\n"
        "## Output format\n"
        "Reply with ONLY a JSON object, no markdown fences, no extra commentary, in exactly "
        "this shape:\n"
        '{"scores": {"Understanding": <1-5 int>, "Trust_and_Reliance": <1-5 int>, '
        '"Usability": <1-5 int>, "Control": <1-5 int>, "Transparency": <1-5 int>}, '
        '"reasons": {"Understanding": "<one short first-person sentence>", '
        '"Trust_and_Reliance": "...", "Usability": "...", "Control": "...", "Transparency": "..."}}'
    )


def run_stage1_for_one(
    persona: dict[str, Any],
    instruction_text: str,
    model: str,
    instruction_id: int,
    api_model: str = "gpt-4o",
    temperature: float = 0.8,
    max_tokens: int = 900,
    use_pqa: bool = True,
) -> dict[str, Any]:
    prompt = build_stage1_prompt(persona, instruction_text, use_pqa=use_pqa)
    image = png_path(model, instruction_id)
    last_error: Exception | None = None
    for parse_attempt in range(1, 3 + 1):
        raw = call_multimodal_with_retry(
            prompt,
            [image],
            model=api_model,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        try:
            parsed = extract_json_block(raw)
            raw_scores = parsed["scores"]
            raw_reasons = parsed.get("reasons", {})
            # Tolerate a missing dimension key (e.g. model typo/omission) by
            # falling back to a neutral score (3) rather than raising, so one
            # malformed response doesn't fail the whole panel item.
            scores = {dim: max(1, min(5, int(raw_scores.get(dim, 3)))) for dim in DIMENSIONS}
            reasons = {dim: str(raw_reasons.get(dim, "")) for dim in DIMENSIONS}
            return {"scores": scores, "reasons": reasons}
        except Exception as e:
            last_error = e
            print(f"    [stage1] parse attempt {parse_attempt} failed: {e}; raw={raw[:200]!r}", flush=True)
    raise RuntimeError(f"Stage1 JSON parsing failed after retries: {last_error}")
