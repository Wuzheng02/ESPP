# -*- coding: utf-8 -*-
"""Stage 2: one round of Semantic Bounded-Confidence opinion exchange.

Classical opinion dynamics (DeGroot 1974; Hegselmann-Krause 2002 "bounded
confidence") update each agent's opinion as a weighted average of neighbors
whose opinion lies within a numeric confidence threshold epsilon; opinions
that are "too far" are simply ignored, which is what allows the model to
produce persistent disagreement/polarization instead of collapsing everyone
to the mean.

We extend this in two ways that are natural once the "agents" are LLM-backed
personas rather than scalar variables:

  1. Susceptibility to social influence is not a free hyperparameter -- it is
     derived from the persona's own Big Five traits: higher agreeableness and
     higher neuroticism both increase how much a persona's stated opinion
     shifts under social pressure (well-established in the persuasion /
     conformity literature, e.g. Costa & McCrae's FFM correlates of
     persuadability). Low-agreeableness / emotionally stable personas are
     modeled as comparatively immovable.

  2. Beyond the classical *numeric* bounded-confidence gate (ignore peers
     whose score differs by more than epsilon), we add a *semantic* gate:
     a peer's reasoning is only allowed to move your opinion if it touches a
     concern this persona actually cares about (matched against this
     persona's `frustration` / `ui_expectation` fields and its own P-Q-A
     evidence for that dimension). This "Semantic Bounded Confidence" lets a
     persuasive, on-topic argument shift someone even across a somewhat large
     score gap, while an off-topic argument does not move them even from a
     small gap -- something a purely numeric bounded-confidence model cannot
     express, and something classical social-judgment theory (Sherif's
     assimilation-contrast) would predict.

Implementation notes:
  - The numeric homophily / bounded-confidence weights and the persona's
    overall receptivity are computed deterministically in Python from trait
    fields and passed into the prompt as explicit behavioral instructions,
    so the *mechanism* is auditable even though the final textual judgment
    (was I actually moved, and to where) is produced by the LLM.
  - Cost control: all 5 dimensions are exchanged in a SINGLE call per
    (instruction, model, persona) rather than one call per dimension, to
    avoid a per-dimension design multiplying the number of multimodal calls
    needed. Bundling dimensions keeps Stage 2 at parity with Stage 1 while
    still letting each dimension's exchange be governed by its own
    numeric/semantic gating.
"""

from __future__ import annotations

from typing import Any

from common import (
    DIMENSIONS,
    call_multimodal_with_retry,
    extract_json_block,
    png_path,
)
from sampling import _LEVEL3

# Numeric bounded-confidence gate: peers whose score differs by more than
# this many points (on the 1-5 scale) are not considered "close enough to
# possibly persuade" via the numeric channel alone.
NUMERIC_EPSILON = 2

# Base receptivity range personas are mapped into based on (agreeableness,
# neuroticism). Both traits are ordinal low/medium/high -> {0, 1, 2}.
_MIN_RECEPTIVITY = 0.05
_MAX_RECEPTIVITY = 0.55


def receptivity(persona: dict[str, Any]) -> float:
    """Map (agreeableness, neuroticism) in {0,1,2}^2 to a [0,1] social
    -influence receptivity coefficient. High agreeableness -> wants to align
    with the group; high neuroticism -> more easily unsettled/persuaded by
    others' stated concerns. Both increase receptivity; low-agreeableness +
    low-neuroticism personas are modeled as the most "stubborn" (independent)
    raters, matching trait-persuadability findings in the FFM literature.
    """
    pers = persona["personality"]
    a = _LEVEL3.get(pers.get("agreeableness"), 1)
    n = _LEVEL3.get(pers.get("neuroticism"), 1)
    raw = (a + n) / 4.0  # in [0, 1]
    return _MIN_RECEPTIVITY + raw * (_MAX_RECEPTIVITY - _MIN_RECEPTIVITY)


def receptivity_stance(persona: dict[str, Any]) -> str:
    r = receptivity(persona)
    if r < 0.2:
        return "You are psychologically quite independent/stubborn and rarely change a rating just because others disagree."
    if r < 0.4:
        return "You are moderately open to reconsidering a rating, but only if a peer raises a concrete point you hadn't weighed."
    return "You are quite receptive to social consensus and readily update your view when peers converge on a different assessment."


def topical_hook_terms(persona: dict[str, Any]) -> list[str]:
    """Terms describing what this persona is known to care about, used as the
    semantic gate: peer reasoning that mentions these concerns is flagged as
    "on-topic" and allowed to influence even across a wider numeric gap.
    """
    terms = []
    frustration = persona.get("frustration")
    if frustration:
        terms.append(frustration.replace("_", " "))
    for exp in persona.get("ui_expectation", []) or []:
        terms.append(exp)
    cog = persona.get("cognitive_style", {})
    if cog.get("control") == "high":
        terms.append("control")
        terms.append("predictability")
    if cog.get("risk") == "conservative":
        terms.append("risk")
        terms.append("safety")
    return terms


def persona_brief(persona: dict[str, Any]) -> str:
    demo = persona["demographic"]
    return f"{demo.get('age_group')} {demo.get('occupation')}"


def build_stage2_prompt(
    persona: dict[str, Any],
    instruction_text: str,
    own_result: dict[str, Any],
    panel_stage1: dict[str, dict[str, Any]],
) -> str:
    """Bundle all 5 dimensions' opinion exchange into a single prompt."""
    pid = persona["id"]
    stance = receptivity_stance(persona)
    hooks = topical_hook_terms(persona)

    dim_blocks = []
    for dim in DIMENSIONS:
        own_score = own_result["scores"][dim]
        own_reason = own_result["reasons"][dim]
        peer_lines = []
        for other_pid, other_result in panel_stage1.items():
            if other_pid == pid:
                continue
            peer_lines.append(
                f"  - {other_pid}: {other_result['scores'][dim]}/5 -- \"{other_result['reasons'][dim]}\""
            )
        dim_blocks.append(
            f"### {dim}\n"
            f"Your first-pass rating: {own_score}/5 -- \"{own_reason}\"\n"
            f"Other panelists:\n" + "\n".join(peer_lines)
        )
    dims_text = "\n\n".join(dim_blocks)

    return (
        "You are still role-playing the same persona from before, now in a small panel "
        "discussion with other evaluators about the SAME interface screenshot (attached "
        "again). Everyone already gave independent first-pass ratings on all 5 dimensions; "
        "now you see everyone else's ratings and reasoning once per dimension, and decide "
        "whether to keep or revise your own rating on each dimension independently.\n\n"
        f"## What you personally care about (for judging if a peer's point is relevant to you)\n"
        f"{', '.join(hooks) if hooks else 'no strong specific hooks; judge relevance generically'}\n\n"
        f"## Your social-influence disposition\n{stance}\n\n"
        f"## Per-dimension discussion\n{dims_text}\n\n"
        "## Rules to follow faithfully in character, for EACH dimension independently\n"
        "1. If a peer's reasoning raises a concern that matches something YOU personally "
        "care about (see 'what you personally care about'), you may be persuaded to shift "
        "your score toward theirs even if their score is quite different from yours.\n"
        "2. If peers' reasoning is off-topic relative to what you care about, do not shift "
        "your score just because their numeric score differs from yours, even if it's close.\n"
        f"3. Ignore any peer whose rating differs from yours by more than {NUMERIC_EPSILON} "
        "points UNLESS their reasoning is clearly on-topic for you (rule 1 overrides this "
        "numeric cutoff only in that case).\n"
        "4. Your overall willingness to move at all is governed by your social-influence "
        "disposition above -- a 'stubborn' persona should rarely change, even when rule 1 applies.\n"
        "5. It is fine and expected to keep some or all of your original scores unchanged.\n\n"
        "Reply with ONLY a JSON object, no markdown fences, in exactly this shape:\n"
        '{"revised_scores": {"Understanding": <1-5 int>, "Trust_and_Reliance": <1-5 int>, '
        '"Usability": <1-5 int>, "Control": <1-5 int>, "Transparency": <1-5 int>}, '
        '"revised_reasons": {"Understanding": "<short first-person sentence>", '
        '"Trust_and_Reliance": "...", "Usability": "...", "Control": "...", "Transparency": "..."}}'
    )


def run_stage2_for_persona(
    persona: dict[str, Any],
    instruction_text: str,
    model: str,
    instruction_id: int,
    panel_stage1: dict[str, dict[str, Any]],
    api_model: str = "gpt-4o",
    temperature: float = 0.7,
    max_tokens: int = 900,
) -> dict[str, Any]:
    """panel_stage1: persona_id -> stage1 result dict ({"scores":..., "reasons":...}).

    Returns {"scores": {dim: revised_score}, "reasons": {dim: revised_reason}},
    same shape as a Stage 1 result so downstream aggregation code is agnostic
    to which stage produced the "final" per-panelist score.
    """
    pid = persona["id"]
    own_result = panel_stage1[pid]
    prompt = build_stage2_prompt(persona, instruction_text, own_result, panel_stage1)
    image = png_path(model, instruction_id)

    last_error: Exception | None = None
    for parse_attempt in range(1, 3 + 1):
        raw = call_multimodal_with_retry(prompt, [image], model=api_model, temperature=temperature, max_tokens=max_tokens)
        try:
            parsed = extract_json_block(raw)
            raw_scores = parsed["revised_scores"]
            raw_reasons = parsed.get("revised_reasons", {})
            # Tolerate a missing dimension key by falling back to this
            # persona's own Stage-1 score/reason for that dimension (i.e.
            # "no revision"), rather than raising and failing the whole item.
            revised_scores = {
                dim: max(1, min(5, int(raw_scores.get(dim, own_result["scores"][dim]))))
                for dim in DIMENSIONS
            }
            revised_reasons = {
                dim: str(raw_reasons.get(dim, own_result["reasons"][dim])) for dim in DIMENSIONS
            }
            return {"scores": revised_scores, "reasons": revised_reasons}
        except Exception as e:
            last_error = e
            print(f"    [stage2] parse attempt {parse_attempt} failed: {e}; raw={raw[:200]!r}", flush=True)
    raise RuntimeError(f"Stage2 JSON parsing failed after retries: {last_error}")
