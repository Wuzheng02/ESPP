# -*- coding: utf-8 -*-
"""Stage 3: non-uniform aggregation of the panel's post-discussion scores.

Instead of a plain mean over the N panelists' revised scores, weight each
panelist by:

  1. Expertise weight (Delphi-method inspired): personas with higher
     domain_experience and higher tech_literacy get a higher weight, on the
     premise that more experienced users give more calibrated (less noisy)
     judgments of a GenUI's Usability/Control/Transparency.
  2. Representativeness weight: personas whose experience.target_domain and
     usage_frequency better match the instruction's scenario get a higher
     weight -- someone who actually deals with this kind of interface
     regularly is a more representative rater for it.

Weights are normalized to sum to 1 per (instruction, dimension) panel, and
are deliberately mild (not winner-take-all) so a single high-weight persona
can't dominate the score; this is a soft re-weighting, not a hard filter.
"""

from __future__ import annotations

from typing import Any

from common import DIMENSIONS
from sampling import _LEVEL_EXP, _LEVEL_TECH, domain_relevance

# How much extra weight expertise/representativeness contribute relative to
# the uniform baseline of 1.0 per panelist. Kept modest by design (see
# module docstring) -- the comparison between Stage-2 opinion dynamics and
# Stage-3 weighting is meant to show a *complementary*, not dominant,
# improvement over simple averaging.
EXPERTISE_WEIGHT_SCALE = 0.5
REPRESENTATIVENESS_WEIGHT_SCALE = 0.5


def expertise_score(persona: dict[str, Any]) -> float:
    exp = persona["experience"]
    domain_exp = _LEVEL_EXP.get(exp.get("domain_experience"), 0) / 2.0  # in [0,1]
    tech = _LEVEL_TECH.get(exp.get("tech_literacy"), 0) / 2.0  # in [0,1]
    return (domain_exp + tech) / 2.0


def usage_frequency_score(persona: dict[str, Any]) -> float:
    freq_map = {"rare": 0.0, "occasional": 0.33, "frequent": 0.67, "regular": 1.0}
    return freq_map.get(persona["experience"].get("usage_frequency"), 0.33)


def panelist_weight(persona: dict[str, Any], scenario: str) -> float:
    base = 1.0
    expertise = expertise_score(persona)
    relevance = domain_relevance(persona, scenario)
    usage = usage_frequency_score(persona)
    representativeness = 0.5 * relevance + 0.5 * usage
    return base + EXPERTISE_WEIGHT_SCALE * expertise + REPRESENTATIVENESS_WEIGHT_SCALE * representativeness


def aggregate_panel(
    panel_personas: list[dict[str, Any]],
    scenario: str,
    final_scores_by_pid: dict[str, dict[str, int]],
) -> dict[str, Any]:
    """final_scores_by_pid: persona_id -> {dimension: revised_score}.

    Returns {"dimension_scores": {dim: weighted_avg}, "overall_score": float,
    "weights": {pid: weight}, "raw_scores": final_scores_by_pid}.
    """
    weights = {p["id"]: panelist_weight(p, scenario) for p in panel_personas}
    weight_sum = sum(weights.values())
    normalized = {pid: w / weight_sum for pid, w in weights.items()}

    dim_scores: dict[str, float] = {}
    for dim in DIMENSIONS:
        acc = 0.0
        for pid, w in normalized.items():
            acc += w * final_scores_by_pid[pid][dim]
        dim_scores[dim] = acc

    overall = sum(dim_scores.values()) / len(DIMENSIONS)

    # Panel disagreement (unweighted std across panelists, averaged over
    # dimensions) -- useful for comparing against human raters' disagreement.
    import statistics

    per_dim_std = {}
    for dim in DIMENSIONS:
        vals = [final_scores_by_pid[pid][dim] for pid in normalized]
        per_dim_std[dim] = statistics.pstdev(vals) if len(vals) > 1 else 0.0
    overall_std = sum(per_dim_std.values()) / len(DIMENSIONS)

    return {
        "dimension_scores": dim_scores,
        "overall_score": overall,
        "weights": normalized,
        "raw_scores": final_scores_by_pid,
        "dimension_std": per_dim_std,
        "overall_std": overall_std,
    }
