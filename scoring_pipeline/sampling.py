# -*- coding: utf-8 -*-
"""Stage 0: quota-based panel sampling.

For each UI-generation instruction we draw ONE panel of N personas (default
N=5, matching a human 5-rater setup) from the persona pool. The SAME panel
is reused across all evaluated models for that instruction, so cross-model
comparisons within an instruction are judged by an identical "jury".

Sampling is not uniform-random: it is a quota / stratified draw designed to
guarantee the panel has enough psychological *heterogeneity* for the later
opinion-dynamics stage to have something to do, while still being reasonably
representative of the demographic pool. Concretely we:

  1. Bucket personas by a coarse demographic stratum (age_group x occupation
     class) and a coarse psychological stratum (dominant Big-Five profile
     signature), so a persona is never picked purely because of what's most
     abundant in the pool (avoids repeatedly drawing the majority mode).
  2. Guarantee within the drawn panel:
       - at least one pair with SMALL psychological distance ("likely to
         cluster / reinforce each other" in the opinion-dynamics stage), and
       - at least one pair with LARGE psychological distance ("likely to
         diverge / polarize").
  3. Are fully deterministic given (instruction_id, seed), so re-running the
     pipeline (or re-running a single failed model) reproduces the exact same
     panel -- this is required for the "same jury across models" property
     to hold across separate process invocations.
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Any

from common import REPO_ROOT, instructions_by_id, load_personas

OUT_PANELS = REPO_ROOT / "outputs" / "panels.json"

PANEL_SIZE_DEFAULT = 5

# Ordinal encodings so we can compute a numeric "psychological distance"
# between two personas from their categorical trait labels.
_LEVEL3 = {"low": 0, "medium": 1, "high": 2}
_LEVEL_RISK = {"conservative": 0, "moderate": 1, "adventurous": 2}
_LEVEL_PROCESSING = {"analytical": 0, "intuitive": 1}  # nominal, distance = 0/1
_LEVEL_TECH = {"low": 0, "medium": 1, "high": 2}
_LEVEL_EXP = {"novice": 0, "intermediate": 1, "expert": 2}

# Coarse occupation groups used only for the demographic stratification key
# (keeps the number of strata manageable while still spreading draws across
# very different life contexts).
_OCC_GROUPS = {
    "student": "student",
    "junior_employee": "early_career",
    "freelancer": "early_career",
    "engineer": "professional",
    "researcher": "professional",
    "designer": "professional",
    "consultant": "professional",
    "senior_engineer": "senior_professional",
    "business_owner": "senior_professional",
    "retired": "retired",
}


def occupation_group(occupation: str) -> str:
    return _OCC_GROUPS.get(occupation, "other")


def demographic_stratum(p: dict[str, Any]) -> tuple[str, str]:
    demo = p["demographic"]
    return (demo.get("age_group", "unknown"), occupation_group(demo.get("occupation", "unknown")))


def big5_vector(p: dict[str, Any]) -> tuple[int, int, int, int, int]:
    pers = p["personality"]
    return tuple(_LEVEL3.get(pers.get(k), 1) for k in
                 ("openness", "conscientiousness", "extraversion", "agreeableness", "neuroticism"))


def cognitive_vector(p: dict[str, Any]) -> tuple[int, int, int]:
    cog = p["cognitive_style"]
    return (
        _LEVEL_PROCESSING.get(cog.get("processing"), 0),
        _LEVEL_RISK.get(cog.get("risk"), 1),
        _LEVEL3.get(cog.get("control"), 1),
    )


def psych_vector(p: dict[str, Any]) -> tuple[int, ...]:
    """8-dim ordinal vector: Big Five (0-2 each) + cognitive style (processing,
    risk, control). Used purely to compute a relative psychological distance
    for panel-construction and later for the opinion-dynamics homophily term.
    """
    return big5_vector(p) + cognitive_vector(p)


def psych_distance(a: dict[str, Any], b: dict[str, Any]) -> float:
    va, vb = psych_vector(a), psych_vector(b)
    return sum(abs(x - y) for x, y in zip(va, vb)) / len(va)


def domain_relevance(p: dict[str, Any], scenario: str) -> float:
    """Heuristic match between a persona's stated target_domain / usage profile
    and the instruction's UI scenario. Used later for representativeness
    weighting in stage 3, but also mildly informs sampling so panels aren't
    100% domain-blind.
    """
    target_domain = p["experience"].get("target_domain", "general_genui")
    scenario_to_domain = {
        "landing_marketing": {"content_creation", "general_genui"},
        "dashboard_analytics": {"data_analysis", "programming_tools"},
        "forms_auth_flow": {"general_genui", "programming_tools"},
        "ecommerce_content_listing": {"content_creation", "design_tools", "general_genui"},
        "social_productivity_app": {"general_genui", "content_creation"},
        "mobile_widget_ui": {"design_tools", "general_genui"},
    }
    relevant = scenario_to_domain.get(scenario, {"general_genui"})
    return 1.0 if target_domain in relevant else 0.0


def build_strata(personas: list[dict[str, Any]]) -> dict[tuple[str, str], list[dict[str, Any]]]:
    strata: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for p in personas:
        strata.setdefault(demographic_stratum(p), []).append(p)
    return strata


def draw_panel(
    personas: list[dict[str, Any]],
    strata: dict[tuple[str, str], list[dict[str, Any]]],
    scenario: str,
    panel_size: int,
    rng: random.Random,
    max_attempts: int = 200,
    close_thresh: float = 0.35,
    far_thresh: float = 1.15,
) -> list[str]:
    """Draw `panel_size` distinct persona ids for one instruction.

    Strategy: sample one persona per distinct demographic stratum (cycling
    through strata keys in a shuffled order) to spread demographic coverage,
    with a mild oversampling bias toward personas whose target_domain matches
    the scenario. Retry up to `max_attempts` times until the resulting panel
    contains both a "close" pair (distance <= close_thresh) and a "far" pair
    (distance >= far_thresh) in psychological space -- this guarantees the
    later opinion-dynamics stage has both a homophily cluster and a
    disagreement axis to work with, rather than N near-identical or N
    maximally-random personas.
    """
    strata_keys = list(strata.keys())

    def sample_once() -> list[dict[str, Any]]:
        rng.shuffle(strata_keys)
        chosen: list[dict[str, Any]] = []
        used_ids: set[str] = set()
        key_cycle = strata_keys * ((panel_size // len(strata_keys)) + 2)
        for key in key_cycle:
            if len(chosen) >= panel_size:
                break
            candidates = [p for p in strata[key] if p["id"] not in used_ids]
            if not candidates:
                continue
            weights = [1.0 + 0.6 * domain_relevance(p, scenario) for p in candidates]
            pick = rng.choices(candidates, weights=weights, k=1)[0]
            chosen.append(pick)
            used_ids.add(pick["id"])
        # Fallback: if strata coverage was too sparse to fill panel_size (shouldn't
        # normally happen with a large enough pool / few dozen strata), top up
        # from the full pool.
        while len(chosen) < panel_size:
            pick = rng.choice(personas)
            if pick["id"] not in used_ids:
                chosen.append(pick)
                used_ids.add(pick["id"])
        return chosen

    best_panel: list[dict[str, Any]] | None = None
    for _ in range(max_attempts):
        panel = sample_once()
        dists = [
            psych_distance(panel[i], panel[j])
            for i in range(len(panel))
            for j in range(i + 1, len(panel))
        ]
        has_close = any(d <= close_thresh for d in dists)
        has_far = any(d >= far_thresh for d in dists)
        if best_panel is None:
            best_panel = panel
        if has_close and has_far:
            return [p["id"] for p in panel]
    # Couldn't satisfy both constraints within max_attempts; use the last
    # candidate we generated so sampling never blocks indefinitely.
    return [p["id"] for p in best_panel]


def build_all_panels(panel_size: int, seed: int) -> dict[int, list[str]]:
    personas = load_personas()
    strata = build_strata(personas)
    instr_by_id = instructions_by_id()

    panels: dict[int, list[str]] = {}
    for instr_id in sorted(instr_by_id.keys()):
        scenario = instr_by_id[instr_id]["scenario"]
        # Deterministic per-instruction RNG so re-runs / partial re-runs
        # reproduce identical panels regardless of iteration order.
        rng = random.Random((seed, instr_id))
        panels[instr_id] = draw_panel(personas, strata, scenario, panel_size, rng)
    return panels


def main() -> None:
    ap = argparse.ArgumentParser(description="Draw one persona panel per instruction.")
    ap.add_argument("--panel-size", type=int, default=PANEL_SIZE_DEFAULT)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", type=Path, default=OUT_PANELS)
    args = ap.parse_args()

    panels = build_all_panels(args.panel_size, args.seed)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump({str(k): v for k, v in panels.items()}, f, ensure_ascii=False, indent=2)
    print(f"Wrote {len(panels)} panels (size={args.panel_size}) to {args.out}")


if __name__ == "__main__":
    main()
