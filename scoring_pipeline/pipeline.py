# -*- coding: utf-8 -*-
"""Orchestrate the full Social-Weighted Persona Panel scoring pipeline.

For each instruction:
  0. Load (or draw, via sampling.py) the fixed persona panel for this
     instruction, shared across all models.
  For each model (using the SAME panel):
    1. Stage 1 -- each panelist independently rates the model's screenshot
       for this instruction on all 5 dimensions, grounded in their own P-Q-A
       evidence.
    2. Stage 2 -- one round of semantic bounded-confidence opinion exchange:
       panelists see each other's Stage-1 ratings/reasoning and may revise.
    3. Stage 3 -- expertise/representativeness-weighted aggregation of the
       post-discussion scores into a single per-dimension + overall score.

Results are written incrementally (append-only JSONL) and are resumable:
already-completed (instruction_id, model) pairs are skipped on rerun.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))

from common import ALL_MODELS, REPO_ROOT, append_jsonl, load_jsonl, load_personas_by_id, instructions_by_id
from sampling import OUT_PANELS, PANEL_SIZE_DEFAULT, build_all_panels
from stage1_initial import run_stage1_for_one
from stage2_dynamics import run_stage2_for_persona
from stage3_aggregate import aggregate_panel

OUT_RESULTS = REPO_ROOT / "outputs" / "scores.jsonl"


def load_or_build_panels(panel_size: int, seed: int) -> dict[int, list[str]]:
    if OUT_PANELS.exists():
        with open(OUT_PANELS, encoding="utf-8") as f:
            raw = json.load(f)
        return {int(k): v for k, v in raw.items()}
    panels = build_all_panels(panel_size, seed)
    OUT_PANELS.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_PANELS, "w", encoding="utf-8") as f:
        json.dump({str(k): v for k, v in panels.items()}, f, ensure_ascii=False, indent=2)
    return panels


def already_done(results_path: Path) -> set[tuple[int, str]]:
    done = set()
    for rec in load_jsonl(results_path):
        done.add((rec["instruction_id"], rec["model"]))
    return done


def run_one_instruction_model(
    instruction_id: int,
    instruction_text: str,
    scenario: str,
    model: str,
    panel_ids: list[str],
    personas_by_id: dict[str, dict[str, Any]],
    api_model: str,
    use_pqa: bool = True,
) -> dict[str, Any]:
    panel_personas = [personas_by_id[pid] for pid in panel_ids]

    # Stage 1: independent ratings.
    stage1_results: dict[str, dict[str, Any]] = {}
    for persona in panel_personas:
        print(f"  [stage1] {model} / instr {instruction_id} / {persona['id']}", flush=True)
        stage1_results[persona["id"]] = run_stage1_for_one(
            persona, instruction_text, model, instruction_id, api_model=api_model, use_pqa=use_pqa
        )

    # Stage 2: one round of opinion exchange.
    stage2_results: dict[str, dict[str, Any]] = {}
    for persona in panel_personas:
        print(f"  [stage2] {model} / instr {instruction_id} / {persona['id']}", flush=True)
        stage2_results[persona["id"]] = run_stage2_for_persona(
            persona, instruction_text, model, instruction_id, stage1_results, api_model=api_model
        )

    # Stage 3: weighted aggregation of the post-discussion (Stage 2) scores.
    final_scores_by_pid = {pid: res["scores"] for pid, res in stage2_results.items()}
    aggregation = aggregate_panel(panel_personas, scenario, final_scores_by_pid)

    return {
        "instruction_id": instruction_id,
        "model": model,
        "panel": panel_ids,
        "stage1": stage1_results,
        "stage2": stage2_results,
        "aggregation": aggregation,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="Run the full social-weighted persona panel scoring pipeline.")
    ap.add_argument("--models", nargs="*", default=ALL_MODELS)
    ap.add_argument("--panel-size", type=int, default=PANEL_SIZE_DEFAULT)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--api-model", type=str, default="gpt-4o", help="Multimodal LLM used to play each persona.")
    ap.add_argument("--no-pqa", action="store_true", help="Ablation: disable P-Q-A evidence grounding in Stage 1.")
    ap.add_argument("--out", type=Path, default=OUT_RESULTS)
    ap.add_argument("--instruction-ids", nargs="*", type=int, default=None, help="Optional subset of instruction ids.")
    args = ap.parse_args()

    personas_by_id = load_personas_by_id()
    instr_by_id = instructions_by_id()
    panels = load_or_build_panels(args.panel_size, args.seed)

    instruction_ids = args.instruction_ids or sorted(instr_by_id.keys())
    done = already_done(args.out)

    total_jobs = len(instruction_ids) * len(args.models)
    job_i = 0
    for instruction_id in instruction_ids:
        instr = instr_by_id[instruction_id]
        panel_ids = panels[instruction_id]
        for model in args.models:
            job_i += 1
            if (instruction_id, model) in done:
                print(f"[{job_i}/{total_jobs}] skip (already done): instr={instruction_id} model={model}")
                continue
            print(f"[{job_i}/{total_jobs}] running: instr={instruction_id} model={model}")
            try:
                result = run_one_instruction_model(
                    instruction_id,
                    instr["instruction"],
                    instr["scenario"],
                    model,
                    panel_ids,
                    personas_by_id,
                    api_model=args.api_model,
                    use_pqa=not args.no_pqa,
                )
                append_jsonl(args.out, result)
            except Exception as e:
                print(f"  [error] instr={instruction_id} model={model}: {e}", flush=True)

    print(f"Done. Results in {args.out}")


if __name__ == "__main__":
    main()
