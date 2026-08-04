# Beyond a Single Judge: The Evidence-Grounded, Social-Weighted Persona Panel for Generative UI Evaluation

![ESPP Framework](https://github.com/Wuzheng02/ESPP/blob/main/pipeline.png)

This repository contains the reference implementation of the **Evidence-Grounded, Social-Weighted Persona Panel (ESPP)**, a three-stage evaluation method for Generative UI (GenUI) that replaces a single LLM-as-a-judge with a panel of psychologically diverse, evidence-grounded personas. Each panelist (i) independently rates a generated interface on five dimensions, (ii) exchanges opinions with the rest of the panel under a trait-derived, semantically-gated bounded-confidence mechanism, and (iii) is aggregated via Delphi-inspired social weighting into a single score.


## Installation

```bash
git clone <this-repository-url>
cd SPP-GenUI-Eval
pip install -r requirements.txt
```

Requires Python 3.10+ (uses `from __future__ import annotations` and PEP 604 `X | None` type hints).

### Configuring the LLM backend

Every script talks to the model through `llm_client.LLMClient`, which is a generic OpenAI-protocol-compatible Chat Completions client. No private endpoint or key is hard-coded; configure it via environment variables so it can point at any compatible provider (OpenAI, a self-hosted vLLM server, or a cloud proxy):

```bash
export LLM_API_BASE_URL="https://api.openai.com/v1/chat/completions"
export LLM_API_KEY="sk-..."
export LLM_DEFAULT_MODEL="gpt-4o"   # must support multimodal (image+text) input for the scoring pipeline
```

## Quick Start (Toy Demo)

`scoring_pipeline/sample_data/` ships a tiny, self-contained demo (3 instructions, 3 rendered screenshots, and the persona pool trimmed to a small subset) so you can exercise the full Stage-1→2→3 pipeline end to end without needing the full 1,000-persona pool or the 7,000-screenshot benchmark:

```bash
cd scoring_pipeline
python pipeline.py --models model_A --instruction-ids 27
```

This will:
1. Draw (or load a cached) 5-persona panel for instruction `27`.
2. Run Stage 1 (independent rating) for each panelist against `sample_data/screenshots/model_A/27.png`.
3. Run Stage 2 (opinion exchange) for each panelist.
4. Run Stage 3 (weighted aggregation) into a final score.
5. Append the full result (including every panelist's raw scores/reasons) to `outputs/scores.jsonl`.

Note: the demo screenshots only cover one model per instruction id (`27` → `model_A`, `104` → `model_C`, `131` → `model_B`); pass `--models` / `--instruction-ids` accordingly, or supply your own `sample_data/screenshots/<model>/<id>.png` files for other combinations.

## Full Pipeline Usage

### Stage 0a: Persona Pool & P-Q-A Construction

```bash
cd pqa_construction
python build_prompts.py --seed 42
python generate_answers.py --model gpt-4o --concurrency 8
python assemble_persona_qa.py
```

`build_prompts.py` samples 1 multiple-choice + 1 open-ended question per dimension (5 dimensions × 2 = 10 questions per persona) from `question_bank/`, and renders a role-play chat prompt per (persona, question) pair. `generate_answers.py` is resume-safe (skips already-answered `(persona_id, local_q_idx)` pairs) and writes incrementally so it can be safely interrupted and restarted. `assemble_persona_qa.py` merges the answers back with the structured persona pool into the final `persona_qa.json` used by the scoring pipeline's evidence retrieval.

### Stage 0b: Panel Sampling

```bash
cd scoring_pipeline
python sampling.py --panel-size 5 --seed 42
```

Draws one panel of `panel-size` personas per instruction via quota/stratified sampling (balancing demographic strata while guaranteeing at least one psychologically "close" pair and one "far" pair per panel, so Stage 2's opinion dynamics has both a homophily cluster and a disagreement axis to work with). Sampling is fully deterministic given `(instruction_id, seed)`, so the same panel is reused across every evaluated model for a given instruction.

### Stages 1-3: Scoring Pipeline

```bash
cd scoring_pipeline
python pipeline.py \
  --models model_A model_B model_C \
  --panel-size 5 \
  --api-model gpt-4o \
  --seed 42
```

`pipeline.py` orchestrates Stages 1-3 for every `(instruction, model)` pair, writing one JSON record per pair (append-only JSONL) to `outputs/scores.jsonl`. It is resumable: already-completed `(instruction_id, model)` pairs are automatically skipped on rerun.

## Ablations

Pass `--no-pqa` to disable P-Q-A evidence grounding in Stage 1, isolating the marginal effect of evidence grounding while holding persona identity, dimension definitions, image input, and decoding parameters fixed:

```bash
python pipeline.py --models model_A --no-pqa
```

## Case Studies

`cases/` contains the three worked examples referenced in the paper's qualitative analysis (Appendix), each with its generating instruction, rendered HTML, screenshot, and — for one example — the full per-persona Stage-1/Stage-2 score and reasoning trace that produced a human-vs-panel scoring gap (`case_study_scoring_example.json`).

## Data & Model Availability

- The full 1,000-persona pool, the 500-instruction UIPersonaBench benchmark, and the 7,000 rendered screenshots (14 models × 500 instructions) used in the paper's experiments are released on Hugging Face: **[link to be added]**.
- The `personas_1000_structured.json` and `sample_data/` files in this repository are a small illustrative subset intended to make the code runnable out of the box; they are **not** the full experimental dataset.
- Model names evaluated in the paper are anonymized here as `model_A` / `model_B` / `model_C` placeholders; see the paper for the actual list of 14 commercial/open-source models.


```bibtex
@article{wu2026beyond,
  title={Beyond a Single Judge: Simulating Social Persona Panels for Generative UI Evaluation},
  author={Wu, Zheng and Luo, Yibo and Zhang, Pu and Yang, Cheng and Zhang, Zhuosheng},
  journal={arXiv preprint arXiv:2607.28439},
  year={2026}
}
```
