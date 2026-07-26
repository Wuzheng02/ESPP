# -*- coding: utf-8 -*-
"""Assemble answers.jsonl into the final persona_qa.json dataset."""

from __future__ import annotations

import argparse
from pathlib import Path

from common import DEFAULT_ANSWERS, DEFAULT_ASSEMBLED, DEFAULT_PERSONAS, assemble_final


def main() -> None:
    ap = argparse.ArgumentParser(description="Assemble persona_qa.json")
    ap.add_argument("--answers", type=str, default=str(DEFAULT_ANSWERS))
    ap.add_argument("--personas", type=str, default=str(DEFAULT_PERSONAS))
    ap.add_argument("--out", type=str, default=str(DEFAULT_ASSEMBLED))
    args = ap.parse_args()

    n = assemble_final(
        answers_path=Path(args.answers),
        personas_path=Path(args.personas),
        out_path=Path(args.out),
    )
    print(f"Assembled {n} personas into {args.out}")


if __name__ == "__main__":
    main()
