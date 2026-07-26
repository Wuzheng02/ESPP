# -*- coding: utf-8 -*-
"""Call an LLM API to generate persona questionnaire answers, concurrently.

Reads outputs/prompts.jsonl (produced by build_prompts.py), calls an
OpenAI-protocol-compatible Chat Completions endpoint (see
llm_client.LLMClient), and writes answers to outputs/answers.jsonl.

Key properties:
  - Resume-safe: already-generated (persona_id, local_q_idx) pairs are skipped.
  - 429 handling: waits and retries on rate limiting.
  - Concurrent: number of parallel requests configurable via --concurrency.
  - Thread-safe writes: a lock guards writes to the output file.
  - Periodic progress reporting.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import threading
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from common import (
    DEFAULT_ANSWERS,
    DEFAULT_ASSEMBLED,
    DEFAULT_PERSONAS,
    DEFAULT_PROMPTS,
    answer_key,
    answer_record,
    assemble_final,
    existing_keys,
    load_jsonl,
)
from llm_client import LLMClient

_write_lock = threading.Lock()
_progress_lock = threading.Lock()


def call_with_retry(llm_client: LLMClient, messages, max_retries=20, temperature=0.7, max_tokens=2048):
    for attempt in range(1, max_retries + 1):
        try:
            return llm_client._call_chat_api(
                model=llm_client.default_model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
            )
        except Exception as e:
            if attempt < max_retries:
                wait_time = min(attempt * 10, 60)
                print(f"    [retry {attempt}] request error: {e}, waiting {wait_time}s...", flush=True)
                time.sleep(wait_time)
            else:
                raise
    raise RuntimeError(f"Still failing after {max_retries} retries")


def worker(worker_id, todo_chunk, llm_client, args, fout, progress):
    for record in todo_chunk:
        pid = record["persona_id"]
        qidx = record["local_q_idx"]
        dim = record["dimension"]
        qtype = record["q_type"]

        print(f"[worker-{worker_id}] persona={pid} Q{qidx} ({dim}/{qtype}) ...", flush=True)

        try:
            answer_text = call_with_retry(
                llm_client,
                record["messages"],
                max_retries=args.max_retries,
                temperature=args.temperature,
                max_tokens=args.max_tokens,
            )
            rec = answer_record(record, answer_text)
            with _write_lock:
                fout.write(json.dumps(rec, ensure_ascii=False) + "\n")
                fout.flush()
                os.fsync(fout.fileno())
                progress["completed"] += 1
                progress["success"] += 1
            with _progress_lock:
                completed = progress["completed"]
            print(f"  [worker-{worker_id}] ok | completed {completed}/{progress['total']}", flush=True)
        except Exception as e:
            print(f"  [worker-{worker_id}] failed: {e}", flush=True)
            rec = answer_record(record, f"[ERROR] {e}")
            with _write_lock:
                fout.write(json.dumps(rec, ensure_ascii=False) + "\n")
                fout.flush()
                os.fsync(fout.fileno())
                progress["completed"] += 1
                progress["fail"] += 1


def run(args):
    prompts = load_jsonl(args.prompts)
    total = len(prompts)
    print(f"[info] loaded {total} prompt records from {args.prompts}")

    done = existing_keys(args.answers)
    if done:
        print(f"[info] resuming: {len(done)} answers already exist in {args.answers}")
    todo = [p for p in prompts if answer_key(p) not in done]
    remaining = len(todo)
    print(f"[info] {remaining} remaining to generate")

    if not todo:
        print("[info] nothing to generate, assembling final dataset...")
        n = assemble_final(args.answers, args.personas, args.assembled)
        print(f"[info] assembled {n} personas -> {args.assembled}")
        return

    llm_client = LLMClient(default_model=args.model, timeout=args.timeout)
    print(f"[info] model: {args.model}, concurrency: {args.concurrency}")

    args.answers.parent.mkdir(parents=True, exist_ok=True)
    progress = {"completed": len(done), "success": len(done), "fail": 0, "total": total}

    chunks = [[] for _ in range(args.concurrency)]
    for i, record in enumerate(todo):
        chunks[i % args.concurrency].append(record)

    fout = open(args.answers, "a", encoding="utf-8")
    try:
        threads = []
        for wid in range(args.concurrency):
            t = threading.Thread(target=worker, args=(wid, chunks[wid], llm_client, args, fout, progress), daemon=True)
            threads.append(t)
            t.start()
        for t in threads:
            t.join()
    finally:
        fout.close()

    print(f"[done] {progress['success'] - len(done)} newly succeeded, {progress['fail']} failed this run")

    print("[info] assembling final dataset...")
    n = assemble_final(args.answers, args.personas, args.assembled)
    print(f"[info] assembled {n} personas -> {args.assembled}")


def main():
    ap = argparse.ArgumentParser(description="Generate persona questionnaire answers via an LLM API.")
    ap.add_argument("--prompts", type=Path, default=DEFAULT_PROMPTS)
    ap.add_argument("--answers", type=Path, default=DEFAULT_ANSWERS)
    ap.add_argument("--assembled", type=Path, default=DEFAULT_ASSEMBLED)
    ap.add_argument("--personas", type=Path, default=DEFAULT_PERSONAS)
    ap.add_argument("--model", type=str, default="gpt-4o", help="Model used to answer the questionnaire.")
    ap.add_argument("--timeout", type=int, default=600)
    ap.add_argument("--max-tokens", type=int, default=2048)
    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument("--max-retries", type=int, default=20)
    ap.add_argument("--concurrency", type=int, default=4, help="Number of concurrent API calls.")
    args = ap.parse_args()

    try:
        run(args)
    except KeyboardInterrupt:
        print("\n[interrupted] stopped manually; rerun this command to resume.", file=sys.stderr, flush=True)
        raise


if __name__ == "__main__":
    main()
