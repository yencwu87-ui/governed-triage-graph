#!/usr/bin/env python3
"""
predict_mlx.py — generate triage predictions from an MLX-served model.

Scores against the SAME held-out split that eval_harness.baseline() uses,
so the LLM and the tfidf+logreg baseline are compared on identical rows.

Usage
-----
  # 1. serve the model (in the MLX venv, separate terminal)
  mlx_lm.server --model mlx-community/Meta-Llama-3.1-8B-Instruct-4bit \
                --adapter-path ./finetune/adapters --port 8080

  # 2. smoke test on 20 tickets
  python3 predict_mlx.py --limit 20 --out preds_smoke.csv

  # 3. full run (resumable — rerun the same command after a crash)
  python3 predict_mlx.py --out preds_finetuned.csv

  # 4. score it
  python3 eval_harness.py --preds preds_finetuned.csv --label "llama3.1-8b finetuned"

Parse failures are recorded as "low", which counts against the model on the
gated under_classification metric. Dropping them would silently flatter the
result by removing the cases the model handled worst.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import re
import sys
import time

import pandas as pd
import requests
from sklearn.model_selection import train_test_split

DATA = "tickets_en.csv"
LABELS = ("low", "medium", "high")
FALLBACK = "low"

DEFAULT_SYSTEM = (
    "You are an IT incident triage classifier. Read the ticket and assign a "
    "priority of exactly one of: low, medium, high. "
    'Respond with JSON only, in the form {"priority": "<label>"}. '
    "No explanation, no preamble."
)


def held_out(path: str) -> pd.DataFrame:
    """Reproduce eval_harness.baseline()'s test split exactly."""
    df = pd.read_csv(path)
    df["txt"] = df.subject.fillna("") + " \n " + df.body.fillna("")
    _, x_te, _, _ = train_test_split(
        df.txt,
        df.priority,
        test_size=0.25,
        random_state=0,
        stratify=df.priority,
    )
    return df.loc[x_te.index, ["ticket_id", "txt"]].reset_index(drop=True)


def parse_label(text: str) -> tuple[str, bool]:
    """Return (label, ok). ok=False means we fell back."""
    try:
        obj = json.loads(re.sub(r"^```(?:json)?|```$", "", text.strip(), flags=re.M))
        val = str(obj.get("priority", "")).strip().lower()
        if val in LABELS:
            return val, True
    except Exception:
        pass

    hits = re.findall(r"\b(low|medium|high)\b", text.lower())
    if hits:
        return hits[-1], True

    return FALLBACK, False


def call_model(url: str, model: str, system: str, ticket: str, timeout: int) -> str:
    r = requests.post(
        url,
        json={
            "model": model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": ticket},
            ],
            "temperature": 0.0,
            "max_tokens": 64,
        },
        timeout=timeout,
    )
    r.raise_for_status()
    return r.json()["choices"][0]["message"]["content"]


def load_done(path: str) -> set:
    if not os.path.exists(path):
        return set()
    try:
        return set(pd.read_csv(path).ticket_id.astype(str))
    except Exception:
        return set()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=DATA)
    ap.add_argument("--out", default="preds_mlx.csv")
    ap.add_argument("--url", default="http://127.0.0.1:8080/v1/chat/completions")
    ap.add_argument("--model", default="local")
    ap.add_argument("--rubric", default="rubric.md",
                    help="file whose contents become the system prompt")
    ap.add_argument("--limit", type=int, default=0, help="0 = all")
    ap.add_argument("--timeout", type=int, default=120)
    a = ap.parse_args()

    system = DEFAULT_SYSTEM
    if os.path.exists(a.rubric):
        system = open(a.rubric).read().strip() + "\n\n" + DEFAULT_SYSTEM
        print(f"  system prompt: {a.rubric} + output-shape instruction")
    else:
        print(f"  system prompt: built-in default ({a.rubric} not found)")

    rows = held_out(a.data)
    if a.limit:
        rows = rows.head(a.limit)

    done = load_done(a.out)
    todo = rows[~rows.ticket_id.astype(str).isin(done)]
    print(f"  held-out split: {len(rows)} tickets, {len(done)} already done, "
          f"{len(todo)} to go")

    if todo.empty:
        print("  nothing to do")
        return

    new_file = not os.path.exists(a.out)
    fails = 0
    t0 = time.time()

    with open(a.out, "a", newline="") as fh:
        w = csv.writer(fh)
        if new_file:
            w.writerow(["ticket_id", "priority"])

        for i, row in enumerate(todo.itertuples(), 1):
            try:
                raw = call_model(a.url, a.model, system, row.txt, a.timeout)
                label, ok = parse_label(raw)
            except Exception as e:
                print(f"  ! {row.ticket_id}: {type(e).__name__}: {e}", file=sys.stderr)
                label, ok = FALLBACK, False

            if not ok:
                fails += 1
            w.writerow([row.ticket_id, label])
            fh.flush()

            if i % 25 == 0:
                rate = i / (time.time() - t0)
                eta = (len(todo) - i) / rate / 60
                print(f"  {i}/{len(todo)}  {rate:.2f}/s  eta {eta:.0f}m  "
                      f"parse-fails {fails}")

    mins = (time.time() - t0) / 60
    print(f"\n  wrote {a.out} in {mins:.1f}m")
    print(f"  parse failures: {fails}/{len(todo)} "
          f"({fails / max(len(todo), 1):.1%}) — recorded as '{FALLBACK}'")
    print(f"\n  next: python3 eval_harness.py --preds {a.out} --label \"<name>\"")


if __name__ == "__main__":
    main()
