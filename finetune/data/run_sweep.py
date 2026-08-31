#!/usr/bin/env python3
"""
Driver for the temperature sweep against a local Ollama model.

    python3 run_sweep.py --quick      # 3 temps x 3 repeats, for a first look
    python3 run_sweep.py              # full 6 x 10 grid

Two things to adapt before first run — both marked ADAPT below:
  1. load_golden_set()  — point at your existing golden set
  2. classify()         — swap in your existing prompt and call
"""

import argparse
import json
import re
import sys
import time
from pathlib import Path

from temperature_sweep import (
    GoldenItem, SweepConfig, run_sweep, write_report, dump_raw, SEVERITY_ORDER
)

MODEL = "llama3.1:8b"
OLLAMA_URL = "http://localhost:11434/api/chat"


# ---------------------------------------------------------------- ADAPT (1)
def load_golden_set(path: Path) -> list[GoldenItem]:
    """Expects JSON: [{"id": "...", "text": "...", "truth": "P2"}, ...]

    If your golden set lives elsewhere or uses different field names, this is
    the only place that needs to change.
    """
    raw = json.loads(path.read_text())
    items = [GoldenItem(str(r["id"]), r["text"], r["truth"]) for r in raw]

    unknown = {i.truth for i in items} - set(SEVERITY_ORDER)
    if unknown:
        sys.exit(f"Labels not in SEVERITY_ORDER: {sorted(unknown)}. "
                 f"Fix SEVERITY_ORDER in temperature_sweep.py to match your rubric.")
    return items


# ---------------------------------------------------------------- ADAPT (2)
PROMPT = """Classify the severity of this incident ticket.
Respond with exactly one label and nothing else: {labels}

Ticket:
{text}

Severity:"""


def classify(text: str, *, temperature: float, seed: int) -> str:
    """Replace the body with your existing call if you already have one.

    The only contract: accept temperature and seed, return a label string.
    """
    import httpx  # or `requests` / the `ollama` package — whatever you use

    r = httpx.post(
        OLLAMA_URL,
        json={
            "model": MODEL,
            "messages": [{
                "role": "user",
                "content": PROMPT.format(labels=", ".join(SEVERITY_ORDER), text=text),
            }],
            "stream": False,
            "options": {
                "temperature": temperature,
                "seed": seed,
                "top_p": 0.95,
                # top_k 1 at temp 0 for genuinely greedy decoding — without it
                # Ollama still samples and your T=0 consistency will not be 1.0
                **({"top_k": 1} if temperature == 0.0 else {}),
            },
        },
        timeout=120,
    )
    r.raise_for_status()
    return extract_label(r.json()["message"]["content"])


def extract_label(raw: str) -> str:
    """Strip reasoning blocks and pull the first valid label.

    Returns the raw text if nothing matches, so score_run counts it as
    unparseable rather than silently guessing.
    """
    body = raw.split("</think>", 1)[-1]
    m = re.search(r"\b(" + "|".join(map(re.escape, SEVERITY_ORDER)) + r")\b", body)
    return m.group(1) if m else raw.strip()[:40]


# --------------------------------------------------------------------------
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--golden", type=Path, default=Path("data/golden_set.json"))
    ap.add_argument("--out", type=Path, default=Path("reports"))
    ap.add_argument("--quick", action="store_true",
                    help="3 temperatures x 3 repeats instead of the full grid")
    args = ap.parse_args()

    items = load_golden_set(args.golden)
    cfg = (SweepConfig(temperatures=(0.0, 0.5, 1.0), repeats=3)
           if args.quick else SweepConfig())

    calls = len(items) * len(cfg.temperatures) * cfg.repeats
    print(f"{len(items)} items x {len(cfg.temperatures)} temps x {cfg.repeats} repeats "
          f"= {calls} inference calls")
    print(f"At ~3s/call that is roughly {calls * 3 / 60:.0f} minutes.\n")

    # Fail fast on a single call rather than 40 minutes in.
    print("Probe call... ", end="", flush=True)
    t0 = time.time()
    probe = classify(items[0].text, temperature=0.0, seed=0)
    dt = time.time() - t0
    print(f"{dt:.1f}s -> {probe!r}")
    if probe not in SEVERITY_ORDER:
        print("\nWARNING: probe did not return a valid label. Every item will "
              "score as unparseable. Fix the prompt before running the sweep.")
        sys.exit(1)
    print(f"Revised estimate: {calls * dt / 60:.0f} minutes.\n")

    started = time.time()
    points = run_sweep(classify, items, cfg, model_name=MODEL)
    print(f"Sweep finished in {(time.time() - started) / 60:.1f} minutes.\n")

    md = write_report(points, args.out / "temperature_sweep.md")
    raw = dump_raw(points, args.out / "temperature_sweep_raw.json")

    for p in points:
        print(f"  T={p.provenance.temperature:.1f}  "
              f"exact {p.exact_mean:.1%}  "
              f"under(worst) {p.under_worst:.1%}  "
              f"consistency {p.consistency:.1%}  "
              f"{'PASS' if p.passes else 'FAIL'}")

    print(f"\nReport:   {md}")
    print(f"Evidence: {raw}")


if __name__ == "__main__":
    main()
