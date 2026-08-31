"""
Real classifier for the triage graph. Wire this into triage_graph.classify().

    export ANTHROPIC_API_KEY=sk-ant-...
    pip install anthropic

Captures tokens, latency and cost on every call, because a severity call you
cannot price is a severity call you cannot govern.
"""
from __future__ import annotations
import json, os, pathlib, time
from typing import Optional

from anthropic import Anthropic

MODEL = os.environ.get("TRIAGE_MODEL", "claude-sonnet-5")

# Per-million-token rates. CHECK CURRENT PRICING and set these yourself —
# stale rates in a cost column are worse than no cost column.
RATE_IN = float(os.environ.get("RATE_IN", "0"))
RATE_OUT = float(os.environ.get("RATE_OUT", "0"))

RUBRIC_PATH = pathlib.Path("rubric.md")
_client: Optional[Anthropic] = None


def client() -> Anthropic:
    global _client
    if _client is None:
        if not os.environ.get("ANTHROPIC_API_KEY"):
            raise RuntimeError("ANTHROPIC_API_KEY is not set")
        _client = Anthropic()
    return _client


def rubric() -> str:
    if not RUBRIC_PATH.exists():
        raise RuntimeError("rubric.md missing — write your rubric before classifying")
    return RUBRIC_PATH.read_text()


SYSTEM = """You are a triage classifier. Apply the rubric below literally.

{rubric}

Return ONLY a JSON object, no prose, no code fences:
{{"priority": "low"|"medium"|"high",
  "rationale": "<=40 words, quoting the specific words in the ticket that drove the call",
  "confidence": <float 0.0-1.0>}}

Confidence rules — these matter more than the priority itself:
- Report your actual uncertainty. A flat 0.9 on everything makes the
  downstream confidence gate useless and is a governance failure, not a style choice.
- Below 0.5 when the ticket is ambiguous, the rubric does not clearly cover it,
  or key context is missing.
- The rationale must cite the ticket. If you cannot point at specific words,
  your confidence is too high."""

USER = """RUBRIC VERSION: {version}

SUBJECT: {subject}
BODY: {body}
{degraded}
{precedents}"""


def build_messages(subject, body, dq=None, precedents=None, version="v0.1"):
    deg = ""
    if dq is not None and getattr(dq, "degraded", False):
        deg = ("\nDEGRADED INPUT — missing: " + ", ".join(dq.degraded_reasons)
               + "\nScore on what is present. Lower your confidence accordingly.")
    prec = ""
    if precedents:
        lines = [f"- [{p.get('priority','?')}] {str(p.get('text',''))[:220]}"
                 for p in precedents[:5]]
        prec = ("\nSIMILAR RESOLVED TICKETS (reference only — a precedent may itself "
                "have been mislabelled; do not follow one that contradicts the rubric):\n"
                + "\n".join(lines))
    return USER.format(version=version, subject=subject or "(missing)",
                       body=body, degraded=deg, precedents=prec)


def classify(subject, body, dq=None, precedents=None, version="v0.1",
             model=MODEL, log_path="calls.jsonl") -> dict:
    """Returns priority, rationale, confidence, plus tokens/latency/cost."""
    t0 = time.time()
    resp = client().messages.create(
        model=model,
        max_tokens=300,
        temperature=0,                      # reproducibility is non-negotiable here
        system=SYSTEM.format(rubric=rubric()),
        messages=[{"role": "user",
                   "content": build_messages(subject, body, dq, precedents, version)}],
    )
    elapsed = time.time() - t0
    raw = "".join(b.text for b in resp.content if b.type == "text").strip()
    raw = raw.removeprefix("```json").removeprefix("```").removesuffix("```").strip()

    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError:
        # A malformed response is a real failure mode. Surface it, don't paper over it.
        parsed = {"priority": None, "rationale": f"UNPARSEABLE: {raw[:160]}",
                  "confidence": 0.0, "parse_error": True}

    meta = {
        "model": model,
        "rubric_version": version,
        "in_tokens": resp.usage.input_tokens,
        "out_tokens": resp.usage.output_tokens,
        "latency_s": round(elapsed, 3),
        "cost_usd": round(resp.usage.input_tokens / 1e6 * RATE_IN
                          + resp.usage.output_tokens / 1e6 * RATE_OUT, 6),
    }
    out = {**parsed, **meta}
    if log_path:
        with open(log_path, "a") as f:
            f.write(json.dumps(out) + "\n")
    return out
