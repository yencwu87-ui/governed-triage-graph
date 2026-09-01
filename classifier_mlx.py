"""
MLX backend — a locally served, optionally LoRA-adapted model.

Start the server first, in an environment with mlx-lm installed:

    mlx_lm.server --model mlx-community/Meta-Llama-3.1-8B-Instruct-4bit \
                  --adapter-path ./finetune/adapters --port 8080

The server holds one adapter, fixed at startup. This module cannot detect
which adapter (if any) is loaded — /v1/models reports the base model either
way. Set MLX_ADAPTER to declare it for the health message:

    export MLX_ADAPTER="lora 500it"

That is a declaration, not a verification. Treat it accordingly.
"""
from __future__ import annotations

import json
import os
import re
import time
import urllib.error
import urllib.request

HOST = os.environ.get("MLX_HOST", "http://127.0.0.1:8080")
MODEL = os.environ.get("MLX_MODEL", "mlx-community/Meta-Llama-3.1-8B-Instruct-4bit")
ADAPTER = os.environ.get("MLX_ADAPTER", "")

LABELS = ("low", "medium", "high")

SYSTEM = (
    "You are an IT incident triage classifier. Read the ticket and assign a "
    "priority of exactly one of: low, medium, high. "
    'Respond with JSON only: {"priority": "<label>", "confidence": <0-1>, '
    '"rationale": "<one sentence>"}. No preamble, no code fences.'
)


def _rubric() -> str:
    """rubric.md is the prompt's source of truth. Fall back if absent."""
    try:
        with open("rubric.md") as fh:
            return fh.read().strip() + "\n\n" + SYSTEM
    except OSError:
        return SYSTEM


def health() -> tuple[bool, str]:
    try:
        with urllib.request.urlopen(f"{HOST}/v1/models", timeout=5) as r:
            data = json.loads(r.read())
    except urllib.error.URLError as e:
        return False, (f"MLX server not reachable at {HOST} ({e.reason}). "
                       f"Run: mlx_lm.server --model {MODEL} --port 8080")
    except Exception as e:
        return False, f"MLX server error at {HOST} — {type(e).__name__}: {e}"

    served = (data.get("data") or [{}])[0].get("id", "")
    if not served:
        return False, f"MLX server up at {HOST} but reports no model"

    note = f", adapter declared: {ADAPTER}" if ADAPTER else ", adapter status unknown"
    return True, f"MLX up at {HOST}, serving {served}{note}"


def _parse(raw: str) -> dict:
    """Extract the label. Returns parse_error=True if nothing usable came back."""
    cleaned = re.sub(r"^```(?:json)?|```$", "", raw.strip(), flags=re.M).strip()
    cleaned = re.sub(r"<\|[a-z_]+\|>", "", cleaned).strip()

    try:
        obj = json.loads(cleaned)
        pri = str(obj.get("priority", "")).strip().lower()
        if pri in LABELS:
            conf = obj.get("confidence", 0.0)
            try:
                conf = float(conf)
            except (TypeError, ValueError):
                conf = 0.0
            return {
                "priority": pri,
                "confidence": max(0.0, min(1.0, conf)),
                "rationale": str(obj.get("rationale", "")).strip(),
                "parse_error": False,
            }
    except (json.JSONDecodeError, AttributeError):
        pass

    # JSON failed. Fall back to a bare label mention, flagged as degraded.
    hits = re.findall(r"\b(low|medium|high)\b", cleaned.lower())
    if hits:
        return {
            "priority": hits[-1],
            "confidence": 0.0,
            "rationale": "recovered from unstructured output",
            "parse_error": True,
        }

    return {
        "priority": "",
        "confidence": 0.0,
        "rationale": f"unparseable response: {cleaned[:200]}",
        "parse_error": True,
    }


def classify(subject: str | None = None, body: str = "",
             dq=None, precedents=None, timeout: int = 120) -> dict:
    """Classify one ticket. Returns the same shape as the other backends."""
    ticket = f"{subject or ''}\n\n{body}".strip()

    system = _rubric()
    if precedents:
        system += "\n\nSimilar past tickets:\n" + str(precedents)

    payload = json.dumps({
        "model": MODEL,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": ticket},
        ],
        "temperature": 0.0,
        "max_tokens": 256,
    }).encode()

    req = urllib.request.Request(
        f"{HOST}/v1/chat/completions", data=payload,
        headers={"Content-Type": "application/json"},
    )

    t0 = time.time()
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            data = json.loads(r.read())
    except Exception as e:
        return {
            "priority": "",
            "confidence": 0.0,
            "rationale": f"{type(e).__name__}: {e}",
            "parse_error": True,
            #"latency_ms": int((time.time() - t0) * 1000),
            #"tokens_in": 0,
            #"tokens_out": 0,
            "latency_s": round(time.time() - t0, 3),
            "in_tokens": 0,
            "out_tokens": 0,
            "cost_usd": 0.0,
            "model": MODEL,
            "adapter": ADAPTER,
        }

    #latency_ms = int((time.time() - t0) * 1000)
    latency_s = round(time.time() - t0, 3)
    raw = (data.get("choices") or [{}])[0].get("message", {}).get("content", "")
    usage = data.get("usage", {})

    out = _parse(raw)
    out.update({
        #"latency_ms": latency_ms,
        #"tokens_in": usage.get("prompt_tokens", 0),
        #"tokens_out": usage.get("completion_tokens", 0),
        "latency_s": latency_s,
        "in_tokens": usage.get("prompt_tokens", 0),
        "out_tokens": usage.get("completion_tokens", 0),
        "cost_usd": 0.0,          # local inference — compute cost, no API spend
        "model": MODEL,
        "adapter": ADAPTER,
        "raw": raw,
    })
    return out


if __name__ == "__main__":
    ok, msg = health()
    print(msg)
    if ok:
        r = classify(subject="Database cluster unreachable",
                     body="All application nodes are failing to connect. "
                          "Customer-facing checkout is down.")
        print(json.dumps({k: v for k, v in r.items() if k != "raw"}, indent=2))
