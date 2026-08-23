"""
Local classifier via Ollama. Same interface as classifier.py — no API key,
no cost, nothing leaves the machine.

    ollama serve            # if it isn't already running
    ollama list             # see what you have
    export TRIAGE_MODEL=llama3.1:8b

Drop-in: anywhere the code says `import classifier`, use
`import classifier_local as classifier` instead.
"""
from __future__ import annotations
import json, os, pathlib, time, urllib.error, urllib.request

HOST = os.environ.get("OLLAMA_HOST", "http://localhost:11434")
MODEL = os.environ.get("TRIAGE_MODEL", "llama3.1:8b")
RUBRIC_PATH = pathlib.Path("rubric.md")


def rubric() -> str:
    if not RUBRIC_PATH.exists():
        raise RuntimeError("rubric.md missing — write your rubric before classifying")
    return RUBRIC_PATH.read_text()


SYSTEM = """You are a triage classifier. Apply the rubric below literally.

{rubric}

Reply with a single JSON object and nothing else:
{{"priority": "low" or "medium" or "high",
  "rationale": "under 40 words, quoting the specific words in the ticket that decided it",
  "confidence": a number between 0.0 and 1.0}}

Confidence matters more than the priority. Report real uncertainty — the same
number on every ticket makes the downstream gate useless. Go below 0.5 when the
ticket is ambiguous or the rubric does not clearly cover it."""


def health() -> tuple[bool, str]:
    """Is Ollama up, and is the model pulled?"""
    try:
        with urllib.request.urlopen(f"{HOST}/api/tags", timeout=5) as r:
            tags = json.load(r)
    except urllib.error.URLError as e:
        return False, f"Ollama not reachable at {HOST} ({e.reason}). Run: ollama serve"
    names = [m["name"] for m in tags.get("models", [])]
    if not names:
        return False, "Ollama is running but has no models. Run: ollama pull llama3.1:8b"
    if MODEL not in names and MODEL.split(":")[0] not in [n.split(":")[0] for n in names]:
        return False, f"Model '{MODEL}' not found. Available: {', '.join(names)}"
    return True, f"Ollama up at {HOST}, using {MODEL}"


def classify(subject=None, body="", dq=None, precedents=None, version="v0.1",
             model=None, log_path="calls.jsonl", api_key=None, rubric_text=None) -> dict:
    model = model or MODEL
    deg = ""
    if dq is not None and getattr(dq, "degraded", False):
        deg = ("\nDEGRADED INPUT — missing: " + ", ".join(dq.degraded_reasons)
               + "\nScore on what is present. Lower your confidence accordingly.")
    prompt = (f"RUBRIC VERSION: {version}\n\nSUBJECT: {subject or '(missing)'}\n"
              f"BODY: {body}{deg}")

    payload = json.dumps({
        "model": model,
        "format": "json",              # Ollama constrains output to valid JSON
        "stream": False,
        "options": {"temperature": 0},  # reproducibility
        "messages": [
            {"role": "system", "content": SYSTEM.format(rubric=rubric_text or rubric())},
            {"role": "user", "content": prompt},
        ],
    }).encode()

    t0 = time.time()
    req = urllib.request.Request(f"{HOST}/api/chat", data=payload,
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=180) as r:
        data = json.load(r)
    elapsed = time.time() - t0

    raw = data.get("message", {}).get("content", "").strip()
    try:
        parsed = json.loads(raw)
        if parsed.get("priority") not in ("low", "medium", "high"):
            raise ValueError(f"priority not in enum: {parsed.get('priority')!r}")
        parsed["confidence"] = float(parsed.get("confidence", 0))
    except (json.JSONDecodeError, ValueError, TypeError) as e:
        parsed = {"priority": None, "rationale": f"UNPARSEABLE ({e}): {raw[:140]}",
                  "confidence": 0.0, "parse_error": True}

    out = {**parsed, "model": model, "rubric_version": version,
           "in_tokens": data.get("prompt_eval_count", 0),
           "out_tokens": data.get("eval_count", 0),
           "latency_s": round(elapsed, 3), "cost_usd": 0.0, "local": True}
    if log_path:
        with open(log_path, "a") as f:
            f.write(json.dumps(out) + "\n")
    return out
