"""
Is the model's stated reason actually about this ticket?

A rationale is a narration the model produces alongside its answer. It is not
a trace of the computation that produced the answer, and it can be fluent,
plausible and entirely disconnected from the input. So it is not an
explanation — but it IS a testable artifact, and this is the test.

Two checks:
  1. verbatim quotes  — anything in "quotes" must appear in the ticket
  2. content overlap  — the substantive words must come from the ticket

An ungrounded rationale is a finding regardless of whether the priority
happened to be right.
"""
from __future__ import annotations
import re

STOP = set("""a an the and or but if then than that this these those is are was were
be been being do does did have has had will would can could should may might must
to of in on at by for with from as it its no not so such very more most other
some any each both all when where which who whom whose what why how i you he she
they we me him her them us our your their there here about into over under again
customer ticket issue priority request response support team user please thank
thanks dear sincerely regards high medium low urgent because due indicates
suggests appears seems given based likely rubric level assigned classification""".split())


def _words(t: str) -> list[str]:
    return re.findall(r"[a-z][a-z0-9'-]{2,}", (t or "").lower())


def check(rationale: str, subject: str | None, body: str) -> dict:
    """Returns a grounding score plus what could not be found in the ticket."""
    source = f"{subject or ''} {body or ''}".lower()
    src_words = set(_words(source))

    # 1. verbatim quotes
    quotes = re.findall(r'"([^"]{4,80})"', rationale or "")
    bad_quotes = [q for q in quotes if q.lower().strip() not in source]

    # 2. content-word overlap
    terms = [w for w in _words(rationale) if w not in STOP]
    if not terms:
        return {"score": 0.0, "n_terms": 0, "missing": [], "bad_quotes": bad_quotes,
                "verdict": "no substantive content"}

    missing = []
    for w in terms:
        # tolerate simple inflection: plural, -ing, -ed
        if not (w in src_words or any(s.startswith(w[:max(4, len(w) - 3)])
                                      for s in src_words)):
            missing.append(w)
    score = 1 - len(missing) / len(terms)

    if bad_quotes:
        verdict = "FABRICATED QUOTE"
    elif score >= 0.6:
        verdict = "grounded"
    elif score >= 0.35:
        verdict = "weak"
    else:
        verdict = "UNGROUNDED"

    return {"score": round(score, 2), "n_terms": len(terms),
            "missing": sorted(set(missing))[:8], "bad_quotes": bad_quotes,
            "verdict": verdict}


def flags(result: dict) -> list[str]:
    """Audit flags for the graph's audit node."""
    f = []
    if result["bad_quotes"]:
        f.append("rationale_quotes_text_not_in_ticket")
    if result["verdict"] == "UNGROUNDED":
        f.append("rationale_ungrounded")
    elif result["verdict"] == "no substantive content":
        f.append("rationale_empty")
    return f
