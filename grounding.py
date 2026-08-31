"""
Is the model's stated reason actually about this ticket?

v2. The v1 check scored lexical overlap between the rationale and the ticket
and called the result "grounding". It penalised correct paraphrase — a
rationale saying "the customer cannot complete a purchase" about a ticket
saying "checkout returns 502" scored zero — so `weak` was close to the default
verdict regardless of model quality. Any signal derived from it would have
taught a model to copy ticket vocabulary rather than to reason.

v2 separates two claims that v1 conflated:

  ANCHORS  — things that must appear literally: error codes, product and
             system names, ticket references, quantities, dates. If the
             rationale invents one, that is fabrication and is decisive.

  SEMANTIC — whether the rationale is about the same subject matter, measured
             by embedding similarity, which survives paraphrase. Falls back to
             a widened lexical check if sentence-transformers is absent, and
             SAYS SO in the result rather than silently changing meaning.

Rubric vocabulary (blocked, impeded, urgent, priority...) is excluded from
both. Those words come from the rubric, not the ticket, and scoring them
measured prompt compliance while labelling it grounding.

Thresholds are NOT set here. Derive them from your gold set with
calibrate_grounding.py and pass them in. A cutoff nobody derived is a cutoff
nobody can defend.
"""
from __future__ import annotations
import json, os, re, urllib.request
from dataclasses import dataclass
from functools import lru_cache

# Words that come from the rubric or from letter-writing convention, not from
# the ticket. Scoring these measured instruction-following, not grounding.
RUBRIC_VOCAB = set("""priority severity urgent urgency critical high medium low
blocked blocking impeded impediment workaround deadline escalate escalation
customer client user ticket issue request enquiry inquiry case incident
indicates suggests appears seems implies reflects consistent classification
classify rated rating assessed assessment because therefore thus hence given
based rubric level category dear sincerely regards hello hi thanks thank you
please kindly team support response reply""".split())

STOP = set("""a an the and or but if then than that this these those is are was were
be been being do does did have has had having will would shall should can could
may might must to of in on at by for with from as it its no not so such very more
most other some any each both all when where which who whom whose what why how
i you he she they we me him her them us our your their there here about into over
under again their they've it's we're""".split())

# Things that must appear verbatim if cited: error codes, versions, refs, IDs.
ANCHOR_RE = re.compile(
    r"""(?:
        \b\d{3}\b                    # HTTP-ish codes
      | \b[A-Z]{2,}[-_]?\d{2,}\b     # TK00042, ERR-500, INC1234
      | \bv?\d+\.\d+(?:\.\d+)?\b     # versions
      | \b\d+(?:[.,]\d+)?\s*(?:%|ms|s|sec|min|hour|hr|day|GB|MB|kb)\b
      | \$\s?\d[\d,.]*               # amounts
      | \b\d{1,2}[:/]\d{2}\b         # times / partial dates
    )""", re.X)


@dataclass
class Bands:
    """Grounded/weak cutoffs. Defaults are PLACEHOLDERS — calibrate them."""
    grounded: float = 0.60
    weak: float = 0.35
    derived: bool = False           # set True only by calibrate_grounding.py

    def label(self, score: float) -> str:
        if score >= self.grounded:
            return "grounded"
        if score >= self.weak:
            return "weak"
        return "UNGROUNDED"


OLLAMA = os.environ.get("OLLAMA_HOST", "http://localhost:11434")
EMBED_MODEL = os.environ.get("GROUNDING_EMBED", "nomic-embed-text")


@lru_cache(maxsize=1)
def _backend() -> str:
    """Ollama first — no extra model stack, stays local. Then
    sentence-transformers. Then lexical, which is clearly labelled as a
    fallback because it does not mean the same thing."""
    try:
        with urllib.request.urlopen(f"{OLLAMA}/api/tags", timeout=3) as r:
            names = [m["name"].split(":")[0] for m in json.load(r).get("models", [])]
        if EMBED_MODEL.split(":")[0] in names:
            return "ollama"
    except Exception:
        pass
    try:
        from sentence_transformers import SentenceTransformer  # noqa: F401
        _st()
        return "st"
    except Exception:
        return "lexical"


@lru_cache(maxsize=1)
def _st():
    from sentence_transformers import SentenceTransformer
    return SentenceTransformer("all-MiniLM-L6-v2")


def _embed(texts: list[str]):
    import numpy as np
    if _backend() == "ollama":
        out = []
        for t in texts:
            payload = json.dumps({"model": EMBED_MODEL, "prompt": t}).encode()
            req = urllib.request.Request(f"{OLLAMA}/api/embeddings", data=payload,
                                         headers={"Content-Type": "application/json"})
            with urllib.request.urlopen(req, timeout=60) as r:
                v = np.array(json.load(r)["embedding"], dtype=float)
            out.append(v / (np.linalg.norm(v) + 1e-9))
        return out
    return list(_st().encode(texts, normalize_embeddings=True))


def _content(t: str) -> list[str]:
    ws = re.findall(r"[a-z][a-z0-9'-]{2,}", (t or "").lower())
    return [w for w in ws if w not in STOP and w not in RUBRIC_VOCAB]


def _anchors(t: str) -> set[str]:
    return {a.lower().strip() for a in ANCHOR_RE.findall(t or "")}


def _lexical(rationale: str, source: str) -> float:
    """Fallback only. Widened: stems to 4 chars and ignores rubric vocab."""
    terms = _content(rationale)
    if not terms:
        return 0.0
    src = set(_content(source))
    hit = sum(1 for w in terms
              if w in src or any(s[:4] == w[:4] for s in src))
    return hit / len(terms)


def check(rationale: str, subject: str | None, body: str,
          bands: Bands | None = None) -> dict:
    bands = bands or Bands()
    source = f"{subject or ''} {body or ''}"
    rationale = rationale or ""

    # --- 1. anchors: decisive, not scored ---
    cited = _anchors(rationale)
    src_anchors = _anchors(source)
    fabricated = sorted(cited - src_anchors)

    quotes = re.findall(r'"([^"]{4,80})"', rationale)
    bad_quotes = [q for q in quotes if q.lower().strip() not in source.lower()]

    # --- 2. semantic similarity, paraphrase-tolerant ---
    method = _backend()
    if method != "lexical" and rationale.strip() and source.strip():
        try:
            import numpy as np
            a, b = _embed([rationale, source])
            score = float(np.dot(a, b))
            method = f"embedding:{method}"
        except Exception:
            score, method = _lexical(rationale, source), "lexical-fallback"
    else:
        score, method = _lexical(rationale, source), "lexical-fallback"

    terms = _content(rationale)
    if not terms:
        return {"score": 0.0, "method": method, "verdict": "no substantive content",
                "fabricated_anchors": fabricated, "bad_quotes": bad_quotes,
                "bands_derived": bands.derived}

    verdict = ("FABRICATED" if (fabricated or bad_quotes) else bands.label(score))

    return {"score": round(score, 3), "method": method, "verdict": verdict,
            "n_terms": len(terms), "fabricated_anchors": fabricated,
            "bad_quotes": bad_quotes, "bands_derived": bands.derived}


def flags(result: dict) -> list[str]:
    f = []
    if result.get("fabricated_anchors"):
        f.append("rationale_cites_absent_identifier")
    if result.get("bad_quotes"):
        f.append("rationale_quotes_text_not_in_ticket")
    if result.get("verdict") == "UNGROUNDED":
        f.append("rationale_ungrounded")
    elif result.get("verdict") == "no substantive content":
        f.append("rationale_empty")
    if not result.get("bands_derived"):
        f.append("grounding_bands_not_calibrated")
    if str(result.get("method", "")).startswith("lexical"):
        f.append("grounding_lexical_fallback_in_use")
    return f
