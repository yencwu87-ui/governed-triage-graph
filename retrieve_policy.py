"""
Policy retrieval channel for the `retrieve` node — backed by a graphify graph.

WHAT THIS IS NOT. This is not precedent retrieval. Precedents are similar
resolved tickets, and similarity over ticket text is an embedding problem —
`grounding.py` already carries that stack (nomic-embed-text via Ollama, with
a labelled lexical fallback). Graphify does not index CSV rows and cannot
answer "which past tickets look like this one". Pointing it at a ticket
corpus means exploding every row into its own markdown file and paying an
LLM semantic pass per row. Do not do that.

WHAT THIS IS. A second, separate channel: given a ticket, which *rubric
clauses, runbook sections and control documents* govern it. That corpus is
small, stable, written in prose, and genuinely relational — which is what a
graph is for and what an embedding index is bad at.

The two channels stay separate all the way into state, because they fail
differently. A wrong precedent teaches the model the wrong label. A wrong
policy hit gives a correct label the wrong justification. Conflating them
in one `precedents` list makes the second failure invisible.

PROVENANCE IS THE POINT. Every graphify edge carries a confidence tag —
EXTRACTED (read literally out of the source) or INFERRED (resolved by
graphify). This module refuses to traverse INFERRED edges by default. An
inferred edge is graphify's guess about what connects to what; letting one
pull a runbook into a severity decision means the model was steered by
something no human wrote. Set `allow_inferred=True` if you want it, and
every hit that used one comes back flagged.

WHAT THE PROVENANCE GATE DOES NOT COVER. It governs traversal only. Seeding
is literal overlap between the ticket and a node label, and a seed has no
edge, so `allow_inferred` has no purchase on it. `test_policy_channel.py`
found this: a P3 ticket seeded straight onto a SEV-1 clause on the single
shared word "affected", at hop 0, tagged EXTRACTED. Refusing inferred edges
did nothing, because no edge was involved. These are two independent hazards
and only one of them is closed. The open one is declared, not hidden — see
`policy_hit_single_term_seed`.

No thresholds are set here that you have not derived. `MIN_SCORE` is a
placeholder and says so in the output until you calibrate it. Note that the
seeding hazard above IS the calibration problem wearing a different hat: the
defensible fix is a derived floor, not a number picked to make a test green.
"""

from __future__ import annotations

import json
import re
import subprocess
from dataclasses import dataclass, field
from pathlib import Path

# Placeholder. Derive it the way calibrate_grounding.py derives the grounding
# bands — from hand-checked hits on your own gold set — then set `derived`.
MIN_SCORE = 0.15
SCORE_DERIVED = False

STOP = set(
    """a an the and or but if then than that this these those is are was were be
been being do does did have has had having will would shall should can could may
might must to of in on at by for with from as it its no not so such very more
most other some any each both all when where which who whom whose what why how i
you he she they we me him her them us our your their there here about into over
under again""".split()
)


def _terms(text: str) -> list[str]:
    ws = re.findall(r"[a-z][a-z0-9'-]{2,}", (text or "").lower())
    return [w for w in ws if w not in STOP]


@dataclass
class PolicyHit:
    """One governing document fragment, with everything needed to audit it."""

    label: str
    source_file: str
    source_location: str
    score: float
    hops: int  # 0 = matched the ticket directly; >0 = reached by traversal
    via: list[str] = field(default_factory=list)  # relation names walked
    confidence: str = "EXTRACTED"  # weakest tag on the path taken
    seed_terms: list[str] = field(default_factory=list)  # words the seed matched on

    def as_dict(self) -> dict:
        return {
            "label": self.label,
            "source": f"{self.source_file}:{self.source_location}",
            "score": round(self.score, 3),
            "hops": self.hops,
            "via": self.via,
            "confidence": self.confidence,
            "seed_terms": self.seed_terms,
        }


class PolicyIndex:
    """
    Loads a graphify graph.json once and answers ticket-shaped questions
    against it.

    Build the index with:

        graphify extract ./policy --backend ollama --no-cluster
        # ./policy holds rubric.md, runbooks, control docs — prose, not CSV

    Ollama keeps the semantic pass local, which matters because the rubric
    and any internal runbooks are not documents to hand to a hosted API.
    """

    def __init__(self, graph_path: str | Path = "policy-out/graph.json", *,
                 repo_root: str | Path = "."):
        self.graph_path = Path(graph_path)
        self.repo_root = Path(repo_root)
        self._nodes: dict[str, dict] = {}
        self._adj: dict[str, list[tuple[str, str, str]]] = {}
        self._built_at_commit: str | None = None
        self.load_error: str | None = None
        self._load()

    def _load(self) -> None:
        try:
            g = json.loads(self.graph_path.read_text())
        except Exception as exc:
            self.load_error = f"{type(exc).__name__}: {exc}"
            return
        self._built_at_commit = g.get("built_at_commit")
        for n in g.get("nodes", []):
            self._nodes[n["id"]] = n
        for e in g.get("links", []):
            s, t = e.get("source"), e.get("target")
            rel = e.get("relation", "?")
            conf = e.get("confidence", "AMBIGUOUS")
            self._adj.setdefault(s, []).append((t, rel, conf))
            self._adj.setdefault(t, []).append((s, rel, conf))

    # -------------------------------------------------------------- staleness

    def stale_against_head(self) -> bool | None:
        """
        True when the graph was built from a different commit than HEAD.

        A retrieval index that lags the thing it indexes is a governance
        problem, not a performance one — it means the rubric clause the model
        was shown may not be the rubric clause in force. Returns None when it
        cannot be determined, which is not the same as False.
        """
        if not self._built_at_commit:
            return None
        try:
            head = subprocess.run(
                ["git", "rev-parse", "HEAD"], cwd=self.repo_root,
                capture_output=True, text=True, timeout=5,
            ).stdout.strip()
        except Exception:
            return None
        if not head:
            return None
        return not head.startswith(self._built_at_commit)

    # ---------------------------------------------------------------- lookup

    def lookup(self, subject: str | None, body: str, *, k: int = 4,
               depth: int = 2, allow_inferred: bool = False) -> list[PolicyHit]:
        """
        Seed on literal term overlap with the ticket, then walk `depth` hops.

        Returns [] rather than low-scoring filler — an empty list is
        information, a bad policy hit is contamination. Same contract the
        retrieve() docstring sets for precedents.
        """
        if self.load_error:
            return []

        allowed = {"EXTRACTED"} | ({"INFERRED"} if allow_inferred else set())
        qt = set(_terms(f"{subject or ''} {body or ''}"))
        if not qt:
            return []

        # --- seeds: literal overlap between ticket and node label ---
        #
        # NOTE: seeding is NOT governed by the provenance gate. A seed has no
        # edge, so `allow_inferred` cannot apply to it. A clause reached on one
        # shared word enters at hop 0 tagged EXTRACTED, and the refusal above
        # does nothing about it. That is a second, independent hazard, and the
        # matched terms are recorded so `flags()` can surface it.
        seeds: dict[str, tuple[float, list[str]]] = {}
        for nid, n in self._nodes.items():
            lt = set(_terms(n.get("label", "")))
            if not lt:
                continue
            shared = qt & lt
            overlap = len(shared) / len(lt)
            if overlap >= MIN_SCORE:
                seeds[nid] = (overlap, sorted(shared))
        if not seeds:
            return []

        # --- bounded walk, carrying the weakest tag seen on the path ---
        best: dict[str, PolicyHit] = {}
        frontier = [
            (nid, sc, 0, [], "EXTRACTED", st) for nid, (sc, st) in seeds.items()
        ]
        seen = set(seeds)
        while frontier:
            nid, sc, hops, via, conf, seed_terms = frontier.pop(0)
            n = self._nodes.get(nid, {})
            hit = PolicyHit(
                label=n.get("label", nid),
                source_file=n.get("source_file", "?"),
                source_location=n.get("source_location", "?"),
                score=sc, hops=hops, via=list(via), confidence=conf,
                seed_terms=list(seed_terms),
            )
            prev = best.get(nid)
            if prev is None or hit.score > prev.score:
                best[nid] = hit
            if hops >= depth:
                continue
            for tgt, rel, econf in self._adj.get(nid, []):
                if econf not in allowed or tgt in seen:
                    continue
                seen.add(tgt)
                # decay per hop: a clause two hops out governs less directly
                frontier.append(
                    (tgt, sc * 0.5, hops + 1, via + [rel],
                     "INFERRED" if econf == "INFERRED" else conf, seed_terms)
                )

        ranked = sorted(best.values(), key=lambda h: (-h.score, h.hops))
        return ranked[:k]

    # ----------------------------------------------------------------- flags

    def flags(self, hits: list[PolicyHit]) -> list[str]:
        """Audit flags, in the same shape grounding.flags() returns."""
        f: list[str] = []
        if self.load_error:
            f.append("policy_index_unavailable")
            return f
        if not hits:
            f.append("policy_no_governing_clause_found")
        if any(h.confidence == "INFERRED" for h in hits):
            f.append("policy_hit_via_inferred_edge")
        if any(h.hops > 0 for h in hits):
            f.append("policy_hit_indirect")
        if any(h.hops == 0 and len(h.seed_terms) < 2 for h in hits):
            # One shared word is not evidence that a clause governs a ticket.
            # Until MIN_SCORE is calibrated this cannot be excluded honestly,
            # so it is declared instead.
            f.append("policy_hit_single_term_seed")
        stale = self.stale_against_head()
        if stale is True:
            f.append("policy_index_stale_vs_head")
        elif stale is None:
            f.append("policy_index_freshness_unknown")
        if not SCORE_DERIVED:
            f.append("policy_score_floor_not_calibrated")
        return f


# --------------------------------------------------------- drop-in for retrieve()

_INDEX: PolicyIndex | None = None


def index() -> PolicyIndex:
    global _INDEX
    if _INDEX is None:
        _INDEX = PolicyIndex()
    return _INDEX


def retrieve(state: dict) -> dict:
    """
    Drop-in body for `retrieve()` in triage_graph.py.

    Reads: subject, body.
    Writes: precedents (your embedding channel — still yours to write),
            policy, policy_provenance, audit_flags.

    Must not: let a policy hit change `precedent_depth`. That field is set at
    ingress because gate_a reads it, and gate_a runs on the precedent channel.
    A policy hit is not a precedent and must not be able to satisfy a gate
    that was written to ask whether this ticket has been seen before.
    """
    idx = index()
    hits = idx.lookup(state.get("subject"), state.get("body", ""))
    return {
        "policy": [h.as_dict() for h in hits],
        "policy_provenance": {
            "graph": str(idx.graph_path),
            "built_at_commit": idx._built_at_commit,
            "inferred_edges_allowed": False,
        },
        "audit_flags": (state.get("audit_flags") or []) + idx.flags(hits),
    }
