"""
Governed incident triage graph — skeleton.

Wiring is complete and runnable. Node bodies are yours to fill.
Each node's docstring is its contract: what it reads from state, what it
must write back, and what it must never do.

Run `python eval_harness.py --baseline` first. Get the red number.
Then fill the nodes until it moves.
"""

from __future__ import annotations

from typing import Annotated, Literal, Optional, TypedDict

from langgraph.graph import END, START, StateGraph
from langgraph.checkpoint.memory import MemorySaver
from langgraph.types import interrupt
from pydantic import BaseModel, Field

# ------------------------------------------------------------- thresholds
#
# Every number here is a PLACEHOLDER and is flagged as such at runtime until
# you derive it. None of them silently defaults to something defensible-
# looking. See WIRE_IN.md step 3 and calibrate_grounding.py for how the
# grounding bands were derived — same method, same discipline.

BODY_FLOOR = 40           # chars below which a ticket carries no signal
BODY_FLOOR_DERIVED = False

GATE_A_FLOOR = None       # confidence below which the machine refuses to
                          # auto-proceed. None means NOT DERIVED, and gate_a
                          # FAILS CLOSED rather than guessing one.

def _load_thresholds() -> None:
    """Read gate_thresholds.json if calibrate_gates.py has written one."""
    global BODY_FLOOR, BODY_FLOOR_DERIVED, GATE_A_FLOOR
    import json, pathlib as _pl
    f = _pl.Path("gate_thresholds.json")
    if not f.exists():
        return
    try:
        d = json.loads(f.read_text())
    except Exception:
        return
    if "body_floor" in d:
        BODY_FLOOR = int(d["body_floor"]["body_floor"])
        BODY_FLOOR_DERIVED = True
    # A null gate_a_floor is a DERIVED refusal, not a missing value: the sweep
    # ran and found no floor that holds the ceiling. Stay closed either way.
    if "gate_a" in d and d["gate_a"].get("gate_a_floor") is not None:
        GATE_A_FLOOR = float(d["gate_a"]["gate_a_floor"])


_load_thresholds()

Priority = Literal["low", "medium", "high"]
Queue = Literal[
    "Technical Support",
    "Product Support",
    "Customer Service",
    "IT Support",
    "Billing and Payments",
    "Returns and Exchanges",
    "Service Outages and Maintenance",
    "Sales and Pre-Sales",
    "Human Resources",
    "General Inquiry",
]


QUEUES = frozenset(Queue.__args__)


# ---------------------------------------------------------------- contracts


class DataQuality(BaseModel):
    """Written by ingress. Read by the audit node. Never overwritten downstream."""

    subject_present: bool
    body_chars: int
    degraded: bool = Field(description="True when scoring proceeds on partial input")
    degraded_reasons: list[str] = Field(default_factory=list)
    precedent_depth: int = Field(
        0, description="Neighbours found for this ticket's shape"
    )


class TriageCall(BaseModel):
    """The model's severity judgement. Shape validation only — not correctness."""

    priority: Priority
    rationale: str = Field(max_length=400)
    confidence: float = Field(ge=0.0, le=1.0)


class RoutingCall(BaseModel):
    queue: Queue
    confidence: float = Field(ge=0.0, le=1.0)


class TriageState(TypedDict, total=False):
    ticket_id: str
    subject: Optional[str]
    body: str
    data_quality: DataQuality
    precedents: list[dict]
    triage: TriageCall
    routing: RoutingCall
    gate_a_passed: bool
    gate_b_decision: Optional[dict]  # {"approved_priority", "resumed_by", "agreed"}
    audit_flags: list[str]
    terminal_reason: Optional[str]


# ------------------------------------------------------------------- nodes


def ingress(state: TriageState) -> dict:
    """
    Reads: subject, body.
    Writes: data_quality, and terminal_reason if the record is unscoreable.

    Contract:
      - Normalise sentinels. Empty string and whitespace are not values.
      - subject is missing on ~16% of this corpus. That is DEGRADED, not reject.
      - body under ~40 chars carries no signal. Reject, don't guess.
      - Set precedent_depth here, not in retrieve() — the gate needs it.

    Must not: infer or estimate priority. Validation only.
    """
    subject = (state.get("subject") or "").strip()
    body = (state.get("body") or "").strip()

    reasons = []
    if not subject:
        reasons.append("no subject")
    if body and body.lower() in {"n/a", "none", "-", "test", "asdf"}:
        reasons.append("sentinel body")

    # precedent_depth is set HERE, not in retrieve(), because gate_a reads it.
    try:
        import precedents
        depth = len(precedents.index().lookup(subject, body))
    except Exception:
        depth = 0

    dq = DataQuality(
        subject_present=bool(subject),
        body_chars=len(body),
        degraded=bool(reasons),
        degraded_reasons=reasons,
        precedent_depth=depth,
    )

    flags = []
    if not BODY_FLOOR_DERIVED:
        flags.append("body_floor_not_calibrated")

    out = {"data_quality": dq, "audit_flags": flags}
    if len(body) < BODY_FLOOR:
        out["terminal_reason"] = "body too short to score"
    return out


def retrieve(state: TriageState) -> dict:
    """
    Reads: subject, body.
    Writes: precedents — list of similar resolved tickets, each with its
            `answer` text and its labelled priority.

    Contract:
      - Similarity over subject+body. Return k with scores attached.
      - Return [] rather than low-scoring filler. An empty list is information;
        a bad neighbour is contamination.

    Break-it target: a precedent whose own label is wrong poisons every
    ticket that retrieves it. Log which precedents were used, per ticket.
    """
    import precedents

    pidx = precedents.index()
    hits = pidx.lookup(state.get("subject"), state.get("body", ""))
    out = {
        "precedents": [h.as_dict() for h in hits],
        "precedents_used": [h.ticket_id for h in hits],
        "audit_flags": (state.get("audit_flags") or []) + pidx.flags(hits),
    }

    # Second, SEPARATE channel: which clause governs this ticket. Optional —
    # absent policy index is a flag, not a failure.
    try:
        import retrieve_policy
        pol = retrieve_policy.retrieve({**state, "audit_flags": []})
        out["policy"] = pol["policy"]
        out["policy_provenance"] = pol["policy_provenance"]
        out["audit_flags"] = out["audit_flags"] + pol["audit_flags"]
    except Exception:
        out["audit_flags"] = out["audit_flags"] + ["policy_channel_unavailable"]

    return out


def classify(state: TriageState) -> dict:
    """
    Reads: subject, body, precedents, data_quality.
    Writes: triage (TriageCall).

    Contract:
      - Rubric in the system prompt, versioned. Put the version in the rationale.
      - When data_quality.degraded, the model must be told what is missing.
      - confidence must be usable by gate_a. A model that always says 0.9
        makes the gate decorative — check the distribution before you trust it.

    Bar to beat: 64.2% accuracy, macro-F1 0.611, under-classification 30.8%.
    """
    import classifier

    r = classifier.classify(
        subject=state.get("subject"),
        body=state["body"],
        dq=state.get("data_quality"),
        precedents=state.get("precedents"),
        version="v0.1",
    )
    # A bad response STOPS the ticket. It does not fall back to "medium".
    # A classifier that silently defaults on parse failure passes your eval
    # and fails in production.
    if r.get("parse_error") or r.get("priority") not in ("low", "medium", "high"):
        return {
            "terminal_reason": "classifier returned an unusable response",
            "audit_flags": (state.get("audit_flags") or []) + ["classifier_parse_error"],
        }
    return {"triage": TriageCall(priority=r["priority"],
                                 rationale=r["rationale"],
                                 confidence=float(r["confidence"]))}


def route(state: TriageState) -> dict:
    """
    Reads: subject, body, triage, precedents.
    Writes: routing (RoutingCall).

    Bar to beat: 57.6% accuracy, macro-F1 0.475 over 10 queues.
    """
    import classifier

    # Queue assignment is a routing decision, not a severity one, and a wrong
    # queue is a delay while a wrong severity is an outage. Precedent queues
    # are the cheapest signal available, so try them before spending a call.
    prec = state.get("precedents") or []
    queues = [p.get("queue") for p in prec if p.get("queue") in QUEUES]
    if queues and len(set(queues)) == 1:
        return {"routing": RoutingCall(queue=queues[0],
                                       confidence=float(prec[0].get("score", 0.0)))}

    r = classifier.route(subject=state.get("subject"), body=state["body"],
                         precedents=prec) if hasattr(classifier, "route") else None
    if not r or r.get("queue") not in QUEUES:
        # No queue is better than a guessed queue — General Inquiry is the
        # corpus's own catch-all, and the flag records that we defaulted.
        return {"routing": RoutingCall(queue="General Inquiry", confidence=0.0),
                "audit_flags": (state.get("audit_flags") or []) + ["routing_defaulted"]}
    return {"routing": RoutingCall(queue=r["queue"],
                                   confidence=float(r.get("confidence", 0.0)))}


def audit(state: TriageState) -> dict:
    """
    Reads: everything.
    Writes: audit_flags.

    This is the second-line node and it does not exist in commercial triage
    tools. Flag, at minimum:
      - scored while degraded
      - precedent_depth == 0 but confidence high
      - rationale does not reference anything present in the body
      - human at gate B disagreed with the machine
    """
    f = list(state.get("audit_flags") or [])
    dq = state.get("data_quality")
    t = state.get("triage")

    if dq and dq.degraded:
        f.append("scored_while_degraded")
    if dq and dq.precedent_depth == 0 and t and t.confidence >= 0.8:
        f.append("high_confidence_without_precedent")

    # Does the rationale actually reference the ticket? grounding.py already
    # answers this, anchors and all.
    if t is not None:
        try:
            import grounding
            g = grounding.check(t.rationale, state.get("subject"), state.get("body", ""))
            f.extend(grounding.flags(g))
            if not g.get("bands_derived", False):
                f.append("grounding_bands_not_calibrated")
        except Exception:
            f.append("grounding_check_unavailable")

    d = state.get("gate_b_decision")
    if d and t and d.get("approved_priority") != t.priority:
        f.append("human_overrode_machine")
    if d and d.get("agreed") is False:
        f.append("human_disagreed")

    return {"audit_flags": sorted(set(f))}


def escalate_ungrounded(state: TriageState) -> dict:
    """Terminal. No precedent, or gate A refused. Queue for manual triage."""
    reason = state.get("terminal_reason") or "gate A refused to auto-proceed"
    return {
        "terminal_reason": reason,
        "audit_flags": sorted(set((state.get("audit_flags") or [])
                                  + ["escalated_to_manual_triage"])),
    }


def gate_b_halt(state: TriageState) -> dict:
    """
    Gate B — human accountability. The graph STOPS here.

    `interrupt()` suspends execution and persists state via the checkpointer.
    Nothing proceeds until a human resumes with a decision. The record of
    who resumed, when, and whether they agreed IS the audit trail — and it
    is also your next round's eval data.

    Timeout behaviour is a design decision, not a default: an unanswered
    high-severity halt must escalate, never auto-close. Getting this
    backwards turns your accountability gate into a silent outage.
    """
    decision = interrupt(
        {
            "ticket_id": state["ticket_id"],
            "machine_priority": state["triage"].priority,
            "rationale": state["triage"].rationale,
            "confidence": state["triage"].confidence,
            "prompt": "Confirm or override this severity call.",
        }
    )
    return {"gate_b_decision": decision}


# ------------------------------------------------------------------- edges


def after_ingress(state: TriageState) -> str:
    """Unscoreable records never reach a model."""
    return "escalate_ungrounded" if state.get("terminal_reason") else "retrieve"


def gate_a(state: TriageState) -> str:
    """
    Gate A — machine safety. A conditional edge the machine evaluates alone.

    Refuse to auto-proceed when confidence is below threshold OR no precedent
    was found. Derive the threshold from the gold set; do not pick a round
    number and defend it later.

    FAILS CLOSED. GATE_A_FLOOR is None until you derive it, and while it is
    None every ticket escalates to manual triage. That is deliberate: an
    undefined safety threshold must not evaluate to "let it through". If this
    node ever looks too conservative, derive the number — do not soften the
    default.
    """
    if GATE_A_FLOOR is None:
        return "escalate_ungrounded"

    t = state.get("triage")
    dq = state.get("data_quality")
    if t is None or state.get("terminal_reason"):
        return "escalate_ungrounded"
    if dq is not None and dq.precedent_depth == 0:
        return "escalate_ungrounded"
    if t.confidence < GATE_A_FLOOR:
        return "escalate_ungrounded"
    return "route"


def needs_human(state: TriageState) -> str:
    """
    Gate B trigger. Every `high` call halts for sign-off.
    Everything else dispatches.
    """
    return "gate_b_halt" if state["triage"].priority == "high" else "audit"


# ------------------------------------------------------------------- build


def build_graph(checkpointer=None):
    g = StateGraph(TriageState)

    g.add_node("ingress", ingress)
    g.add_node("retrieve", retrieve)
    g.add_node("classify", classify)
    g.add_node("route", route)
    g.add_node("gate_b_halt", gate_b_halt)
    g.add_node("audit", audit)
    g.add_node("escalate_ungrounded", escalate_ungrounded)

    g.add_edge(START, "ingress")
    g.add_conditional_edges(
        "ingress",
        after_ingress,
        {"retrieve": "retrieve", "escalate_ungrounded": "escalate_ungrounded"},
    )
    g.add_edge("retrieve", "classify")
    g.add_conditional_edges(
        "classify",
        gate_a,
        {"route": "route", "escalate_ungrounded": "escalate_ungrounded"},
    )
    g.add_conditional_edges(
        "route", needs_human, {"gate_b_halt": "gate_b_halt", "audit": "audit"}
    )
    g.add_edge("gate_b_halt", "audit")
    g.add_edge("audit", END)
    g.add_edge("escalate_ungrounded", END)

    return g.compile(checkpointer=checkpointer or MemorySaver())


if __name__ == "__main__":
    print(build_graph().get_graph().draw_mermaid())
