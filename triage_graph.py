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
    # TODO: any other degradation you decided counts

    dq = DataQuality(
        subject_present=bool(subject),
        body_chars=len(body),
        degraded=bool(reasons),
        degraded_reasons=reasons,
        precedent_depth=0,  # TODO: your proxy count
    )

    out = {"data_quality": dq, "audit_flags": []}
    if len(body) < YOUR_FLOOR:  # TODO: the number you derived, not mine
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
    raise NotImplementedError


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
    raise NotImplementedError


def route(state: TriageState) -> dict:
    """
    Reads: subject, body, triage, precedents.
    Writes: routing (RoutingCall).

    Bar to beat: 57.6% accuracy, macro-F1 0.475 over 10 queues.
    """
    raise NotImplementedError


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
    raise NotImplementedError


def escalate_ungrounded(state: TriageState) -> dict:
    """Terminal. No precedent, or gate A refused. Queue for manual triage."""
    raise NotImplementedError


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
    """
    raise NotImplementedError


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
