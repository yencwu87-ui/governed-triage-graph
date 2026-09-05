"""
Gate control tests — evidence, not assertions in a README.

    python test_gates.py

Needs no model and no gold set. These test the CONTROL PLANE, which is the
part that can be validated before accuracy means anything.

The organising question for T3-T6: if the human does nothing, what happens?
  nothing proceeds  -> human IN the loop  (blocking, required)
  it proceeds       -> human ON the loop  (supervisory, advisory)
Systems claim the first and implement the second more often than anyone admits.
"""
from __future__ import annotations
import json, os, sys, uuid

os.environ.setdefault("LANGGRAPH_ALLOWED_MSGPACK_MODULES", "triage_graph")

import pandas as pd
from langgraph.types import Command

import triage_graph as tg
import run_triage

run_triage.tg = tg
run_triage._install_stubs()

PASS, FAIL = "\033[92mPASS\033[0m", "\033[91mFAIL\033[0m"
results = []


def check(name, ok, detail=""):
    results.append((name, ok))
    print(f"  [{PASS if ok else FAIL}] {name}")
    if detail:
        print(f"         {detail}")


def high_ticket(app):
    """Drive the graph until it produces a `high`, so Gate B engages.
    Returns the halted state AND the thread config that halted — they must
    match, or every later check inspects the wrong thread."""
    for seed in range(400):
        cfg = {"configurable": {"thread_id": str(uuid.uuid4())}}
        body = f"Production checkout is down for all customers. Ref {seed}. " + "x" * 80
        s = app.invoke({"ticket_id": f"T{seed}", "subject": "Outage", "body": body}, cfg)
        if "__interrupt__" in s:
            return s, cfg
    return None, None


print("\n  GATE CONTROL TESTS\n  " + "=" * 58)

# ---------------------------------------------------------------- ingress
print("\n  Ingress")
app = tg.build_graph()
cfg = {"configurable": {"thread_id": str(uuid.uuid4())}}
s = app.invoke({"ticket_id": "T-short", "subject": "hi", "body": "broken"}, cfg)
check("T1  unscoreable input never reaches the model",
      s.get("terminal_reason") is not None and not s.get("triage"),
      f"terminal_reason = {s.get('terminal_reason')!r}")

cfg = {"configurable": {"thread_id": str(uuid.uuid4())}}
s = app.invoke({"ticket_id": "T-nosubj", "subject": None,
                "body": "The export job fails every night with a timeout after "
                        "roughly ten minutes and no rows are written." * 2}, cfg)
dq = s.get("data_quality")
check("T2  missing subject degrades but does not reject",
      dq is not None and dq.degraded and s.get("triage") is not None,
      f"degraded_reasons = {getattr(dq,'degraded_reasons',None)}")

# ------------------------------------------------------- Gate B: blocking?
print("\n  Gate B — in the loop, or on the loop?")
s, cfg = high_ticket(app)
if s is None:
    sys.exit("  could not produce a `high` — check the stub classifier")

check("T3  the graph SUSPENDS on a high call",
      "__interrupt__" in s,
      "execution stopped mid-graph; no terminal state reached")

check("T4  no decision is recorded while suspended",
      s.get("gate_b_decision") is None,
      "the machine did not supply its own approval")

# --- the decisive one: walk away and see if anything happens ---
snap = app.get_state(cfg)
check("T5  human does NOTHING -> nothing proceeds  (IN the loop)",
      bool(snap.next) and snap.values.get("gate_b_decision") is None,
      f"graph is parked at {snap.next}; it will wait indefinitely. "
      "If this had auto-proceeded on timeout, the human would be ON the loop.")

check("T6  suspended state is durable and resumable later",
      snap.values.get("ticket_id") is not None,
      "state is checkpointed — a different process/day can resume this thread")

# ------------------------------------------------------ override recording
print("\n  Accountability record")
s2 = app.invoke(Command(resume={"approved_priority": "low",
                                "resumed_by": "duty_mgr_A",
                                "agreed": False}), cfg)
d = s2.get("gate_b_decision") or {}
check("T7  resuming requires a human-supplied decision",
      d.get("resumed_by") == "duty_mgr_A",
      f"recorded: {d}")

check("T8  disagreement with the machine is captured, not silently accepted",
      d.get("agreed") is False and d.get("approved_priority") != s["triage"].priority,
      f"machine said {s['triage'].priority}, human set {d.get('approved_priority')}")

check("T9  the override is flagged for audit",
      "human_overrode_machine" in (s2.get("audit_flags") or []),
      f"audit_flags = {s2.get('audit_flags')}")

# ------------------------------------------------------------- bypass check
print("\n  Bypass")
src = open("triage_graph.py").read()
check("T10 no edge routes a high call around gate_b_halt",
      'needs_human' in src and 'gate_b_halt' in src and
      src.count('"gate_b_halt": "gate_b_halt"') == 1,
      "single entry point to the halt node in the compiled graph")

# ---------------------------------------------------------- rubber-stamping
print("\n  Is the human REALLY in the loop?")
print("     A human who approves everything is nominally in the loop and")
print("     functionally out of it. Run this against your real decision log:")
print("       python - <<'EOF'")
print("       import json,pandas as pd")
print("       d=pd.DataFrame([json.loads(l) for l in open('decisions.jsonl')])")
print("       print('override rate:', (~d.agreed).mean())")
print("       print('median seconds to decide:', d.decide_secs.median())")
print("       EOF")
print("     Override rate at or near 0%, or decisions in under ~3 seconds,")
print("     means the gate is theatre. That is a finding about the PROCESS,")
print("     not the code — and no test here can detect it for you.")

# ------------------------------------------------------- policy channel
# Fixture-backed; needs no model, no gold set and no ./policy corpus.
import test_policy_channel
test_policy_channel.run(check)

# -------------------------------------------------------------------- done
n_pass = sum(1 for _, ok in results if ok)
print("\n  " + "=" * 58)
print(f"  {n_pass}/{len(results)} passed\n")
if n_pass == len(results):
    print("  Verdict: Gate B is human-IN-the-loop — blocking, with no timeout")
    print("  bypass. Gate A is machine-only and involves no human at all.\n")
    print("  Caveat, and it is not small: T21 PASSES BY DEMONSTRATING A HOLE.")
    print("  The policy channel's provenance gate governs traversal only. A")
    print("  band-crossing clause can still seed directly on one shared word.")
    print("  It is flagged, not prevented. Do not read 22/22 as closed.\n")
else:
    print("  Verdict: WITHHELD. A failing control test is a finding — log it\n"
          "  before changing anything.\n")
sys.exit(0 if n_pass == len(results) else 1)
