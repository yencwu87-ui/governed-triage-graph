"""
Runner for the governed triage graph.

    python run_triage.py --stub --ticket TK00042     watch the wiring work
    python run_triage.py --stub --batch 20           end-to-end, auto-approving
    python run_triage.py --ticket TK00042            run YOUR nodes
    python run_triage.py --batch 200 --out preds.csv then: eval_harness.py --preds preds.csv

--stub swaps in placeholder node bodies so you can see the graph execute, the
gates fire and the interrupt halt before you have written anything. Nothing in
stub mode is a real answer. Delete the flag as soon as your own nodes exist.

Without --stub, any unfilled node stops the run and the runner tells you which
one to write next. That is the intended loop.
"""
from __future__ import annotations
import argparse, os, random, sys, time, uuid, warnings

# The checkpointer serialises state to msgpack. Pydantic models in state are
# unregistered custom types, which LangGraph will refuse in a future version.
# Declaring them is the fix; suppressing the warning is not. Do this properly
# before you rely on the checkpoint as an audit record.
os.environ.setdefault("LANGGRAPH_ALLOWED_MSGPACK_MODULES", "triage_graph")
warnings.filterwarnings("ignore", message="Deserializing unregistered type")

import pandas as pd
from langgraph.types import Command

import triage_graph as tg


# ------------------------------------------------------- stub node bodies
def _install_stubs() -> None:
    """Placeholders. Deterministic nonsense, so you can see the machine move."""
    def ingress(s):
        body = (s.get("body") or "").strip()
        dq = tg.DataQuality(
            subject_present=bool((s.get("subject") or "").strip()),
            body_chars=len(body),
            degraded=not bool((s.get("subject") or "").strip()),
            degraded_reasons=[] if (s.get("subject") or "").strip() else ["no subject"],
            precedent_depth=3,
        )
        out = {"data_quality": dq, "audit_flags": []}
        if len(body) < 40:
            out["terminal_reason"] = "body too short to score"
        return out

    def retrieve(s):
        return {"precedents": [{"ticket_id": "TK-STUB", "priority": "medium", "score": 0.5}]}

    def classify(s):
        rnd = random.Random(s["ticket_id"])
        p = rnd.choice(["low", "medium", "high"])
        return {"triage": tg.TriageCall(
            priority=p, rationale="STUB — not a real judgement",
            confidence=round(rnd.uniform(0.4, 0.95), 2))}

    def route(s):
        return {"routing": tg.RoutingCall(queue="IT Support", confidence=0.5)}

    def audit(s):
        f = list(s.get("audit_flags") or [])
        if s.get("data_quality") and s["data_quality"].degraded:
            f.append("scored_while_degraded")
        d = s.get("gate_b_decision")
        if d and d.get("approved_priority") != s["triage"].priority:
            f.append("human_overrode_machine")
        return {"audit_flags": f}

    def escalate(s):
        return {"terminal_reason": s.get("terminal_reason") or "gate A refused"}

    def gate_a(s):
        t = s["triage"]
        return "route" if (t.confidence >= 0.6 and s.get("precedents")) else "escalate_ungrounded"

    tg.ingress, tg.retrieve, tg.classify = ingress, retrieve, classify
    tg.route, tg.audit = route, audit
    tg.escalate_ungrounded, tg.gate_a = escalate, gate_a


# ------------------------------------------------------------------ driver
def run_one(app, row, auto_approve=None, verbose=True):
    """Invoke the graph. Handle the Gate B halt. Return the final state."""
    cfg = {"configurable": {"thread_id": str(uuid.uuid4())}}
    payload = {"ticket_id": row["ticket_id"],
               "subject": None if pd.isna(row.get("subject")) else row["subject"],
               "body": row["body"]}
    t0 = time.time()
    state = app.invoke(payload, cfg)

    while "__interrupt__" in state:
        ask = state["__interrupt__"][0].value
        if verbose:
            print(f"\n  ⏸  GATE B HALT — {ask['ticket_id']}")
            print(f"     machine says : {ask['machine_priority']} "
                  f"(confidence {ask['confidence']})")
            print(f"     rationale    : {ask['rationale'][:100]}")
        if auto_approve is not None:
            decision = {"approved_priority": auto_approve or ask["machine_priority"],
                        "resumed_by": "auto", "agreed": True}
        else:
            valid = {"low", "medium", "high"}
            while True:
                got = input(f"      [{ask['ticket_id']}] Enter=accept {ask['machine_priority']}, "
                        f"or type low/medium/high: ").strip().lower()
                if got == "" or got in valid: break
                print(f"not a severity — expected one of {sorted(valid)}, or Enter to accept")
            decision = {"approved_priority": got or ask["machine_priority"],"resumed_by": "operator", "agreed": (not got) or got == ask["machine_priority"]}


        state = app.invoke(Command(resume=decision), cfg)

    state["_elapsed_s"] = round(time.time() - t0, 3)
    return state


def summarise(s):
    if s.get("terminal_reason"):
        return f"  {s['ticket_id']}  →  STOPPED: {s['terminal_reason']}"
    t, d = s["triage"], s.get("gate_b_decision")
    line = f"  {s['ticket_id']}  →  {t.priority:6s} conf {t.confidence:.2f}"
    if d:
        line += f" | human: {d['approved_priority']}" + ("" if d["agreed"] else "  ⚠ OVERRIDE")
    if s.get("audit_flags"):
        line += f" | flags: {','.join(s['audit_flags'])}"
    return line + f" | {s['_elapsed_s']}s"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stub", action="store_true")
    ap.add_argument("--ticket")
    ap.add_argument("--batch", type=int)
    ap.add_argument("--data", default="tickets_en.csv")
    ap.add_argument("--out")
    ap.add_argument("--auto-approve", action="store_true",
                    help="never prompt at Gate B (required for batch)")
    a = ap.parse_args()

    if a.stub:
        _install_stubs()
        print("  ⚠  STUB MODE — placeholder nodes. No output here is a real answer.")

    app = tg.build_graph()
    df = pd.read_csv(a.data)

    if a.ticket:
        rows = df[df.ticket_id == a.ticket]
        if rows.empty:
            sys.exit(f"no ticket {a.ticket}")
        auto = "" if a.auto_approve else None
    elif a.batch:
        rows = df.sample(a.batch, random_state=3)
        auto = ""      # batch never blocks on a human
    else:
        sys.exit("pass --ticket or --batch")

    results, halted = [], 0
    for _, row in rows.iterrows():
        try:
            s = run_one(app, row, auto_approve=auto, verbose=bool(a.ticket))
        except NotImplementedError:
            import traceback
            fn = traceback.extract_tb(sys.exc_info()[2])[-1].name
            print(f"\n  ✗  node `{fn}` is not written yet.")
            print(f"     Open triage_graph.py, read its docstring, fill it in.")
            print(f"     Or run with --stub to watch the wiring work first.\n")
            sys.exit(1)
        if s.get("gate_b_decision"):
            halted += 1
        results.append(s)
        if a.ticket:
            print(summarise(s))

    if a.batch:
        for s in results[:15]:
            print(summarise(s))
        if len(results) > 15:
            print(f"  ... {len(results)-15} more")
        n = len(results)
        stopped = sum(1 for s in results if s.get("terminal_reason"))
        print(f"\n  {n} tickets | {stopped} stopped before scoring "
              f"| {halted} halted at gate B | "
              f"mean {sum(s['_elapsed_s'] for s in results)/n:.3f}s")

    if a.out:
        keep = [s for s in results if not s.get("terminal_reason")]
        pd.DataFrame([{
            "ticket_id": s["ticket_id"],
            "priority": (s.get("gate_b_decision") or {}).get("approved_priority")
                        or s["triage"].priority,
            "machine_priority": s["triage"].priority,
            "confidence": s["triage"].confidence,
            "flags": "|".join(s.get("audit_flags") or []),
            "elapsed_s": s["_elapsed_s"],
        } for s in keep]).to_csv(a.out, index=False)
        print(f"  wrote {len(keep)} rows to {a.out}")


if __name__ == "__main__":
    main()
