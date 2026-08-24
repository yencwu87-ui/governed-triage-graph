"""
Visual test bench for the governed triage graph.

    pip install streamlit
    streamlit run app.py

Works with a half-finished graph. Unwritten nodes are shown as gaps rather
than tracebacks, so you can watch the wiring before the logic exists.
Tick "stub mode" in the sidebar to fill every node with a placeholder.
"""
from __future__ import annotations
import os, uuid

os.environ.setdefault("LANGGRAPH_ALLOWED_MSGPACK_MODULES", "triage_graph")

import pandas as pd
import streamlit as st
from langgraph.types import Command

import triage_graph as tg
import run_triage

st.set_page_config(page_title="Governed triage — test bench", layout="wide")

NODE_ORDER = ["ingress", "retrieve", "classify", "route", "gate_b_halt", "audit"]
PILL = {"high": "🔴", "medium": "🟠", "low": "🟢"}


@st.cache_data
def load(path="tickets_en.csv"):
    return pd.read_csv(path)


def fresh_app(stub: bool, clf=None):
    """Stubs fill the unwritten nodes. When a real classifier is chosen it
    replaces the stub `classify` only — the rest of the control plane is
    unchanged, which is the point of keeping them separable."""
    import importlib
    importlib.reload(tg)
    run_triage.tg = tg
    run_triage._install_stubs()          # ingress/retrieve/route/audit/gate_a
    if clf is not None:
        def classify(s):
            r = clf.classify(subject=s.get("subject"), body=s["body"],
                             dq=s.get("data_quality"),
                             precedents=s.get("precedents"))
            if r.get("parse_error") or r["priority"] not in ("low","medium","high"):
                return {"terminal_reason": "classifier returned an unusable response",
                        "audit_flags": (s.get("audit_flags") or []) + ["classifier_parse_error"]}
            import grounding
            gr = grounding.check(r["rationale"], s.get("subject"), s["body"])
            st.session_state.setdefault("meta", []).append(
                {**{k: r.get(k) for k in ("latency_s","in_tokens","out_tokens","confidence")},
                 "ticket_id": s["ticket_id"], "priority": r["priority"],
                 "rationale": r["rationale"], "grounding": gr["score"],
                 "verdict": gr["verdict"], "missing": ", ".join(gr["missing"]),
                 "subject": s.get("subject"), "body": s["body"]})
            return {"triage": tg.TriageCall(priority=r["priority"],
                                            rationale=r["rationale"],
                                            confidence=float(r["confidence"]))}
        tg.classify = classify
    return tg.build_graph()


# ------------------------------------------------------------------ sidebar
st.sidebar.header("Test bench")
backend = st.sidebar.radio(
    "Classifier", ["Stub", "Local (Ollama)", "Anthropic API"],
    help="Stub = placeholder, no model. Local = Ollama on this machine, free. "
         "API = hosted, needs a key.")
stub = backend == "Stub"

clf = None
if backend == "Local (Ollama)":
    import classifier_local as clf
    ok, msg = clf.health()
    (st.sidebar.success if ok else st.sidebar.error)(msg)
    if not ok:
        st.stop()
    st.sidebar.caption(f"Model: {clf.MODEL}. Nothing leaves this machine.")
elif backend == "Anthropic API":
    if not os.environ.get("ANTHROPIC_API_KEY"):
        st.sidebar.error("ANTHROPIC_API_KEY is not set in this shell.")
        st.stop()
    import classifier as clf
    st.sidebar.success(f"API · {clf.MODEL}")
df = load()
mode = st.sidebar.radio("Mode", ["Single ticket", "Batch", "Label gold set"])

if stub:
    st.sidebar.warning("Stub mode — outputs are placeholders.")

# ------------------------------------------------------------- single ticket
if mode == "Single ticket":
    left, right = st.columns([1, 1])

    with left:
        st.subheader("Input")
        pick = st.selectbox("Ticket", df.ticket_id.tolist(), index=41)
        row = df[df.ticket_id == pick].iloc[0]
        subj = row.subject if pd.notna(row.subject) else None
        st.text_input("Subject", subj or "(missing)", disabled=True)
        st.text_area("Body", row.body, height=180, disabled=True)
        st.caption(f"dataset label: {PILL.get(row.priority,'')} {row.priority}  ·  "
                   f"queue: {row.queue}  ·  {len(str(row.body))} chars")

        if st.button("Run through graph", type="primary", use_container_width=True):
            st.session_state.clear()
            st.session_state.app = fresh_app(stub, clf)
            st.session_state.cfg = {"configurable": {"thread_id": str(uuid.uuid4())}}
            st.session_state.truth = row.priority
            try:
                st.session_state.state = st.session_state.app.invoke(
                    {"ticket_id": pick, "subject": subj, "body": row.body},
                    st.session_state.cfg)
            except NotImplementedError as e:
                import sys, traceback
                st.session_state.missing = traceback.extract_tb(sys.exc_info()[2])[-1].name
            except Exception as e:
                st.session_state.err = f"{type(e).__name__}: {e}"

    with right:
        st.subheader("Path through the graph")

        if m := st.session_state.get("missing"):
            st.error(f"Stopped at `{m}` — node not written yet.")
            st.caption("Fill it in triage_graph.py, or switch on stub mode.")
        elif e := st.session_state.get("err"):
            st.error(e)
        elif (s := st.session_state.get("state")) is not None:

            dq = s.get("data_quality")
            if dq:
                cols = st.columns(3)
                cols[0].metric("Body chars", dq.body_chars)
                cols[1].metric("Subject", "present" if dq.subject_present else "MISSING")
                cols[2].metric("Precedents", dq.precedent_depth)
                if dq.degraded:
                    st.warning("Degraded input: " + ", ".join(dq.degraded_reasons))

            if s.get("terminal_reason") and not s.get("triage"):
                st.error(f"Stopped before scoring — {s['terminal_reason']}")
                st.caption("No model call was made. This is the control working.")

            if t := s.get("triage"):
                st.markdown(f"### {PILL.get(t.priority,'')} {t.priority.upper()}")
                st.progress(t.confidence, text=f"confidence {t.confidence:.2f}")
                st.info(t.rationale)
                try:
                    import grounding
                    row = st.session_state.get("meta", [{}])[-1]
                    gr = grounding.check(t.rationale, row.get("subject"), row.get("body", ""))
                    icon = {"grounded": "✅", "weak": "⚠️"}.get(gr["verdict"], "❌")
                    st.caption(f"{icon} Rationale grounding {gr['score']:.0%} — {gr['verdict']}"
                               + (f" · not in ticket: {', '.join(gr['missing'])}"
                                  if gr["missing"] else ""))
                    if gr["bad_quotes"]:
                        st.error("The rationale quotes text that is not in the ticket: "
                                 + "; ".join(f'"{q}"' for q in gr["bad_quotes"]))
                except Exception:
                    pass
                truth = st.session_state.get("truth")
                if truth:
                    ok = truth == t.priority
                    st.caption(("✅ agrees with" if ok else "❌ differs from")
                               + f" dataset label ({truth})")

            if s.get("routing"):
                st.caption(f"routed to **{s['routing'].queue}** "
                           f"(confidence {s['routing'].confidence:.2f})")

            # ------------------------------------------------ Gate B halt
            if "__interrupt__" in s:
                ask = s["__interrupt__"][0].value
                st.divider()
                st.markdown("### ⏸ Gate B — the graph has stopped")
                st.caption("Execution is suspended. State is checkpointed. "
                           "Nothing proceeds without a human write.")
                choice = st.radio("Your call",
                                  ["Confirm the machine", "Override to high",
                                   "Override to medium", "Override to low"],
                                  horizontal=False)
                who = st.text_input("Resumed by", "duty_manager")
                if st.button("Resume graph", type="primary"):
                    approved = (ask["machine_priority"] if choice.startswith("Confirm")
                                else choice.split()[-1])
                    st.session_state.state = st.session_state.app.invoke(
                        Command(resume={"approved_priority": approved,
                                        "resumed_by": who,
                                        "agreed": approved == ask["machine_priority"]}),
                        st.session_state.cfg)
                    st.rerun()

            if d := s.get("gate_b_decision"):
                st.divider()
                if d["agreed"]:
                    st.success(f"Gate B cleared by {d['resumed_by']} — machine call confirmed.")
                else:
                    st.error(f"Gate B OVERRIDE by {d['resumed_by']}: "
                             f"{s['triage'].priority} → {d['approved_priority']}")
                    st.caption("This disagreement is your next round's eval data.")

            if flags := s.get("audit_flags"):
                st.divider()
                st.markdown("**Audit flags**")
                for f in flags:
                    st.markdown(f"- `{f}`")

            with st.expander("Raw state"):
                st.json({k: str(v) for k, v in s.items() if k != "__interrupt__"})
        else:
            st.caption("Pick a ticket and press Run.")

# --------------------------------------------------------------------- batch
else:
    n = st.sidebar.slider("Tickets", 10, 300, 50, step=10)
    if st.sidebar.button("Run batch", type="primary"):
        app = fresh_app(stub, clf)
        st.session_state.meta = []
        rows, bar = [], st.progress(0.0)
        sample = df.sample(n, random_state=3)
        for i, (_, r) in enumerate(sample.iterrows()):
            try:
                s = run_triage.run_one(app, r, auto_approve="", verbose=False)
            except NotImplementedError as e:
                import sys, traceback
                st.error(f"Node `{traceback.extract_tb(sys.exc_info()[2])[-1].name}` "
                         "is not written yet.")
                st.stop()
            rows.append({
                "ticket_id": r.ticket_id,
                "truth": r.priority,
                "predicted": None if s.get("terminal_reason") else s["triage"].priority,
                "confidence": None if s.get("terminal_reason") else s["triage"].confidence,
                "stopped": s.get("terminal_reason"),
                "gate_b": bool(s.get("gate_b_decision")),
                "flags": ",".join(s.get("audit_flags") or []),
                "secs": s["_elapsed_s"],
            })
            bar.progress((i + 1) / n)
        st.session_state.batch = pd.DataFrame(rows)

    if (b := st.session_state.get("batch")) is not None:
        scored = b[b.predicted.notna()]
        c = st.columns(4)
        c[0].metric("Scored", f"{len(scored)}/{len(b)}")
        c[1].metric("Stopped at ingress/gate A", int(b.predicted.isna().sum()))
        c[2].metric("Halted at gate B", int(b.gate_b.sum()))
        if len(scored):
            c[3].metric("Agreement with dataset",
                        f"{(scored.truth == scored.predicted).mean():.1%}")
            st.caption("Agreement is not accuracy — the dataset labels are "
                       "generator-assigned. Score against your gold set instead.")
            st.subheader("Confusion")
            st.dataframe(pd.crosstab(scored.truth, scored.predicted), use_container_width=True)
        if b.confidence.notna().any():
            st.subheader("Confidence distribution")
            st.caption("Gate A routes on this number. A narrow spread means the "
                       "gate never fires and the machine-safety control is decorative.")
            cc = b.confidence.dropna()
            k = st.columns(4)
            k[0].metric("min", f"{cc.min():.2f}")
            k[1].metric("median", f"{cc.median():.2f}")
            k[2].metric("max", f"{cc.max():.2f}")
            k[3].metric("spread", f"{cc.max()-cc.min():.2f}",
                        delta="too flat" if cc.max()-cc.min() < 0.15 else "usable",
                        delta_color="inverse" if cc.max()-cc.min() < 0.15 else "normal")
            st.bar_chart(cc.round(1).value_counts().sort_index())

        if m := st.session_state.get("meta"):
            md = pd.DataFrame(m)
            st.caption(f"{len(md)} model calls · median {md.latency_s.median():.2f}s · "
                       f"{int(md.in_tokens.sum()):,} in / {int(md.out_tokens.sum()):,} out tokens")

        if m2 := st.session_state.get("meta"):
            md2 = pd.DataFrame(m2)
            if "grounding" in md2:
                st.subheader("Reasoning")
                st.caption("A rationale is the model's narration, not a trace of its "
                           "computation — so it is not an explanation. It is testable: "
                           "the words it cites should come from the ticket.")
                v = md2.verdict.value_counts()
                g1 = st.columns(4)
                g1[0].metric("Median grounding", f"{md2.grounding.median():.0%}")
                g1[1].metric("Grounded", int(v.get("grounded", 0)))
                g1[2].metric("Weak", int(v.get("weak", 0)))
                g1[3].metric("Ungrounded / fabricated",
                             int(v.get("UNGROUNDED", 0) + v.get("FABRICATED QUOTE", 0)))
                for _, q in md2.sort_values("grounding").head(20).iterrows():
                    icon = {"grounded": "✅", "weak": "⚠️"}.get(q.verdict, "❌")
                    with st.expander(f"{icon} {q.ticket_id} · {q.priority} · "
                                     f"grounding {q.grounding:.0%}"):
                        st.markdown(f"**Model said:** {q.rationale}")
                        if q.missing:
                            st.caption(f"Terms not found in the ticket: {q.missing}")
                        st.text_area("Ticket", str(q.body)[:1200], height=140,
                                     disabled=True, key=f"tb{q.ticket_id}")
                st.download_button("Download reasoning log",
                                   md2.drop(columns=["body"], errors="ignore").to_csv(index=False),
                                   "reasoning_log.csv", "text/csv")

        st.subheader("Runs")
        st.dataframe(b, use_container_width=True, height=340)
        st.download_button("Download predictions",
                           b[["ticket_id", "predicted"]].rename(
                               columns={"predicted": "priority"}).to_csv(index=False),
                           "preds.csv", "text/csv")


# ---------------------------------------------------------------- labelling
if mode == "Label gold set":
    import pathlib
    GOLD_IN, GOLD_OUT = "gold_set_template.csv", "gold_set_labelled.csv"

    st.header("Gold set labelling")
    st.caption("Blinded — the dataset's own priority is deliberately not shown. "
               "Your labels become the reference every later number is scored against. "
               "Saves after each ticket; close and resume any time.")

    tmpl = pd.read_csv(GOLD_IN)
    if pathlib.Path(GOLD_OUT).exists():
        done = pd.read_csv(GOLD_OUT)
    else:
        done = pd.DataFrame(columns=["ticket_id","my_priority","my_rationale",
                                     "confidence","borderline"])

    remaining = tmpl[~tmpl.ticket_id.isin(done.ticket_id)]
    st.progress(len(done)/len(tmpl), text=f"{len(done)} of {len(tmpl)} labelled")

    if remaining.empty:
        st.success("All 100 labelled.")
        agree = None
        full = pd.read_csv("tickets_en.csv")[["ticket_id","priority"]]
        j = done.merge(full, on="ticket_id")
        agree = (j.my_priority == j.priority).mean()
        st.metric("Agreement with the dataset's generator labels", f"{agree:.1%}")
        st.caption("Disagreement is a finding, not an error. Where you and the "
                   "generator differ, YOUR label is the reference.")
        st.dataframe(pd.crosstab(j.priority, j.my_priority), use_container_width=True)
        st.dataframe(done, use_container_width=True, height=300)
        st.download_button("Download gold set", done.to_csv(index=False),
                           GOLD_OUT, "text/csv")
        if st.button("Start over (deletes labels)"):
            pathlib.Path(GOLD_OUT).unlink(); st.rerun()
    else:
        r = remaining.iloc[0]
        st.divider()
        st.markdown(f"**{r.ticket_id}** · {len(str(r.body))} chars")
        st.markdown(f"#### {r.subject if pd.notna(r.subject) else '(no subject)'}")
        st.text_area("Body", r.body, height=240, disabled=True, key=f"b{r.ticket_id}")

        c = st.columns([1,1,1])
        choice = None
        if c[0].button("🟢  LOW", use_container_width=True):  choice = "low"
        if c[1].button("🟠  MEDIUM", use_container_width=True): choice = "medium"
        if c[2].button("🔴  HIGH", use_container_width=True):   choice = "high"

        why = st.text_input("Why — the words in the ticket that decided it",
                            key=f"w{r.ticket_id}")
        conf = st.slider("How sure are you?", 0.0, 1.0, 0.8, 0.05, key=f"c{r.ticket_id}")
        border = st.checkbox("Borderline — I could argue the other way",
                             key=f"x{r.ticket_id}")

        if choice:
            done.loc[len(done)] = [r.ticket_id, choice, why, conf, border]
            done.to_csv(GOLD_OUT, index=False)
            st.rerun()

        st.caption("Tip: if you hesitate more than ~20 seconds, tick borderline and "
                   "move on. The borderline cases are the most informative rows in the set.")
