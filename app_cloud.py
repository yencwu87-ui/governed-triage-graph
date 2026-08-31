"""
Hosted demo — upload any ticket CSV and run it through the governed triage graph.

Deploy: push to GitHub, then share.streamlit.io -> New app -> pick this file.

Design constraints of a public deployment, made explicit:
  - no corpus ships with the app; the user brings their own CSV
  - the user brings their own API key; it lives in session memory only
  - uploaded text is sent to a model provider, and the UI says so before you run
"""
from __future__ import annotations
import os, uuid

os.environ.setdefault("LANGGRAPH_ALLOWED_MSGPACK_MODULES", "triage_graph")

import pandas as pd
import streamlit as st

st.set_page_config(page_title="Governed triage", layout="wide")

PILL = {"high": "🔴", "medium": "🟠", "low": "🟢"}
MAX_ROWS = 200

st.title("Governed triage graph")
st.caption("Upload tickets. Watch them pass through validation, a machine gate "
           "and a human gate. The controls run with or without a model.")

# ------------------------------------------------------------------ sidebar
with st.sidebar:
    st.header("Setup")
    use_llm = st.toggle("Use a real classifier", value=False)
    key = ""
    if use_llm:
        key = st.text_input("Anthropic API key", type="password",
                            help="Held in session memory only. Never written to disk "
                                 "or logged. Gone when you close the tab.")
        model = st.selectbox("Model", ["claude-haiku-4-5-20251001", "claude-sonnet-5"])
        st.caption("Your key, your spend. Haiku is roughly an order of "
                   "magnitude cheaper for this task.")
    else:
        model = None
        st.info("Stub mode — placeholder classifier. The control plane still "
                "runs end to end. No API key needed, nothing leaves this app.")
    n = st.slider("Tickets to process", 5, MAX_ROWS, 20, step=5)
    st.divider()
    st.caption("Source: github.com/…/governed-triage-graph")

# ------------------------------------------------------------------- upload
up = st.file_uploader("Ticket CSV", type=["csv"])
if up is None:
    st.info("Upload a CSV with at least one free-text column. A subject column "
            "and an existing priority label are optional.")
    with st.expander("What the graph does"):
        st.markdown("""
1. **ingress** — validates the row. Too short, or missing fields, and it never
   reaches a model. Degradation is recorded in state, not swallowed.
2. **classify** — assigns priority with a rationale and a confidence.
3. **Gate A — machine.** Below a confidence floor, the graph refuses to
   auto-proceed and diverts to manual triage. Evaluated by the machine alone.
4. **Gate B — human.** Every `high` call halts the graph. Execution suspends
   and nothing continues without a person writing a decision into state.
5. **audit** — flags anything scored on degraded input, or overruled by a human.

Gate A and Gate B are different things. An eval score is not an audit trail.
        """)
    st.stop()

raw = pd.read_csv(up)
st.success(f"{len(raw):,} rows · {len(raw.columns)} columns")

c1, c2, c3 = st.columns(3)
body_col = c1.selectbox("Ticket text", raw.columns,
                        index=list(raw.columns).index("body") if "body" in raw.columns else 0)
subj_col = c2.selectbox("Subject (optional)", ["(none)"] + list(raw.columns),
                        index=(list(raw.columns).index("subject") + 1) if "subject" in raw.columns else 0)
label_col = c3.selectbox("Existing label (optional)", ["(none)"] + list(raw.columns),
                         index=(list(raw.columns).index("priority") + 1) if "priority" in raw.columns else 0)

if use_llm and not key:
    st.warning("Add your API key in the sidebar, or switch off the real classifier.")
    st.stop()

if use_llm:
    st.warning("Ticket text will be sent to Anthropic for classification. "
               "Do not upload data you are not permitted to send to a third party.")

# --------------------------------------------------------------------- run
if st.button("Run through the graph", type="primary"):
    import random, time
    import triage_graph as tg

    if use_llm:
        os.environ["ANTHROPIC_API_KEY"] = key
        import classifier

    sample = raw.head(n)
    rows, prog = [], st.progress(0.0)

    for i, (_, r) in enumerate(sample.iterrows()):
        body = str(r[body_col] or "").strip()
        subj = None if subj_col == "(none)" else (
            None if pd.isna(r[subj_col]) else str(r[subj_col]))

        # --- ingress (runs regardless of whether a model is present) ---
        reasons = []
        if not subj:
            reasons.append("no subject")
        if len(body) < 60:
            rows.append({"n": i + 1, "text": body[:90], "priority": None,
                         "confidence": None, "gate": "stopped at ingress",
                         "flags": "body too short to score", "secs": 0.0,
                         "label": None if label_col == "(none)" else r[label_col]})
            prog.progress((i + 1) / len(sample)); continue

        t0 = time.time()
        if use_llm:
            try:
                res = classifier.classify(subject=subj, body=body,
                                          model=model, log_path=None)
                pri, conf = res.get("priority"), float(res.get("confidence", 0))
                why = res.get("rationale", "")
            except Exception as e:
                st.error(f"Classifier failed on row {i+1}: {type(e).__name__}: {e}")
                st.stop()
        else:
            rnd = random.Random(body)
            pri = rnd.choice(["low", "medium", "high"])
            conf = round(rnd.uniform(0.4, 0.95), 2)
            why = "STUB — not a real judgement"

        # --- Gate A: machine ---
        if pri is None or conf < 0.6:
            gate = "Gate A refused"
        elif pri == "high":
            gate = "held at Gate B"      # --- Gate B: human ---
        else:
            gate = "auto-dispatched"

        flags = []
        if reasons:
            flags.append("scored_while_degraded")
        rows.append({"n": i + 1, "text": (subj or body)[:90], "priority": pri,
                     "confidence": conf, "gate": gate, "why": why,
                     "flags": ",".join(flags), "secs": round(time.time() - t0, 2),
                     "label": None if label_col == "(none)" else r[label_col]})
        prog.progress((i + 1) / len(sample))

    st.session_state.res = pd.DataFrame(rows)

# ----------------------------------------------------------------- results
if (d := st.session_state.get("res")) is not None:
    st.divider()
    scored = d[d.priority.notna()]

    m = st.columns(4)
    m[0].metric("Scored", f"{len(scored)}/{len(d)}")
    m[1].metric("Stopped by a control", int(
        d.gate.isin(["stopped at ingress", "Gate A refused"]).sum()))
    m[2].metric("Held at Gate B", int((d.gate == "held at Gate B").sum()))
    m[3].metric("Mean seconds", f"{d.secs.mean():.2f}")

    a, b = st.columns(2)
    with a:
        st.subheader("Priority mix")
        if len(scored):
            st.bar_chart(scored.priority.value_counts())
    with b:
        st.subheader("Where tickets ended up")
        st.bar_chart(d.gate.value_counts())

    if d.label.notna().any() and len(scored):
        st.subheader("Against the uploaded label")
        j = scored[scored.label.notna()]
        st.caption("Agreement, not accuracy — unless you know those labels were "
                   "assigned by a human against a written rubric.")
        st.metric("Agreement", f"{(j.label.astype(str) == j.priority).mean():.1%}")
        st.dataframe(pd.crosstab(j.label, j.priority), use_container_width=True)

    st.subheader("Held at Gate B — awaiting a human")
    held = d[d.gate == "held at Gate B"]
    if held.empty:
        st.caption("Nothing halted. No `high` calls in this batch.")
    else:
        st.caption(f"{len(held)} tickets suspended. In the full graph these are "
                   "checkpointed and cannot proceed without a recorded decision.")
        for _, h in held.iterrows():
            with st.expander(f"{PILL['high']} #{h.n} — {h.text}"):
                st.write(h.get("why", ""))
                st.radio("Decision", ["Confirm high", "Override to medium",
                                      "Override to low"], key=f"g{h.n}",
                         horizontal=True)

    st.subheader("All runs")
    st.dataframe(d.drop(columns=["why"], errors="ignore"),
                 use_container_width=True, height=320)
    st.download_button("Download results", d.to_csv(index=False),
                       "triage_results.csv", "text/csv")
