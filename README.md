# governed-triage-graph

A support-ticket triage system built as an explicit LangGraph state machine,
with the governance controls as first-class nodes rather than wrappers.

The point is not the classifier. The point is that **the control plane is
separable from the model** — the whole graph runs with the model swapped out,
so the governance can be tested independently of the thing it governs.

## The graph

```
ingress ──▶ retrieve ──▶ classify ──▶ [Gate A] ──▶ route ──▶ [Gate B] ──▶ audit
    │                                     │                     │
    └────────────▶ escalate_ungrounded ◀──┘                     │
                                              human writes to state
```

**Gate A — machine safety.** A conditional edge. Refuses to auto-proceed below
a confidence floor or with no precedent. Evaluated by the machine alone.

**Gate B — human accountability.** A LangGraph `interrupt()`. Every `high` call
halts the graph. Execution suspends, state is checkpointed, and nothing
proceeds without a human writing a decision into state. The checkpoint records
who resumed it and whether they agreed.

Those are two different things and most systems conflate them. An eval score
is not an audit trail; a signature is not accountability.

## Results

Scored against the hand-labelled gold set, not the dataset's synthetic labels.

| run | accuracy | macro-F1 | under-class. | cost/run |
|---|---|---|---|---|
| majority-class floor | 0.4051 | — | — | 0 |
| TF-IDF + logistic regression | 0.6421 | 0.6112 | 0.1383 | ~0 |
| LLM classifier (rubric v0.1) | _tbd_ | _tbd_ | _tbd_ | _tbd_ |

Thresholds: accuracy ≥ 0.700, macro-F1 ≥ 0.650, under-classification ≤ 0.200.

**Read the confusion matrix, not the headline.** The baseline's population-wide
under-classification of 13.8% clears the ceiling while it under-scores 30.8% of
genuinely high-priority tickets. Population metrics hide asymmetric failure.

An LLM that scores ~0.64 has not beaten this baseline — it has matched a bag of
words at several thousand times the cost per run. Cost belongs in the same row
as accuracy or the comparison is dishonest.

## Run it

```bash
pip install -r requirements.txt
python make_datasets.py          # see DATA.md — corpus is not in this repo
streamlit run app.py             # visual test bench
```

The test bench has three modes: single-ticket trace, batch with confusion
matrix, and blind gold-set labelling. Stub mode runs the full control plane
with placeholder node bodies, so the gates can be exercised with no API key.

```bash
python eval_harness.py --baseline
python eval_harness.py --preds preds.csv
python run_triage.py --ticket TK00042
```

## Findings

Things this build surfaced, kept here because negative results are the point.

1. **Rejected the first corpus.** The UCI/Kaggle ServiceNow event log (141,712
   rows, 24,918 incidents) turned out to have no free-text field, `cmdb_ci`
   99.7% missing, and a `priority` that matched the ITIL impact×urgency matrix
   **100.0% of the time** — a lookup table, not a judgement, with 95% of
   incidents sitting at the Medium/Medium default. Predicting it would have
   been predicting a formula. Killed at diligence, before any build.

2. **Checkpoint serialisation is undeclared.** LangGraph warns that the Pydantic
   models in state are unregistered msgpack types and will be refused in a
   future version. The Gate B approval record depends on that checkpoint, so the
   audit trail currently rests on a deprecation-flagged format. Open.

3. **Confidence may not be measuring anything.** Gate A routes on the model's
   self-reported confidence. If the distribution is narrow, the gate is
   decorative. Checked explicitly rather than assumed — see `WIRE_IN.md`.

4. **Parse failures stop the ticket.** A malformed classifier response flags and
   halts rather than defaulting to `medium`. A silent fallback passes evals and
   fails in production.

## Limitations

- Corpus is synthetic and its labels are generator-assigned. See `DATA.md`.
- Customer support, not ITIL incidents. "Priority" means commercial urgency.
- Retrieval similarity is lexical; no embedding model is wired in yet.
- No per-node cost or latency instrumentation yet.
- Gold set is n=100, single annotator, no inter-rater agreement.

## Files

| | |
|---|---|
| `triage_graph.py` | state schema, nodes, edges, gates |
| `classifier.py` | LLM call with token/latency/cost capture |
| `rubric.md` | versioned severity rubric — the prompt's source of truth |
| `run_triage.py` | CLI runner, handles the Gate B interrupt/resume |
| `app.py` | Streamlit test bench and labelling UI |
| `eval_harness.py` | metrics, thresholds, ledger |
| `make_datasets.py` | regenerates the working sets |
| `gold/labels.csv` | hand-labelled reference set |
# governed-triage-graph
