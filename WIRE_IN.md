# Wiring the real classifier in

## 1. Install and set the key

```bash
pip install anthropic
export ANTHROPIC_API_KEY=sk-ant-...
```

Optional, for the cost column — check current pricing and set your own rates:

```bash
export RATE_IN=3.0        # USD per million input tokens
export RATE_OUT=15.0      # USD per million output tokens
export TRIAGE_MODEL=claude-sonnet-5
```

Leave the rates unset and cost logs as 0. That is honest. A guessed rate is not.

## 2. Replace the body of `classify()` in `triage_graph.py`

Keep the docstring. Replace `raise NotImplementedError` with:

```python
    import classifier
    r = classifier.classify(
        subject=state.get("subject"),
        body=state["body"],
        dq=state.get("data_quality"),
        precedents=state.get("precedents"),
        version="v0.1",
    )
    if r.get("parse_error") or r["priority"] not in ("low", "medium", "high"):
        return {"terminal_reason": "classifier returned an unusable response",
                "audit_flags": (state.get("audit_flags") or []) + ["classifier_parse_error"]}
    return {"triage": TriageCall(priority=r["priority"],
                                 rationale=r["rationale"],
                                 confidence=float(r["confidence"]))}
```

Note what that does on a bad response: it stops the ticket and flags it,
rather than substituting a default. A classifier that silently falls back to
"medium" on parse failure will pass your eval and fail in production.

## 3. Write `gate_a()`

Still yours. It needs a confidence floor, and you cannot pick one until you
have seen the distribution:

```bash
python - << 'EOF'
import json, pandas as pd
d = pd.DataFrame([json.loads(l) for l in open('calls.jsonl')])
print(d.confidence.describe())
print(d.groupby('priority').confidence.describe()[['count','mean','min','max']])
EOF
```

Run 50 tickets first, then look. **If the spread is narrow — say everything
between 0.8 and 0.9 — the gate is decorative and the model is not reporting
real uncertainty.** That is a finding to write down, not a number to tune around.

## 4. Order of operations

1. Label the gold set (`streamlit run app.py` → Label gold set). Do this first.
2. Rewrite `rubric.md` from your own rationales. Bump to v0.2.
3. Wire in the classifier. Run 20 tickets in the test bench with stub mode off.
4. Check the confidence distribution. Set the gate A floor. Write your reason.
5. Batch 100. Download predictions. `python eval_harness.py --preds preds.csv`
6. Compare against ledger row 0: accuracy 0.6421, macro-F1 0.6112.

If step 5 lands near 0.64, the LLM has matched a bag of words at a few
thousand times the cost. Record cost per run in the same ledger row and let
the comparison stand — that finding is worth more than a higher score.
