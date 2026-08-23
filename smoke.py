"""
Terminal smoke test — five tickets, real model, no graph, no Streamlit.

    python smoke.py            # 5 tickets
    python smoke.py 20         # more

Confirms the key works, the rubric loads, output parses, and — most usefully —
shows the confidence spread. If that spread is narrow, Gate A is decorative.
"""
import os, sys, statistics as stats
import pandas as pd

BACKEND = os.environ.get("TRIAGE_BACKEND", "local")   # local | api

if BACKEND == "local":
    import classifier_local as classifier
    ok, msg = classifier.health()
    print(f"\n  {msg}")
    if not ok:
        sys.exit("")
else:
    if not os.environ.get("ANTHROPIC_API_KEY"):
        sys.exit("Set your key first:  export ANTHROPIC_API_KEY=sk-ant-...")
    import classifier

n = int(sys.argv[1]) if len(sys.argv) > 1 else 5
model = os.environ.get("TRIAGE_MODEL",
    "llama3.1:8b" if BACKEND == "local" else "claude-haiku-4-5-20251001")
df = pd.read_csv("tickets_en.csv").sample(n, random_state=5)

print(f"\n  model: {model}   rubric: {classifier.RUBRIC_PATH}   tickets: {n}\n")
print(f"  {'ticket':<9} {'model':<8} {'label':<8} {'conf':>5}  {'sec':>5}  {'tok':>6}")
print("  " + "-" * 52)

rows, tin, tout = [], 0, 0
for _, r in df.iterrows():
    try:
        res = classifier.classify(
            subject=None if pd.isna(r.subject) else r.subject,
            body=r.body, model=model)
    except Exception as e:
        m = str(e).lower()
        if "authentication" in m or "401" in m:
            sys.exit("\n  Key rejected. Check ANTHROPIC_API_KEY.")
        if "credit" in m or "billing" in m:
            sys.exit("\n  No credit on the account. Top up at console.anthropic.com.")
        sys.exit(f"\n  {type(e).__name__}: {e}")

    agree = "=" if res["priority"] == r.priority else "x"
    print(f"  {r.ticket_id:<9} {str(res['priority']):<8} {r.priority:<8} "
          f"{res['confidence']:>5.2f}  {res['latency_s']:>5.2f}  "
          f"{res['in_tokens']+res['out_tokens']:>6} {agree}")
    rows.append(res); tin += res["in_tokens"]; tout += res["out_tokens"]
    if res.get("parse_error"):
        print(f"           ^ PARSE FAILURE: {res['rationale'][:70]}")

confs = [r["confidence"] for r in rows]
agree = sum(a["priority"] == b for a, b in zip(rows, df.priority)) / len(rows)

print("\n  " + "-" * 52)
print(f"  agreement with dataset labels : {agree:.0%}   (agreement, not accuracy)")
print(f"  confidence  min {min(confs):.2f}  median {stats.median(confs):.2f}  max {max(confs):.2f}")
print(f"  spread      {max(confs)-min(confs):.2f}")
print(f"  tokens      {tin:,} in / {tout:,} out")
print(f"  logged to   calls.jsonl\n")

if max(confs) - min(confs) < 0.15:
    print("  ⚠  Confidence is nearly flat. Gate A routes on this number, so the\n"
          "     machine-safety gate is currently decorative. That is a finding —\n"
          "     write it in the failure log before you tune anything.\n")

print("  Sample rationale:")
print(f"    {rows[0]['rationale'][:200]}\n")
