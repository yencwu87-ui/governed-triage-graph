"""
Derive the grounded/weak cutoffs from YOUR judgements. Do not inherit them.

    python calibrate_grounding.py --label      # you judge 40 rationales blind
    python calibrate_grounding.py --fit        # cutoffs from those judgements

Produces grounding_bands.json. Until that file exists, every grounding result
carries the flag `grounding_bands_not_calibrated`, because a threshold nobody
derived is a threshold nobody can defend.
"""
from __future__ import annotations
import argparse, json, pathlib, random, sys
import pandas as pd

LOG, OUT, JUDG = "calls.jsonl", "grounding_bands.json", "grounding_judgements.csv"


def load_calls():
    p = pathlib.Path(LOG)
    if not p.exists():
        sys.exit(f"{LOG} not found — run the classifier over some tickets first")
    rows = [json.loads(l) for l in p.read_text().splitlines() if l.strip()]
    return [r for r in rows if r.get("rationale") and not r.get("parse_error")]


def label():
    tickets = pd.read_csv("tickets_en.csv").set_index("ticket_id")
    calls = load_calls()
    done = set()
    if pathlib.Path(JUDG).exists():
        done = set(pd.read_csv(JUDG).idx)
    random.seed(3); random.shuffle(calls)

    print("\n  For each one: would you ACCEPT this rationale as being about this\n"
          "  ticket? Paraphrase is fine. Judge the substance, not the wording.\n"
          "  [y] accept   [n] reject   [s] skip   [q] save and quit\n")
    out = []
    for i, c in enumerate(calls):
        if i in done:
            continue
        tid = c.get("ticket_id")
        t = tickets.loc[tid] if tid in tickets.index else None
        if t is None:
            continue
        print("  " + "-" * 66)
        print(f"  TICKET  {str(t.subject)[:70]}")
        print(f"          {str(t.body)[:300]}")
        print(f"\n  MODEL SAID: {c['rationale']}")
        a = input("\n  accept? [y/n/s/q] ").strip().lower()
        if a == "q":
            break
        if a in ("y", "n"):
            out.append({"idx": i, "ticket_id": tid, "rationale": c["rationale"],
                        "accept": a == "y"})
        if len(out) >= 40:
            break
    if out:
        df = pd.DataFrame(out)
        if pathlib.Path(JUDG).exists():
            df = pd.concat([pd.read_csv(JUDG), df], ignore_index=True)
        df.to_csv(JUDG, index=False)
        print(f"\n  {len(df)} judgements saved to {JUDG}. "
              f"Aim for 40+, then run --fit\n")


def fit():
    import grounding
    if not pathlib.Path(JUDG).exists():
        sys.exit("no judgements yet — run --label first")
    j = pd.read_csv(JUDG)
    tickets = pd.read_csv("tickets_en.csv").set_index("ticket_id")
    if len(j) < 20:
        print(f"  ⚠  only {len(j)} judgements. Cutoffs from this few are weak "
              f"evidence — 40+ is the bar.\n")

    scored = []
    for _, r in j.iterrows():
        t = tickets.loc[r.ticket_id]
        res = grounding.check(r.rationale, t.subject, t.body)
        scored.append({"score": res["score"], "accept": bool(r.accept),
                       "method": res["method"]})
    d = pd.DataFrame(scored)
    acc, rej = d[d.accept].score, d[~d.accept].score
    print(f"\n  method: {d.method.iloc[0]}   n={len(d)} "
          f"({d.accept.sum()} accepted / {(~d.accept).sum()} rejected)")
    if acc.empty or rej.empty:
        sys.exit("  need both accepted and rejected examples to separate them")
    print(f"  accepted  median {acc.median():.3f}   10th pct {acc.quantile(.10):.3f}")
    print(f"  rejected  median {rej.median():.3f}   90th pct {rej.quantile(.90):.3f}")

    # grounded floor: catch 90% of what you accepted.
    # weak floor: below 90% of what you rejected.
    grounded = round(float(acc.quantile(0.10)), 3)
    weak = round(float(rej.quantile(0.90)), 3)
    if weak >= grounded:                      # overlapping distributions
        mid = (acc.median() + rej.median()) / 2
        grounded, weak = round(float(mid) + 0.03, 3), round(float(mid) - 0.03, 3)
        print("\n  ⚠  Accepted and rejected scores OVERLAP. The checker only weakly\n"
              "     separates what you accept from what you reject. Bands were set\n"
              "     around the midpoint, but the honest reading is that this metric\n"
              "     is not yet discriminative. Record that.")

    bands = {"grounded": grounded, "weak": weak, "derived": True,
             "n_judgements": int(len(d)), "method": d.method.iloc[0],
             "separation": round(float(acc.median() - rej.median()), 3)}
    pathlib.Path(OUT).write_text(json.dumps(bands, indent=2))
    print(f"\n  grounded >= {grounded}   weak >= {weak}   -> {OUT}")
    print(f"  separation (median accepted - median rejected): {bands['separation']:+.3f}")
    print("  Below about 0.10 the metric is close to noise.\n")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--label", action="store_true")
    ap.add_argument("--fit", action="store_true")
    a = ap.parse_args()
    label() if a.label else fit() if a.fit else ap.print_help()
