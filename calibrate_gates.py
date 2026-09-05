"""
Derive BODY_FLOOR, GATE_A_FLOOR and MIN_SIM from YOUR judgements and YOUR
gold set. Same discipline as calibrate_grounding.py — nothing here inherits
a default.

    python calibrate_gates.py --floor        # you judge 30 short tickets
    python calibrate_gates.py --sim          # you judge 30 retrieved neighbours
    python calibrate_gates.py --gate-a       # swept against the gold labels
    python calibrate_gates.py --fit          # write gate_thresholds.json

Produces gate_thresholds.json. Until it exists, triage_graph and precedents
flag every run as uncalibrated, because a threshold nobody derived is a
threshold nobody can defend.

WHAT IS AND IS NOT A JUDGEMENT CALL. --floor and --sim ask you things only
you can answer: is this ticket scoreable, is this neighbour relevant.
--gate-a asks nothing. It sweeps candidate floors against your gold labels
and reports what each one costs, because "which floor keeps
under-classification under the ceiling" is arithmetic, not taste. You still
pick the floor. The sweep just stops you picking a round number and
defending it later.
"""

from __future__ import annotations

import argparse
import json
import pathlib
import random
import sys

OUT = "gate_thresholds.json"
CALLS = "calls.jsonl"
GOLD = "gold_set_labelled.csv"
TICKETS = "tickets_en.csv"
J_FLOOR, J_SIM = "floor_judgements.csv", "sim_judgements.csv"

# From eval_harness.THRESHOLDS — the ceiling the gate exists to hold.
UNDER_CLASS_CEILING = 0.200
ORDER = {"low": 0, "medium": 1, "high": 2}


def _load(path: str):
    p = pathlib.Path(path)
    if not p.exists():
        sys.exit(f"{path} not found")
    return p


def _ask(prompt: str, keys: str) -> str:
    while True:
        r = input(prompt).strip().lower()
        if r in keys:
            return r


# ------------------------------------------------------------------- floor


def label_floor(n: int = 30) -> None:
    """Is this ticket scoreable from its body alone? You decide, blind."""
    import pandas as pd

    df = pd.read_csv(_load(TICKETS))
    df["chars"] = df.body.fillna("").str.len()
    # Sample across the short tail, where the cutoff actually lives.
    short = df[df.chars <= df.chars.quantile(0.25)].copy()
    random.seed(5)
    rows = short.sample(min(n, len(short)), random_state=5).to_dict("records")

    print("\n  Could a competent triager assign a severity from this body alone?\n"
          "  Judge the ticket, not its length — you will not be shown the count.\n"
          "  [y] scoreable   [n] not scoreable   [s] skip   [q] save and quit\n")
    out = []
    for r in rows:
        print("  " + "-" * 66)
        print(f"  {str(r['body'])[:400]}")
        k = _ask("  > ", "ynsq")
        if k == "q":
            break
        if k == "s":
            continue
        out.append({"ticket_id": r["ticket_id"], "chars": int(r["chars"]),
                    "scoreable": k == "y"})
    if out:
        pd.DataFrame(out).to_csv(J_FLOOR, index=False)
        print(f"\n  wrote {len(out)} judgements to {J_FLOOR}")


def fit_floor() -> dict | None:
    import pandas as pd

    p = pathlib.Path(J_FLOOR)
    if not p.exists():
        return None
    d = pd.read_csv(p)
    if d.scoreable.nunique() < 2:
        print(f"  BODY_FLOOR: you judged every sampled ticket "
              f"{'scoreable' if d.scoreable.all() else 'unscoreable'} — no cutoff "
              f"can be derived from that. Widen the sample.")
        return None

    # Lowest cutoff that rejects nothing you called scoreable.
    floor = int(d[d.scoreable].chars.min())
    lost = int((d[~d.scoreable].chars >= floor).sum())
    print(f"  BODY_FLOOR = {floor} chars")
    print(f"    shortest ticket you called scoreable: {floor}")
    print(f"    unscoreable tickets this floor still lets through: {lost} "
          f"of {int((~d.scoreable).sum())}")
    if lost:
        print("    those are the ones to look at — length is not the whole signal")
    return {"body_floor": floor, "n_judged": int(len(d))}


# --------------------------------------------------------------------- sim


def label_sim(n: int = 30) -> None:
    """Is this retrieved neighbour actually a precedent for this ticket?"""
    import pandas as pd
    import precedents

    gold = pd.read_csv(_load(GOLD))
    tickets = pd.read_csv(_load(TICKETS)).set_index("ticket_id")
    idx = precedents.PrecedentIndex()
    if idx.load_error:
        sys.exit(f"precedent index unavailable: {idx.load_error}")

    saved = precedents.MIN_SIM
    precedents.MIN_SIM = 0.0  # judge the whole range, not the survivors

    print("\n  Would you accept this as a precedent for the ticket above it?\n"
          "  Scores are hidden on purpose.\n"
          "  [y] relevant   [n] not relevant   [s] skip   [q] save and quit\n")
    out = []
    try:
        for _, g in gold.iterrows():
            if str(g.ticket_id) not in tickets.index:
                continue
            t = tickets.loc[str(g.ticket_id)]
            hits = idx.lookup(t.get("subject"), t.get("body", ""), k=3)
            for h in hits:
                print("  " + "=" * 66)
                print(f"  TICKET:    {str(t.get('body',''))[:220]}")
                print(f"  PRECEDENT: {h.text[:220]}")
                k = _ask("  > ", "ynsq")
                if k == "q":
                    raise KeyboardInterrupt
                if k == "s":
                    continue
                out.append({"ticket_id": str(g.ticket_id), "precedent_id": h.ticket_id,
                            "score": h.score, "relevant": k == "y"})
                if len(out) >= n:
                    raise KeyboardInterrupt
    except KeyboardInterrupt:
        pass
    finally:
        precedents.MIN_SIM = saved

    if out:
        pd.DataFrame(out).to_csv(J_SIM, index=False)
        print(f"\n  wrote {len(out)} judgements to {J_SIM}")


def fit_sim() -> dict | None:
    import pandas as pd

    p = pathlib.Path(J_SIM)
    if not p.exists():
        return None
    d = pd.read_csv(p)
    if d.relevant.nunique() < 2:
        print("  MIN_SIM: your judgements are all one way — no cutoff derivable.")
        return None

    # Sweep for the cutoff that best separates accept from reject.
    best, best_acc = None, -1.0
    for c in sorted(d.score.unique()):
        acc = ((d.score >= c) == d.relevant).mean()
        if acc > best_acc:
            best, best_acc = float(c), float(acc)

    overlap = d[d.relevant].score.min() <= d[~d.relevant].score.max()
    print(f"  MIN_SIM = {best:.3f}   (separates {best_acc:.0%} of your judgements)")
    print(f"    relevant   scores: {d[d.relevant].score.min():.3f}"
          f" – {d[d.relevant].score.max():.3f}")
    print(f"    irrelevant scores: {d[~d.relevant].score.min():.3f}"
          f" – {d[~d.relevant].score.max():.3f}")
    if overlap:
        print("    RANGES OVERLAP — no cutoff separates them cleanly. The score")
        print("    is not measuring what you are judging. That is a finding about")
        print("    the retriever, not a number to split the difference on.")
    return {"min_sim": best, "separation": best_acc,
            "ranges_overlap": bool(overlap), "n_judged": int(len(d))}


# ------------------------------------------------------------------ gate A


def sweep_gate_a() -> dict | None:
    """
    Sweep candidate floors against the gold labels. No judgement asked.

    For each floor: of the tickets that WOULD auto-proceed, how many were
    under-classified, and how many tickets get escalated to buy that.
    """
    import pandas as pd

    calls = pathlib.Path(CALLS)
    gold = pathlib.Path(GOLD)
    if not calls.exists() or not gold.exists():
        print(f"  GATE_A_FLOOR: needs both {CALLS} and {GOLD}.")
        print("    Run the classifier over the gold tickets first "
              "(WIRE_IN.md steps 1-4).")
        return None

    rows = [json.loads(l) for l in calls.read_text().splitlines() if l.strip()]
    c = pd.DataFrame([r for r in rows if not r.get("parse_error")])
    g = pd.read_csv(gold).rename(columns={"my_priority": "truth"})
    d = c.merge(g[["ticket_id", "truth"]], on="ticket_id", how="inner")
    d = d[d.truth.isin(ORDER) & d.priority.isin(ORDER)]
    if d.empty:
        print("  GATE_A_FLOOR: no overlap between calls.jsonl and the gold set.")
        return None

    spread = float(d.confidence.max() - d.confidence.min())
    print(f"  confidence spread: {d.confidence.min():.2f} – {d.confidence.max():.2f}"
          f"  (n={len(d)})")
    if spread < 0.20:
        print("    SPREAD IS NARROW. The model is not reporting real uncertainty,")
        print("    so any floor you set here is decorative. That is a finding to")
        print("    write down, not a number to tune around. — WIRE_IN.md step 3")

    print(f"\n  {'floor':>7} {'auto%':>7} {'under-class':>12} {'escalated':>10}")
    curve, chosen = [], None
    for f in [round(x * 0.05, 2) for x in range(int(d.confidence.min() / 0.05), 21)]:
        auto = d[d.confidence >= f]
        if auto.empty:
            continue
        under = float((auto.priority.map(ORDER) < auto.truth.map(ORDER)).mean())
        rate = len(auto) / len(d)
        curve.append({"floor": f, "auto_rate": rate, "under_class": under})
        ok = under <= UNDER_CLASS_CEILING
        if ok and chosen is None:
            chosen = f
        print(f"  {f:>7.2f} {rate:>6.0%} {under:>11.3f}{' OK' if ok else '   '}"
              f" {len(d)-len(auto):>9}")

    if chosen is None:
        print(f"\n  NO FLOOR holds under-classification at or below "
              f"{UNDER_CLASS_CEILING}.")
        print("    Escalating everything is the only safe setting, which means")
        print("    the classifier is not fit for auto-proceed yet. Leave")
        print("    GATE_A_FLOOR as None and say so.")
        return {"gate_a_floor": None, "curve": curve, "spread": spread,
                "reason": "no floor meets the under-classification ceiling"}

    print(f"\n  GATE_A_FLOOR = {chosen}  (lowest floor holding under-class "
          f"<= {UNDER_CLASS_CEILING})")
    print("    Lowest, not safest — a higher floor escalates more and is more")
    print("    conservative. Pick deliberately and record why.")
    return {"gate_a_floor": chosen, "curve": curve, "spread": spread,
            "n": int(len(d))}


# --------------------------------------------------------------------- fit


def fit() -> None:
    print("\n  DERIVING GATE THRESHOLDS\n  " + "=" * 58 + "\n")
    out: dict = {}

    f = fit_floor()
    print()
    s = fit_sim()
    print()
    a = sweep_gate_a()

    if f:
        out["body_floor"] = f
    if s:
        out["min_sim"] = s
    if a:
        out["gate_a"] = a

    if not out:
        sys.exit("\n  nothing derivable yet — run --floor / --sim first, and the "
                 "classifier over the gold set for --gate-a\n")

    pathlib.Path(OUT).write_text(json.dumps(out, indent=2))
    print(f"\n  wrote {OUT}")
    print("  triage_graph and precedents read it on import; the "
          "'not_calibrated' flags\n  clear for whichever thresholds it contains.\n")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--floor", action="store_true", help="judge short tickets")
    ap.add_argument("--sim", action="store_true", help="judge retrieved neighbours")
    ap.add_argument("--gate-a", action="store_true", help="sweep against gold labels")
    ap.add_argument("--fit", action="store_true", help="write gate_thresholds.json")
    a = ap.parse_args()

    if a.floor:
        label_floor()
    elif a.sim:
        label_sim()
    elif a.gate_a:
        sweep_gate_a()
    elif a.fit:
        fit()
    else:
        ap.print_help()


if __name__ == "__main__":
    main()
