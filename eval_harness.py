"""
Eval harness. Complete and runnable — this is infrastructure, not the lesson.

    python eval_harness.py --baseline          reproduce the bar to beat
    python eval_harness.py --preds preds.csv   score your graph

Governance thresholds (edit in THRESHOLDS, don't edit them to pass):
  exact accuracy          >= 0.700
  under-classification    <= 0.200
  macro-F1                >= 0.650
"""
from __future__ import annotations
import argparse, json, sys
import numpy as np, pandas as pd

DATA = "tickets_en.csv"
ORDER = {"low": 0, "medium": 1, "high": 2}
#THRESHOLDS = {"accuracy": 0.700, "under_classification": 0.200, "macro_f1": 0.650}
THRESHOLDS = {"accuracy": 0.700, "under_classification": 0.200,
              "macro_f1": 0.650, "kappa": 0.200}

def under_classification(y_true, y_pred) -> float:
    """Share of tickets scored LOWER than truth. The asymmetric error:
    a high called medium is an outage; a low called high is a wasted page."""
    t = np.array([ORDER[v] for v in y_true])
    p = np.array([ORDER[v] for v in y_pred])
    return float((p < t).mean())

from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, cohen_kappa_score
def score(y_true, y_pred, label="run") -> dict:
    from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
    m = {
        "label": label,
        "n": len(y_true),
        "accuracy": round(accuracy_score(y_true, y_pred), 4),
        "macro_f1": round(f1_score(y_true, y_pred, average="macro"), 4),"kappa": round(cohen_kappa_score(y_true, y_pred), 4),
        "under_classification": round(under_classification(y_true, y_pred), 4),
        "over_classification": round(under_classification(y_pred, y_true), 4),
    }
    m["passes"] = (m["accuracy"] >= THRESHOLDS["accuracy"]
                   and m["under_classification"] <= THRESHOLDS["under_classification"]
                   and m["macro_f1"] >= THRESHOLDS["macro_f1"])
    labs = ["low", "medium", "high"]
    m["confusion"] = pd.DataFrame(
        confusion_matrix(y_true, y_pred, labels=labs),
        index=[f"t_{l}" for l in labs], columns=[f"p_{l}" for l in labs]
    ).to_dict()
    return m


def report(m: dict) -> None:
    print(f"\n=== {m['label']}  (n={m['n']}) ===")
    for k in ("accuracy", "macro_f1", "kappa", "under_classification", "over_classification"):
        thr = THRESHOLDS.get(k)
        mark = ""
        if thr is not None:
            ok = m[k] <= thr if k == "under_classification" else m[k] >= thr
            mark = f"   [{'PASS' if ok else 'FAIL'} vs {thr}]"
        print(f"  {k:22s} {m[k]:.4f}{mark}")
    print(f"  {'VERDICT':22s} {'PASS' if m['passes'] else 'FAIL'}")
    print("\n  confusion:")
    print(pd.DataFrame(m["confusion"]).to_string(header=True).replace("\n", "\n  "))


def baseline(path: str = DATA) -> dict:
    """TF-IDF + logistic regression. The bar an LLM must clear to be worth its cost."""
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import train_test_split
    df = pd.read_csv(path)
    df["txt"] = df.subject.fillna("") + " \n " + df.body.fillna("")
    Xtr, Xte, ytr, yte = train_test_split(
        df.txt, df.priority, test_size=0.25, random_state=0, stratify=df.priority)
    v = TfidfVectorizer(max_features=30000, ngram_range=(1, 2), min_df=2)
    m = LogisticRegression(max_iter=2000, C=2).fit(v.fit_transform(Xtr), ytr)
    out = score(yte, m.predict(v.transform(Xte)), "tfidf+logreg baseline")
    maj = yte.value_counts(normalize=True).iloc[0]
    print(f"\n  majority-class floor: {maj:.4f}")
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--baseline", action="store_true")
    ap.add_argument("--preds", help="csv with ticket_id,priority")
    ap.add_argument("--truth", default=DATA)
    ap.add_argument("--data", default=DATA)
    ap.add_argument("--ledger", default="ledger.jsonl")
    ap.add_argument("--label", default="run")
    a = ap.parse_args()

    if a.baseline:
        m = baseline(a.data)
    elif a.preds:
        p = pd.read_csv(a.preds)[["ticket_id", "priority"]].rename(columns={"priority": "pred"})
        t = pd.read_csv(a.truth)[["ticket_id", "priority"]]
        j = t.merge(p, on="ticket_id", how="inner")
        if j.empty:
            sys.exit("no overlapping ticket_id between preds and truth")
        print(f"\n  matched {len(j)} of {len(p)} predictions")
        m = score(j.priority, j.pred, a.label)
    else:
        sys.exit("pass --baseline or --preds")

    report(m)
    with open(a.ledger, "a") as f:
        f.write(json.dumps({k: v for k, v in m.items() if k != "confusion"}) + "\n")
    print(f"\n  appended to {a.ledger}")


if __name__ == "__main__":
    main()
