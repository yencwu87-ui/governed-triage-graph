"""
Build the LoRA training split. Run from the project root.

    python finetune/prepare_data.py

Writes finetune/data/{train,valid,test}.jsonl in MLX chat format, plus a
manifest recording exactly which tickets went where.

Two rules enforced here, not left to discipline:
  1. The gold set is EXCLUDED from every split. You cannot train on your
     own reference set and then quote a number against it.
  2. test.jsonl is sealed. It is written once and read once, at the end.
     If you look at it while iterating it stops being a test set.
"""
from __future__ import annotations
import hashlib, json, pathlib
import numpy as np, pandas as pd

ROOT = pathlib.Path(__file__).resolve().parent
OUT = ROOT / "data"; OUT.mkdir(exist_ok=True)
RUBRIC = (ROOT.parent / "rubric.md").read_text()

SYSTEM = ("You are a triage classifier. Apply this rubric literally.\n\n" + RUBRIC +
          '\n\nReply with one JSON object: {"priority": ..., "rationale": ..., '
          '"confidence": ...}')

df = pd.read_csv(ROOT.parent / "tickets_en.csv")
gold = set(pd.read_csv(ROOT.parent / "gold_set_template.csv").ticket_id)
pool = df[~df.ticket_id.isin(gold)].copy()
print(f"  {len(df):,} tickets · {len(gold)} held out as gold · {len(pool):,} available")

rng = np.random.RandomState(17)
pool = pool.iloc[rng.permutation(len(pool))].reset_index(drop=True)
n_test = 1300
n_valid = 2000
test, valid, train = pool[:n_test], pool[n_test:n_test + n_valid], pool[n_test + n_valid:]


def record(r):
    subj = "(missing)" if pd.isna(r.subject) else r.subject
    # The rationale is a TEMPLATE, not a model judgement. We are distilling the
    # dataset's LABEL, and inventing reasoning text would train the model to
    # fabricate. Confidence is deliberately omitted from the target for the
    # same reason — see finetune/README.md.
    target = {"priority": r.priority}
    return {"messages": [
        {"role": "system", "content": SYSTEM},
        {"role": "user", "content": f"SUBJECT: {subj}\nBODY: {r.body}"},
        {"role": "assistant", "content": json.dumps(target)},
    ]}


for name, part in (("train", train), ("valid", valid), ("test", test)):
    p = OUT / f"{name}.jsonl"
    with p.open("w") as f:
        for _, r in part.iterrows():
            f.write(json.dumps(record(r)) + "\n")
    print(f"  {name:<6} {len(part):>6,}  {p}")

manifest = {
    "seed": 17,
    "gold_excluded": sorted(gold),
    "counts": {"train": len(train), "valid": len(valid), "test": len(test)},
    "test_ids_sha256": hashlib.sha256(
        ",".join(sorted(test.ticket_id)).encode()).hexdigest()[:16],
    "label_source": "dataset priority column — GENERATOR-ASSIGNED, not human",
    "what_this_trains": "distils the dataset generator's labelling function "
                        "into an 8B model. It does not make triage more correct.",
}
(OUT / "manifest.json").write_text(json.dumps(manifest, indent=2))
print(f"\n  manifest written. test set fingerprint {manifest['test_ids_sha256']}")
print("  Seal test.jsonl. One read, at the end.\n")
