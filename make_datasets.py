"""Regenerate the working sets from the downloaded source CSV. See DATA.md."""
import pathlib, numpy as np, pandas as pd

SRC = pathlib.Path("raw/aa_dataset-tickets-multi-lang-5-2-50-version.csv")
if not SRC.exists():
    raise SystemExit(f"missing {SRC} — download it first, see DATA.md")

df = pd.read_csv(SRC, low_memory=False)
en = df[df.language == "en"].copy().reset_index(drop=True)
en.insert(0, "ticket_id", ["TK%05d" % i for i in range(1, len(en) + 1)])
tags = [c for c in en.columns if c.startswith("tag_")]
en["tags"] = en[tags].apply(lambda r: "|".join(str(x) for x in r if pd.notna(x)), axis=1)
en["subject_missing"] = en.subject.isna()
cols = ["ticket_id","subject","body","answer","type","queue","priority","tags","subject_missing"]
en[cols].to_csv("tickets_en.csv", index=False)

rng = np.random.RandomState(7)
idx = [i for _, g in en.groupby("priority")
       for i in rng.choice(g.index.values, min(34, len(g)), replace=False)]
gold = en.loc[idx].sample(frac=1, random_state=7).head(100)
g = gold[["ticket_id","subject","body","type","queue"]].copy()   # priority withheld
for c in ("my_priority","my_rationale","confidence","borderline"):
    g[c] = ""
g.to_csv("gold_set_template.csv", index=False)

pool = en[~en.ticket_id.isin(gold.ticket_id)]
rng = np.random.RandomState(11)
pidx = [i for _, gg in pool.groupby("priority")
        for i in rng.choice(gg.index.values, min(200, len(gg)), replace=False)]
p = pool.loc[pidx].sample(frac=1, random_state=11).reset_index(drop=True)
p["text"] = ("TICKET " + p.ticket_id
    + "\nSUBJECT: " + p.subject.fillna("(none)")
    + "\nBODY: " + p.body.fillna("")
    + "\nTYPE: " + p.type + "   QUEUE: " + p.queue
    + "\nRESOLVED PRIORITY: " + p.priority
    + "\nAGENT RESPONSE: " + p.answer.fillna(""))
p[["ticket_id","text","priority","queue","type"]].to_csv("precedent_pool.csv", index=False)

assert not set(p.ticket_id) & set(gold.ticket_id), "gold leaked into precedent pool"
print(f"tickets_en {len(en)} | gold {len(g)} | precedents {len(p)} | no leakage")
