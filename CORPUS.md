# External corpus — preparation runbook

How the three public Hugging Face datasets become `corpus.csv`, and how to
run them through the graph and the Streamlit bench.

This corpus is for **operability testing**, not scoring. None of the three
sources carries usable ground truth — see "Why none of these are eval sets"
at the end.

---

## 0. Prerequisites

Use the project venv, not the one in `Downloads/`. They diverge.

```bash
cd governed-triage-graph
source .venv/bin/activate
pip install pandas huggingface_hub fsspec
```

`fsspec` is what lets pandas resolve `hf://` URIs. `huggingface_hub` is
needed on top of it for parquet reads. `read_json` works without the
latter, which is why the JSON sources loaded before the parquet one did.

No Hugging Face account or token is needed — all three datasets are public.
If a fetch returns **429 Too Many Requests**, that is the anonymous rate
limit. Wait ten minutes and retry rather than setting up auth.

---

## 1. Download the three sources

Run each as a single shell line. Outer quotes single, inner double — that
is what stops zsh treating `hf://` as a glob pattern.

**Loukh1 — the primary corpus.** 3,770 rows, Alpaca format
(`instruction` / `input` / `response`).

```bash
python -c 'import pandas as pd; pd.read_json("hf://datasets/Loukh1/incidentsV1/combined_incidents.json").to_csv("incidents.csv", index=False)'
```

**ServiceNow — second schema.** 500 rows, parquet.

```bash
python -c 'import pandas as pd; pd.read_parquet("hf://datasets/6StringNinja/synthetic-servicenow-incidents/data/train-00000-of-00001.parquet").to_csv("servicenow_incidents.csv", index=False)'
```

**SOC alerts — degenerate-input case.** 100 rows, JSON Lines. Drop
`analyst_name` at download; it is a person-shaped field with no reason to
enter a test corpus.

```bash
python -c 'import pandas as pd; pd.read_json("hf://datasets/HaseebDev/soc_alert_incident_tickets/demo_soc_alert_incident_tickets.jsonl", lines=True).drop(columns=["analyst_name"]).to_csv("soc_alerts.csv", index=False)'
```

For reproducibility later, pin the commit rather than tracking `main` —
these authors push updates:

```python
from huggingface_hub import hf_hub_download
hf_hub_download(repo_id="...", filename="...", repo_type="dataset",
                revision="<commit-sha>", local_dir="data/raw")
```

Note `repo_type="dataset"`. The default is models and you get a 404
without it.

**Security note.** Use `read_json` / `read_parquet` against explicit file
paths, not `load_dataset()` on unknown repos. `load_dataset` will execute a
loading script if the repo ships one. Hugging Face disclosed an incident in
July 2026 where a malicious dataset abused exactly that path for code
execution in their processing infrastructure. Treat third-party dataset
repos as untrusted input.

---

## 2. Sniff-test before trusting anything

Run this on every new corpus **before** building on it. The SOC set looked
fine until this check ran.

```bash
python -c '
import pandas as pd
d = pd.read_csv("FILE.csv")
print(d.shape)
for c in d.columns:
    print(c, d[c].nunique(), f"{d[c].isna().mean():.0%} null")
print()
print(d.iloc[0].to_dict())'
```

What kills a corpus:

- **Distinct body count far below row count.** Loukh1: 3,687 of 3,770,
  fine. ServiceNow: 97 of 500, so dedupe. SOC: effectively 1 of 100,
  unusable.
- **A sentence repeated across most rows.** Check any phrase that looks
  templated: `d.body.str.contains("<phrase>").mean()`. The SOC set returned
  1.0 for "ransomware file signature" while its `threat_category` varied
  randomly — labels uncorrelated with text.
- **A label that is a deterministic function of other columns.** If
  priority equals impact × urgency every time, it is a lookup table, not a
  judgement. Predicting it is predicting a formula.
- **Post-triage fields.** Anything written *after* the incident was worked
  (`response`, `resolution`, `analyst_assessment`, `containment_steps`)
  leaks the answer and must be excluded from model input.

---

## 3. Build the corpus

```bash
python build_corpus.py
```

Merges the three into `corpus.csv`, normalised to the graph's state schema:
separate `subject` and `body`, never a concatenated blob. The `ingress`
node checks `subject_present` to set `degraded`, so collapsing the fields
would make every row look identical to that check.

Per-source handling:

| source | subject | body |
|---|---|---|
| Loukh1 | text before `Log:`, `Ticket:` prefix stripped | the log excerpt |
| ServiceNow | `short_description` | `description` |
| SOC | empty, deliberately | `threat_category` + affected system |

Also done here: mojibake normalisation (`â€"` → `-`), a `non_ascii` flag,
dedupe on body, and a `source` column so any failure is traceable to its
origin.

Expected output — 3,884 rows: 3,687 Loukh1, 97 ServiceNow, 100 SOC.

Verify the header before going further:

```bash
head -1 corpus.csv
# source,ticket_id,subject,body,has_label,urgency,impact,priority,non_ascii
```

**Known artifact.** 183 Loukh1 rows (5%) have an empty subject because
they write "See ELK log:" rather than "Log:", so the split marker misses
and everything falls to body. Those rows land on the degraded path. Fix by
widening the split to a regex if it matters; left as-is for operability
testing, but it means degraded-path counts are ~183 higher than genuinely
subject-less tickets would give.

---

## 4. CLI run

```bash
python run_triage.py --stub --data corpus.csv --batch 3884 \
    --out runs/full_out.csv --auto-approve
```

Flags that matter:

- `--stub` installs deterministic placeholder nodes seeded on `ticket_id`.
  No model calls, no cost, reproducible.
- `--batch` is **a sample size, not a cap**. It calls
  `df.sample(n, random_state=3)` without replacement, so a value above the
  row count raises `ValueError`, and a smaller value draws a fixed random
  subset — not the first n rows.
- `--auto-approve` is required for batch. Without it every `high` call
  blocks on stdin at Gate B.

Start with `--batch 20` before committing to the full run.

Analyse the output — note `d["flags"]`, not `d.flags`, which collides with
a pandas property:

```bash
python -c '
import pandas as pd
d = pd.read_csv("runs/full_out.csv")
print(d.shape)
print(d.machine_priority.value_counts())
print(d["flags"].value_counts(dropna=False))
print(d["elapsed_s"].describe()[["mean","max"]])'
```

### Gate B, interactively

The batch run rubber-stamps. To exercise the override path:

```bash
python run_triage.py --stub --data corpus.csv --ticket LK-01752
```

At the prompt, **type `low` then Enter**. Pressing Enter alone accepts the
machine's call and tests nothing. A correct override prints:

```
LK-01752 → high  conf 0.69 | human: low  ⚠ OVERRIDE | flags: human_overrode_machine
```

`--ticket` needs `--data` alongside it, or it searches the default file and
reports "no ticket".

---

## 5. Streamlit bench

Two edits to `app.py`, both in the `load()` function near line 29.

```python
@st.cache_data
def load(path="tickets_en.csv"):
    d = pd.read_csv(path)
    for col in ("priority", "queue", "type", "tags", "answer"):
        if col not in d.columns:
            d[col] = None          # columns tickets_en.csv has, corpus.csv doesn't
    return d

source = st.sidebar.selectbox("Corpus", ["tickets_en.csv", "corpus.csv"])
df = load(source)
```

Delete the original bare `df = load()` further down the file — it runs
after the selectbox and silently reverts to the default.

```bash
grep -n "df = load" app.py    # should show exactly one line
streamlit run app.py
```

If a change does not take effect, `@st.cache_data` is holding the old
frame. Clear cache from the top-right menu, or restart the server.

### What works on `corpus.csv`

| mode | works | note |
|---|---|---|
| Single ticket | yes | full node trace and both gates. The useful one |
| Batch | runs | `truth` is `None` for unlabelled rows, so the confusion matrix is empty |
| Label gold set | no | line ~296 joins against `tickets_en.csv` regardless of selection |

The gold-set labelling path needs separate work before Loukh1 tickets can
be labelled through the UI.

---

## Why none of these are eval sets

Recorded so this is not relitigated later.

**SOC alerts** — no severity field, and the same scenario sentence in 100%
of rows with `threat_category` attached at random. `analyst_assessment` is
Faker word-salad. Kept only because its absent subject exercises the
degraded path.

**ServiceNow** — no `priority` field. It has `urgency` and `impact`, from
which priority must be derived through the ITIL matrix, making it a lookup
table 100% of the time. That is the same property that killed the
UCI/Kaggle corpus at diligence. Labels are SDV-generated and sampled from a
joint model rather than reasoned from the text; spot checks found a blocked
merchant onboarding at urgency 3 / impact 1.

**Loukh1** — genuinely good text (asymmetric routing, SIGTERM grace-period
exhaustion, NetworkPolicy audit gaps) with no labels at all. It is an
input corpus and a candidate hand-labelling pool. Note it is IT
infrastructure, while this repo's own corpus is customer support where
"priority" means commercial urgency — so it is out-of-*domain*, not just
out-of-distribution. A performance drop on it would not cleanly separate
poor generalisation from domain shift.
