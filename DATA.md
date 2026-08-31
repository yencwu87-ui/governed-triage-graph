# Data provenance

This repo contains no corpus. Everything is regenerated locally.

## Source

`Tobi-Bueck/customer-support-tickets` on Hugging Face, file
`aa_dataset-tickets-multi-lang-5-2-50-version.csv`. Licensed **CC-BY-NC-4.0**,
so it is not redistributed here.

**The labels are synthetic.** The tickets and their priority values were
generated, not assigned by human triagers. Accuracy against them is therefore
*agreement with a generator*, not accuracy. Every number in this repo is scored
against the hand-labelled gold set in `gold/` instead.

## Regenerate

```bash
# download the CSV from the dataset page, put it in ./raw/
python make_datasets.py
```

Produces `tickets_en.csv` (16,338 English tickets), `gold_set_template.csv`
(100 blinded, stratified) and `precedent_pool.csv` (600 balanced, with the
gold tickets excluded so no graded ticket can retrieve its own answer).

## Profile of the English subset

| | |
|---|---|
| rows | 16,338 |
| priority | medium 6,618 / high 6,346 / low 3,374 |
| majority-class floor | 40.5% |
| queue | 10 classes, 29% majority |
| body | 0% missing, median 377 chars |
| subject | **16% missing** — a natural degraded-input case |
| label leakage | priority words appear as whole words in ≤2% of own text |

## Scope note

This is a **customer support** corpus. "Priority" means commercial urgency,
not ITIL service impact. The graph is an incident-triage architecture applied
to support tickets; don't read severity semantics into the labels.
