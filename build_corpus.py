"""
Merge the downloaded incident CSVs into a corpus matching the graph's
state schema: separate `subject` and `body`, not a single text blob.

    python build_corpus.py

Output columns:
    source          which dataset the row came from
    ticket_id       identifier within that source
    subject         maps to state["subject"] - may be empty
    body            maps to state["body"]
    has_label       whether ground truth exists
    urgency/impact  ITIL inputs, servicenow rows only
    priority        derived P1-P4, servicenow rows only
    non_ascii       flag for encoding artifacts

Post-triage fields are excluded from subject/body because they give away
the answer: response (loukh1), resolution (servicenow),
analyst_assessment and containment_steps (soc_alerts).

Note on soc_alerts: those rows have no natural subject line, so they
exercise the ingress `degraded` path. Their body text is also
generator-collapsed - the same scenario sentence in all 100 rows - so
treat them as a degenerate-input case, not as coverage.
"""

from pathlib import Path

import pandas as pd

# ITIL priority matrix. In this dataset 1 = highest on both axes.
PRIORITY = {
    (1, 1): "P1", (1, 2): "P2", (1, 3): "P3",
    (2, 1): "P2", (2, 2): "P3", (2, 3): "P4",
    (3, 1): "P3", (3, 2): "P4", (3, 3): "P4",
}


def load_loukh1(path=Path("incidents.csv")) -> pd.DataFrame:
    """Ticket line becomes subject, log excerpt becomes body.

    The `input` field looks like:
        Ticket: <one line>. Log: <json or log text>
    Split on the first 'Log:' so the graph sees a real subject/body pair.
    """
    d = pd.read_csv(path).drop_duplicates("input").reset_index(drop=True)
    raw = d["input"].astype(str)

    parts = raw.str.split("Log:", n=1, expand=True)
    subject = (parts[0]
               .str.replace(r"^\s*Ticket:\s*", "", regex=True)
               .str.strip()
               .str.rstrip("."))

    if parts.shape[1] > 1:
        body = parts[1].fillna("").str.strip()
        # No 'Log:' marker - keep the whole string as body instead.
        missing = parts[1].isna()
        body = body.mask(missing, raw)
        subject = subject.mask(missing, "")
    else:
        body = raw
        subject = pd.Series([""] * len(d))

    return pd.DataFrame({
        "source": "loukh1",
        "ticket_id": [f"LK-{i:05d}" for i in range(len(d))],
        "subject": subject.values,
        "body": body.values,
        "has_label": False,
    })


def load_servicenow(path=Path("servicenow_incidents.csv")) -> pd.DataFrame:
    """Native subject/body split. Deduped - ~97 distinct descriptions of 500."""
    d = pd.read_csv(path).drop_duplicates("description")
    return pd.DataFrame({
        "source": "servicenow",
        "ticket_id": d["number"].astype(str).values,
        "subject": d["short_description"].astype(str).values,
        "body": d["description"].astype(str).values,
        "has_label": True,
        "urgency": d["urgency"].values,
        "impact": d["impact"].values,
        "priority": [PRIORITY.get((u, i), "UNMAPPED")
                     for u, i in zip(d["urgency"], d["impact"])],
    })


def load_soc(path=Path("soc_alerts.csv")) -> pd.DataFrame:
    """No subject field exists - left empty on purpose to hit the ingress
    degraded branch. analyst_name is dropped entirely."""
    d = pd.read_csv(path)
    body = (d["threat_category"].astype(str)
            + ". Affected system: " + d["system_impacted"].astype(str))
    return pd.DataFrame({
        "source": "soc_alerts",
        "ticket_id": d["alert_id"].astype(str).values,
        "subject": "",
        "body": body.values,
        "has_label": False,
    })


def main(include_soc: bool = True) -> None:
    frames = []
    for name, loader in (("incidents.csv", load_loukh1),
                         ("servicenow_incidents.csv", load_servicenow)):
        if Path(name).exists():
            frames.append(loader())
        else:
            print(f"skipped, not found: {name}")

    if include_soc and Path("soc_alerts.csv").exists():
        frames.append(load_soc())

    c = pd.concat(frames, ignore_index=True)

    for col in ("subject", "body"):
        c[col] = (c[col].fillna("").astype(str)
                  .str.replace("\u00e2\u20ac\u201d", "-", regex=False)
                  .str.replace("\u00e2\u20ac\u2122", "'", regex=False)
                  .str.strip())

    c["non_ascii"] = ~(c.subject + c.body).map(str.isascii)
    c.to_csv("corpus.csv", index=False)

    print(f"\nwrote corpus.csv - {len(c)} rows")
    print(c.groupby(["source", "has_label"]).size().to_string())
    print(f"\nempty subject (hits degraded path): {(c.subject == '').sum()}")
    print(f"non-ascii rows: {c.non_ascii.sum()}")
    print(f"median body chars: {c.body.str.len().median():.0f}")
    lab = c[c.has_label]
    if len(lab):
        print(f"\nlabelled subset (n={len(lab)}):")
        print(lab.priority.value_counts().sort_index().to_string())


if __name__ == "__main__":
    main()
