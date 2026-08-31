"""
Temperature sweep and eval report generation for governed-incident-triage.

Drop-in module. Inject your existing classifier as a callable:

    def classify(ticket_text: str, *, temperature: float, seed: int) -> str:
        # your existing Ollama / HTTP call, returns a severity label
        ...

    records = run_sweep(classify, golden_set, SweepConfig())
    write_report(records, Path("reports/temperature_sweep.md"))

Design notes:
  - Every prediction is retained. Dispersion is the measurement, so nothing
    is collapsed at collection time.
  - Thresholds are ceilings, so they are evaluated against the WORST run in
    the repeat set, never the mean.
  - Provenance is captured per sweep point so any figure can be traced back
    to the exact configuration that produced it.
"""

from __future__ import annotations

import hashlib
import json
import statistics
import subprocess
from collections import Counter
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Sequence

# --------------------------------------------------------------------------
# Domain
# --------------------------------------------------------------------------

# Ordinal severity. Index position defines rank, so under-classification is
# simply a lower index than ground truth. Adjust to match your rubric.
SEVERITY_ORDER: list[str] = ["P4", "P3", "P2", "P1"]

# Governance thresholds. These are the numbers the harness passes or fails on.
THRESHOLD_EXACT_MIN = 0.60
THRESHOLD_UNDER_MAX = 0.20
THRESHOLD_CONSISTENCY_MIN = 0.90


def rank(label: str) -> int:
    try:
        return SEVERITY_ORDER.index(label)
    except ValueError:
        return -1  # unparseable — counted as a failure, not silently dropped


@dataclass(frozen=True)
class GoldenItem:
    id: str
    text: str
    truth: str


# --------------------------------------------------------------------------
# Configuration and provenance
# --------------------------------------------------------------------------

@dataclass
class SweepConfig:
    temperatures: Sequence[float] = (0.0, 0.2, 0.4, 0.6, 0.8, 1.0)
    repeats: int = 10
    top_p: float = 0.95
    base_seed: int = 1000


@dataclass
class Provenance:
    model: str
    temperature: float
    top_p: float
    seeds: list[int]
    golden_set_hash: str
    golden_set_size: int
    git_commit: str
    timestamp_utc: str


def _git_commit() -> str:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True, text=True, timeout=5,
        )
        return out.stdout.strip() or "unknown"
    except Exception:
        return "unknown"


def _hash_golden_set(items: Sequence[GoldenItem]) -> str:
    blob = json.dumps(
        [{"id": i.id, "text": i.text, "truth": i.truth} for i in items],
        sort_keys=True,
    ).encode()
    return hashlib.sha256(blob).hexdigest()[:16]


# --------------------------------------------------------------------------
# Metrics
# --------------------------------------------------------------------------

@dataclass
class RunMetrics:
    """Metrics for a single pass over the golden set."""
    seed: int
    exact: float
    under: float
    over: float
    unparseable: float


@dataclass
class SweepPoint:
    """All repeats at one temperature, plus derived statistics."""
    provenance: Provenance
    runs: list[RunMetrics]
    consistency: float
    predictions: dict[str, list[str]] = field(default_factory=dict)

    # -- central tendency ---------------------------------------------------
    @property
    def exact_mean(self) -> float:
        return statistics.fmean(r.exact for r in self.runs)

    @property
    def exact_sd(self) -> float:
        return statistics.stdev(r.exact for r in self.runs) if len(self.runs) > 1 else 0.0

    # -- worst case: what the thresholds are actually judged against --------
    @property
    def exact_worst(self) -> float:
        return min(r.exact for r in self.runs)

    @property
    def under_worst(self) -> float:
        return max(r.under for r in self.runs)

    @property
    def under_mean(self) -> float:
        return statistics.fmean(r.under for r in self.runs)

    @property
    def passes(self) -> bool:
        return (
            self.exact_worst >= THRESHOLD_EXACT_MIN
            and self.under_worst <= THRESHOLD_UNDER_MAX
            and self.consistency >= THRESHOLD_CONSISTENCY_MIN
        )

    def failures(self) -> list[str]:
        out = []
        if self.exact_worst < THRESHOLD_EXACT_MIN:
            out.append(f"exact {self.exact_worst:.1%} < {THRESHOLD_EXACT_MIN:.0%}")
        if self.under_worst > THRESHOLD_UNDER_MAX:
            out.append(f"under {self.under_worst:.1%} > {THRESHOLD_UNDER_MAX:.0%}")
        if self.consistency < THRESHOLD_CONSISTENCY_MIN:
            out.append(f"consistency {self.consistency:.1%} < {THRESHOLD_CONSISTENCY_MIN:.0%}")
        return out


def score_run(preds: dict[str, str], items: Sequence[GoldenItem], seed: int) -> RunMetrics:
    n = len(items)
    exact = under = over = bad = 0
    for item in items:
        p = preds[item.id]
        pr, tr = rank(p), rank(item.truth)
        if pr < 0:
            bad += 1
        elif pr == tr:
            exact += 1
        elif pr < tr:
            under += 1
        else:
            over += 1
    return RunMetrics(seed, exact / n, under / n, over / n, bad / n)


def modal_agreement(predictions: dict[str, list[str]]) -> float:
    """Mean fraction of repeats landing on each item's most common label.

    1.0 means every item produced the same answer every time. This is the
    determinism measure — it is independent of whether the answer is right.
    """
    if not predictions:
        return 0.0
    scores = []
    for labels in predictions.values():
        if not labels:
            continue
        top = Counter(labels).most_common(1)[0][1]
        scores.append(top / len(labels))
    return statistics.fmean(scores) if scores else 0.0


# --------------------------------------------------------------------------
# Sweep
# --------------------------------------------------------------------------

ClassifyFn = Callable[..., str]


def run_sweep(
    classify: ClassifyFn,
    items: Sequence[GoldenItem],
    config: SweepConfig = SweepConfig(),
    model_name: str = "unknown",
) -> list[SweepPoint]:
    gs_hash = _hash_golden_set(items)
    commit = _git_commit()
    points: list[SweepPoint] = []

    for temp in config.temperatures:
        seeds = [config.base_seed + i for i in range(config.repeats)]
        per_item: dict[str, list[str]] = {i.id: [] for i in items}
        runs: list[RunMetrics] = []

        for seed in seeds:
            preds = {}
            for item in items:
                label = classify(item.text, temperature=temp, seed=seed)
                preds[item.id] = label
                per_item[item.id].append(label)
            runs.append(score_run(preds, items, seed))

        points.append(SweepPoint(
            provenance=Provenance(
                model=model_name,
                temperature=temp,
                top_p=config.top_p,
                seeds=seeds,
                golden_set_hash=gs_hash,
                golden_set_size=len(items),
                git_commit=commit,
                timestamp_utc=datetime.now(timezone.utc).isoformat(timespec="seconds"),
            ),
            runs=runs,
            consistency=modal_agreement(per_item),
            predictions=per_item,
        ))

    return points


# --------------------------------------------------------------------------
# Report
# --------------------------------------------------------------------------

def write_report(points: list[SweepPoint], path: Path) -> Path:
    if not points:
        raise ValueError("no sweep points to report")

    p0 = points[0].provenance
    n = p0.golden_set_size
    k = len(points[0].runs)

    passing = [p for p in points if p.passes]
    recommended = max(passing, key=lambda p: p.exact_mean) if passing else None

    L: list[str] = []
    L.append("# Temperature sweep — eval report\n")
    L.append(f"Generated {p0.timestamp_utc} · commit `{p0.git_commit}`\n")

    L.append("## 1. Scope and method\n")
    L.append(
        f"Golden set `{p0.golden_set_hash}`, {n} items. Model `{p0.model}`. "
        f"Each temperature was run {k} times with fixed seeds and all predictions "
        f"retained. Accuracy ceilings are evaluated against the worst observed run, "
        f"not the mean, because a control that breaches on one run in {k} has not passed.\n"
    )

    L.append("## 2. Thresholds\n")
    L.append("| Control | Threshold | Basis |")
    L.append("|---|---|---|")
    L.append(f"| Exact match | ≥ {THRESHOLD_EXACT_MIN:.0%} | worst run |")
    L.append(f"| Under-classification | ≤ {THRESHOLD_UNDER_MAX:.0%} | worst run |")
    L.append(f"| Self-consistency | ≥ {THRESHOLD_CONSISTENCY_MIN:.0%} | modal agreement |")
    L.append("")

    L.append("## 3. Results\n")
    L.append("| Temp | Exact (mean ± sd) | Exact (worst) | Under (mean) | Under (worst) | Consistency | Verdict |")
    L.append("|---|---|---|---|---|---|---|")
    for p in points:
        verdict = "PASS" if p.passes else "FAIL"
        L.append(
            f"| {p.provenance.temperature:.1f} "
            f"| {p.exact_mean:.1%} ± {p.exact_sd:.1%} "
            f"| {p.exact_worst:.1%} "
            f"| {p.under_mean:.1%} "
            f"| {p.under_worst:.1%} "
            f"| {p.consistency:.1%} "
            f"| {verdict} |"
        )
    L.append("")

    L.append("## 4. Failures by sweep point\n")
    any_fail = False
    for p in points:
        f = p.failures()
        if f:
            any_fail = True
            L.append(f"- **T={p.provenance.temperature:.1f}** — " + "; ".join(f))
    if not any_fail:
        L.append("No sweep point breached a threshold.")
    L.append("")

    L.append("## 5. Selected configuration\n")
    if recommended:
        t = recommended.provenance.temperature
        L.append(
            f"Temperature **{t:.1f}** selected — highest mean exact match among "
            f"configurations clearing all three thresholds on worst-run basis "
            f"({recommended.exact_mean:.1%} mean, {recommended.under_worst:.1%} worst-case "
            f"under-classification, {recommended.consistency:.1%} consistency).\n"
        )
    else:
        L.append(
            "**No configuration cleared all three thresholds.** The classifier "
            "should not be treated as a passing control at any temperature in this "
            "sweep. See section 4.\n"
        )

    L.append("## 6. Interpretation limits\n")
    L.append(
        f"- At n={n}, one item flipping moves exact match by {1/n:.1%}. Differences "
        f"smaller than roughly {2/n:.1%} between adjacent temperatures should not be "
        f"read as a trend.\n"
        f"- Consistency below 100% at T=0.0 indicates non-determinism from sources "
        f"other than sampling — floating-point ordering, batching, or expert routing. "
        f"This is expected behaviour, not a defect, but it bounds reproducibility.\n"
        f"- The golden set is synthetic. Results transfer to production only to the "
        f"extent the synthetic distribution matches live ticket text.\n"
    )

    L.append("## 7. Change control\n")
    L.append(
        "Temperature is a configuration item that changes control effectiveness "
        "without changing the model artefact. Any change to the selected value "
        "invalidates this report and requires a re-run before the thresholds can "
        "be treated as met.\n"
    )

    L.append("## 8. Provenance\n")
    L.append("```json")
    L.append(json.dumps([asdict(p.provenance) for p in points], indent=2))
    L.append("```")

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(L), encoding="utf-8")
    return path


def dump_raw(points: list[SweepPoint], path: Path) -> Path:
    """Persist every prediction. The report is a view; this is the evidence."""
    payload = [
        {
            "provenance": asdict(p.provenance),
            "runs": [asdict(r) for r in p.runs],
            "consistency": p.consistency,
            "predictions": p.predictions,
        }
        for p in points
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return path
