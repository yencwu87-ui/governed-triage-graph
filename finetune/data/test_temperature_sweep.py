"""
Tests for temperature_sweep.

Deterministic fixtures with hand-computed expected values. The point is to
verify the arithmetic behind the threshold claims, not to test a model.

Run:  python3 test_temperature_sweep.py
"""

from pathlib import Path
import tempfile

from temperature_sweep import (
    GoldenItem, SweepConfig, Provenance, RunMetrics, SweepPoint,
    run_sweep, score_run, modal_agreement, rank, write_report,
    SEVERITY_ORDER, THRESHOLD_EXACT_MIN, THRESHOLD_UNDER_MAX,
    THRESHOLD_CONSISTENCY_MIN,
)

FAILURES: list[str] = []


def check(name: str, actual, expected, tol: float = 1e-9):
    ok = abs(actual - expected) < tol if isinstance(expected, float) else actual == expected
    print(f"  {'PASS' if ok else 'FAIL'}  {name}: {actual!r} (expected {expected!r})")
    if not ok:
        FAILURES.append(name)


# 10 items, 5 x P2 and 5 x P3. Hand-checkable denominators.
ITEMS = (
    [GoldenItem(f"A{i}", f"text-a-{i}", "P2") for i in range(5)]
    + [GoldenItem(f"B{i}", f"text-b-{i}", "P3") for i in range(5)]
)
TRUTH = {i.id: i.truth for i in ITEMS}


# --------------------------------------------------------------------------
print("\n[1] rank ordering")
# --------------------------------------------------------------------------
check("P4 is lowest", rank("P4"), 0)
check("P1 is highest", rank("P1"), len(SEVERITY_ORDER) - 1)
check("P2 outranks P3", rank("P2") > rank("P3"), True)
check("garbage is -1", rank("NOT_A_LABEL"), -1)


# --------------------------------------------------------------------------
print("\n[2] score_run — perfect classifier")
# --------------------------------------------------------------------------
m = score_run(dict(TRUTH), ITEMS, seed=0)
check("exact", m.exact, 1.0)
check("under", m.under, 0.0)
check("over", m.over, 0.0)
check("unparseable", m.unparseable, 0.0)


# --------------------------------------------------------------------------
print("\n[3] score_run — always one band too low")
# --------------------------------------------------------------------------
# P2 -> P3, P3 -> P4. Every item under-classified.
one_low = {i.id: SEVERITY_ORDER[rank(i.truth) - 1] for i in ITEMS}
m = score_run(one_low, ITEMS, seed=0)
check("exact", m.exact, 0.0)
check("under", m.under, 1.0)
check("over", m.over, 0.0)


# --------------------------------------------------------------------------
print("\n[4] score_run — mixed, hand-counted")
# --------------------------------------------------------------------------
# 6 correct, 2 under (P2->P3), 1 over (P3->P2), 1 unparseable.
mixed = dict(TRUTH)
mixed["A0"] = "P3"          # under
mixed["A1"] = "P3"          # under
mixed["B0"] = "P2"          # over
mixed["B1"] = "SEV-BANANA"  # unparseable
m = score_run(mixed, ITEMS, seed=0)
check("exact 6/10", m.exact, 0.6)
check("under 2/10", m.under, 0.2)
check("over 1/10", m.over, 0.1)
check("unparseable 1/10", m.unparseable, 0.1)
check("components sum to 1", m.exact + m.under + m.over + m.unparseable, 1.0)


# --------------------------------------------------------------------------
print("\n[5] score_run — unparseable is NOT counted as correct")
# --------------------------------------------------------------------------
all_junk = {i.id: "<think>hmm</think>" for i in ITEMS}
m = score_run(all_junk, ITEMS, seed=0)
check("exact", m.exact, 0.0)
check("unparseable", m.unparseable, 1.0)


# --------------------------------------------------------------------------
print("\n[6] modal_agreement")
# --------------------------------------------------------------------------
check("identical repeats -> 1.0",
      modal_agreement({"x": ["P2"] * 5}), 1.0)
check("even split of 2 -> 0.5",
      modal_agreement({"x": ["P2", "P3", "P2", "P3"]}), 0.5)
check("4 of 5 agree -> 0.8",
      modal_agreement({"x": ["P2", "P2", "P2", "P2", "P3"]}), 0.8)
check("averages across items",
      modal_agreement({"x": ["P2"] * 4, "y": ["P2", "P2", "P3", "P3"]}), 0.75)
check("empty -> 0.0", modal_agreement({}), 0.0)

# Consistency is independent of correctness: consistently wrong scores 1.0.
check("consistently wrong is still consistent",
      modal_agreement({i.id: ["P4"] * 5 for i in ITEMS}), 1.0)


# --------------------------------------------------------------------------
print("\n[7] worst-case vs mean — the core governance claim")
# --------------------------------------------------------------------------
# Four runs at 15% under, one at 25%. Mean = 17% (passes 20% ceiling),
# worst = 25% (breaches). The point must FAIL.
prov = Provenance("test", 0.0, 0.95, [0], "hash", 10, "abc", "now")
runs = [RunMetrics(s, 0.80, u, 0.0, 0.0)
        for s, u in enumerate([0.15, 0.15, 0.15, 0.15, 0.25])]
pt = SweepPoint(provenance=prov, runs=runs, consistency=0.95)

check("mean under would pass", pt.under_mean <= THRESHOLD_UNDER_MAX, True)
check("worst under breaches", pt.under_worst > THRESHOLD_UNDER_MAX, True)
check("point FAILS on worst case", pt.passes, False)
check("failure names the breach",
      any("under" in f for f in pt.failures()), True)


# --------------------------------------------------------------------------
print("\n[8] threshold logic — each control fails independently")
# --------------------------------------------------------------------------
good = RunMetrics(0, 0.90, 0.05, 0.05, 0.0)

pt_ok = SweepPoint(prov, [good], consistency=0.99)
check("all clear -> pass", pt_ok.passes, True)
check("no failures listed", pt_ok.failures(), [])

pt_low_exact = SweepPoint(prov, [RunMetrics(0, 0.50, 0.05, 0.0, 0.0)], consistency=0.99)
check("low exact fails", pt_low_exact.passes, False)

pt_low_consistency = SweepPoint(prov, [good], consistency=0.50)
check("low consistency fails", pt_low_consistency.passes, False)
check("consistency breach is named",
      any("consistency" in f for f in pt_low_consistency.failures()), True)


# --------------------------------------------------------------------------
print("\n[9] run_sweep — deterministic stub, end to end")
# --------------------------------------------------------------------------
def perfect(text, *, temperature, seed):
    return next(i.truth for i in ITEMS if i.text == text)

pts = run_sweep(perfect, ITEMS, SweepConfig(temperatures=(0.0, 0.5), repeats=3),
                model_name="stub")
check("one point per temperature", len(pts), 2)
check("repeats recorded", len(pts[0].runs), 3)
check("perfect stub -> exact 1.0", pts[0].exact_mean, 1.0)
check("perfect stub -> consistency 1.0", pts[0].consistency, 1.0)
check("sd is zero for deterministic stub", pts[0].exact_sd, 0.0)
check("all predictions retained",
      sum(len(v) for v in pts[0].predictions.values()), 30)  # 10 items x 3 repeats
check("seeds differ across repeats",
      len(set(pts[0].provenance.seeds)), 3)
check("golden set hash stable across temps",
      pts[0].provenance.golden_set_hash == pts[1].provenance.golden_set_hash, True)


# --------------------------------------------------------------------------
print("\n[10] report — no passing config must say so")
# --------------------------------------------------------------------------
def always_under(text, *, temperature, seed):
    truth = next(i.truth for i in ITEMS if i.text == text)
    return SEVERITY_ORDER[rank(truth) - 1]

bad_pts = run_sweep(always_under, ITEMS, SweepConfig(temperatures=(0.0,), repeats=2))
with tempfile.TemporaryDirectory() as d:
    p = write_report(bad_pts, Path(d) / "r.md")
    body = p.read_text()
check("report states no passing config",
      "No configuration cleared all three thresholds" in body, True)
check("report does not fabricate a recommendation",
      "selected —" not in body, True)

good_pts = run_sweep(perfect, ITEMS, SweepConfig(temperatures=(0.0,), repeats=2))
with tempfile.TemporaryDirectory() as d:
    p = write_report(good_pts, Path(d) / "r.md")
    body = p.read_text()
check("passing report names a temperature", "Temperature **0.0** selected" in body, True)
check("provenance embedded", "golden_set_hash" in body, True)
check("change control section present", "Change control" in body, True)


# --------------------------------------------------------------------------
print("\n" + "=" * 60)
if FAILURES:
    print(f"{len(FAILURES)} FAILED: {FAILURES}")
    raise SystemExit(1)
print("All checks passed.")
