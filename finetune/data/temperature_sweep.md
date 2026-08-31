# Temperature sweep — eval report

Generated 2026-08-30T02:22:23+00:00 · commit `unknown`

## 1. Scope and method

Golden set `467a4cd4032155af`, 30 items. Model `llama3.1:8b`. Each temperature was run 5 times with fixed seeds and all predictions retained. Accuracy ceilings are evaluated against the worst observed run, not the mean, because a control that breaches on one run in 5 has not passed.

## 2. Thresholds

| Control | Threshold | Basis |
|---|---|---|
| Exact match | ≥ 60% | worst run |
| Under-classification | ≤ 20% | worst run |
| Self-consistency | ≥ 90% | modal agreement |

## 3. Results

| Temp | Exact (mean ± sd) | Exact (worst) | Under (mean) | Under (worst) | Consistency | Verdict |
|---|---|---|---|---|---|---|
| 0.0 | 75.3% ± 11.2% | 63.3% | 24.7% | 36.7% | 80.0% | FAIL |
| 0.2 | 74.0% ± 10.4% | 56.7% | 26.0% | 43.3% | 79.3% | FAIL |
| 0.4 | 62.0% ± 8.0% | 53.3% | 38.0% | 46.7% | 74.7% | FAIL |
| 0.6 | 64.0% ± 9.2% | 53.3% | 36.0% | 46.7% | 71.3% | FAIL |
| 0.8 | 61.3% ± 11.2% | 46.7% | 38.7% | 53.3% | 75.3% | FAIL |
| 1.0 | 60.7% ± 8.6% | 50.0% | 39.3% | 50.0% | 73.3% | FAIL |

## 4. Failures by sweep point

- **T=0.0** — under 36.7% > 20%; consistency 80.0% < 90%
- **T=0.2** — exact 56.7% < 60%; under 43.3% > 20%; consistency 79.3% < 90%
- **T=0.4** — exact 53.3% < 60%; under 46.7% > 20%; consistency 74.7% < 90%
- **T=0.6** — exact 53.3% < 60%; under 46.7% > 20%; consistency 71.3% < 90%
- **T=0.8** — exact 46.7% < 60%; under 53.3% > 20%; consistency 75.3% < 90%
- **T=1.0** — exact 50.0% < 60%; under 50.0% > 20%; consistency 73.3% < 90%

## 5. Selected configuration

**No configuration cleared all three thresholds.** The classifier should not be treated as a passing control at any temperature in this sweep. See section 4.

## 6. Interpretation limits

- At n=30, one item flipping moves exact match by 3.3%. Differences smaller than roughly 6.7% between adjacent temperatures should not be read as a trend.
- Consistency below 100% at T=0.0 indicates non-determinism from sources other than sampling — floating-point ordering, batching, or expert routing. This is expected behaviour, not a defect, but it bounds reproducibility.
- The golden set is synthetic. Results transfer to production only to the extent the synthetic distribution matches live ticket text.

## 7. Change control

Temperature is a configuration item that changes control effectiveness without changing the model artefact. Any change to the selected value invalidates this report and requires a re-run before the thresholds can be treated as met.

## 8. Provenance

```json
[
  {
    "model": "llama3.1:8b",
    "temperature": 0.0,
    "top_p": 0.95,
    "seeds": [
      1000,
      1001,
      1002,
      1003,
      1004
    ],
    "golden_set_hash": "467a4cd4032155af",
    "golden_set_size": 30,
    "git_commit": "unknown",
    "timestamp_utc": "2026-08-30T02:22:23+00:00"
  },
  {
    "model": "llama3.1:8b",
    "temperature": 0.2,
    "top_p": 0.95,
    "seeds": [
      1000,
      1001,
      1002,
      1003,
      1004
    ],
    "golden_set_hash": "467a4cd4032155af",
    "golden_set_size": 30,
    "git_commit": "unknown",
    "timestamp_utc": "2026-08-30T02:22:23+00:00"
  },
  {
    "model": "llama3.1:8b",
    "temperature": 0.4,
    "top_p": 0.95,
    "seeds": [
      1000,
      1001,
      1002,
      1003,
      1004
    ],
    "golden_set_hash": "467a4cd4032155af",
    "golden_set_size": 30,
    "git_commit": "unknown",
    "timestamp_utc": "2026-08-30T02:22:23+00:00"
  },
  {
    "model": "llama3.1:8b",
    "temperature": 0.6,
    "top_p": 0.95,
    "seeds": [
      1000,
      1001,
      1002,
      1003,
      1004
    ],
    "golden_set_hash": "467a4cd4032155af",
    "golden_set_size": 30,
    "git_commit": "unknown",
    "timestamp_utc": "2026-08-30T02:22:23+00:00"
  },
  {
    "model": "llama3.1:8b",
    "temperature": 0.8,
    "top_p": 0.95,
    "seeds": [
      1000,
      1001,
      1002,
      1003,
      1004
    ],
    "golden_set_hash": "467a4cd4032155af",
    "golden_set_size": 30,
    "git_commit": "unknown",
    "timestamp_utc": "2026-08-30T02:22:23+00:00"
  },
  {
    "model": "llama3.1:8b",
    "temperature": 1.0,
    "top_p": 0.95,
    "seeds": [
      1000,
      1001,
      1002,
      1003,
      1004
    ],
    "golden_set_hash": "467a4cd4032155af",
    "golden_set_size": 30,
    "git_commit": "unknown",
    "timestamp_utc": "2026-08-30T02:22:23+00:00"
  }
]
```