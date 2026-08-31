# Fine-tuning data

Not committed — derived from `tickets_en.csv`, same licence terms as the corpus.

Regenerate with `make_datasets.py`. Expected contents:

- `train.jsonl` — 12,938 rows, the 75% training split (`random_state=0`)
- `valid.jsonl` — validation split, required by `mlx_lm.lora`

Verify no overlap with the held-out eval set before training:

    python3 check_leakage.py     # expect: overlap: 0 of 12938
