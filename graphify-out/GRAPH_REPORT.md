# Graph Report - governed-triage-graph  (2026-09-05)

## Corpus Check
- cluster-only mode — file stats not available

## Summary
- 342 nodes · 476 edges · 18 communities (15 shown, 2 thin omitted)
- Extraction: 96% EXTRACTED · 4% INFERRED · 0% AMBIGUOUS · INFERRED: 17 edges (avg confidence: 0.85)
- Token cost: 0 input · 0 output

## Graph Freshness
- Built from commit: `f8e37248`
- Run `git rev-parse HEAD` and compare to check if the graph is stale.
- Run `graphify update .` after code changes (no API cost).

## Community Hubs (Navigation)
- gold_excluded
- triage_graph.py
- temperature_sweep.py
- grounding.py
- PolicyIndex
- classifier.py
- build_corpus.py
- classifier_mlx.py
- eval_harness.py
- manifest.json
- predict_mlx.py
- classifier_old.py
- grounding_old.py
- calibrate_grounding.py
- run_triage.py
- prepare_data.py
- make_datasets.py

## God Nodes (most connected - your core abstractions)
1. `gold_excluded` - 101 edges
2. `build_graph()` - 17 edges
3. `run_sweep()` - 14 edges
4. `SweepPoint` - 13 edges
5. `TriageState` - 12 edges
6. `_install_stubs()` - 10 edges
7. `PolicyIndex` - 8 edges
8. `ingress()` - 8 edges
9. `route()` - 8 edges
10. `main()` - 8 edges

## Surprising Connections (you probably didn't know these)
- `_install_stubs()` --indirect_call--> `audit()`  [INFERRED]
  run_triage.py → triage_graph.py
- `_install_stubs()` --indirect_call--> `gate_a()`  [INFERRED]
  run_triage.py → triage_graph.py
- `_install_stubs()` --indirect_call--> `ingress()`  [INFERRED]
  run_triage.py → triage_graph.py
- `_install_stubs()` --indirect_call--> `route()`  [INFERRED]
  run_triage.py → triage_graph.py
- `main()` --calls--> `build_graph()`  [EXTRACTED]
  merge_langgraph_topology.py → triage_graph.py

## Import Cycles
- None detected.

## Communities (18 total, 2 thin omitted)

### Community 0 - "gold_excluded"
Cohesion: 0.02
Nodes (101): gold_excluded, TK00198, TK00526, TK00751, TK00798, TK00816, TK00836, TK00848 (+93 more)

### Community 1 - "triage_graph.py"
Cohesion: 0.11
Nodes (33): BaseModel, main(), Inject the compiled LangGraph topology into graphify's graph.json. WHY THIS…, after_ingress(), audit(), build_graph(), classify(), DataQuality (+25 more)

### Community 2 - "temperature_sweep.py"
Cohesion: 0.10
Nodes (30): ClassifyFn, classify(), extract_label(), load_golden_set(), main(), Path, Expects JSON: [{"id": "...", "text": "...", "truth": "P2"}, ...] If your golden…, Replace the body with your existing call if you already have one. The only… (+22 more)

### Community 3 - "grounding.py"
Cohesion: 0.16
Nodes (18): bands(), line(), main(), profile(), Before/after: did optimising accuracy degrade the governance properties? python…, run(), _anchors(), _backend() (+10 more)

### Community 4 - "PolicyIndex"
Cohesion: 0.14
Nodes (13): index(), PolicyHit, PolicyIndex, Path, Policy retrieval channel for the `retrieve` node — backed by a graphify graph.…, True when the graph was built from a different commit than HEAD. A retrieval…, Seed on literal term overlap with the ticket, then walk `depth` hops. Returns…, Audit flags, in the same shape grounding.flags() returns. (+5 more)

### Community 5 - "classifier.py"
Cohesion: 0.10
Nodes (20): Hosted demo — upload any ticket CSV and run it through the governed triage…, fresh_app(), load(), cache_data, Visual test bench for the governed triage graph. pip install streamlit…, Stubs fill the unwritten nodes. When a real classifier is chosen it replaces…, build_messages(), classify() (+12 more)

### Community 6 - "build_corpus.py"
Cohesion: 0.31
Nodes (9): load_loukh1(), load_servicenow(), load_soc(), main(), DataFrame, Merge the downloaded incident CSVs into a corpus matching the graph's state…, Ticket line becomes subject, log excerpt becomes body. The `input` field looks…, Native subject/body split. Deduped - ~97 distinct descriptions of 500. (+1 more)

### Community 7 - "classifier_mlx.py"
Cohesion: 0.28
Nodes (7): classify(), _parse(), MLX backend — a locally served, optionally LoRA-adapted model. Start the server…, Classify one ticket. Returns the same shape as the other backends., rubric.md is the prompt's source of truth. Fall back if absent., Extract the label. Returns parse_error=True if nothing usable came back., _rubric()

### Community 8 - "eval_harness.py"
Cohesion: 0.36
Nodes (8): baseline(), main(), Eval harness. Complete and runnable — this is infrastructure, not the lesson.…, Share of tickets scored LOWER than truth. The asymmetric error: a high called…, TF-IDF + logistic regression. The bar an LLM must clear to be worth its cost., report(), score(), under_classification()

### Community 9 - "manifest.json"
Cohesion: 0.22
Nodes (8): counts, test, train, valid, label_source, seed, test_ids_sha256, what_this_trains

### Community 10 - "predict_mlx.py"
Cohesion: 0.33
Nodes (8): call_model(), held_out(), load_done(), main(), parse_label(), DataFrame, Reproduce eval_harness.baseline()'s test split exactly., Return (label, ok). ok=False means we fell back.

### Community 11 - "classifier_old.py"
Cohesion: 0.39
Nodes (7): build_messages(), classify(), client(), Anthropic, Real classifier for the triage graph. Wire this into triage_graph.classify().…, Returns priority, rationale, confidence, plus tokens/latency/cost., rubric()

### Community 12 - "grounding_old.py"
Cohesion: 0.33
Nodes (6): check(), flags(), Is the model's stated reason actually about this ticket? A rationale is a…, Returns a grounding score plus what could not be found in the ticket., Audit flags for the graph's audit node., _words()

### Community 13 - "calibrate_grounding.py"
Cohesion: 0.50
Nodes (3): label(), load_calls(), Derive the grounded/weak cutoffs from YOUR judgements. Do not inherit them.…

### Community 14 - "run_triage.py"
Cohesion: 0.10
Nodes (19): fresh_app(), load(), cache_data, Visual test bench for the governed triage graph. pip install streamlit…, Stubs fill the unwritten nodes. When a real classifier is chosen it replaces…, fresh_app(), load(), cache_data (+11 more)

## Knowledge Gaps
- **107 isolated node(s):** `TK00198`, `TK00526`, `TK00751`, `TK00798`, `TK00816` (+102 more)
  These have ≤1 connection - possible missing edges or undocumented components. (Counts symbols only; 207 node(s) total have ≤1 connection when file, concept and rationale nodes are included.)
- **2 thin communities (<3 nodes) omitted from report** — run `graphify query` to explore isolated nodes.

## Suggested Questions
_Questions this graph is uniquely positioned to answer:_

- **Why does `gold_excluded` connect `gold_excluded` to `manifest.json`?**
  _High betweenness centrality (0.101) - this node is a cross-community bridge._
- **Why does `build_graph()` connect `triage_graph.py` to `classifier.py`, `run_triage.py`?**
  _High betweenness centrality (0.010) - this node is a cross-community bridge._
- **Are the 10 inferred relationships involving `build_graph()` (e.g. with `after_ingress()` and `audit()`) actually correct?**
  _`build_graph()` has 10 INFERRED edges - model-reasoned connections that need verification._
- **What connects `TK00198`, `TK00526`, `TK00751` to the rest of the system?**
  _107 weakly-connected nodes found - possible documentation gaps or missing edges._
- **Should `gold_excluded` be split into smaller, more focused modules?**
  _Cohesion score 0.019801980198019802 - nodes in this community are weakly interconnected._
- **Should `triage_graph.py` be split into smaller, more focused modules?**
  _Cohesion score 0.11092436974789915 - nodes in this community are weakly interconnected._
- **Should `temperature_sweep.py` be split into smaller, more focused modules?**
  _Cohesion score 0.09523809523809523 - nodes in this community are weakly interconnected._