# graphify in this repo — what it maps, what it doesn't, and where it goes

Two separate uses, done separately because they are separate things. Part 1
maps this repository. Part 2 puts a graph inside the pipeline as a retrieval
channel. Neither depends on the other.

---

## Part 1 — Repo map

### Running it

Code is parsed locally with tree-sitter. No API key, no LLM, nothing leaves
the machine — which is why `--code-only` is the right flag here even though
the repo has markdown worth indexing. The docs pass needs a model, and the
rubric is not a document to hand to a hosted API without deciding to.

```bash
pip install graphifyy                          # package is graphifyy, command is graphify
graphify extract . --code-only --no-label
graphify cluster-only . --no-label
graphify export callflow-html
```

`--no-label` keeps communities as `Community N`. Naming them needs a backend;
`graphify label . --backend ollama` will do it locally once you point it at a
model.

Add to `.gitignore` and `.claudeignore`:

```
graphify-out/cost.json
```

```
graph.json
graphify-out/
```

The second one matters. Graphify writes into the workspace on every rebuild,
and if Claude Code can see those paths it re-uploads the prompt cache each
time.

### What it found

319 nodes, 439 edges, 17 communities. 96% EXTRACTED, 4% INFERRED, no
AMBIGUOUS. Zero token cost.

**The top god node is not an abstraction.** `gold_excluded` came back with 101
edges — an order of magnitude above `build_graph()` at 15. It is the
leakage-exclusion list in `finetune/data/manifest.json`, and graphify
promoted all 101 gold-set ticket IDs to first-class nodes. That is a data
hub, not a core abstraction, and it distorts the ranking.

`--exclude-hubs 99` removes it but shattered the graph into 121 communities,
which is worse. Leave it, and read the god-node list knowing row 1 is an
artefact. The real ranking starts at `build_graph()`.

Worth noting the finding is not useless: the manifest records which gold rows
were held out of LoRA training, and the graph makes that provenance chain
visible as structure. `check_leakage.py` asserts it; the graph shows it.

**graphify could not see the state machine.** This is the important one.

`graphify path "ingress()" "audit()"` returned *no directed path*. The
pipeline topology is not in the source in any form tree-sitter can read — it
lives in string literals passed to `add_edge("retrieve", "classify")` inside
`build_graph()`. So the extracted graph knew all seven node functions
existed, knew `build_graph()` mentioned them, and knew nothing about the
order they run in.

That is a category error in the question, not a bug in the tool. graphify
maps static structure; a compiled state machine is runtime structure. But it
means the out-of-the-box graph is silently wrong about the one thing this
repo is *for*, and a reader who trusted it would conclude the governance
nodes are unconnected.

**Fix: `merge_langgraph_topology.py`.** LangGraph already knows the answer —
`build_graph().get_graph()` returns the real topology, which is what the
`draw_mermaid()` line at the bottom of `triage_graph.py` prints. The script
injects those 8 edges into `graph.json` as `state_transition` and
`conditional_transition`, tagged `_origin: langgraph-compiled` so AST edges
and compiled edges stay distinguishable. After merging:

```
$ graphify path "ingress()" "audit()"
Shortest path (4 hops):
  ingress() --conditional_transition [EXTRACTED]--> retrieve()
            --state_transition [EXTRACTED]--> classify()
            --conditional_transition [EXTRACTED]--> route()
            --conditional_transition [EXTRACTED]--> audit()
```

Re-run it after every `graphify update .`, or the transition edges are
dropped on the next rebuild.

One nice confirmation: `gate_a()` does not appear as a node on that path. It
appears as the condition attached to the `classify → route` edge, which is
exactly the claim the README makes — Gate A is a conditional edge, not a
step. The graph structure and the architecture claim agree.

### What it's actually good for here

The call-flow HTML and `GRAPH_REPORT.md` are usable repo artefacts as-is.
`graphify claude install` makes Claude Code query the graph before reading
files, which is worth having while the node bodies are still being filled.

What it is not is a substitute for the ASCII diagram in the README. That
diagram carries information the extracted graph did not have until it was
told.

---

## Part 2 — A retrieval channel inside the pipeline

### The thing to not do

The obvious read of "wire graphify into the graph" is: back `retrieve()` with
it, and have precedents come from a graph instead of an embedding index.
That does not work, for a boring reason.

**graphify does not index CSV.** The extraction run skipped
`gold_set_labelled.csv`, `gold_set_template.csv` and `ledger.jsonl` outright
— no supported extension. Getting a ticket corpus into a graphify graph means
exploding every row into its own markdown file and paying an LLM semantic
pass per row. For 4,000 tickets that is slow, expensive, and produces a graph
whose nodes are tickets connected by whatever an LLM inferred.

And it would be solving a problem that is already solved better. Precedent
retrieval is similarity over ticket text. `grounding.py` already carries that
stack — nomic-embed-text via Ollama, with a labelled lexical fallback and a
calibration script. Precedents should stay there.

### The thing to do instead

Graphify is right for a *second* channel: given a ticket, which **rubric
clauses, runbook sections and control documents** govern it. That corpus is
small, stable, prose, and genuinely relational — a clause cites another
clause, a runbook references a control. That is what a graph is for, and what
an embedding index handles badly.

`retrieve_policy.py` implements it. Build the index over prose, not tickets:

```bash
graphify extract ./policy --backend ollama --no-cluster
```

Ollama keeps the pass local, which matters for internal runbooks.

Then in `triage_graph.py`:

```python
def retrieve(state: TriageState) -> dict:
    import retrieve_policy
    out = retrieve_policy.retrieve(state)
    out["precedents"] = your_embedding_lookup(state)   # still yours
    return out
```

### The design choices worth defending

**The two channels stay separate all the way into state.** They fail
differently. A wrong precedent teaches the model the wrong label. A wrong
policy hit gives a correct label the wrong justification. Merged into one
`precedents` list, the second failure is invisible.

**INFERRED edges are not traversed by default.** Every graphify edge is
tagged EXTRACTED (read literally out of the source) or INFERRED (resolved by
graphify). An inferred edge is the tool's guess about what connects to what.
Letting one pull a runbook into a severity decision means the model was
steered by a link no human wrote. `allow_inferred=True` turns it on, and
every hit that used one comes back flagged
`policy_hit_via_inferred_edge`.

This is the part that is worth more than the retrieval itself. It is a
retrieval step where the provenance of every returned fragment is a first-
class field, and where the confidence tag of the weakest edge on the path is
carried through to the audit record. Most RAG cannot say which of its
retrieved chunks was reached by a guess.

**Staleness is a control, not a performance concern.** `graph.json` records
`built_at_commit`. `stale_against_head()` compares it to HEAD and flags
`policy_index_stale_vs_head`. An index that lags means the clause the model
was shown may not be the clause in force. It returns `None` when it cannot
tell, which is flagged separately from False — unknown freshness is not
fresh.

**`MIN_SCORE` is a placeholder and says so.** Every result carries
`policy_score_floor_not_calibrated` until you derive a floor from hand-checked
hits, the way `calibrate_grounding.py` derives the grounding bands. Same
reasoning as `WIRE_IN.md` step 3: a cutoff nobody derived is a cutoff nobody
can defend.

**A policy hit must not touch `precedent_depth`.** That field is set at
ingress because Gate A reads it, and Gate A asks whether this ticket has been
seen before. A governing clause is not a precedent. Letting one raise
`precedent_depth` would let the policy channel satisfy a gate written about
the precedent channel — the gate would still pass, and would no longer mean
what its docstring says.

### The eval consequence

Part 1 changes nothing about how the system runs. Part 2 puts a retrieval
step inside the decision path, before classification, which makes it eval
surface. Two questions that need numbers before this ships:

1. Does policy context move accuracy and under-classification at all? Current
   held-out baseline is TF-IDF at 0.6404 / 0.1386 under-class. Run the same
   split with and without the channel.
2. Can a policy hit push a ticket across a band? Specifically: with
   `allow_inferred=True`, can an inferred edge pull in a clause that moves a
   P3 to a P1? If yes, that is the finding, and the default stays off.

The second one is the break-it test, and it belongs in `test_gates.py`
alongside the existing gate tests.

---

## Housekeeping

Graphify logs every query to `~/.cache/graphify-queries.log` by default
(timestamp, question, corpus path). Every command above was run with
`GRAPHIFY_QUERY_LOG_DISABLE=1`. Decide deliberately rather than inheriting the
default, particularly if you ever demo this against a real corpus.
