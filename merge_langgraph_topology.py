"""
Inject the compiled LangGraph topology into graphify's graph.json.

WHY THIS EXISTS. graphify parses source with tree-sitter. The triage
pipeline's topology is not in the source in any form tree-sitter can see —
it lives in string literals handed to `add_edge("retrieve", "classify")`.
So the extracted graph knows every node function exists and knows
`build_graph()` mentions them, and knows nothing about the order they run
in. `graphify path "ingress()" "audit()"` returns no path, which is not a
bug in graphify so much as a category error in asking it: the state machine
is runtime structure, and graphify maps static structure.

LangGraph already knows the answer. `compiled.get_graph()` returns the real
topology. This merges it in as `state_transition` edges tagged with their
own origin, so a reader can always tell which edges came from the AST and
which came from the compiled graph.

Run after every `graphify extract` or `graphify update`, or the transition
edges are silently dropped on the next rebuild:

    python -m graphify update . && python merge_langgraph_topology.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

GRAPH = Path("graphify-out/graph.json")
NODE_PREFIX = "triage_graph_"  # graphify's id scheme for triage_graph.py symbols


def main() -> int:
    if not GRAPH.exists():
        print(f"no graph at {GRAPH} — run `graphify extract . --code-only` first")
        return 1

    from triage_graph import build_graph

    compiled = build_graph().get_graph()
    g = json.loads(GRAPH.read_text())
    ids = {n["id"] for n in g["nodes"]}

    def nid(name: str) -> str | None:
        """Map a LangGraph node name to a graphify node id, if it has one."""
        if name in ("__start__", "__end__"):
            return None
        cand = f"{NODE_PREFIX}{name}"
        return cand if cand in ids else None

    existing = {(e["source"], e["target"], e.get("relation")) for e in g["links"]}
    added = skipped = 0

    for edge in compiled.edges:
        s, t = nid(edge.source), nid(edge.target)
        if not s or not t:
            skipped += 1
            continue
        # LangGraph marks conditional edges; they are branches, not guarantees,
        # and a reader of the graph should be able to tell the difference.
        conditional = bool(getattr(edge, "conditional", False))
        relation = "conditional_transition" if conditional else "state_transition"
        if (s, t, relation) in existing:
            continue
        g["links"].append({
            "source": s,
            "target": t,
            "relation": relation,
            "_origin": "langgraph-compiled",
            # Declared literally in build_graph(), not resolved by anything.
            "confidence": "EXTRACTED",
            "confidence_score": 1.0,
            "context": "conditional edge" if conditional else "edge",
            "source_file": "triage_graph.py",
            "source_location": "build_graph()",
            "weight": 1.0,
        })
        added += 1

    GRAPH.write_text(json.dumps(g, indent=1))
    print(f"merged {added} transition edge(s); {skipped} terminal edge(s) skipped")
    print("re-run `graphify cluster-only . --no-label` to refresh the report")
    return 0


if __name__ == "__main__":
    sys.exit(main())
