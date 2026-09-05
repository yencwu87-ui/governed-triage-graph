"""
Policy channel control tests — the break-it case for retrieve_policy.py.

    python test_policy_channel.py

Needs no model, no gold set, and no policy corpus. The graph is a fixture
built here, so these run today, before ./policy exists and before classify()
is implemented.

The organising question, matching test_gates.py's:
  if an edge nobody wrote pulls in a clause, can that clause change the call?
    it cannot reach the prompt  -> provenance is a control
    it reaches the prompt       -> provenance is a label on a thing that
                                   already happened

T13 is the one that matters. It does not assert the model behaves — it
asserts the inferred-edge content never gets far enough to find out.
"""
from __future__ import annotations
import json, sys, tempfile
from pathlib import Path

import retrieve_policy as rp

PASS, FAIL = "\033[92mPASS\033[0m", "\033[91mFAIL\033[0m"


def _node(nid, label, file, loc):
    return {"id": nid, "label": label, "source_file": file,
            "source_location": loc, "file_type": "doc", "_origin": "semantic"}


def _edge(s, t, rel, conf):
    return {"source": s, "target": t, "relation": rel, "confidence": conf,
            "confidence_score": 1.0 if conf == "EXTRACTED" else 0.85,
            "source_file": "rubric.md", "source_location": "L1", "weight": 1.0}


def fixture_graph() -> dict:
    """
    A minimal two-clause rubric with one honest link and one guess.

    The ticket below seeds on `clause_routine`. From there:
      - an EXTRACTED `references` edge reaches the escalation-exception note
      - an INFERRED `related_to` edge reaches the SEV-1 clause

    The INFERRED edge is the hazard. Nothing in the source says a routine
    single-customer timeout relates to a total outage. graphify resolved it.
    """
    return {
        "directed": True, "multigraph": False, "graph": {},
        "built_at_commit": "deadbeefdeadbeefdeadbeefdeadbeefdeadbeef",
        "hyperedges": [],
        "nodes": [
            _node("clause_routine",
                  "routine single customer timeout request",
                  "rubric.md", "L20"),
            _node("clause_exception",
                  "escalation exception when workaround exists",
                  "rubric.md", "L34"),
            # Deliberately shares NO word with the ticket, so the only way in
            # is the inferred edge. An earlier draft of this clause shared
            # "affected" and seeded directly — see fixture_seed_graph().
            _node("clause_sev1",
                  "priority zero catastrophic estatewide degradation",
                  "rubric.md", "L52"),
        ],
        "links": [
            _edge("clause_routine", "clause_exception", "references", "EXTRACTED"),
            _edge("clause_routine", "clause_sev1", "related_to", "INFERRED"),
        ],
    }


def fixture_seed_graph() -> dict:
    """
    The same SEV-1 clause, worded the way a real rubric would word it.

    It now shares exactly one word with the ticket — "affected" — which is
    enough to clear the uncalibrated MIN_SCORE and seed at hop 0. No edge is
    involved, so the provenance gate is not consulted. This is the hazard the
    first draft of this test found by accident.
    """
    g = fixture_graph()
    for n in g["nodes"]:
        if n["id"] == "clause_sev1":
            n["label"] = "severity one total outage all customers affected"
    return g


# A P3-shaped ticket. Nothing in it is a SEV-1.
TICKET = {
    "ticket_id": "T-policy-1",
    "subject": "routine timeout",
    "body": ("A single customer reports a routine timeout on the export "
             "request. A workaround exists and no other customer is affected."),
}


def run(check) -> None:
    """Run the policy checks against `check(name, ok, detail)`."""
    tmp = Path(tempfile.mkdtemp()) / "graph.json"
    tmp.write_text(json.dumps(fixture_graph()))
    # repo_root points at a non-repo dir on purpose: stale_against_head()
    # must return None there, and None must be flagged, not treated as fresh.
    idx = rp.PolicyIndex(graph_path=tmp, repo_root=tmp.parent)

    print("\n  Policy channel — provenance as a control")

    default = idx.lookup(TICKET["subject"], TICKET["body"], k=8, depth=2)
    opened = idx.lookup(TICKET["subject"], TICKET["body"], k=8, depth=2,
                        allow_inferred=True)
    dlabels = {h.label for h in default}
    olabels = {h.label for h in opened}
    sev1 = "priority zero catastrophic estatewide degradation"

    check("T11 the fixture is live — the guess DOES reach a band-crossing clause",
          sev1 in olabels,
          f"allow_inferred=True returns {sorted(olabels)}")

    check("T12 default traversal refuses the INFERRED edge",
          sev1 not in dlabels and all(h.confidence == "EXTRACTED" for h in default),
          f"default returns {sorted(dlabels)}")

    # --- the break-it case ---
    check("T13 no inferred content can reach the classifier prompt",
          not (olabels - dlabels) & dlabels and (olabels - dlabels) == {sev1},
          "the only clause the guess adds is the one that would move the band; "
          "it is excluded by default, so classify() never sees it")

    check("T14 opening the gate is visible in the audit record",
          "policy_hit_via_inferred_edge" in idx.flags(opened)
          and "policy_hit_via_inferred_edge" not in idx.flags(default),
          f"flags(opened) = {idx.flags(opened)}")

    print("\n  Policy channel — the honest-placeholder flags")

    check("T15 uncalibrated score floor is declared on every result",
          "policy_score_floor_not_calibrated" in idx.flags(default),
          "MIN_SCORE is a placeholder until derived from hand-checked hits")

    check("T16 unknown index freshness is flagged, not read as fresh",
          idx.stale_against_head() is None
          and "policy_index_freshness_unknown" in idx.flags(default),
          "None is not False — an index of unknown age is not a current index")

    check("T17 an empty result is returned rather than filler",
          idx.lookup(None, "") == [] and idx.lookup("zzz", "qqq") == [],
          "no governing clause found returns [], and says so in the flags")

    print("\n  Policy channel — separation from the precedent channel")

    rp._INDEX = idx  # otherwise retrieve() loads the absent default index
                     # and T18-T20 pass vacuously on an empty result
    out = rp.retrieve(dict(TICKET))
    check("T18 the policy channel writes its own key, not `precedents`",
          "policy" in out and "precedents" not in out,
          "a wrong precedent teaches the wrong label; a wrong policy hit gives "
          "a right label the wrong justification — merged, the second is invisible")

    check("T19 a policy hit cannot raise precedent_depth",
          "data_quality" not in out,
          "precedent_depth is set at ingress because gate_a reads it; a "
          "governing clause must not be able to satisfy a gate written about "
          "whether this ticket has been seen before")

    check("T20 every returned fragment carries its own provenance",
          all({"source", "confidence", "hops", "via", "seed_terms"}
              <= set(h.as_dict()) for h in opened)
          and out["policy_provenance"]["built_at_commit"],
          "source file, weakest tag on the path, hop count, relations walked, "
          "and the words the seed matched on")

    print("\n  Policy channel — the hazard the gate does NOT cover")

    tmp2 = Path(tempfile.mkdtemp()) / "graph.json"
    tmp2.write_text(json.dumps(fixture_seed_graph()))
    idx2 = rp.PolicyIndex(graph_path=tmp2, repo_root=tmp2.parent)
    seeded = idx2.lookup(TICKET["subject"], TICKET["body"], k=8, depth=2)
    real_sev1 = "severity one total outage all customers affected"
    hit = next((h for h in seeded if h.label == real_sev1), None)

    check("T21 a band-crossing clause CAN still enter as a hop-0 seed",
          hit is not None and hit.hops == 0 and hit.confidence == "EXTRACTED"
          and hit.seed_terms == ["affected"],
          "refusing inferred edges did nothing here — no edge was walked. "
          "The provenance gate governs traversal, not seeding.")

    check("T22 a single-term seed is declared in the audit record",
          "policy_hit_single_term_seed" in idx2.flags(seeded)
          and "policy_hit_single_term_seed" not in idx2.flags(default),
          f"flags = {idx2.flags(seeded)}")


# ------------------------------------------------------------------ standalone

if __name__ == "__main__":
    results = []

    def check(name, ok, detail=""):
        results.append((name, ok))
        print(f"  [{PASS if ok else FAIL}] {name}")
        if detail:
            print(f"         {detail}")

    print("\n  POLICY CHANNEL CONTROL TESTS\n  " + "=" * 58)
    run(check)

    n_pass = sum(1 for _, ok in results if ok)
    print("\n  " + "=" * 58)
    print(f"  {n_pass}/{len(results)} passed\n")
    if n_pass == len(results):
        print("  Verdict: PARTIAL, and read T21 before quoting the rest.\n")
        print("  Closed: traversal. The channel is EXTRACTED-only by default,")
        print("  the one clause an inferred edge adds is the one that moves")
        print("  the band, and `allow_inferred=True` does not ship.\n")
        print("  OPEN: seeding. T21 passes by DEMONSTRATING the hole — a P3")
        print("  ticket seeds onto a SEV-1 clause on one shared word, at hop 0,")
        print("  with no edge walked. The provenance gate cannot reach it. It")
        print("  is flagged, not prevented. Closing it means deriving MIN_SCORE")
        print("  from hand-checked hits, the way calibrate_grounding.py derives")
        print("  the grounding bands — not picking a number that greens T21.\n")
        print("  Also unproven, and not provable here: whether policy context")
        print("  moves accuracy at all. That needs classify() implemented and")
        print("  the same split run with and without the channel.\n")
    else:
        print("  Verdict: WITHHELD. A failing control test is a finding — log\n"
              "  it before changing anything.\n")
    sys.exit(0 if n_pass == len(results) else 1)
