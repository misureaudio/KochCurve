"""Verification for borel_hierarchy_v1.py (hook-ready version)."""
import json
import sys
from pathlib import Path

sys.path.insert(0, r"D:\Source\hermes-dir")
import borel_hierarchy_v1 as bh
from borel_hierarchy_v1 import (
    BorelGraph,
    Edge,
    load_graph,
    annotate_svg,
    ancestors,
    descendants,
    level,
    complement,
    build,
    contains,
    shortest_path,
    witness_grid,
)

HERE = Path(r"D:\Source\hermes-dir")
PLAIN = HERE / "borel_hierarchy_lattice.svg"
V1SVG = HERE / "borel_hierarchy_lattice_v1.svg"
failures = []


def check(label, got, want):
    ok = got == want
    print(f"{'PASS' if ok else 'FAIL'}  {label}" + ("" if ok else f": {got!r} (want {want!r})"))
    if not ok:
        failures.append(label)


g = load_graph(PLAIN)

# ============================ structure (as v0) ============================
check("title", g.title, "Borel hierarchy")
check("n classes", len(g.nodes), 9)
check("n edges", len(g.edges), 10)
check("dashed count", sum(1 for e in g.edges.values() if e.dashed), 2)
check("classes order", g.classes(),
      ["Σ⁰₁", "Π⁰₁", "Σ⁰₂", "Δ⁰₂", "Π⁰₂", "Σ⁰₃", "Δ⁰₃", "Π⁰₃", "Borel sets"])
check("subtitles", (g.node("Σ⁰₂").subtitle, g.node("Π⁰₂").subtitle,
                    g.node("Borel sets").subtitle),
      ("F_σ sets", "G_δ sets", "union of all levels α < ω₁"))
check("level F_σ (alias)", level("F_σ"), 2)
check("level Borel", level("Borel sets"), "omega1")

# ============================ navigation (as v0) ===========================
check("ancestors Σ⁰₁", g.ancestors("Σ⁰₁"),
      {"Δ⁰₂", "Σ⁰₂", "Π⁰₂", "Δ⁰₃", "Σ⁰₃", "Π⁰₃", "Borel sets"})
check("descendants Borel", g.descendants("Borel"),
      {"Σ⁰₁", "Π⁰₁", "Δ⁰₂", "Σ⁰₂", "Π⁰₂", "Δ⁰₃", "Σ⁰₃", "Π⁰₃"})
check("Π⁰₂ ⊇ Σ⁰₁", contains("Π⁰₂", "Σ⁰₁"), True)
check("Σ⁰₂ ⊆ Π⁰₂? NO", contains("Π⁰₂", "Σ⁰₂"), False)
check("path open -> G_δ", shortest_path("open", "G_δ"),
      ["Σ⁰₁", "Δ⁰₂", "Π⁰₂"])
check("complement F_σ", complement("F_σ"), "Π⁰₂")
check("build Π⁰₃", build("Π⁰₃"), "countable intersections of Σ⁰₂ sets")
check("successors Δ⁰₂", g.successors("Δ⁰₂"), ["Σ⁰₂", "Π⁰₂"])
check("predecessors Δ⁰₂", g.predecessors("Δ⁰₂"), ["Σ⁰₁", "Π⁰₁"])

# ============================ v1: edge objects =============================
check("edges are Edge dataclasses",
      all(isinstance(e, Edge) for e in g.edges.values()), True)
e = g.edge("edge-sigma-1-delta-2")
check("edge lookup by id", (e.src, e.dst, e.dashed), ("Σ⁰₁", "Δ⁰₂", False))
check("edge lookup by Edge", g.edge(e) is e, True)
try:
    g.edge("nope")
    check("bad edge id raises", "no error", "KeyError")
except KeyError:
    check("bad edge id raises", "KeyError", "KeyError")

# ============================ v1: walk() ===================================
events = list(g.walk("open"))
node_seq = [name for kind, name in events if kind == "node"]
edge_seq = [e for kind, e in events if kind == "edge"]
check("walk node order (BFS, visual)",
      node_seq,
      ["Σ⁰₁", "Δ⁰₂", "Σ⁰₂", "Π⁰₂", "Δ⁰₃", "Σ⁰₃", "Π⁰₃", "Borel sets"])
check("walk edge count (all crossings)", len(edge_seq), 9)
check("walk yields Edge objects", all(isinstance(e, Edge) for e in edge_seq), True)
# BFS crossings: from Σ⁰₁: (Σ⁰₁→Δ⁰₂); from Δ⁰₂: (Δ⁰₂→Σ⁰₂, Δ⁰₂→Π⁰₂);
# from Σ⁰₂: (Σ⁰₂→Δ⁰₃); from Π⁰₂: (Π⁰₂→Δ⁰₃); from Δ⁰₃: (Δ⁰₃→Σ⁰₃, Δ⁰₃→Π⁰₃);
# from Σ⁰₃: (Σ⁰₃→Borel); from Π⁰₃: (Π⁰₃→Borel).  = 9 crossings.
cross_pairs = [(e.src, e.dst) for e in edge_seq]
check("dashed arrows crossed",
      sorted(e.src for e in edge_seq if e.dashed), ["Π⁰₃", "Σ⁰₃"])
# every edge out of a visited node is yielded, even when the far end was
# already visited: (Π⁰₂→Δ⁰₃) and (Π⁰₃→Borel) cross to seen nodes.
check("crossings to already-seen nodes still yielded",
      (sum(1 for e in edge_seq if e.dst == "Δ⁰₃"),
       sum(1 for e in edge_seq if e.dst == "Borel sets")), (2, 2))

# direction down
down = [name for kind, name in g.walk("Borel sets", direction="down") if kind == "node"]
check("walk down from Borel", down,
      ["Borel sets", "Σ⁰₃", "Π⁰₃", "Δ⁰₃", "Σ⁰₂", "Π⁰₂", "Δ⁰₂", "Σ⁰₁", "Π⁰₁"])
try:
    list(g.walk("open", direction="sideways"))
    check("bad direction raises", "no error", "ValueError")
except ValueError:
    check("bad direction raises", "ValueError", "ValueError")

# ============================ v1: hooks + dispatch =========================
g2 = load_graph(PLAIN)
log = []
g2.on_enter("Σ⁰₂", lambda n: log.append(("enter", n)))
g2.on_traverse(lambda e: e.dashed, lambda e: log.append(("dashed", e.id)))
g2.on_traverse("edge-sigma-1-delta-2", lambda e: log.append(("edge", e.id)))
g2.on_traverse(("Σ⁰₁", "Δ⁰₂"), lambda e: log.append(("pair", e.id)))
g2.run_walk("open")
check("hook: enter fired", ("enter", "Σ⁰₂") in log, True)
check("hook: predicate (dashed) fired twice",
      log.count(("dashed", "edge-sigma-3-borel")) + log.count(("dashed", "edge-pi-3-borel")), 2)
check("hook: edge id fired", ("edge", "edge-sigma-1-delta-2") in log, True)
check("hook: (src,dst) pair fired", ("pair", "edge-sigma-1-delta-2") in log, True)
# non-matching nodes never fire
check("hook: enter only for Σ⁰₂",
      [x for x in log if x[0] == "enter"], [("enter", "Σ⁰₂")])
# dispatch works standalone
g2._last_hook_result = "sentinel"
g2.dispatch("node", "Σ⁰₂")
check("dispatch fires enter hook", any(x == ("enter", "Σ⁰₂") for x in log), True)
g2.dispatch("edge", "edge-pi-3-borel")
check("dispatch fires dashed predicate",
      log.count(("dashed", "edge-pi-3-borel")), 2)
try:
    g2.dispatch("warp", None)
    check("dispatch bad kind raises", "no error", "ValueError")
except ValueError:
    check("dispatch bad kind raises", "ValueError", "ValueError")
# chainability
check("on_enter chainable", g2.on_enter("Π⁰₂", lambda n: None) is g2, True)
check("on_traverse chainable", g2.on_traverse(lambda e: False, lambda e: None) is g2, True)
# one-argument form: the predicate itself is the hook (the review's usage)
onearg = []
g2.on_traverse(lambda e: onearg.append(e.id))
g2.dispatch("edge", "edge-sigma-1-delta-2")
check("on_traverse one-arg form", onearg, ["edge-sigma-1-delta-2"])
g2.clear_hooks()
check("clear_hooks empties registry", g2.hooks, {"enter": [], "traverse": []})

# ============================ v1: trace ====================================
g3 = load_graph(PLAIN)
g3.on_enter("Σ⁰₂", lambda n: {"example": "ℚ ∩ [0,1]"})
g3.on_traverse(lambda e: e.dashed, lambda e: {"note": "transfinite stretch"})
steps = g3.run_walk("open")
check("trace step count = events", len(steps), len(list(g3.walk("open"))))
check("trace first step", steps[0]["enter"], "Σ⁰₁")
check("trace second step", steps[1]["cross"], ["Σ⁰₁", "Δ⁰₂"])
check("trace enter payload", (steps[0]["payload"]["level"], steps[0]["payload"]["family"]),
      (1, "sigma"))
check("trace build recipe in payload", steps[0]["payload"]["build"], "open sets")
check("trace examples in payload", "open set" in " ".join(steps[0]["payload"]["examples"]),
      True)
enter_sigma2 = next(s for s in steps if s.get("enter") == "Σ⁰₂")
check("hook payload merged into step", enter_sigma2["payload"].get("example"),
      "ℚ ∩ [0,1]")
cross_dash = [s for s in steps if s.get("cross", [None, None])[0] == "Σ⁰₃"]
check("dashed step flag", cross_dash[0]["dashed"], True)
check("dashed hook payload merged", cross_dash[0]["payload"].get("note"),
      "transfinite stretch")
check("trace is JSON-serializable",
      json.loads(bh.BorelGraph.to_trace_json(steps)) is not None, True)
# down-direction trace
down_steps = g3.run_walk("Borel sets", direction="down")
check("down trace first enter", down_steps[0]["enter"], "Borel sets")
check("down trace first cross", down_steps[1]["cross"], ["Borel sets", "Σ⁰₃"])

# ============================ v1: loud failures ============================
try:
    BorelGraph.from_svg(HERE / "Borel-Sets-Presentation_2.md")  # not XML
    check("non-XML raises", "no error", "ValueError")
except ValueError:
    check("non-XML raises", "ValueError", "ValueError")

# zero-classes file: a valid SVG with boxes but no known titles
tmp0 = HERE / "_tmp_no_classes.svg"
tmp0.write_text(
    '<svg xmlns="http://www.w3.org/2000/svg" width="100" height="100">'
    '<g><rect x="0" y="0" width="10" height="10"/>'
    '<text x="5" y="5" text-anchor="middle">?⁰_?</text></g></svg>',
    encoding="utf-8",
)
try:
    BorelGraph.from_svg(tmp0)
    check("zero classes raises", "no error", "ValueError")
except ValueError:
    check("zero classes raises", "ValueError", "ValueError")

# namespace-less SVG: must still parse AND warn
tmp_ns = HERE / "_tmp_no_ns.svg"
tmp_ns.write_text(
    '<svg width="100" height="100">'
    '<title>t</title>'
    '<g><rect x="0" y="0" width="30" height="20"/>'
    '<text x="15" y="10" text-anchor="middle">Σ⁰₁</text></g>'
    '<g><rect x="50" y="0" width="30" height="20"/>'
    '<text x="65" y="10" text-anchor="middle">Δ⁰₂</text></g>'
    '<line x1="30" y1="10" x2="50" y2="10" marker-end="url(#a)"/></svg>',
    encoding="utf-8",
)
gns = BorelGraph.from_svg(tmp_ns)
check("no-ns svg parses", len(gns.nodes), 2)
check("no-ns svg warns", any("namespace" in w for w in gns.warnings), True)
check("no-ns svg edge found", len(gns.edges), 1)

# arrow missing a box warns
tmp_miss = HERE / "_tmp_miss_arrow.svg"
tmp_miss.write_text(
    '<svg xmlns="http://www.w3.org/2000/svg" width="100" height="100">'
    '<g><rect x="0" y="0" width="30" height="20"/>'
    '<text x="15" y="10" text-anchor="middle">Σ⁰₁</text></g>'
    '<line x1="30" y1="10" x2="90" y2="10" marker-end="url(#a)"/></svg>',
    encoding="utf-8",
)
gmiss = BorelGraph.from_svg(tmp_miss)
check("missed arrow warns", any("does not touch" in w for w in gmiss.warnings), True)
check("missed arrow not in edges", len(gmiss.edges), 0)

# unknown title warns (alongside a known box, so the parse itself succeeds)
tmp_unk = HERE / "_tmp_unk_title.svg"
tmp_unk.write_text(
    '<svg xmlns="http://www.w3.org/2000/svg" width="100" height="100">'
    '<g><rect x="0" y="0" width="30" height="20"/>'
    '<text x="15" y="10" text-anchor="middle">Σ⁰₁</text></g>'
    '<g><rect x="50" y="0" width="30" height="20"/>'
    '<text x="65" y="10" text-anchor="middle">Ω⁰_9</text></g></svg>',
    encoding="utf-8",
)
gunk = BorelGraph.from_svg(tmp_unk)
check("unknown title warns", any("unknown title" in w for w in gunk.warnings), True)
check("unknown title skipped", len(gunk.nodes), 1)

# ============================ v1: ids / data-* =============================
g1 = load_graph(V1SVG)
check("v1 svg: node ids present",
      {n.id for n in g1.nodes.values()} == {
          "node-sigma-1", "node-pi-1", "node-sigma-2", "node-delta-2",
          "node-pi-2", "node-sigma-3", "node-delta-3", "node-pi-3",
          "node-borel"}, True)
check("v1 svg: edge ids present",
      "edge-sigma-3-borel" in g1.edges and "edge-delta-2-pi-2" in g1.edges, True)
check("v1 svg: data-prompt read back",
      "F_σ" in g1.node("Σ⁰₂").data.get("prompt", ""), True)
check("v1 svg: data-family read back",
      g1.node("Π⁰₂").data.get("family"), "pi")
check("v1 svg: data-level read back",
      g1.node("Σ⁰₃").data.get("level"), "3")
check("v1 svg: dashed metadata read back",
      g1.edges["edge-pi-3-borel"].metadata.get("dashed"), "true")
check("v1 svg: still validates", g1.validate()["ok"], True)
# annotated file is parseable and idempotent-ish (re-annotate keeps ids)
out2 = annotate_svg(V1SVG, HERE / "_tmp_reannotated.svg")
g1b = BorelGraph.from_svg(out2)
check("re-annotated keeps ids", g1b.node("Σ⁰₂").id, "node-sigma-2")
check("re-annotated validates", g1b.validate()["ok"], True)

# ============================ v1: validate() ===============================
v = g.validate()
check("validate ok", v["ok"], True)
check("validate no missing/extra", (v["missing"], v["extra"]), ([], []))
# tamper: drop an edge -> validate catches it
g_bad = load_graph(PLAIN)
del g_bad.edges["edge-delta-2-sigma-2"]
g_bad._build_adjacency()
v_bad = g_bad.validate()
check("validate catches missing edge", v_bad["missing"], [("Δ⁰₂", "Σ⁰₂")])
check("validate not ok after tamper", v_bad["ok"], False)
# expected edges for the full 3-level diagram: 2 climb + 2 same-level per level
check("expected edge count", len(g.expected_edges()), 10)

# ============================ witness grid (depth-limited) =================
check("witness exists-all True",
      witness_grid([[True, True, True], [False, True, False]]), True)
check("witness exists-all depth limit",
      witness_grid([[True, True, True]], "exists-all", depth=0), False)
check("witness all-exists depth limit",
      witness_grid([[True, False], [False, True]], "all-exists", depth=1), True)
try:
    witness_grid([[True]], "forall")
    check("witness bad pattern raises", "no error", "ValueError")
except ValueError:
    check("witness bad pattern raises", "ValueError", "ValueError")

# ============================ serialization ================================
d = g1.as_dict()
check("as_dict has ids", all("id" in c for c in d["classes"]), True)
check("as_dict edges have id+metadata",
      all("id" in e and "metadata" in e for e in d["edges"]), True)
check("as_dict warnings list", isinstance(d["warnings"], list), True)
check("to_json round-trip", json.loads(g1.to_json())["classes"][0]["name"], "Σ⁰₁")
dot = g1.to_dot()
check("dot still renders", '"Σ⁰₁" -> "Δ⁰₂"' in dot, True)

# ============================ cleanup + summary ============================
for p in (tmp0, tmp_ns, tmp_miss, tmp_unk, HERE / "_tmp_reannotated.svg"):
    p.unlink(missing_ok=True)

print()
if failures:
    print(f"{len(failures)} FAILURES: {failures}")
    sys.exit(1)
print("ALL CHECKS PASSED")
