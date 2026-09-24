"""Verification for borel_hierarchy.py against the parsed SVG graph."""
import json
import sys

sys.path.insert(0, r"D:\Source\hermes-dir")
from borel_hierarchy import (
    BorelGraph,
    load_graph,
    ancestors,
    descendants,
    level,
    complement,
    build,
    contains,
    shortest_path,
    witness_grid,
)

g = load_graph(r"D:\Source\hermes-dir\borel_hierarchy_lattice.svg")
failures = []


def check(label, got, want):
    ok = got == want
    print(f"{'PASS' if ok else 'FAIL'}  {label}: {got!r}" + ("" if ok else f"  (want {want!r})"))
    if not ok:
        failures.append(label)


# --- structure -------------------------------------------------------------
check("title", g.title, "Borel hierarchy")
check("n classes", len(g.nodes), 9)
check("n edges", len(g.edges), 10)
check("n dashed", len(g.dashed_edges), 2)
check("classes order", g.classes(),
      ["Σ⁰₁", "Π⁰₁", "Σ⁰₂", "Δ⁰₂", "Π⁰₂", "Σ⁰₃", "Δ⁰₃", "Π⁰₃", "Borel sets"])
check("subtitles", (g.node("Σ⁰₂").subtitle, g.node("Π⁰₂").subtitle,
                    g.node("Borel sets").subtitle),
      ("F_σ sets", "G_δ sets", "union of all levels α < ω₁"))

# --- levels / families -----------------------------------------------------
check("level Σ⁰₁", level("Σ⁰₁"), 1)
check("level F_σ (alias)", level("F_σ"), 2)
check("level Borel", level("Borel sets"), "omega1")
check("family G_δ", g.family("G_δ"), "pi")
check("family Δ⁰₂", g.family("Δ⁰₂"), "delta")

# --- direct neighbours -----------------------------------------------------
check("successors Π⁰₁", g.successors("Π⁰₁"), {"Δ⁰₂"})
check("successors Δ⁰₂", g.successors("Δ⁰₂"), {"Σ⁰₂", "Π⁰₂"})
check("successors Σ⁰₃", g.successors("Σ⁰₃"), {"Borel sets"})
check("predecessors Δ⁰₂", g.predecessors("Δ⁰₂"), {"Σ⁰₁", "Π⁰₁"})
check("predecessors Borel", g.predecessors("Borel"), {"Σ⁰₃", "Π⁰₃"})

# --- transitive navigation -------------------------------------------------
check("ancestors Σ⁰₁", g.ancestors("Σ⁰₁"),
      {"Δ⁰₂", "Σ⁰₂", "Π⁰₂", "Δ⁰₃", "Σ⁰₃", "Π⁰₃", "Borel sets"})
check("ancestors G_δ (alias)", ancestors("G_δ"),
      {"Δ⁰₃", "Σ⁰₃", "Π⁰₃", "Borel sets"})
check("descendants Borel", g.descendants("Borel"),
      {"Σ⁰₁", "Π⁰₁", "Δ⁰₂", "Σ⁰₂", "Π⁰₂", "Δ⁰₃", "Σ⁰₃", "Π⁰₃"})
check("descendants Δ⁰₂", descendants("Δ⁰₂"), {"Σ⁰₁", "Π⁰₁"})
check("ancestors Borel", g.ancestors("Borel"), set())

# --- contains (transitive inclusion) ---------------------------------------
check("Borel ⊇ Π⁰₁", contains("Borel sets", "closed"), True)
check("Π⁰₂ ⊇ Σ⁰₁ (open is G_δ)", contains("Π⁰₂", "Σ⁰₁"), True)
check("Σ⁰₂ ⊇ Π⁰₁", contains("Σ⁰₂", "Π⁰₁"), True)
check("Δ⁰₂ ⊇ Π⁰₁", contains("Δ⁰₂", "Π⁰₁"), True)
check("Δ⁰₂ ⊇ Σ⁰₁", contains("Δ⁰₂", "Σ⁰₁"), True)
check("Σ⁰₂ ⊆ Π⁰₂? NO", contains("Π⁰₂", "Σ⁰₂"), False)
check("self-contains", contains("Σ⁰₂", "F_σ"), True)

# --- shortest paths --------------------------------------------------------
check("path Π⁰₁ -> Borel", shortest_path("Π⁰₁", "Borel sets"),
      ["Π⁰₁", "Δ⁰₂", "Π⁰₂", "Δ⁰₃", "Π⁰₃", "Borel sets"])
check("path open -> G_δ", shortest_path("open", "G_δ"),
      ["Σ⁰₁", "Δ⁰₂", "Π⁰₂"])
try:
    shortest_path("Borel sets", "Σ⁰₁")
    check("path Borel->Σ⁰₁ raises", "no error", "LookupError")
except LookupError:
    check("path Borel->Σ⁰₁ raises", "LookupError", "LookupError")

# --- complement / build ----------------------------------------------------
check("complement F_σ", complement("F_σ"), "Π⁰₂")
check("complement G_δ", complement("G_δ"), "Σ⁰₂")
check("complement Σ⁰₃", complement("Σ⁰₃"), "Π⁰₃")
check("complement Δ⁰₂", complement("Δ⁰₂"), "Δ⁰₂")
check("complement Borel", complement("Borel"), "Borel sets")
check("build Σ⁰₁", build("Σ⁰₁"), "open sets")
check("build Π⁰₁", build("Π⁰₁"), "closed sets")
check("build Σ⁰₂", build("Σ⁰₂"), "countable unions of Π⁰₁ sets")
check("build Π⁰₂", build("Π⁰₂"), "countable intersections of Σ⁰₁ sets")
check("build Σ⁰₃", build("Σ⁰₃"), "countable unions of Π⁰₂ sets")
check("build Π⁰₃", build("Π⁰₃"), "countable intersections of Σ⁰₂ sets")
check("build Δ⁰₂", build("Δ⁰₂"), "the sets that are both Σ⁰₂ and Π⁰₂")
check("build Borel", build("Borel sets"),
      "the union of all levels Σ⁰_α, Π⁰_α for countable α < ω₁")

# --- examples from the presentation ----------------------------------------
check("examples Σ⁰₂ nonempty", len(g.examples("F_σ")) >= 1, True)
check("examples Π⁰₃ mentions convergence",
      "converg" in " ".join(g.examples("Π⁰₃")), True)

# --- witness grid (the teaching device) ------------------------------------
# exists-all (F_σ): row 1 all True  -> member
check("witness exists-all True",
      witness_grid([[True, True, True], [False, True, False]]), True)
check("witness exists-all False",
      witness_grid([[True, False, True], [False, True, False]]), False)
# all-exists (G_δ): every row has a True -> member
check("witness all-exists True",
      witness_grid([[False, True, False], [True, False, False]], "all-exists"),
      True)
check("witness all-exists False",
      witness_grid([[True, False, False], [False, False, False]], "all-exists"),
      False)

# --- serialization ---------------------------------------------------------
d = g.as_dict()
js = g.to_json()
check("json round-trip", json.loads(js)["classes"][0]["name"], "Σ⁰₁")
check("json edge count", len(d["edges"]), 10)
check("json dashed flag", any(e["dashed"] for e in d["edges"]), True)
dot = g.to_dot()
check("dot has edge", '"Σ⁰₁" -> "Δ⁰₂"' in dot, True)
check("dot has dashed", "[style=dashed]" in dot, True)

# --- alias robustness ------------------------------------------------------
for alias in ["open sets", "F-sigma", "G_delta", "sigma 1", "pi 2",
              "Borel", "Δ2", "Σ3", "F_sigma_delta"]:
    try:
        g.node(alias)
        check(f"alias {alias!r}", "ok", "ok")
    except KeyError:
        check(f"alias {alias!r}", "KeyError", "ok")
try:
    g.node("not a class")
    check("unknown name raises", "no error", "KeyError")
except KeyError:
    check("unknown name raises", "KeyError", "KeyError")

print()
if failures:
    print(f"{len(failures)} FAILURES: {failures}")
    sys.exit(1)
print("ALL CHECKS PASSED")
