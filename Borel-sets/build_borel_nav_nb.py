"""build_borel_nav_nb.py — Build Borel-Hierarchy-Navigation_v1.ipynb.

A JupyterLab notebook that walks the Borel hierarchy lattice parsed by
borel_hierarchy_v1.py: queries, traversal, hooks on nodes and edges,
and the JSON trace for a browser player.  Assembled via nbformat and
executed headlessly with nbclient against the project .venv.
"""
import json
import os
import sys
import tempfile
from pathlib import Path

import nbformat

HERE = Path(__file__).parent
NB_PATH = HERE / "Borel-Hierarchy-Navigation_v1.ipynb"
VENV_PY = HERE / ".venv" / "Scripts" / "python.exe"


def md(text: str) -> dict:
    return {
        "cell_type": "markdown",
        "metadata": {},
        "source": text,
        "id": _uid(),
    }


def py(text: str) -> dict:
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": text,
        "id": _uid(),
    }


_counter = 0


def _uid() -> str:
    global _counter
    _counter += 1
    return f"borel-nav-{_counter:04d}"


cells = []

cells.append(md(r'''
# Navigating the Borel hierarchy lattice

This notebook drives **`borel_hierarchy_v1.py`**: a module that reads the
lattice diagram (`borel_hierarchy_lattice_v1.svg`, the id-annotated copy of
`borel_hierarchy_lattice.svg`) and builds a navigable graph of the Borel
classes.

The graph model:

- **nodes** are the classes — Σ⁰ₙ (countable unions, purple), Π⁰ₙ
  (countable intersections, teal), Δ⁰ₙ = Σ⁰ₙ ∩ Π⁰ₙ (gray), and the Borel
  sets at the top (the union of all levels below ω₁);
- an **edge** `u → v` means *u ⊆ v* (the arrow points upward);
- the two **dashed** edges (Σ⁰₃, Π⁰₃ → Borel sets) stand for the
  transfinite stretch through every countable ordinal.

What is new in v1 (code-review driven): a real **`walk()`** event stream,
**`Edge` objects** with metadata, stable **ids** on nodes and edges
(`node-sigma-2`, `edge-sigma-1-delta-2`), **hooks** (`on_enter` /
`on_traverse` / `dispatch`), a JSON **trace** for a browser player,
**`validate()`** against the taxonomy, and loud failures (parse warnings,
hard error on an empty graph).
'''))

cells.append(py(r'''
from pathlib import Path
import sys

HERE = Path.cwd()
if not (HERE / "borel_hierarchy_v1.py").exists():
    sys.path.insert(0, str(HERE.parent))  # run from a subfolder?
from borel_hierarchy_v1 import load_graph, witness_grid

SVG = HERE / "borel_hierarchy_lattice_v1.svg"   # id-annotated diagram
g = load_graph(SVG)
print(repr(g))
print("parse warnings:", g.warnings or "none")
'''))

cells.append(md(r'''
## 1. The graph as data

Every class and edge has a stable id — the anchor point for hooks and for
the animation player.  `data-*` attributes from the SVG (family, level,
the expansion `prompt`) are read back onto the node.
'''))

cells.append(py(r'''
import pandas as pd

classes = pd.DataFrame([
    {"name": n.name, "id": n.id, "subtitle": n.subtitle,
     "family": n.family, "level": n.level,
     "contained_in": ", ".join(g.successors(n.name)) or "— (top)"}
    for n in sorted(g.nodes.values(), key=lambda n: (n.rank, n.x))
])
edges = pd.DataFrame([
    {"id": e.id, "from": e.src, "to": e.dst, "means": f"{e.src} ⊆ {e.dst}",
     "dashed": e.dashed, "metadata": e.metadata or {}}
    for e in sorted(g.edges.values(), key=lambda e: e.id)
])
classes
'''))

cells.append(py(r'''
edges
'''))

cells.append(py(r'''
# data-* attributes read back from the annotated SVG
node = g.node("Σ⁰₂")
print(node.id, "|", node.subtitle, "|", node.family, "level", node.level)
print("prompt:", node.data.get("prompt"))
'''))

cells.append(md(r'''
## 2. Queries (v0, unchanged)

`ancestors` / `descendants` are transitive; `contains` answers
*u ⊆ v?*; `shortest_path` gives an inclusion chain; `complement` and
`build` give the semantics.
'''))

cells.append(py(r'''
print("ancestors(open)   =", sorted(g.ancestors("open")))
print("descendants(Δ⁰₂)  =", sorted(g.descendants("Δ⁰₂")))
print()
print("Π⁰₂ contains Σ⁰₁ (every open set is G_δ)?", g.contains("Π⁰₂", "Σ⁰₁"))
print("Π⁰₂ contains Σ⁰₂ (F_σ ⊆ G_δ)?           ", g.contains("Π⁰₂", "Σ⁰₂"))
print()
print("shortest_path(open → G_δ):", " → ".join(g.shortest_path("open", "G_δ")))
print("shortest_path(Π⁰₁ → Borel):", " → ".join(g.shortest_path("Π⁰₁", "Borel sets")))
print()
print("complement(F_σ) =", g.complement("F_σ"))
print("build(Σ⁰₂)      =", g.build("Σ⁰₂"))
print("build(Π⁰₃)      =", g.build("Π⁰₃"))
'''))

cells.append(md(r'''
## 3. Traversal: `walk()`

`walk(start, direction)` is a **generator of events**:

- `("node", name)` — the walk is *at* a class;
- `("edge", Edge)` — the walk *crosses* an arrow (every outgoing edge of
  the current node is crossed, even when the far end was already seen).

Breadth-first, deterministic, in the visual reading order of the diagram.
`direction="up"` climbs toward the Borel sets; `"down"` descends.
'''))

cells.append(py(r'''
events = list(g.walk("open"))
for kind, item in events:
    if kind == "node":
        print(f"  enter  {item}")
    else:
        mark = "  (dashed)" if item.dashed else ""
        print(f"  cross  {item.src} -> {item.dst}{mark}")
'''))

cells.append(py(r'''
# descending from the top: the same lattice, walked in reverse
down = [(kind, item) for kind, item in g.walk("Borel sets", direction="down")]
print(" → ".join(name for kind, name in down if kind == "node"))
'''))

cells.append(md(r'''
## 4. Hooks: activities attached to nodes and edges

Hooks are the "further process" the review asked for.  They key on the
stable ids, so they survive re-parsing and map 1:1 onto the SVG elements:

- `on_enter(name, fn)` — `fn(node_name)` when the walk enters a class;
- `on_traverse(match, fn)` — `fn(edge)` when the walk crosses a matching
  edge.  `match` is an edge id, a `(src, dst)` pair, or a predicate
  (e.g. `lambda e: e.dashed`).

A hook may return a dict: `run_walk` merges it into the payload of the
trace step, so an activity can *annotate* the walk for the browser player.
'''))

cells.append(py(r'''
g2 = load_graph(SVG)
log = []

# activity on a node: announce the canonical example when the walk arrives
def enter_sigma2(name):
    log.append(f"enter {name}: show example")
    return {"activity": "show_example", "example": "ℚ ∩ [0,1]"}

# activity on a vertex/edge: pulse every arrow as it is crossed
def cross_all(e):
    log.append(f"cross {e.id}")
    return None

# activity on specific edges: flag the transfinite stretch
def cross_dashed(e):
    log.append(f"cross {e.id}  -> transfinite stretch (levels continue to ω₁)")
    return {"activity": "flag_transfinite"}

_ = g2.on_enter("Σ⁰₂", enter_sigma2)
_ = g2.on_enter("Π⁰₃", lambda n: log.append(f"enter {n}: convergence set ⋂ₖ⋃_N⋂{{n,m≥N}}"))
_ = g2.on_traverse(cross_all)
_ = g2.on_traverse(lambda e: e.dashed, cross_dashed)

steps = g2.run_walk("open")          # walk + dispatch + record
for line in log:
    print(line)
'''))

cells.append(py(r'''
# dispatch() can fire hooks on a single event, standalone
g2.dispatch("node", "Σ⁰₂")
g2.dispatch("edge", "edge-pi-3-borel")
print("after manual dispatch:", log[-2:])
g2.clear_hooks()
'''))

cells.append(md(r'''
## 5. The trace: JSON for the browser player

"Trace, then replay" (Answer 3): Python records the walk as a JSON list of
steps — the mathematics stays in one testable place, a small JS player
animates the steps on the SVG (targeting the `id` attributes):

```json
[{"enter": "Σ⁰₁", "payload": {...}},
 {"cross": ["Σ⁰₁", "Δ⁰₂"], "edge": "edge-sigma-1-delta-2",
  "dashed": false, "payload": {}}, ...]
```

`enter` payloads carry level, family, the build recipe and the canonical
examples; `cross` payloads carry the edge metadata **plus whatever the
hooks returned** (the `activity` annotations from §4).
'''))

cells.append(py(r'''
import json

g3 = load_graph(SVG)
_ = g3.on_enter("Σ⁰₂", lambda n: {"activity": "show_example", "example": "ℚ ∩ [0,1]"})
_ = g3.on_traverse(lambda e: e.dashed, lambda e: {"activity": "flag_transfinite"})
steps = g3.run_walk("open")

TRACE = HERE / "Borel-Hierarchy-trace_v1.json"
TRACE.write_text(g3.to_trace_json(steps), encoding="utf-8")
print(f"wrote {TRACE.name} with {len(steps)} steps")
print()
print(json.dumps(steps[:6], ensure_ascii=False, indent=2))
'''))

cells.append(py(r'''
# the hook-annotated steps, mid-walk
for s in steps:
    act = s.get("payload", {}).get("activity")
    if act:
        target = s.get("enter") or " -> ".join(s["cross"])
        print(f"{s.get('enter') or 'cross ' + ' -> '.join(s['cross']):28s} activity: {act}")
'''))

cells.append(md(r'''
## 6. The witness grid (depth-limited verdict)

The presentation's teaching device, evaluated by the module.  **The grid
is a finite truncation**: "some row is all green" means all green *across
the columns shown* (and, with `depth`, only the first `depth` rows).  It
is a stage of the limit — a row green "forever" can never be certified by
any finite grid, so a lesson must show the depth, not a settled verdict.
'''))

cells.append(py(r'''
# F_σ (Σ⁰₂):  x ∈ ⋃ₙ ⋂ₘ Aₙ,ₘ  <=>  some row is all green
grid_fs = [[True,  True,  True,  True],
           [False, True,  False, True],
           [True,  False, True,  False]]
for depth in (1, 2, 3):
    print(f"depth {depth}: exists-all =",
          witness_grid(grid_fs, "exists-all", depth=depth))

print()
# G_δ (Π⁰₂):  x ∈ ⋂ₙ ⋃ₘ Aₙ,ₘ  <=>  green in every row
grid_gd = [[False, False, True],
           [True,  False, False],
           [False, False, False]]
for depth in (1, 2, 3):
    print(f"depth {depth}: all-exists =",
          witness_grid(grid_gd, "all-exists", depth=depth))
print("(at depth 3 the third row has no green cell -> not a member —")
print(" at depth 2 the verdict was still 'member': the limit is not settled)")
'''))

cells.append(md(r'''
## 7. `validate()`: the taxonomy is the source of truth

The structure follows entirely from the taxonomy (Σ, Π, Δ at each level);
the SVG only supplies geometry.  `validate()` cross-checks the parsed
edges against the taxonomy-derived expectation:

- per level *n*: Σ⁰ₙ → Δ⁰ₙ₊₁ and Π⁰ₙ → Δ⁰ₙ₊₁ (climbing), plus
  Δ⁰ₙ → Σ⁰ₙ and Δ⁰ₙ → Π⁰ₙ (same level — Δ sits inside both);
- the top level's Σ⁰ₙ/Π⁰ₙ point up to the Borel sets (dashed).
'''))

cells.append(py(r'''
v = g.validate()
print("validate():", "OK — matches the taxonomy" if v["ok"] else "MISMATCH")
print("missing:", v["missing"] or "none", "| extra:", v["extra"] or "none")

# tamper with the graph -> validate() catches it
g_bad = load_graph(SVG)
del g_bad.edges["edge-delta-2-sigma-2"]
g_bad._build_adjacency()
print()
print("after deleting edge Δ⁰₂ -> Σ⁰₂:")
print("  validate():", "OK" if g_bad.validate()["ok"] else "MISMATCH",
      "| missing:", g_bad.validate()["missing"])
'''))

cells.append(py(r'''
# loud failures: an SVG with no known boxes raises instead of returning
# a silent empty graph
try:
    from borel_hierarchy_v1 import BorelGraph
    BorelGraph.from_svg(HERE / "Borel-Sets-Presentation_2.md")
except ValueError as exc:
    print("ValueError:", exc)
'''))

cells.append(md(r'''
## 8. The id-annotated SVG

`borel_hierarchy_lattice_v1.svg` is the diagram with the anchors the
player needs: each box group carries `id="node-<family>-<level>"` plus
`data-family`, `data-level`, `data-prompt` (the expansion prompt asked
when a box is clicked); each arrow carries `id="edge-<srcslug>-<dstslug>"`
and `data-dashed="true"` for the transfinite ones.  Regenerate it with:

```bash
python borel_hierarchy_v1.py --annotate
```

(then rename the output to `borel_hierarchy_lattice_v1.svg`).
'''))

cells.append(py(r'''
import re
raw = SVG.read_text(encoding="utf-8")
node_ids = re.findall(r'id="(node-[^"]+)"', raw)
edge_ids = re.findall(r'id="(edge-[^"]+)"', raw)
print("node ids:", len(node_ids))
for i in node_ids:
    print("  ", i)
print("edge ids:", len(edge_ids))
for i in edge_ids:
    print("  ", i)
'''))

cells.append(md(r'''
## Recap

| need (from the review) | v1 answer |
| --- | --- |
| traversal, not just queries | `walk()` yields `("node", name)` / `("edge", Edge)` in order |
| edge objects to hang callbacks on | `Edge` dataclass (id, src, dst, dashed, metadata) in `_out`/`_in` |
| stable ids for hooks | `node-sigma-2`, `edge-sigma-1-delta-2`, … |
| parser keeps presentation data | `id` + `data-*` read back onto nodes/edges; `annotate_svg()` writes the annotated diagram |
| no silent failures | `g.warnings` list; `ValueError` on non-XML or zero classes |
| corrected statements | docstring no longer claims strict n→n+1 grading; no `Σ¹₀` alias; `witness_grid` is declared depth-limited; `validate()` cross-checks the SVG against the taxonomy |
| trace for the browser player | `run_walk()` → JSON steps with hook-merged payloads (`Borel-Hierarchy-trace_v1.json`) |
'''))

nb_dict = {
    "cells": cells,
    "metadata": {
        "kernelspec": {
            "display_name": "Python 3 (venv)",
            "language": "python",
            "name": "python3",
        },
        "language_info": {"name": "python", "version": "3.11"},
    },
    "nbformat": 4,
    "nbformat_minor": 5,
}
nb = nbformat.from_dict(nb_dict)
nbformat.validate(nb)
NB_PATH.write_text(nbformat.writes(nb), encoding="utf-8")
print(f"wrote {NB_PATH} with {len(cells)} cells")

# ---------------------------------------------------------------------------
# Headless execution with nbclient (custom kernel spec -> venv python).
# ---------------------------------------------------------------------------

import nbclient

tmpdir = Path(tempfile.mkdtemp(prefix="borel-nav-kernel-"))
(spec_dir := tmpdir / "kernels" / "venv-py").mkdir(parents=True)
(spec_dir / "kernel.json").write_text(json.dumps({
    "argv": [str(VENV_PY), "-m", "ipykernel_launcher", "-f", "{connection_file}"],
    "display_name": "Python 3 (venv)",
    "language": "python",
}), encoding="utf-8")

os.environ["JUPYTER_DATA_DIR"] = str(tmpdir)
client = nbclient.NotebookClient(nb, timeout=300, kernel_name="venv-py")
client.execute()

# save executed notebook (outputs included)
executed = nbformat.from_dict(client.nb)
nbformat.validate(executed)
NB_PATH.write_text(nbformat.writes(executed), encoding="utf-8")

# verify: no error outputs
errors = [
    out
    for cell in executed.cells
    if cell.cell_type == "code"
    for out in cell.get("outputs", [])
    if out.get("output_type") == "error"
]
print(f"executed {sum(1 for c in executed.cells if c.cell_type == 'code')} code cells, "
      f"{len(errors)} errors")
if errors:
    for e in errors:
        print(e.get("ename"), e.get("evalue"))
    sys.exit(1)
print("NOTEBOOK OK")
