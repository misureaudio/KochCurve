"""borel_hierarchy_v1.py — The Borel hierarchy lattice, hook-ready.

Reads the Borel hierarchy SVG (the teaching diagram from the
Borel-sets presentation: Σ/Pi/Delta classes as boxes, arrows meaning
"is contained in", the Borel sets at the top, dashed arrows standing
for the transfinite stretch through all countable ordinals) and builds
a structured, navigable, *hookable* representation.

What changed in v1 (code-review driven)
---------------------------------------
1. **Traversal, not just queries.**  ``walk()`` is a generator that
   yields an ordered event stream — ``("node", name)`` and
   ``("edge", Edge)`` — so a hook has a well-defined "now at node X,
   crossing edge Y" sequence.  ``run_walk()`` dispatches every event
   through the hook registry and records the walk as a JSON-friendly
   trace (``[{"enter": ...}, {"cross": ...}, ...]``) for replay in a
   browser player.
2. **Real edge objects.**  Edges are ``Edge`` dataclasses (id, src,
   dst, dashed, metadata) stored in adjacency dicts (``_out``/``_in``),
   so callbacks and metadata can be attached.
3. **Stable ids.**  Every node and edge has a stable string id
   (``node-sigma-2``, ``edge-sigma-1-delta-2``).  ``ClassNode`` is no
   longer frozen and carries a mutable ``state`` dict plus a ``data``
   dict with the ``data-*`` attributes read back from the SVG.
4. **Parser keeps presentation data.**  ``id`` and ``data-*``
   attributes (and ``onclick``) are read from the SVG.  ``annotate_svg``
   produces an id-annotated copy of the diagram when the source lacks
   ids.
5. **Loud failures.**  Parse problems land in ``graph.warnings``; a
   file that yields zero classes raises ``ValueError`` instead of
   silently returning an empty graph.
6. **Corrected statements.**  The lattice is *not* strictly graded
   n → n+1: besides the level-climbing edges (Σ⁰ₙ → Δ⁰ₙ₊₁,
   Π⁰ₙ → Δ⁰₊₁, and the dashed Σ⁰ₙ/Π⁰ₙ → Borel), there are
   same-level edges Δ⁰ₙ → Σ⁰ₙ and Δ⁰ₙ → Π⁰ₙ (Δ⁰ₙ = Σ⁰ₙ ∩ Π⁰ₙ sits
   inside both).  The taxonomy — not the SVG — is the source of truth
   for the structure; ``validate()`` cross-checks the parsed edges
   against the taxonomy-derived expectation.  ``witness_grid`` now says
   plainly that it evaluates a finite, depth-limited truncation.

Graph model
-----------
A directed acyclic graph.  Edge ``u -> v`` means "u is contained in v"
(the arrow points upward, from the smaller class to the larger one).
Levels: open/closed = 1, F_σ/G_δ = 2, G_δσ/F_σδ = 3, Borel = ω₁.

Quick start
-----------
>>> from borel_hierarchy_v1 import load_graph
>>> g = load_graph("borel_hierarchy_lattice.svg")
>>> g.classes()
['Σ⁰₁', 'Π⁰₁', 'Σ⁰₂', 'Δ⁰₂', 'Π⁰₂', 'Σ⁰₃', 'Δ⁰₃', 'Π⁰₃', 'Borel sets']
>>> g.contains("Δ⁰₂", "Π⁰₁")
True
>>> g.complement("F_σ")
'Π⁰₂'
>>> g.build("Σ⁰₂")
'countable unions of Π⁰₁ sets'
>>> [name for kind, name in g.walk("open") if kind == "node"]
['Σ⁰₁', 'Δ⁰₂', 'Σ⁰₂', 'Π⁰₂', 'Δ⁰₃', 'Σ⁰₃', 'Π⁰₃', 'Borel sets']
>>> g.validate()["ok"]
True

Hooks and traces
----------------
>>> g2 = load_graph("borel_hierarchy_lattice.svg")
>>> seen = []
>>> _ = g2.on_enter("Σ⁰₂", lambda n: seen.append(n))      # hooks are chainable
>>> _ = g2.on_traverse(lambda e: e.dashed, lambda e: seen.append(e.id))
>>> steps = g2.run_walk("open")
>>> seen
['Σ⁰₂', 'edge-sigma-3-borel', 'edge-pi-3-borel']
>>> steps[0]["enter"]
'Σ⁰₁'
>>> steps[1]["cross"]
['Σ⁰₁', 'Δ⁰₂']

Run ``python borel_hierarchy_v1.py`` for a printable summary of the
graph (``--json`` / ``--dot`` for other renderings).
"""

from __future__ import annotations

import json
import re
import sys
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from xml.etree import ElementTree as ET

__all__ = [
    "BorelGraph",
    "ClassNode",
    "Edge",
    "load_graph",
    "annotate_svg",
    "witness_grid",
    # module-level navigation (name-based)
    "ancestors",
    "descendants",
    "level",
    "complement",
    "build",
    "contains",
    "shortest_path",
]

SVG_NS = "http://www.w3.org/2000/svg"

# ---------------------------------------------------------------------------
# Class taxonomy — the meaning of every box in the diagram.  This, not the
# SVG geometry, is the source of truth for the lattice structure.
# ---------------------------------------------------------------------------

#: family: "sigma" (countable unions), "pi" (countable intersections),
#: "delta" (= Sigma ∩ Pi) or "borel" (the union of all levels).
FAMILY_SIGMA = "sigma"
FAMILY_PI = "pi"
FAMILY_DELTA = "delta"
FAMILY_BOREL = "borel"

#: (family, level) for every class that can appear in the lattice.
#: level is 1, 2, 3, ... for the concrete levels and "omega1" for the top.
_TAXONOMY = {
    "Borel sets": (FAMILY_BOREL, "omega1"),
    "Σ⁰₁": (FAMILY_SIGMA, 1),
    "Π⁰₁": (FAMILY_PI, 1),
    "Σ⁰₂": (FAMILY_SIGMA, 2),
    "Π⁰₂": (FAMILY_PI, 2),
    "Δ⁰₂": (FAMILY_DELTA, 2),
    "Σ⁰₃": (FAMILY_SIGMA, 3),
    "Π⁰₃": (FAMILY_PI, 3),
    "Δ⁰₃": (FAMILY_DELTA, 3),
}

#: Accepted aliases for each canonical class name.  Matching is
#: case-insensitive and tolerant of inner spacing.
_ALIASES = {
    "Borel sets": {
        "borel", "borel sets", "borel sigma-algebra", "borel algebra",
        "borel sets (borel)", "the borel sets", "borel σ-algebra",
    },
    "Σ⁰₁": {"open", "open sets", "Σ0_1", "sigma 1", "Σ₁", "Σ1"},
    "Π⁰₁": {"closed", "closed sets", "Π0_1", "pi 1", "Π₁", "Π1"},
    "Σ⁰₂": {"F_σ", "F_sigma", "Fσ", "F-sigma", "F_σ sets", "Σ0_2", "Σ₂", "Σ2", "sigma 2"},
    "Π⁰₂": {"G_δ", "G_delta", "Gδ", "G-delta", "G_δ sets", "Π0_2", "Π₂", "Π2", "pi 2"},
    "Δ⁰₂": {"Δ0_2", "Delta_2", "Δ₂", "Δ2"},
    "Σ⁰₃": {"G_δσ", "G_ds", "G_delta_sigma", "Gδσ", "Σ0_3", "Σ₃", "Σ3", "sigma 3"},
    "Π⁰₃": {"F_σδ", "F_sd", "F_sigma_delta", "Fσδ", "Π0_3", "Π₃", "Π3", "pi 3"},
    "Δ⁰₃": {"Δ0_3", "Delta_3", "Δ₃", "Δ3"},
}

#: Lowercased alias -> canonical name, built once at import.
_ALIAS_LOOKUP: dict[str, str] = {
    alias.lower(): canonical
    for canonical, aliases in _ALIASES.items()
    for alias in aliases
}
_ALIAS_LOOKUP.update({c.lower(): c for c in _TAXONOMY})

#: Expansion prompts carried as ``data-prompt`` in the annotated SVG —
#: the text asked when a box is clicked in the presentation.
_PROMPTS = {
    "Borel sets": (
        "Explain the Borel σ-algebra as the union of all levels below ω₁, "
        "and why the hierarchy stabilizes at ω₁ (a countable set of countable "
        "ordinals is bounded)."
    ),
    "Σ⁰₁": "What are the open sets (Σ⁰₁) in ℝ?",
    "Π⁰₁": "What are the closed sets (Π⁰₁) in ℝ?",
    "Σ⁰₂": (
        "Explain F_σ sets: countable unions of closed sets. "
        "Example: ℚ ∩ [0,1], a countable union of points."
    ),
    "Π⁰₂": (
        "Explain G_δ sets: countable intersections of open sets. "
        "Example: the irrationals, dense open sets shrinking down while "
        "never becoming empty."
    ),
    "Δ⁰₂": "Which sets are both F_σ and G_δ (Δ⁰₂)?",
    "Σ⁰₃": "Explain G_δσ sets: countable unions of G_δ sets.",
    "Π⁰₃": (
        "Explain F_σδ sets: countable intersections of F_σ sets. "
        "Example: the set where a sequence of continuous functions converges."
    ),
    "Δ⁰₃": "Which sets are both G_δσ and F_σδ (Δ⁰₃)?",
}


def _normalize(name: str) -> str:
    """Map any accepted spelling of a class name to its canonical form."""
    key = re.sub(r"\s+", " ", str(name).strip().lower())
    if key in _ALIAS_LOOKUP:
        return _ALIAS_LOOKUP[key]
    compact = key.replace(" ", "")
    for alias, canonical in _ALIAS_LOOKUP.items():
        if alias.replace(" ", "") == compact:
            return canonical
    raise KeyError(
        f"unknown Borel class {name!r}; known classes: "
        + ", ".join(sorted(_TAXONOMY))
    )


def _subscript(n: "int | str") -> str:
    """Render a level as a unicode subscript (1 -> ₁, 'omega1' -> ω₁)."""
    if n == "omega1":
        return "ω₁"
    digits = "₀₁₂₃₄₅₆₇₈₉"
    return "".join(digits[int(c)] for c in str(n))


def _sigma(level: "int | str") -> str:
    return f"Σ⁰{_subscript(level)}"


def _pi(level: "int | str") -> str:
    return f"Π⁰{_subscript(level)}"


def _delta(level: "int | str") -> str:
    return f"Δ⁰{_subscript(level)}"


def _slug(name: str) -> str:
    """Stable ASCII id fragment for a class: sigma-1, pi-2, delta-3, borel."""
    canon = _normalize(name)
    if canon == "Borel sets":
        return "borel"
    family, level = _TAXONOMY[canon]
    return f"{family}-{level}"


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------


@dataclass
class ClassNode:
    """One box of the lattice: a Borel class plus its geometry and state.

    Mutable on purpose (v1): hooks attach per-node state here, and the
    ``data`` dict holds the ``data-*`` attributes read back from the SVG.
    """

    name: str                      # canonical name, e.g. "Σ⁰₂"
    id: str = ""                   # stable id, e.g. "node-sigma-2"
    subtitle: str = ""             # text in the box, e.g. "F_σ sets"
    family: str = ""               # sigma / pi / delta / borel
    level: "int | str" = ""        # 1, 2, 3, ... or "omega1"
    x: float = 0.0
    y: float = 0.0
    width: float = 0.0
    height: float = 0.0
    data: dict = field(default_factory=dict)   # data-* attributes from the SVG
    state: dict = field(default_factory=dict)  # mutable state for hooks

    @property
    def rank(self) -> float:
        """Level as a sortable number: concrete levels by value, top = ∞."""
        return float(self.level) if isinstance(self.level, int) else float("inf")

    @property
    def label(self) -> str:
        return f"{self.name} ({self.subtitle})" if self.subtitle else self.name

    def __str__(self) -> str:  # pragma: no cover - cosmetic
        return self.label


@dataclass
class Edge:
    """One inclusion arrow.  ``src -> dst`` means *src ⊆ dst* (upward).

    ``dashed`` marks the transfinite "levels continue" stretch to the
    Borel top; ``metadata`` carries the ``data-*`` attributes read from
    the SVG (and anything hooks want to attach).
    """

    src: str
    dst: str
    id: str = ""
    dashed: bool = False
    metadata: dict = field(default_factory=dict)

    def __str__(self) -> str:  # pragma: no cover - cosmetic
        return f"{self.src} -> {self.dst}" + (" (dashed)" if self.dashed else "")


@dataclass
class BorelGraph:
    """Directed graph of the Borel hierarchy parsed from the SVG.

    ``edges`` is keyed by edge id; ``_out``/``_in`` are the adjacency
    dicts (name -> list[Edge]) used by ``walk``.  ``warnings`` collects
    non-fatal parse problems; a file with zero classes raises instead
    of returning an empty graph.
    """

    title: str = ""
    description: str = ""
    source_file: str = ""
    nodes: dict[str, ClassNode] = field(default_factory=dict)
    edges: dict[str, Edge] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)
    _out: dict[str, list[Edge]] = field(default_factory=dict)
    _in: dict[str, list[Edge]] = field(default_factory=dict)
    _hooks: dict[str, list] = field(
        default_factory=lambda: {"enter": [], "traverse": []}
    )
    _last_hook_result: object = None

    # ------------------------------------------------------------------
    # construction / parsing
    # ------------------------------------------------------------------

    @classmethod
    def from_svg(cls, path: "str | Path") -> "BorelGraph":
        """Parse the lattice SVG at *path* into a BorelGraph.

        Raises ``ValueError`` if the file is not parseable XML or yields
        zero classes (a silent empty graph is a bug, not a result).
        """
        path = Path(path)
        raw = path.read_bytes()
        try:
            root = ET.fromstring(raw)
        except ET.ParseError as exc:
            raise ValueError(f"could not parse {path}: {exc}") from exc
        return cls._parse(root, source_file=str(path))

    @staticmethod
    def _parse(root: ET.Element, source_file: str = "") -> "BorelGraph":
        g = BorelGraph(source_file=source_file)

        # --- namespace detection: the SVG may or may not declare one ---
        tag = root.tag
        if isinstance(tag, str) and tag.startswith("{"):
            ns = tag[: tag.index("}") + 1]  # e.g. "{http://www.w3.org/2000/svg}"
        else:
            ns = ""
            g.warnings.append(
                "no XML namespace on the <svg> root; parsed without a "
                "namespace (fragile — re-export the SVG with xmlns)"
            )

        def q(tagname: str) -> str:
            return f"{ns}{tagname}"

        g.title = _text_of(root.find(q("title")))
        g.description = _text_of(root.find(q("desc")))

        # --- collect boxes: each <g> containing a <rect> and <text>s ---
        boxes: list[dict] = []
        for grp in root.iter(q("g")):
            rect = grp.find(q("rect"))
            if rect is None:
                continue
            texts = [
                t.text.strip()
                for t in grp.findall(q("text"))
                if t.text and t.text.strip()
            ]
            if not texts:
                continue
            boxes.append(
                {
                    "g": grp,
                    "rect": rect,
                    "x": float(rect.get("x") or 0),
                    "y": float(rect.get("y") or 0),
                    "w": float(rect.get("width") or 0),
                    "h": float(rect.get("height") or 0),
                    "texts": texts,
                }
            )

        for box in boxes:
            title = box["texts"][0]
            try:
                canon = _normalize(title)
            except KeyError:
                g.warnings.append(
                    f"box with unknown title {title!r} was skipped"
                )
                continue
            family, level = _TAXONOMY[canon]
            if canon in g.nodes:
                g.warnings.append(f"duplicate box for {canon!r}; keeping the first")
                continue
            # stable id: read from the SVG (g or rect), else synthesize
            node_id = box["g"].get("id") or box["rect"].get("id")
            if not node_id:
                node_id = f"node-{_slug(canon)}"
            # data-* attributes (g wins over rect), plus onclick
            data: dict[str, str] = {}
            for el in (box["rect"], box["g"]):
                for attr, value in el.attrib.items():
                    if attr.startswith("data-"):
                        data[attr[5:]] = value
            if box["g"].get("onclick"):
                data["onclick"] = box["g"].get("onclick")
            g.nodes[canon] = ClassNode(
                name=canon,
                id=node_id,
                subtitle=" ".join(box["texts"][1:]) if len(box["texts"]) > 1 else "",
                family=family,
                level=level,
                x=box["x"],
                y=box["y"],
                width=box["w"],
                height=box["h"],
                data=data,
            )

        if not g.nodes:
            raise ValueError(
                f"no Borel classes found in {source_file or '<string>'} — "
                "is this the Borel lattice diagram? (expected boxes titled "
                "like Σ⁰₁, Π⁰₂, …, Borel sets)"
            )

        # --- collect edges: <line> elements with an arrow marker --------
        for idx, line in enumerate(root.iter(q("line"))):
            x1, y1 = _f(line.get("x1")), _f(line.get("y1"))
            x2, y2 = _f(line.get("x2")), _f(line.get("y2"))
            if None in (x1, y1, x2, y2):
                g.warnings.append(f"line #{idx} has incomplete coordinates; skipped")
                continue
            if line.get("marker-end") is None:
                continue  # a plain line, not an arrow
            src = _nearest_box(boxes, x1, y1, slack=8.0)
            dst = _nearest_box(boxes, x2, y2, slack=8.0)
            if src is None or dst is None or src == dst:
                g.warnings.append(
                    f"arrow from ({x1:g},{y1:g}) to ({x2:g},{y2:g}) does not "
                    "touch two known boxes; skipped"
                )
                continue
            dashed = bool(line.get("stroke-dasharray"))
            edge_id = line.get("id") or f"edge-{_slug(src)}-{_slug(dst)}"
            metadata: dict[str, str] = {
                attr[5:]: value
                for attr, value in line.attrib.items()
                if attr.startswith("data-")
            }
            edge = Edge(
                src=src, dst=dst, id=edge_id, dashed=dashed, metadata=metadata
            )
            if edge_id in g.edges:
                g.warnings.append(f"duplicate edge id {edge_id!r}; renaming")
                edge_id = f"{edge_id}-{idx}"
                edge.id = edge_id
            g.edges[edge_id] = edge

        g._build_adjacency()
        return g

    def _build_adjacency(self) -> None:
        """(Re)build the deterministic adjacency dicts from ``edges``."""
        self._out = {name: [] for name in self.nodes}
        self._in = {name: [] for name in self.nodes}
        for e in self.edges.values():
            if e.src in self._out:
                self._out[e.src].append(e)
            if e.dst in self._in:
                self._in[e.dst].append(e)
        for name in self._out:
            self._out[name].sort(
                key=lambda e: (self.nodes[e.dst].rank, self.nodes[e.dst].x, e.dst)
            )
        for name in self._in:
            self._in[name].sort(
                key=lambda e: (self.nodes[e.src].rank, self.nodes[e.src].x, e.src)
            )

    # ------------------------------------------------------------------
    # basic accessors
    # ------------------------------------------------------------------

    def canon(self, name: str) -> str:
        """Canonical class name for any accepted spelling (raises KeyError)."""
        return _normalize(name)

    def classes(self) -> list[str]:
        """All class names, ordered by level (lowest first); within a level,
        left to right as in the diagram (Σ on the left, Π on the right)."""
        return [
            n.name
            for n in sorted(
                self.nodes.values(), key=lambda n: (n.rank, n.x, n.name)
            )
        ]

    def node(self, name: str) -> ClassNode:
        return self.nodes[self.canon(name)]

    def has(self, name: str) -> bool:
        try:
            _normalize(name)
            return True
        except KeyError:
            return False

    def edge(self, ref: "str | Edge") -> Edge:
        """Look up an edge by its id or by an ``Edge`` itself."""
        if isinstance(ref, Edge):
            return ref
        if ref in self.edges:
            return self.edges[ref]
        raise KeyError(f"no edge with id {ref!r}")

    def successors(self, name: str) -> list[str]:
        """Classes that directly contain *name* (upward neighbours, ordered)."""
        n = self.canon(name)
        return [e.dst for e in self._out.get(n, [])]

    def predecessors(self, name: str) -> list[str]:
        """Classes that *name* directly contains (downward neighbours, ordered)."""
        n = self.canon(name)
        return [e.src for e in self._in.get(n, [])]

    # ------------------------------------------------------------------
    # navigation
    # ------------------------------------------------------------------

    def ancestors(self, name: str) -> set[str]:
        """Everything *name* is contained in — all classes strictly above it
        (transitive closure of the upward edges)."""
        n = self.canon(name)
        seen: set[str] = set()
        stack = [n]
        while stack:
            cur = stack.pop()
            for e in self._out.get(cur, []):
                if e.dst not in seen:
                    seen.add(e.dst)
                    stack.append(e.dst)
        return seen

    def descendants(self, name: str) -> set[str]:
        """Everything contained in *name* — all classes strictly below it."""
        n = self.canon(name)
        seen: set[str] = set()
        stack = [n]
        while stack:
            cur = stack.pop()
            for e in self._in.get(cur, []):
                if e.src not in seen:
                    seen.add(e.src)
                    stack.append(e.src)
        return seen

    def level(self, name: str) -> "int | str":
        """The Borel level of *name*: 1, 2, 3, ... or ``"omega1"`` for the
        Borel sets (the union of all levels below ω₁)."""
        return self.node(name).level

    def rank(self, name: str) -> float:
        return self.node(name).rank

    def family(self, name: str) -> str:
        """``"sigma"``, ``"pi"``, ``"delta"`` or ``"borel"``."""
        return self.node(name).family

    def contains(self, big: str, small: str) -> bool:
        """True if the class *small* is contained in the class *big*
        (transitively).  Every class contains itself."""
        b, s = self.canon(big), self.canon(small)
        return b == s or s in self.descendants(b)

    def shortest_path(self, start: str, end: str) -> list[str]:
        """A shortest inclusion chain from *start* (bottom) to *end* (top).

        Returns ``[start, ..., end]``; raises ``LookupError`` if *end* is
        not reachable from *start* (it sits below, or on a side branch).
        """
        a, b = self.canon(start), self.canon(end)
        if a not in self.nodes or b not in self.nodes:
            raise KeyError(f"unknown class in shortest_path({start!r}, {end!r})")
        if b != a and b not in self.ancestors(a):
            raise LookupError(
                f"{b!r} is not contained in {a!r}: no inclusion chain exists"
            )
        parent: dict[str, str] = {a: a}
        q: deque[str] = deque([a])
        while q:
            cur = q.popleft()
            if cur == b:
                break
            for e in self._out.get(cur, []):
                if e.dst not in parent:
                    parent[e.dst] = cur
                    q.append(e.dst)
        if b not in parent:  # pragma: no cover - guarded above
            raise LookupError(f"no path from {a!r} to {b!r}")
        chain = [b]
        while chain[-1] != a:
            chain.append(parent[chain[-1]])
        chain.reverse()
        return chain

    # ------------------------------------------------------------------
    # traversal and hooks (the v1 core)
    # ------------------------------------------------------------------

    def walk(self, start: str, direction: str = "up"):
        """Ordered event stream of a breadth-first traversal.

        Yields ``("node", name)`` when the walk is *at* a class and
        ``("edge", Edge)`` when it crosses an arrow.  Every edge out of
        the current node is yielded (even when the far end was already
        visited — the crossing still happens), but each node's event is
        yielded once.  ``direction`` is ``"up"`` (toward the Borel top,
        the default) or ``"down"`` (toward open/closed).

        The order is deterministic: BFS, neighbours sorted by (level,
        x-position) — the visual reading order of the diagram.
        """
        if direction not in ("up", "down"):
            raise ValueError(f"direction must be 'up' or 'down', got {direction!r}")
        s = self.canon(start)
        adj = self._out if direction == "up" else self._in
        seen = {s}
        queue = deque([s])
        while queue:
            cur = queue.popleft()
            yield ("node", cur)
            for e in adj.get(cur, []):
                yield ("edge", e)
                nxt = e.dst if direction == "up" else e.src
                if nxt not in seen:
                    seen.add(nxt)
                    queue.append(nxt)

    def on_enter(self, name: str, fn) -> "BorelGraph":
        """Register a hook called when the walk enters *name*.

        ``fn(node_name)`` may return a dict, which is merged into the
        payload of the corresponding trace step.  Chainable.
        """
        self._hooks["enter"].append((self.canon(name), fn))
        return self

    def on_traverse(self, match, fn=None) -> "BorelGraph":
        """Register a hook called when the walk crosses a matching edge.

        *match* is one of:

        * an edge id (string), e.g. ``"edge-sigma-3-borel"``;
        * a ``(src, dst)`` pair of class names;
        * a predicate ``edge -> bool``, e.g. ``lambda e: e.dashed``.

        ``fn(edge)`` may return a dict, merged into the trace step.  With
        one argument the callable is a plain observer, invoked on every
        edge the walk crosses (``on_traverse(lambda e: print(e))``).
        Chainable.
        """
        if fn is None:
            if not callable(match) or isinstance(match, str):
                raise TypeError(
                    "on_traverse with one argument needs a callable "
                    f"observer; got {match!r}"
                )
            pred, fn = (lambda e: True), match
        elif callable(match) and not isinstance(match, str):
            pred = match
        elif isinstance(match, str):
            pred = lambda e: e.id == match
        elif isinstance(match, (tuple, list)) and len(match) == 2:
            s, d = self.canon(match[0]), self.canon(match[1])
            pred = lambda e: (e.src, e.dst) == (s, d)
        else:
            raise TypeError(
                "on_traverse match must be an edge id, a (src, dst) pair, "
                f"or a predicate; got {match!r}"
            )
        self._hooks["traverse"].append((pred, fn))
        return self

    def clear_hooks(self) -> None:
        self._hooks["enter"].clear()
        self._hooks["traverse"].clear()

    @property
    def hooks(self) -> dict[str, list]:
        """The live hook registry (for inspection/testing)."""
        return self._hooks

    def dispatch(self, kind: str, item) -> None:
        """Fire the hooks for one walk event: ``kind`` is ``"node"`` (item
        a class name) or ``"edge"`` (item an ``Edge`` or edge id).

        A hook's dict return value, if any, is kept in
        ``_last_hook_result`` so ``run_walk`` can merge it into the step.
        """
        self._last_hook_result = None
        if kind == "node":
            name = item if isinstance(item, str) else item.name
            for hook_name, fn in self._hooks["enter"]:
                if hook_name == name:
                    self._last_hook_result = fn(name)
        elif kind == "edge":
            e = item if isinstance(item, Edge) else self.edge(item)
            for pred, fn in self._hooks["traverse"]:
                if pred(e):
                    self._last_hook_result = fn(e)
        else:
            raise ValueError(f"unknown event kind {kind!r}")

    def run_walk(self, start: str, direction: str = "up") -> list[dict]:
        """Walk + dispatch + record: the walk as a JSON-friendly trace.

        Returns a list of steps, one per event:

        * ``{"enter": name, "payload": {...}}`` — the walk enters a class
          (payload: level, family, build recipe, canonical examples);
        * ``{"cross": [from, to], "edge": edge_id, "dashed": bool,
          "payload": {...}}`` — the walk crosses an arrow, oriented in the
          direction of travel (for a down-walk ``[from, to]`` reverses the
          edge's stored orientation); payload carries the edge's metadata.

        Hook return values (dicts) are merged into the step's payload,
        so hooks can enrich the trace for the browser player.
        """
        steps: list[dict] = []
        cur: "str | None" = None
        for kind, item in self.walk(start, direction):
            if kind == "node":
                cur = item
                node = self.nodes[item]
                step: dict = {
                    "enter": item,
                    "payload": {
                        "level": node.level,
                        "family": node.family,
                        "build": self.build(item),
                        "examples": self.examples(item),
                    },
                }
            else:
                e = item
                nxt = e.dst if direction == "up" else e.src
                step = {
                    "cross": [cur, nxt],
                    "edge": e.id,
                    "dashed": e.dashed,
                    "payload": dict(e.metadata),
                }
            self.dispatch(kind, item)
            if isinstance(self._last_hook_result, dict):
                step["payload"].update(self._last_hook_result)
            steps.append(step)
        return steps

    @staticmethod
    def to_trace_json(steps: list[dict], indent: int = 2) -> str:
        """Serialize a ``run_walk`` trace for the browser player."""
        return json.dumps(steps, ensure_ascii=False, indent=indent)

    # ------------------------------------------------------------------
    # semantics: complement, build, examples, validate
    # ------------------------------------------------------------------

    def complement(self, name: str) -> str:
        """The complementary class of the same level.

        Σ⁰ₙ and Π⁰ₙ are complements of each other; Δ⁰ₙ is
        self-complement; the Borel sets are self-complement.
        """
        node = self.node(name)
        if node.family == FAMILY_SIGMA:
            return _pi(node.level)
        if node.family == FAMILY_PI:
            return _sigma(node.level)
        return node.name

    def build(self, name: str) -> str:
        """The construction recipe for *name*: which operation on which
        lower class builds it.

        Σ⁰ₙ₊₁ = countable unions of Π⁰ₙ sets
        Π⁰ₙ₊₁ = countable intersections of Σ⁰ₙ sets
        Δ⁰ₙ   = Σ⁰ₙ ∩ Π⁰ₙ (the sets in both)
        Borel = union of all levels α < ω₁
        """
        node = self.node(name)
        fam, lev = node.family, node.level
        if fam == FAMILY_BOREL:
            return "the union of all levels Σ⁰_α, Π⁰_α for countable α < ω₁"
        if fam == FAMILY_DELTA:
            return (
                f"the sets that are both Σ⁰{_subscript(lev)} "
                f"and Π⁰{_subscript(lev)}"
            )
        if lev == 1:
            return "open sets" if fam == FAMILY_SIGMA else "closed sets"
        lower = _pi(lev - 1) if fam == FAMILY_SIGMA else _sigma(lev - 1)
        op = "countable unions" if fam == FAMILY_SIGMA else "countable intersections"
        return f"{op} of {lower} sets"

    def examples(self, name: str) -> list[str]:
        """Canonical examples from the presentation, if any are known for
        this class (e.g. ℚ for Σ⁰₂)."""
        return _EXAMPLES.get(self.canon(name), [])

    def expected_edges(self) -> set[tuple[str, str]]:
        """The inclusion edges the taxonomy *requires* for the levels
        present in this graph.  The taxonomy is the source of truth; the
        SVG is a rendering to be cross-checked.

        For each level n: Σ⁰ₙ → Δ⁰ₙ₊₁, Π⁰ₙ → Δ⁰ₙ₊₁ (climbing), and
        Δ⁰ₙ → Σ⁰ₙ, Δ⁰ₙ → Π⁰ₙ (same level — Δ sits inside both).  The
        top level's Σ⁰ₙ/Π⁰ₙ point up to the Borel sets (dashed).
        """
        exp: set[tuple[str, str]] = set()
        levels = sorted(
            {n.level for n in self.nodes.values() if isinstance(n.level, int)}
        )
        for n in levels:
            d_next = _delta(n + 1)
            if d_next in self.nodes:
                exp.add((_sigma(n), d_next))
                exp.add((_pi(n), d_next))
            d = _delta(n)
            if d in self.nodes:
                if _sigma(n) in self.nodes:
                    exp.add((d, _sigma(n)))
                if _pi(n) in self.nodes:
                    exp.add((d, _pi(n)))
        if "Borel sets" in self.nodes and levels:
            top = max(levels)
            exp.add((_sigma(top), "Borel sets"))
            exp.add((_pi(top), "Borel sets"))
        return exp

    def validate(self) -> dict:
        """Cross-check the parsed edges against the taxonomy.

        Returns ``{"ok", "missing", "extra", "warnings"}`` where
        ``missing``/``extra`` are lists of ``(src, dst)`` pairs.
        """
        exp = self.expected_edges()
        got = {(e.src, e.dst) for e in self.edges.values()}
        missing = sorted(exp - got)
        extra = sorted(got - exp)
        return {
            "ok": not missing and not extra,
            "missing": missing,
            "extra": extra,
            "warnings": list(self.warnings),
        }

    # ------------------------------------------------------------------
    # visualization / serialization
    # ------------------------------------------------------------------

    def as_dict(self) -> dict:
        """Plain-dict view of the whole graph (JSON-serializable)."""
        return {
            "title": self.title,
            "description": self.description,
            "source_file": self.source_file,
            "classes": [
                {
                    "name": n.name,
                    "id": n.id,
                    "subtitle": n.subtitle,
                    "family": n.family,
                    "level": n.level,
                    "data": n.data,
                    "contained_in": self.successors(n.name),
                }
                for n in sorted(
                    self.nodes.values(), key=lambda n: (n.rank, n.x, n.name)
                )
            ],
            "edges": [
                {
                    "id": e.id,
                    "from": e.src,
                    "to": e.dst,
                    "dashed": e.dashed,
                    "metadata": e.metadata,
                }
                for e in sorted(self.edges.values(), key=lambda e: e.id)
            ],
            "warnings": list(self.warnings),
        }

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.as_dict(), ensure_ascii=False, indent=indent)

    def to_dot(self) -> str:
        """Graphviz DOT rendering (edge u -> v means u ⊆ v)."""
        fam_style = {
            FAMILY_SIGMA: ("#EEEDFE", "#534AB7"),
            FAMILY_PI: ("#E1F5EE", "#0F6E56"),
            FAMILY_DELTA: ("#F1EFE8", "#5F5E5A"),
            FAMILY_BOREL: ("#F1EFE8", "#5F5E5A"),
        }
        lines = [
            "digraph BorelHierarchy {",
            "  rankdir=BT;",
            '  node [shape=box, style="rounded,filled", fontname="sans"];',
            f'  label="{self.title}";',
        ]
        for n in sorted(
            self.nodes.values(), key=lambda n: (n.rank, n.x, n.name)
        ):
            fill, stroke = fam_style[n.family]
            label = n.name if not n.subtitle else f"{n.name}\\n{escape_dot(n.subtitle)}"
            lines.append(
                f'  "{n.name}" [label="{label}", fillcolor="{fill}", '
                f'color="{stroke}"];'
            )
        for e in sorted(self.edges.values(), key=lambda e: e.id):
            style = " [style=dashed]" if e.dashed else ""
            lines.append(f'  "{e.src}" -> "{e.dst}"{style};')
        lines.append("}")
        return "\n".join(lines)

    def print_summary(self) -> str:
        """A human-readable summary of the parsed graph (also used by the
        CLI).  Returns the text so it can be captured in tests."""
        out = [
            f"Borel hierarchy lattice — {self.title}",
            f"source: {self.source_file or '<in-memory>'}",
            f"{len(self.nodes)} classes, {len(self.edges)} inclusion arrows",
        ]
        dashed = [e for e in self.edges.values() if e.dashed]
        if dashed:
            out[-1] += f" ({len(dashed)} dashed = the transfinite stretch)"
        if self.warnings:
            out.append(f"warnings: {len(self.warnings)}")
            for w in self.warnings:
                out.append(f"  ! {w}")
        out.append("")
        by_level: dict["int | str", list[ClassNode]] = {}
        for n in self.nodes.values():
            by_level.setdefault(n.level, []).append(n)
        for lev in sorted(by_level, key=lambda L: (L == "omega1", L)):
            names = ", ".join(
                sorted((n.name for n in by_level[lev]), key=lambda s: self.nodes[s].x)
            )
            out.append(f"level {_subscript(lev)}: {names}")
        out.append("")
        out.append("Inclusion arrows (u -> v  means  u ⊆ v):")
        for e in sorted(self.edges.values(), key=lambda e: (e.src, e.dst)):
            dash = "  (dashed)" if e.dashed else ""
            out.append(f"  {e.src} -> {e.dst}{dash}")
        out.append("")
        v = self.validate()
        out.append(
            f"validate(): {'OK — matches the taxonomy' if v['ok'] else 'MISMATCH'}"
        )
        if v["missing"]:
            out.append(f"  missing: {v['missing']}")
        if v["extra"]:
            out.append(f"  extra:   {v['extra']}")
        return "\n".join(out)

    def __repr__(self) -> str:  # pragma: no cover - cosmetic
        return (
            f"BorelGraph(classes={len(self.nodes)}, "
            f"edges={len(self.edges)}, source={self.source_file!r})"
        )


# ---------------------------------------------------------------------------
# Parsing helpers
# ---------------------------------------------------------------------------


def _text_of(el: "ET.Element | None") -> str:
    return (el.text or "").strip() if el is not None and el.text else ""


def _f(attr: "str | None") -> "float | None":
    try:
        return float(attr) if attr is not None else None
    except (TypeError, ValueError):
        return None


def _nearest_box(
    boxes: list[dict],
    px: float,
    py: float,
    slack: float,
) -> "str | None":
    """Return the canonical name of the box whose border is within *slack*
    of point (px, py), or None.  A line endpoint touching a box's border
    (with a few px of tolerance) is considered attached to that box."""
    best: "tuple[float, str] | None" = None
    for box in boxes:
        x, y, w, h = box["x"], box["y"], box["w"], box["h"]
        dx = max(x - px, 0.0, px - (x + w))
        dy = max(y - py, 0.0, py - (y + h))
        dist = (dx * dx + dy * dy) ** 0.5
        if dist <= slack:
            try:
                name = _normalize(box["texts"][0])
            except KeyError:
                continue
            if best is None or dist < best[0]:
                best = (dist, name)
    return best[1] if best else None


def escape_dot(s: str) -> str:
    return s.replace("\\", "\\\\").replace('"', '\\"')


# ---------------------------------------------------------------------------
# Canonical examples from the presentation (Questions/Answers 1 & 2).
# ---------------------------------------------------------------------------

_EXAMPLES: dict[str, list[str]] = {
    "Σ⁰₁": ["any open set, e.g. (0, 1)"],
    "Π⁰₁": ["any closed set, e.g. [0, 1]"],
    "Σ⁰₂": [
        "ℚ ∩ [0,1]: a countable union of points (singletons are closed)",
    ],
    "Π⁰₂": [
        "the irrationals: a countable intersection of dense open sets "
        "(a preview of the Baire category theorem)",
    ],
    "Π⁰₃": [
        "the set where a sequence of continuous functions converges: "
        "⋂ₖ ⋃_N ⋂_{n,m≥N} {|fₙ − fₘ| < 1/k} — 'ε shrinks, N is found'; "
        "in fact every Π⁰₃ set is the convergence set of some sequence "
        "of continuous functions (Hausdorff)",
    ],
    "Borel sets": [
        "limsup / liminf of sets (Borel–Cantelli)",
        "every F_σ, G_δ and every level Σ⁰_α, Π⁰_α for α < ω₁",
    ],
}


# ---------------------------------------------------------------------------
# SVG annotation: add the ids / data-* attributes the parser and the
# animation player need to target individual boxes and arrows.
# ---------------------------------------------------------------------------


def annotate_svg(src: "str | Path", dst: "str | Path | None" = None) -> Path:
    """Write an id-annotated copy of the lattice SVG.

    Each box group gets ``id="node-<slug>"`` plus ``data-family``,
    ``data-level`` and ``data-prompt`` (the expansion prompt from the
    presentation); each arrow gets ``id="edge-<srcslug>-<dstslug>"`` and
    ``data-dashed`` when it is one of the transfinite arrows.  Existing
    ``id``/``data-*`` attributes are preserved.  Returns the output path
    (default: ``<src stem>_annotated.svg`` next to the source).
    """
    src = Path(src)
    root = ET.fromstring(src.read_bytes())
    tag = root.tag
    if isinstance(tag, str) and tag.startswith("{"):
        ns = tag[: tag.index("}") + 1]  # e.g. "{http://www.w3.org/2000/svg}"
    else:
        ns = ""

    def q(tagname: str) -> str:
        return f"{ns}{tagname}"

    # boxes
    boxes: list[dict] = []
    for grp in root.iter(q("g")):
        rect = grp.find(q("rect"))
        if rect is None:
            continue
        texts = [
            t.text.strip()
            for t in grp.findall(q("text"))
            if t.text and t.text.strip()
        ]
        if not texts:
            continue
        boxes.append(
            {
                "g": grp,
                "rect": rect,
                "x": float(rect.get("x") or 0),
                "y": float(rect.get("y") or 0),
                "w": float(rect.get("width") or 0),
                "h": float(rect.get("height") or 0),
                "texts": texts,
            }
        )

    for box in boxes:
        try:
            canon = _normalize(box["texts"][0])
        except KeyError:
            continue
        family, level = _TAXONOMY[canon]
        slug = _slug(canon)
        box["g"].set("id", box["g"].get("id") or f"node-{slug}")
        box["g"].set("data-family", family)
        box["g"].set("data-level", str(level))
        box["g"].set("data-prompt", _PROMPTS.get(canon, ""))
        box["slug"] = slug

    # arrows
    for line in root.iter(q("line")):
        x1, y1 = _f(line.get("x1")), _f(line.get("y1"))
        x2, y2 = _f(line.get("x2")), _f(line.get("y2"))
        if None in (x1, y1, x2, y2) or line.get("marker-end") is None:
            continue
        src_name = _nearest_box(boxes, x1, y1, slack=8.0)
        dst_name = _nearest_box(boxes, x2, y2, slack=8.0)
        if src_name is None or dst_name is None:
            continue
        s = next(b["slug"] for b in boxes if b["texts"][0] == src_name)
        d = next(b["slug"] for b in boxes if b["texts"][0] == dst_name)
        line.set("id", line.get("id") or f"edge-{s}-{d}")
        if line.get("stroke-dasharray"):
            line.set("data-dashed", "true")

    ET.register_namespace("", SVG_NS)
    ET.register_namespace("c2pa", "http://c2pa.org/manifest")
    out = Path(dst) if dst is not None else src.with_name(
        f"{src.stem}_annotated.svg"
    )
    out.write_text(ET.tostring(root, encoding="unicode"), encoding="utf-8")
    return out


# ---------------------------------------------------------------------------
# Module-level navigation: name-based convenience wrappers.
# They load (and cache) the lattice SVG next to this module on first use
# (the annotated v1 diagram if present, else the original), so
# `from borel_hierarchy_v1 import ancestors; ancestors("F_σ")` just works.
# ---------------------------------------------------------------------------

_DEFAULT_GRAPH: "BorelGraph | None" = None
_HERE = Path(__file__).parent
_CANDIDATES = [
    _HERE / "borel_hierarchy_lattice_v1.svg",
    _HERE / "borel_hierarchy_lattice.svg",
]


def default_path() -> Path:
    for p in _CANDIDATES:
        if p.exists():
            return p
    raise FileNotFoundError(
        "no lattice SVG found next to this module; pass an explicit path "
        "to load_graph()"
    )


def default_graph() -> BorelGraph:
    """Load (once) the default lattice SVG next to this module."""
    global _DEFAULT_GRAPH
    if _DEFAULT_GRAPH is None:
        _DEFAULT_GRAPH = BorelGraph.from_svg(default_path())
    return _DEFAULT_GRAPH


def load_graph(path: "str | Path | None" = None) -> BorelGraph:
    """Parse *path* (or the default SVG next to this module) into a
    BorelGraph.  Explicit paths are always re-parsed; the default file
    is cached, so repeated calls are cheap."""
    if path is None:
        return default_graph()
    return BorelGraph.from_svg(path)


def ancestors(name: str) -> set[str]:
    """Classes above *name* (what it is contained in)."""
    return default_graph().ancestors(name)


def descendants(name: str) -> set[str]:
    """Classes below *name* (what it contains)."""
    return default_graph().descendants(name)


def level(name: str) -> "int | str":
    return default_graph().level(name)


def family(name: str) -> str:
    return default_graph().family(name)


def complement(name: str) -> str:
    return default_graph().complement(name)


def build(name: str) -> str:
    return default_graph().build(name)


def contains(big: str, small: str) -> bool:
    return default_graph().contains(big, small)


def shortest_path(start: str, end: str) -> list[str]:
    return default_graph().shortest_path(start, end)


# ---------------------------------------------------------------------------
# The witness grid — the conceptual bridge from the presentation.
# ---------------------------------------------------------------------------


def witness_grid(
    grid: list[list[bool]],
    pattern: str = "exists-all",
    depth: "int | None" = None,
) -> bool:
    """Evaluate a point's membership in a two-level Borel set from its
    witness grid (the teaching device of the presentation).

    *grid* is a list of rows; ``grid[n][m]`` is True when ``x ∈ Aₙ,ₘ``.

    ``pattern`` is the quantifier shape:

    * ``"exists-all"`` (Σ⁰₂, F_σ):  ``⋃ₙ ⋂ₘ Aₙ,ₘ`` — some row is all
      True within the shown depth (a row that is green *so far*).
    * ``"all-exists"`` (Π⁰₂, G_δ):  ``⋂ₙ ⋃ₘ Aₙ,ₘ`` — every row shown
      has at least one True.

    **The verdict is depth-limited.**  The grid is a finite truncation of
    a countable family: "some row is all True" means all True *across the
    columns actually shown*, and — if *depth* is given — only the first
    *depth* rows are considered at all.  It is a stage of the limit, not
    the limit itself; a row that is green "forever" cannot be certified
    by any finite grid.  Report the depth to the student (the presentation
    calls for an explicit "depth n" indicator) rather than presenting the
    verdict as settled.

    Returns the membership verdict at the given depth.
    """
    rows = grid if depth is None else grid[:depth]
    if pattern == "exists-all":
        return any(all(row) for row in rows)
    if pattern == "all-exists":
        return all(any(row) for row in rows)
    raise ValueError(f"unknown witness-grid pattern {pattern!r}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _cli(argv: list[str]) -> int:
    import argparse

    p = argparse.ArgumentParser(
        description="Parse the Borel hierarchy lattice SVG and navigate it."
    )
    p.add_argument(
        "svg",
        nargs="?",
        default=None,
        help="path to the lattice SVG (default: the one next to this module)",
    )
    p.add_argument(
        "--json", action="store_true",
        help="emit the structured graph as JSON instead of a summary",
    )
    p.add_argument(
        "--dot", action="store_true",
        help="emit a Graphviz DOT rendering instead of a summary",
    )
    p.add_argument(
        "--annotate", action="store_true",
        help="write an id-annotated copy of the SVG and exit",
    )
    args = p.parse_args(argv)

    if args.annotate:
        src = args.svg or default_path()
        out = annotate_svg(src)
        print(f"wrote {out}")
        return 0

    g = BorelGraph.from_svg(args.svg or default_path())
    if args.json:
        print(g.to_json())
    elif args.dot:
        print(g.to_dot())
    else:
        print(g.print_summary())
    return 0


if __name__ == "__main__":
    sys.exit(_cli(sys.argv[1:]))
