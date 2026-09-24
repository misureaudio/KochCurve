"""borel_hierarchy.py — Parse the Borel hierarchy lattice SVG and navigate it.

Reads ``borel_hierarchy_lattice.svg`` (the teaching diagram from the
Borel-sets presentation: Σ/Pi/Delta classes as boxes, arrows meaning
"is contained in", the Borel sets at the top, dashed arrows standing for
the transfinite stretch through all countable ordinals) and builds a
structured, navigable representation of it.

Graph model
-----------
A directed acyclic graph.  Edge ``u -> v`` means "u is contained in v"
(the arrow points upward, from the smaller class to the larger one).
The lattice is graded: the *rank* of a class is its Borel level
(open/closed = 1, F_σ/G_δ = 2, G_δσ/F_σδ = 3, Borel = ω₁).  Every edge
goes from rank n to rank n+1 (or to the Borel top), so:

* ``ancestors(c)``  = the classes c is contained in (everything above),
* ``descendants(c)``= the classes contained in c (everything below),
* ``level(c)``      = the Borel level (rank),
* ``complement(c)`` = the complementary class of the same level,
* ``build(c)``      = the construction recipe: c is obtained from the
  class below it by the countable set operation of the opposite family
  (Σ⁰ₙ₊₁ = countable unions of Π⁰ₙ sets, and symmetrically).

The parser is tolerant of small layout variations (it locates each box
by its label text, not by position, and recovers the diamond-shaped
inclusion arrows from the geometry of the boxes).

Navigation is provided both as object methods (``graph.ancestors(node)``)
and as module-level functions that take a class name (``ancestors("F_σ")``).

Quick start
-----------
>>> from borel_hierarchy import load_graph
>>> g = load_graph("borel_hierarchy_lattice.svg")
>>> g.classes()
['Σ⁰₁', 'Π⁰₁', 'Σ⁰₂', 'Δ⁰₂', 'Π⁰₂', 'Σ⁰₃', 'Δ⁰₃', 'Π⁰₃', 'Borel sets']
>>> sorted(g.ancestors("Σ⁰₁"))
['Borel sets', 'Δ⁰₂', 'Δ⁰₃', 'Π⁰₂', 'Π⁰₃', 'Σ⁰₂', 'Σ⁰₃']
>>> g.contains("Δ⁰₂", "Π⁰₁")
True
>>> g.complement("F_σ")
'Π⁰₂'
>>> g.build("Σ⁰₂")
'countable unions of Π⁰₁ sets'
>>> g.shortest_path("Π⁰₁", "Borel sets")
['Π⁰₁', 'Δ⁰₂', 'Π⁰₂', 'Δ⁰₃', 'Π⁰₃', 'Borel sets']

Run ``python borel_hierarchy.py`` for a printable summary of the graph.
"""

from __future__ import annotations

import re
import sys
from dataclasses import dataclass, field
from pathlib import Path
from xml.etree import ElementTree as ET

__all__ = [
    "BorelGraph",
    "ClassNode",
    "load_graph",
    # module-level navigation (name-based)
    "ancestors",
    "descendants",
    "level",
    "complement",
    "build",
    "contains",
    "shortest_path",
    "witness_grid",
]

_NS = "{http://www.w3.org/2000/svg}"

# ---------------------------------------------------------------------------
# Class taxonomy — the meaning of every box in the diagram.
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

#: Accepted aliases for each canonical class name.  The alias map is
#: deliberately rich so that navigation can be addressed with whatever
#: spelling the user (or a downstream lesson script) reaches for.
_ALIASES = {
    "Borel sets": {
        "borel", "borel sets", "borel sigma-algebra", "borel algebra",
        "borel sets (borel)", "the borel sets", "borel σ-algebra",
    },
    "Σ⁰₁": {"open", "open sets", "Σ0_1", "sigma 1", "Σ₁", "Σ¹₀"},
    "Π⁰₁": {"closed", "closed sets", "Π0_1", "pi 1", "Π₁", "Π¹₀"},
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


def _normalize(name: str) -> str:
    """Map any accepted spelling of a class name to its canonical form.

    Matching is case-insensitive and tolerant of inner spacing, so
    ``"f-sigma"``, ``"F_σ"``, ``"Σ0_2"`` and ``"open sets"`` all resolve.
    """
    key = re.sub(r"\s+", " ", str(name).strip().lower())
    if key in _ALIAS_LOOKUP:
        return _ALIAS_LOOKUP[key]
    # Tolerate stray spaces inside tokens ("Σ 0 _2" -> "Σ0_2").
    compact = key.replace(" ", "")
    for alias, canonical in _ALIAS_LOOKUP.items():
        if alias.replace(" ", "") == compact:
            return canonical
    raise KeyError(
        f"unknown Borel class {name!r}; known classes: "
        + ", ".join(sorted(_TAXONOMY))
    )


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ClassNode:
    """One box of the lattice: a Borel class plus its geometry."""

    name: str                      # canonical name, e.g. "Σ⁰₂"
    subtitle: str = ""             # text in the box, e.g. "F_σ sets"
    family: str = ""               # sigma / pi / delta / borel
    level: "int | str" = ""        # 1, 2, 3, ... or "omega1"
    x: float = 0.0
    y: float = 0.0
    width: float = 0.0
    height: float = 0.0

    @property
    def rank(self) -> float:
        """Grading of the lattice: concrete levels by number, top = ω₁."""
        return float(self.level) if isinstance(self.level, int) else float("inf")

    @property
    def label(self) -> str:
        return f"{self.name} ({self.subtitle})" if self.subtitle else self.name

    def __str__(self) -> str:  # pragma: no cover - cosmetic
        return self.label


@dataclass
class BorelGraph:
    """Directed graph of the Borel hierarchy parsed from the SVG.

    ``edges`` is a set of ``(source, target)`` pairs where the edge means
    *source is contained in target* (arrow points up).
    """

    title: str = ""
    description: str = ""
    source_file: str = ""
    nodes: dict[str, ClassNode] = field(default_factory=dict)
    edges: set[tuple[str, str]] = field(default_factory=set)
    #: dashed edges: the transfinite "levels continue" stretch to the top.
    dashed_edges: set[tuple[str, str]] = field(default_factory=set)

    # ------------------------------------------------------------------
    # construction / lookup
    # ------------------------------------------------------------------

    @classmethod
    def from_svg(cls, path: "str | Path") -> "BorelGraph":
        """Parse the lattice SVG at *path* into a BorelGraph."""
        path = Path(path)
        raw = path.read_bytes()
        root = ET.fromstring(raw)
        return cls._parse(root, source_file=str(path))

    @staticmethod
    def _parse(root: ET.Element, source_file: str = "") -> "BorelGraph":
        g = BorelGraph(source_file=source_file)
        g.title = _text_of(root.find(f"{_NS}title"))
        g.description = _text_of(root.find(f"{_NS}desc"))

        # --- collect boxes: each <g> containing a <rect> and <text>s ---
        boxes: list[tuple[float, float, float, float, list[str]]] = []
        for grp in root.iter(f"{_NS}g"):
            rect = grp.find(f"{_NS}rect")
            if rect is None:
                continue
            texts = [
                t.text.strip()
                for t in grp.findall(f"{_NS}text")
                if t.text and t.text.strip()
            ]
            if not texts:
                continue
            boxes.append(
                (
                    float(rect.get("x")),
                    float(rect.get("y")),
                    float(rect.get("width")),
                    float(rect.get("height")),
                    texts,
                )
            )

        for x, y, w, h, texts in boxes:
            title = texts[0]
            node = _classify_box(title, x, y, w, h, texts[1:] if len(texts) > 1 else [])
            if node is not None:
                g.nodes[node.name] = node

        # --- collect edges: <line> elements with an arrow marker --------
        # A line is an inclusion arrow if it starts near the border of one
        # box (its source) and ends near the border of another (its target).
        for line in root.iter(f"{_NS}line"):
            x1, y1 = _f(line.get("x1")), _f(line.get("y1"))
            x2, y2 = _f(line.get("x2")), _f(line.get("y2"))
            if None in (x1, y1, x2, y2):
                continue
            src = _nearest_box(boxes, x1, y1, slack=8.0)
            dst = _nearest_box(boxes, x2, y2, slack=8.0)
            if src is None or dst is None or src == dst:
                continue
            edge = (src, dst)
            g.edges.add(edge)
            if line.get("stroke-dasharray"):
                g.dashed_edges.add(edge)

        return g

    # ------------------------------------------------------------------
    # basic accessors
    # ------------------------------------------------------------------

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
        return self.nodes[_normalize(name)]

    def has(self, name: str) -> bool:
        try:
            _normalize(name)
            return True
        except KeyError:
            return False

    def successors(self, name: str) -> set[str]:
        """Classes that directly contain *name* (upward neighbours)."""
        n = _normalize(name)
        return {t for s, t in self.edges if s == n}

    def predecessors(self, name: str) -> set[str]:
        """Classes that *name* directly contains (downward neighbours)."""
        n = _normalize(name)
        return {s for s, t in self.edges if t == n}

    # ------------------------------------------------------------------
    # navigation
    # ------------------------------------------------------------------

    def ancestors(self, name: str) -> set[str]:
        """Everything *name* is contained in — all classes strictly above it
        (transitive closure of the upward edges)."""
        n = _normalize(name)
        seen: set[str] = set()
        stack = [n]
        while stack:
            cur = stack.pop()
            for t in (t for s, t in self.edges if s == cur):
                if t not in seen:
                    seen.add(t)
                    stack.append(t)
        return seen

    def descendants(self, name: str) -> set[str]:
        """Everything contained in *name* — all classes strictly below it."""
        n = _normalize(name)
        seen: set[str] = set()
        stack = [n]
        while stack:
            cur = stack.pop()
            for s in (s for s, t in self.edges if t == cur):
                if s not in seen:
                    seen.add(s)
                    stack.append(s)
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
        b, s = _normalize(big), _normalize(small)
        return b == s or s in self.descendants(b) or b in self.ancestors(s)

    def shortest_path(self, start: str, end: str) -> list[str]:
        """A shortest inclusion chain from *start* (bottom) to *end* (top).

        Returns ``[start, ..., end]``; raises ``LookupError`` if *end* is
        not reachable from *start* (e.g. it sits below or on a side branch).
        """
        a, b = _normalize(start), _normalize(end)
        if a not in self.nodes or b not in self.nodes:
            raise KeyError(f"unknown class in shortest_path({start!r}, {end!r})")
        if b in self.ancestors(a) or a == b:
            pass
        else:
            raise LookupError(
                f"{b!r} is not contained in {a!r}: no inclusion chain exists"
            )
        # BFS from a to b over upward edges.
        from collections import deque

        parent: dict[str, str] = {a: a}
        q = deque([a])
        while q:
            cur = q.popleft()
            if cur == b:
                break
            for t in sorted(x for s, x in self.edges if s == cur):
                if t not in parent:
                    parent[t] = cur
                    q.append(t)
        if b not in parent:  # pragma: no cover - guarded above
            raise LookupError(f"no path from {a!r} to {b!r}")
        chain = [b]
        while chain[-1] != a:
            chain.append(parent[chain[-1]])
        chain.reverse()
        return chain

    def complement(self, name: str) -> str:
        """The complementary class of the same level.

        Σ⁰ₙ and Π⁰ₙ are complements of each other; Δ⁰ₙ is self-complement;
        the Borel sets are self-complement.
        """
        node = self.node(name)
        if node.family == FAMILY_SIGMA:
            return f"Π⁰{_subscript(node.level)}"
        if node.family == FAMILY_PI:
            return f"Σ⁰{_subscript(node.level)}"
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
        lower = "Π⁰" if fam == FAMILY_SIGMA else "Σ⁰"
        op = "countable unions" if fam == FAMILY_SIGMA else "countable intersections"
        return f"{op} of {lower}{_subscript(lev - 1)} sets"

    def examples(self, name: str) -> list[str]:
        """Canonical examples from the presentation, if any are known for
        this class (e.g. ℚ for Σ⁰₂)."""
        return _EXAMPLES.get(_normalize(name), [])

    # ------------------------------------------------------------------
    # visualization
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
                    "subtitle": n.subtitle,
                    "family": n.family,
                    "level": n.level,
                    "contained_in": sorted(self.successors(n.name)),
                }
                for n in sorted(
                    self.nodes.values(), key=lambda n: (n.rank, n.x, n.name)
                )
            ],
            "edges": [
                {"from": s, "to": t, "dashed": (s, t) in self.dashed_edges}
                for s, t in sorted(self.edges)
            ],
        }

    def to_json(self, indent: int = 2) -> str:
        import json

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
        for s, t in sorted(self.edges):
            style = ' [style=dashed]' if (s, t) in self.dashed_edges else ""
            lines.append(f'  "{s}" -> "{t}"{style};')
        lines.append("}")
        return "\n".join(lines)

    def print_summary(self) -> str:
        """A human-readable summary of the parsed graph (also used by the
        CLI).  Returns the text so it can be captured in tests."""
        out = [
            f"Borel hierarchy lattice — {self.title}",
            f"source: {self.source_file or '<in-memory>'}",
            f"{len(self.nodes)} classes, {len(self.edges)} inclusion arrows"
            f" ({len(self.dashed_edges)} dashed = the transfinite stretch)",
            "",
        ]
        # group by level
        by_level: dict["int | str", list[ClassNode]] = {}
        for n in self.nodes.values():
            by_level.setdefault(n.level, []).append(n)
        for lev in sorted(by_level, key=lambda L: (L == "omega1", L)):
            names = ", ".join(sorted(n.name for n in by_level[lev]))
            out.append(f"level {_subscript(lev) if lev != 'omega1' else 'ω₁'}: {names}")
        out.append("")
        out.append("Inclusion arrows (u -> v  means  u ⊆ v):")
        for s, t in sorted(self.edges):
            dash = "  (dashed)" if (s, t) in self.dashed_edges else ""
            out.append(f"  {s} -> {t}{dash}")
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


def _subscript(n: "int | str") -> str:
    """Render a level as a unicode subscript (1 -> ₁, 'omega1' -> ω₁)."""
    if n == "omega1":
        return "ω₁"
    digits = "₀₁₂₃₄₅₆₇₈₉"
    return "".join(digits[int(c)] for c in str(n))


def _classify_box(
    title: str,
    x: float,
    y: float,
    w: float,
    h: float,
    subs: list[str],
) -> "ClassNode | None":
    """Turn one <g> box (its title text) into a ClassNode, or None if the
    title is not a known class."""
    # Normalize the title: the SVG uses unicode superscripts/subscripts.
    key = title.strip()
    node = None
    try:
        canon = _normalize(key)
        family, level = _TAXONOMY[canon]
        node = ClassNode(
            name=canon,
            subtitle=" ".join(subs),
            family=family,
            level=level,
            x=x,
            y=y,
            width=w,
            height=h,
        )
    except KeyError:
        # The top box is titled "Borel sets"; tolerate near-misses on the
        # Σ/Pi/Delta labels (e.g. extra spaces).
        cleaned = re.sub(r"\s+", "", key)
        for canon in _TAXONOMY:
            if re.sub(r"\s+", "", canon) == cleaned:
                family, level = _TAXONOMY[canon]
                node = ClassNode(
                    name=canon,
                    subtitle=" ".join(subs),
                    family=family,
                    level=level,
                    x=x,
                    y=y,
                    width=w,
                    height=h,
                )
                break
    return node


def _nearest_box(
    boxes: list[tuple[float, float, float, float, list[str]]],
    px: float,
    py: float,
    slack: float,
) -> "str | None":
    """Return the canonical name of the box whose border is within *slack*
    of point (px, py), or None.  A line endpoint touching a box's border
    (with a few px of tolerance) is considered attached to that box."""
    best: "tuple[float, str] | None" = None
    for x, y, w, h, texts in boxes:
        # distance from point to the rectangle's border
        dx = max(x - px, 0.0, px - (x + w))
        dy = max(y - py, 0.0, py - (y + h))
        dist = (dx * dx + dy * dy) ** 0.5
        if dist <= slack:
            try:
                name = _normalize(texts[0])
            except KeyError:
                continue
            if best is None or dist < best[0]:
                best = (dist, name)
    return best[1] if best else None


def escape_dot(s: str) -> str:
    return s.replace("\\", "\\\\").replace('"', '\\"')


# ---------------------------------------------------------------------------
# Canonical examples from the presentation (Question/Answer 1 & 2).
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
# Module-level navigation: name-based convenience wrappers.
# They load (and cache) the default SVG next to this module on first use,
# so `from borel_hierarchy import ancestors; ancestors("F_σ")` just works.
# ---------------------------------------------------------------------------

_DEFAULT_GRAPH: "BorelGraph | None" = None
_DEFAULT_PATH = Path(__file__).with_name("borel_hierarchy_lattice.svg")


def default_graph() -> BorelGraph:
    """Load (once) the lattice SVG that sits next to this module."""
    global _DEFAULT_GRAPH
    if _DEFAULT_GRAPH is None:
        if not _DEFAULT_PATH.exists():
            raise FileNotFoundError(
                f"default lattice SVG not found at {_DEFAULT_PATH}; "
                "pass an explicit path to load_graph()"
            )
        _DEFAULT_GRAPH = BorelGraph.from_svg(_DEFAULT_PATH)
    return _DEFAULT_GRAPH


def load_graph(path: "str | Path | None" = None) -> BorelGraph:
    """Parse *path* (or the default SVG next to this module) into a
    BorelGraph.  The default file is cached, so repeated calls are cheap."""
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
) -> bool:
    """Evaluate a point's membership in a two-level Borel set from its
    witness grid (the teaching device of the presentation).

    *grid* is a list of rows; ``grid[n][m]`` is True when ``x ∈ Aₙ,ₘ``.

    ``pattern`` is the quantifier shape:

    * ``"exists-all"`` (Σ⁰₂, F_σ):  ``⋃ₙ ⋂ₘ Aₙ,ₘ`` — some row is all True
      (a row that is green forever).
    * ``"all-exists"`` (Π⁰₂, G_δ):  ``⋂ₙ ⋃ₘ Aₙ,ₘ`` — every row has at
      least one True (green in every row).

    Returns the membership verdict.
    """
    if pattern == "exists-all":
        return any(all(row) for row in grid)
    if pattern == "all-exists":
        return all(any(row) for row in grid)
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
        default=str(_DEFAULT_PATH),
        help="path to the lattice SVG (default: next to this module)",
    )
    p.add_argument(
        "--json",
        action="store_true",
        help="emit the structured graph as JSON instead of a summary",
    )
    p.add_argument(
        "--dot",
        action="store_true",
        help="emit a Graphviz DOT rendering instead of a summary",
    )
    args = p.parse_args(argv)

    g = BorelGraph.from_svg(args.svg)
    if args.json:
        print(g.to_json())
    elif args.dot:
        print(g.to_dot())
    else:
        print(g.print_summary())
    return 0


if __name__ == "__main__":
    sys.exit(_cli(sys.argv[1:]))
