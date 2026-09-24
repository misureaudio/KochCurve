# Borel sets - presentation

## Question 1

I was reflecting about Borel Sets and how they are formalized. Just to say, there are a lot of numerable intersection operators involved in the notation. It's probably very difficult to think about a graphical presentation. possibly animated, of a numerable iteration of a sets operator on a second numerable iteration of the same or a different sets operator on typical subsets of a topological space of use in Measure Theory, however I'd like to explore the feasibility of such a presentation in a teaching context. What's your idea?

## Answer 1

I think it's feasible, and the notation is what makes it look harder than it is. The nested unions and intersections are quantifiers in disguise, and quantifiers can be shown.

**1. Show the witness for a point, not the whole set.**
A point x is in ⋃ₙ ⋂ₘ Aₙ,ₘ exactly when there is an n such that for all m, x ∈ Aₙ,ₘ. So draw a grid of cells (rows n, columns m), colored green if x ∈ Aₙ,ₘ and red otherwise. Membership in the F_σ set means some row is all green forever. Membership in the G_δ set (⋂ₙ ⋃ₘ) means every row has at least one green cell. Let the student drag x along the line and watch the grid recolor. The abstract "∃∀ vs ∀∃" difference becomes a pattern they can see.

**2. Animate one operation as a filmstrip.**
Any countable union can be rewritten as an increasing union, and any countable intersection as a decreasing one, so each operation is a monotone process with time index n. Unions are layers stacking up, and intersections are a filter or sieve tightening. For a two-level operation, the inner limit becomes a set Bₙ, and the outer operation runs over the Bₙ as a second, slower film. Showing the two time scales separately is what keeps it readable.

**3. Use motivating examples where the iteration does real work.**
- ℚ∩[0,1] is F_σ, since it is a countable union of points. Its complement, the irrationals, is G_δ: dense open sets shrinking down while never becoming empty. This is a nice preview of Baire.
- The set where a sequence of functions converges is ⋂ₖ ⋃_N ⋂_{n,m≥N} {|fₙ − fₘ| < 1/k}. Animating "ε shrinks, N is found" shows why this is Borel.
- limsup and liminf of sets, tied to Borel–Cantelli, so the picture connects to measure theory.

**4. Consider Cantor space 2^ω instead of ℝ.**
Points are infinite paths in a binary tree, open sets are unions of cylinders (subtrees), and quantifiers become choices along the path. Dyadic zoom animations work well here because every set is decided by finite prefixes.

**5. Be honest about the limits.**
- Pictures are always finite approximations. Dense sets with empty interior, like ℚ, render as "everything" or "nothing" at any resolution. Handle this with explicit zoom and a "depth n" indicator so students know it's a stage of a limit.
- Beyond level 3 or so, concrete pictures stop being informative. The hierarchy runs through all countable ordinals (ω₁ levels), and that part is better shown as an abstract diagram of the Σ⁰_α, Π⁰_α, Δ⁰_α inclusions. You can add a short animated sketch of why the hierarchy is proper (the diagonal argument with a universal set) and why it stabilizes at ω₁, because a countable set of countable ordinals is bounded.

**Possible lesson arc:**
1. Open and closed sets as pixel-stage approximations.
2. Σ⁰₂ and Π⁰₂ through the witness grid, with ℚ and its complement.
3. Convergence sets as a "real" three-level example.
4. Zoom out to the hierarchy diagram, then the σ-algebra as the closure of all of this.

The most valuable prototype is probably the witness grid from point 1, since it carries the conceptual load. I could build it as an interactive page where you drag a point and switch between examples (ℚ, limsup of intervals, a convergence set). Would you prefer that, or a sketch of the whole hierarchy diagram first?

## Question 2

Hierarchy diagram helps.

## Answer 2

Here is the hierarchy as a lattice, with the simplest sets at the bottom.Purple boxes are the Σ classes (countable unions), teal boxes are the Π classes (countable intersections, the complements of Σ), and gray boxes are Δ = Σ ∩ Π. An arrow means "is contained in". Clicking a box asks me to expand on that class.

### **How to read it**

- **Each level is built from the one below.** Σ⁰ₙ₊₁ is the countable unions of Π⁰ₙ sets, and Π⁰ₙ₊₁ is the countable intersections of Σ⁰ₙ sets. So the F_σ class is a countable union of closed sets, and G_δ is a countable intersection of open sets.
- **The diamonds are the inclusions.** Σ⁰ₙ and Π⁰ₙ both sit inside Δ⁰ₙ₊₁, which sits inside both Σ⁰ₙ₊₁ and Π⁰ₙ₊₁. Each class is therefore contained in everything above it.
- **Δ⁰₁ is missing.** It is the clopen sets, which in ℝ are just ∅ and ℝ, so it is uninformative there. It becomes rich in Cantor space 2^ω, where the clopen sets are the finite unions of cylinders.

### **Why the diagram is more than bookkeeping**

- **Every inclusion is strict.** For an uncountable Polish space, each step up contains sets that the level below lacks. ℚ is Σ⁰₂ but not Π⁰₂, and its complement, the irrationals, is Π⁰₂ but not Σ⁰₂. The proof that this pattern continues upward is a diagonal argument on a universal set.
- **Nothing new appears at ω₁.** A countable union of sets from levels below ω₁ has all its pieces below some countable ordinal, because countably many countable ordinals have a countable supremum. So the union of all levels is already closed under countable unions and complements, and that union is the Borel σ-algebra. The dashed arrows and the ellipsis stand for this transfinite stretch.

**In the lesson**
Connect this back to the witness grid. Each row of the grid corresponds to one climb from a class to the one above it. A point in a Σ⁰₂ set has a row that is green forever, and a point in a Π⁰₂ set has green in every row. The diagram tells students which quantifier pattern belongs to which box.

I could animate the diamond, so that picking a set like ℚ highlights the lowest box containing it and grays out the classes it fails to reach. Or I could build the witness grid and link the two, so that dragging the point moves you through the hierarchy.
