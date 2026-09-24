Overall, the note is **conceptually solid and mostly accurate** for a refresher, but it is **not error-free**. It contains several genuine mathematical inaccuracies, a few overgeneralizations, and one nonstandard notation choice that would mislead a careful reader. With corrections, it would be quite good.

## Major inaccuracies / false statements

1. **Borel hierarchy notation is nonstandard and wrong.**  
   The note writes \(\Sigma^1_0,\Pi^1_0,\Sigma^1_1,\dots\). Standard notation uses a superscript \(0\) for the Borel hierarchy:
   \[
   \Sigma^0_1=\text{open},\quad \Pi^0_1=\text{closed},\quad
   \Sigma^0_2=F_\sigma,\quad \Pi^0_2=G_\delta,\quad
   \Sigma^0_3=G_{\delta\sigma},\quad \Pi^0_3=F_{\sigma\delta},\dots
   \]
   Superscript \(1\) is reserved for the **projective hierarchy** (analytic, coanalytic, etc.). The note’s table is shifted and uses the wrong superscript.

2. **“On \(\mathbb R\) (or any second-countable space), compact sets generate the Borel \(\sigma\)-algebra.”**  
   This is false for general second-countable spaces. Compact sets need not generate the Borel \(\sigma\)-algebra unless the space is sufficiently nice, e.g. locally compact second-countable, or \(\mathbb R^d\). In an infinite-dimensional separable Hilbert space, compact sets do not generate the Borel \(\sigma\)-algebra.

3. **Borel’s 1895 theorem and the class of the convergence set.**  
   The note says the Cauchy criterion gives a set of class \(\Sigma^1_2\) (which in its notation is \(G_{\delta\sigma}\)). For the set of convergence of a power series on the circle of convergence, the correct class is \(F_\sigma\) (standard \(\Sigma^0_2\)). For a general sequence of continuous functions, the convergence set is typically \(F_{\sigma\delta}\) (\(\Pi^0_3\)), not \(G_{\delta\sigma}\). So the class is misidentified.

4. **“No strictly larger \(\sigma\)-algebra than \(\mathcal B(\mathbb R)\) supports a translation-invariant, countably additive, normalized measure.”**  
   This is false as written. The Lebesgue \(\sigma\)-algebra is strictly larger than \(\mathcal B(\mathbb R)\) and supports Lebesgue measure, which is translation-invariant, countably additive, and normalized. The correct statement is that there is no translation-invariant, countably additive, normalized measure on the **full power set** of \(\mathbb R\). The maximality statement needs to be phrased much more carefully.

5. **Regularity of Borel probability measures.**  
   The note says: “Every Borel probability measure on a metric space is regular: outer regular by open sets, inner regular by compact sets.”  
   Outer regularity by open sets is fine for finite Borel measures on metric spaces. But **inner regularity by compact sets fails** in general metric spaces. It holds on Polish spaces (or more generally Radon spaces). The note should restrict to Polish/Radon spaces.

6. **\(\mathcal S'(\mathbb R^d)\) is not Polish.**  
   In the Euclidean QFT item, the note says the reconstructed measure lives on \(\mathcal S'(\mathbb R^d)\), “again a Borel measure on a Polish space.” The space \(\mathcal S'(\mathbb R^d)\) with its usual weak-* or strong topology is **not** a Polish space; it is not metrizable. It is a Suslin/nuclear space, and measures on it are handled by a more general theory. So this is inaccurate.

## Minor imprecisions

- **“On any second-countable space \(|\mathcal B(X)|=\mathfrak c\).”**  
  This is true for uncountable second-countable spaces (and for countably infinite ones), but false for finite spaces. The context is clearly uncountable, so it is a minor qualification issue.

- **“Every set of cardinality \(\le\aleph_0\) is Borel.”**  
  True in metric spaces like \(\mathbb R\), but not in an arbitrary second-countable space if singletons are not closed. Minor.

- **Path integrals and uncountable products.**  
  The pitfall about uncountable products is correct: the cylinder \(\sigma\)-algebra is strictly smaller than the Borel \(\sigma\)-algebra of the product topology. But the parenthetical “(path integrals)” is slightly misleading: Wiener measure on \(C[0,1]\) is a Polish-space example where cylinder and Borel \(\sigma\)-algebras **do** coincide. The uncountable-product issue is relevant for other infinite-dimensional settings, not \(C[0,1]\).

## What is accurate

Most of the note is correct:

- Definition of the Borel \(\sigma\)-algebra as the smallest \(\sigma\)-algebra containing open sets.
- Examples: \(\mathbb Q\) is \(F_\sigma\), irrationals are \(G_\delta\), Cantor set is closed, uncountable, measure zero.
- Vitali set is non-Borel.
- Cardinality: \(|\mathcal B(\mathbb R)|=\mathfrak c < 2^{\mathfrak c}=|\mathcal P(\mathbb R)|\).
- Borel vs. Lebesgue: completion, Cantor set subsets, \(L^p\) equivalence classes.
- Standard Borel spaces, Borel isomorphism theorem, measure isomorphism theorem.
- Spectral theorem and projection-valued measures.
- Wiener measure, Feynman–Kac, Gibbs measures, Prokhorov, Portmanteau.
- Analytic sets and the projective hierarchy.

## Verdict

As a **conceptual refresher**, the note is quite good: it captures the main ideas and uses the right examples. As a **precise mathematical reference**, it has enough errors that it should not be trusted without correction. I would rate it roughly **85% accurate**. The main fixes are:

- Use \(\Sigma^0_\alpha,\Pi^0_\alpha\) for the Borel hierarchy.
- Restrict the “compact generators” claim to \(\mathbb R^d\) or locally compact second-countable spaces.
- Correct the class of Borel’s convergence-set theorem.
- Fix the maximality statement about translation-invariant measures.
- Restrict regularity to Polish/Radon spaces.
- Replace “Polish” with a correct description for \(\mathcal S'(\mathbb R^d)\).

With those corrections, the note would be mathematically reliable.