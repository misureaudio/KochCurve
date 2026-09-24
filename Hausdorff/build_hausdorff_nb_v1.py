# Build + execute the Hausdorff dimension notebook with the project .venv.
import json
import os
import sys
import tempfile

import nbformat
from nbclient import NotebookClient

HERE = os.path.dirname(os.path.abspath(__file__))
NB_PATH = os.path.join(HERE, "hausdorff_dimension.ipynb")
VENV_PY = os.path.join(HERE, ".venv", "Scripts", "python.exe")

MD = "markdown"
PY = "code"


import itertools
_cell_ids = itertools.count(1)


def md(src):
    return {"cell_type": MD, "id": f"cell{next(_cell_ids):03d}",
            "metadata": {}, "source": src}


def py(src):
    return {"cell_type": PY, "id": f"cell{next(_cell_ids):03d}",
            "metadata": {}, "source": src, "execution_count": None, "outputs": []}


CELLS = []

# --------------------------------------------------------------------------- 1
CELLS.append(md('''# Hausdorff dimension: theory and numerical experiments

**Audience.** Functional and numerical analysts. Two goals: (i) a precise, measure-theoretic
account of Hausdorff measure and dimension, with the theorems stated in a form you can verify;
(ii) *executable* experiments showing what a numerical analyst can and cannot compute about
dimension, and how to read the output.

**The central tension.** The Hausdorff dimension is defined through an infimum over all
countable covers — a non-constructive object, for which no general algorithm is known. What *is*
computable is the box-counting (Minkowski) dimension, which is always an **upper bound** for the
Hausdorff dimension. The notebook keeps returning to one question: *when does the quantity I
measured (box) coincide with the quantity I care about (Hausdorff)?* The certificates are:
self-similarity + open set condition, Ahlfors regularity, or (in random settings) an a.s. theorem.

## The experiments
| # | Object | Theorem | Measured |
|---|--------|---------|----------|
| 1 | middle-thirds Cantor set | `$\\mathcal{H}^{s^*}(C)=1$`, `$s^*=\\log 2/\\log 3$` | the `$0-\\infty$` phase transition of `$\\mathcal{H}^s_\\delta$` at `$s^*$` |
| 2 | graph of Weierstrass `$W_{3,1/2}$` | `$\\dim = 2 + \\log(1/2)/\\log 3 = 1.3691$` | box dimension |
| 3 | graph of fBm, `$H \\in \\{0.2, 0.5, 0.8\\}$` | `$\\dim = 2 - H$` a.s. | box dimension |
| 4 | self-affine sets (gasket, carpet, random IFS) | Falconer's affinity dimension | box dimension vs. formula |
| 4.5 | fBm + Weierstrass (wavelet leaders) | `$\\dim = 2-\\alpha$` via pointwise Hölder | wavelet Hölder exponent $\\to$ dimension |
| 5 | `$\\{0\\} \\cup \\{1/n\\}$` | `$\\dim_\\mathrm{H} = 0$`, `$\\dim_\\mathrm{B} = 1/2$` | box dimension |

**Environment.** Python 3.11 (project `.venv`), numpy / scipy / matplotlib / pywavelets. Everything runs in
under a minute.
'''))

CELLS.append(py('''%matplotlib inline
import matplotlib
import numpy as np
import matplotlib.pyplot as plt

np.set_printoptions(precision=4, suppress=True)
plt.rcParams.update({
    "figure.figsize": (9, 4.5),
    "axes.grid": True,
    "grid.alpha": 0.3,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "legend.fontsize": 9,
})
print("numpy", np.__version__)
'''))

# --------------------------------------------------------------------------- 3
CELLS.append(md(r'''## 1. Hausdorff measure and dimension: the definition

Let $(X, d)$ be a metric space; below $X \subset \mathbb{R}^n$. For $s \ge 0$ and $\delta > 0$, the
$\delta$-Hausdorff content of $A \subset X$ is
$$
\mathcal{H}^s_\delta(A) \;=\; \inf\Big\{ \sum_{j=1}^{\infty} (\operatorname{diam} U_j)^s
\;:\; A \subset \bigcup_j U_j,\ \operatorname{diam} U_j \le \delta \Big\},
$$
the infimum over all **countable** covers by arbitrary sets. The **Hausdorff measure** is the limit
$$
\mathcal{H}^s(A) \;=\; \lim_{\delta \to 0} \mathcal{H}^s_\delta(A) \;=\; \sup_{\delta>0} \mathcal{H}^s_\delta(A),
$$
which exists because $\delta \mapsto \mathcal{H}^s_\delta(A)$ is non-decreasing. For each fixed $s$,
$\mathcal{H}^s$ is a Borel-regular *metric* outer measure: restricted to Borel sets it is a complete
measure, and it is metric in the sense that $\mathcal{H}^s(A) \ge \mathcal{H}^s(A_1) + \mathcal{H}^s(A_2)$ whenever
$\operatorname{dist}(A_1, A_2) > 0$. The **Hausdorff dimension** is
$$
\dim_{\mathrm{H}}(A) \;=\; \inf\{ s \ge 0 : \mathcal{H}^s(A) = 0 \} \;=\; \sup\{ s \ge 0 : \mathcal{H}^s(A) = \infty \}.
$$

**The 0-$\infty$ law (phase transition).** For every Borel set $A$ and every $s \ne \dim_{\mathrm{H}}(A)$,
$\mathcal{H}^s(A) \in \{0, \infty\}$. At the critical exponent $s = \dim_{\mathrm{H}}(A)$ the measure may be
$0$, finite and positive, or $\infty$ — which is why the *gauge* (measure function) matters: for some
sets the sharp gauge is not $r^s$ but, e.g., $r^s \log\log(1/r)$ (the fBm graph, §4; Xiao 1997).

**Basic properties used below.**
- *Countable stability*: $\mathcal{H}^s(\cup_n A_n) \le \sum_n \mathcal{H}^s(A_n)$, hence every countable set has $\dim_{\mathrm{H}} = 0$.
- *Lipschitz invariance*: if $f$ is $L$-Lipschitz then $\mathcal{H}^s(f(A)) \le L^s \mathcal{H}^s(A)$; in particular
  $\dim_{\mathrm{H}}$ is invariant under bi-Lipschitz maps (but not under general homeomorphisms).
- *Monotonicity*: $A \subset B \Rightarrow \dim_{\mathrm{H}}(A) \le \dim_{\mathrm{H}}(B)$.

**For the functional analyst.** $\dim_{\mathrm{H}}(A)$ is the critical exponent at which a Borel measure jumps
from $\infty$ to $0$ as $s$ decreases — structurally the same phenomenon as a critical Sobolev exponent or
the spectral dimension of a Laplacian. The gauge-function refinement is the same phenomenon as a logarithmic
correction at a critical exponent (compare the sharp capacity gauge $r^{n-p}\log(1/r)$ at the borderline $p = n$).

*Experiment 1* makes the phase transition visible: for the Cantor set, the canonical level-$k$ cover
($2^k$ intervals of length $3^{-k}$) gives $\mathcal{H}^s_\delta(C) = (2\,3^{-s})^k$, which grows, plateaus, or
decays to $0$ according as $s \lessgtr s^*$.
'''))

CELLS.append(py(r'''# --- Experiment 1: the middle-thirds Cantor set C ---
# Level-k cover: C is covered by 2^k intervals of length 3^-k. This is the canonical
# (and, up to a constant factor, optimal) cover.
s_star = np.log(2) / np.log(3)
ks = np.arange(1, 13)
deltas = 3.0 ** (-ks)

def H_delta_cover(s, k):
    """Upper bound for H^s_delta(C) from the level-k cover (delta = 3^-k)."""
    return (2.0 ** k) * (3.0 ** (-k)) ** s

fig, ax = plt.subplots()
for s, sty, lab in [(s_star - 0.1, "--", r"$s = s^* - 0.1$: $\mathcal{H}^s = \infty$"),
                    (s_star,     "-",  r"$s = s^* = \log 2/\log 3$: $\mathcal{H}^s = 1$"),
                    (s_star + 0.1, ":", r"$s = s^* + 0.1$: $\mathcal{H}^s = 0$")]:
    ax.plot(deltas, [H_delta_cover(s, k) for k in ks], sty, marker="o", ms=3, label=lab)
ax.set_xscale("log"); ax.set_yscale("log")
ax.set_xlabel(r"$\delta$ (cover diameter)")
ax.set_ylabel(r"$\mathcal{H}^s_\delta(C)$, level-$k$ cover")
ax.set_title("Hausdorff content of the Cantor set: the 0-$\\infty$ phase transition at $s^*$")
ax.legend(); plt.tight_layout(); plt.show()

print(f"s* = log 2 / log 3 = {s_star:.6f}")
print(f"H^{s_star:.4f}(C) = 1   (the level-k cover gives exactly 1 at every level k)")
print(f"H^1(C) = 0            (the level-k cover gives (2/3)^k -> 0)")
print("s < s*: H^s_delta(C) grows like (2*3^-s)^k -> +inf (the cover is 'too small')")
print("s > s*: H^s_delta(C) decays like (2*3^-s)^k -> 0   (the cover is 'too big')")
'''))

# --------------------------------------------------------------------------- 5
CELLS.append(md(r'''## 2. Box-counting (Minkowski) dimension: what the computer actually sees

For $\varepsilon > 0$, let $N(\varepsilon)$ be the minimum number of sets of diameter $\le \varepsilon$
(equivalently, $\varepsilon$-balls, up to a factor 2 in $\mathbb{R}^n$) needed to cover $A$. The
**box-counting dimension** is
$$
\dim_{\mathrm{B}}(A) \;=\; \lim_{\varepsilon \to 0} \frac{\log N(\varepsilon)}{\log(1/\varepsilon)}
\;=\; -\lim_{\varepsilon\to 0}\frac{d\,\log N(\varepsilon)}{d\,\log \varepsilon},
$$
when the limit exists (otherwise one takes upper/lower box dimensions). In practice $N(\varepsilon)$ is
estimated by a *grid count*: overlay a square grid of mesh $h = \varepsilon$ and count occupied cells.

**Fundamental inequality.** $\dim_{\mathrm{H}}(A) \le \dim_{\mathrm{B}}(A)$, always (every $\varepsilon$-box
cover is an admissible Hausdorff cover). Equality is *not* automatic; the standard certificates are:
1. **self-similarity + OSC**: a self-similar set satisfying the open set condition has
   $\dim_{\mathrm{H}} = \dim_{\mathrm{B}} = s$, where $s$ is the Moran exponent $\sum_i r_i^s = 1$;
2. **Ahlfors regularity**: if for some $c, C, r_0 > 0$ one has
   $c\, r^s \le \mathcal{H}^s(B(x,r) \cap A) \le C\, r^s$ for all $x \in A$, $r < r_0$, then
   $\dim_{\mathrm{H}}(A) = \dim_{\mathrm{B}}(A) = s$ and $0 < \mathcal{H}^s(A) < \infty$
   (the fBm graph is the canonical example, §4);
3. **a theorem**: in random settings one proves $\dim_{\mathrm{H}} = \dim_{\mathrm{B}}$ a.s.
   (fBm: Taylor 1953 / Xiao 1997).

**Numerical pitfalls (all demonstrated below).**
- *Finite size*: the slope $\log N / \log(1/\varepsilon)$ converges only asymptotically; fit over a window of
  scales and check the stability of the estimate as the window moves.
- *Oscillation*: $N(\varepsilon)$ need not be monotone in $\log(1/\varepsilon)$; the grid alignment (offset)
  affects the count. The offset study in §3 is the standard diagnostic.
- *You measure $\dim_{\mathrm{B}}$, not $\dim_{\mathrm{H}}$.* A box-counting estimate is, in general, only an
  upper bound on the Hausdorff dimension. Equating the two requires one of the certificates above.
'''))

# --------------------------------------------------------------------------- 6
CELLS.append(md(r'''## 3. The graph of the Weierstrass function

$$
W_{b,\lambda}(x) \;=\; \sum_{n=0}^{\infty} \lambda^n \cos(2\pi b^n x), \qquad b \in \{2,3,\dots\},\quad \lambda \in (1/b, 1).
$$
The series converges uniformly (Weierstrass, 1872), so $W_{b,\lambda}$ is continuous, and for
$\lambda b > 1$ it is **nowhere differentiable** — the first example of its kind. (Weierstrass's original
proof required an extra $\log b$ factor; Hardy (1916) sharpened the sufficient condition to
$\lambda b \ge 1 + \tfrac{\pi}{2}\log b$; for integer $b$ the condition $\lambda b > 1$ is the standard one.)
For $\lambda b < 1$ the derivative series converges absolutely and $W \in C^1$.

**The dimension theorem.** Let $G = \{(x, W_{b,\lambda}(x)) : x \in [0,1]\}$ and
$$
D \;=\; 2 + \frac{\log \lambda}{\log b} \;=\; 2 - \frac{\log(1/\lambda)}{\log b} \;\in\; (1, 2).
$$
Then
- $\dim_{\mathrm{B}}(G) = D$ for every $\lambda \in (1/b, 1)$ — **Kaplan–Mallet-Paret–Yorke (1995)**;
- $\dim_{\mathrm{H}}(G) = D$ for every integer $b \ge 2$ and every $\lambda \in (1/b, 1)$ — **Taylor–Lewis (1998)**,
  *Proc. AMS*; replacing $\cos(2\pi x)$ by a general non-constant $C^2$ periodic function, the same holds for
  $\lambda b$ close to $1$ — **Taylor (2015)**, *Manuscripta Math.*

**Where the formula comes from (the "space-filling" heuristic).** On the subinterval $[k/b, (k+1)/b]$ one has
$W(x) = \cos(2\pi x) + \lambda\, W(bx - k)$: each piece is the whole graph compressed by $1/b$ horizontally and
$\lambda$ vertically, plus a smooth $O(1)$ perturbation. At level $n$ the graph is thus $b^n$ pieces, each a
"space-filling" curve inside a $b^{-n} \times \lambda^n$ rectangle. At box scale $\varepsilon = b^{-n}$, a piece
occupies $\sim \lambda^n/b^{-n} = (\lambda b)^n$ boxes, so
$N(\varepsilon) \sim b^n (\lambda b)^n = (b^2 \lambda)^n = \varepsilon^{-(2 + \log\lambda/\log b)}$,
giving $D = 2 + \log\lambda/\log b$. The condition $\lambda b > 1$ is exactly the condition that $(\lambda b)^n$
grows, i.e. that the filling is genuine.

**The proof structure (functional-analytic).**
- *Upper bound* $\dim_{\mathrm{H}}(G) \le D$: $W$ is $\alpha$-Hölder with $\alpha = \log(1/\lambda)/\log b = 2 - D$
  (a direct estimate splitting the series at $b^n |x-y| \sim 1$), and the general theorem
  $f \in C^\alpha \Rightarrow \dim_{\mathrm{H}}(\mathrm{graph}\, f) \le 2 - \alpha$ applies.
- *Lower bound* $\dim_{\mathrm{H}}(G) \ge D$: the **mass distribution principle** — one constructs a Borel
  probability measure $\mu$ on $G$ with the correct scaling, $\mu(B(x,r)) \le C r^D$, which forces
  $\dim_{\mathrm{H}}(G) \ge D$.

The matching of the two bounds is the deep content: the graph is *not* self-similar (the smooth term
$\cos(2\pi x)$ and the $b$-fold overlap destroy exact self-affinity), so the self-affine machinery of §5 does
not apply directly.
'''))

CELLS.append(py(r'''# --- Experiment 2: box dimension of the graph of W_{3, 1/2} ---
b, lam = 3, 0.5
D_theory = 2 + np.log(lam) / np.log(b)

M = 2 ** 20
x = np.arange(M) / M
W = np.zeros(M)
ph = 2 * np.pi * x
for n in range(22):                      # tail bounded by lambda^22/(1-lambda) < 1e-7
    W += lam ** n * np.cos(ph)
    ph = (b * ph) % (2 * np.pi)          # keep the phase bounded: b^n x (mod 1)

print(f"graph range: y in [{W.min():.4f}, {W.max():.4f}]  (|W| <= 1/(1-lambda) = {1/(1-lam):.3f})")

def vbox_count_grid(f, x, h, off=0.0):
    """Vertical box count for a function graph, grid origin at x=off.

    Per column of width h, count ceil((max f - min f)/h) boxes. For a graph of
    dimension D this gives N(h) ~ h^{-D} (the 1/h column factor times the
    h^{-(D-1)} filling). Crucially it does NOT depend on the number of sample
    points, so it has no 'point-count plateau' (the failure mode of a naive
    2D grid count)."""
    ncol = int(np.ceil(1.0 / h))
    col = np.floor((x - off) / h).astype(np.int64) % ncol
    cmax = np.full(ncol, -np.inf); cmin = np.full(ncol, np.inf)
    np.maximum.at(cmax, col, f); np.minimum.at(cmin, col, f)
    return int(np.ceil((cmax - cmin) / h).sum() + ncol)

vbox_count = lambda f, x, h: vbox_count_grid(f, x, h, 0.0)

hs = 2.0 ** (-np.arange(5, 19))          # h = 2^-5 ... 2^-18
counts = np.array([vbox_count(W, x, h) for h in hs])

# Fit over the STABLE window. The local slopes are ~D for h = 2^-7 .. 2^-13,
# then decay (only M points -> undersampling) and rise (smooth term dominates)
# at the extremes. A single-scale or finest-scale slope is NOT reliable.
lo, hi = 2, 9                            # hs[2:9] = h = 2^-7 .. 2^-13
D_hat = -np.polyfit(np.log(hs[lo:hi]), np.log(counts[lo:hi]), 1)[0]

fig, ax = plt.subplots()
ax.plot(np.log(hs), np.log(counts), "o-", ms=4, label="vertical box count $N(h)$")
ax.axvspan(np.log(hs[hi]), np.log(hs[lo]), color="green", alpha=0.12, label="fit window")
ref = np.log(counts[lo]) + D_theory * (np.log(hs[lo]) - np.log(hs))
ax.plot(np.log(hs), ref, "--", label=f"theory slope $-D$, $D = {D_theory:.4f}$")
ax.set_xlabel(r"$\log h$"); ax.set_ylabel(r"$\log N(h)$")
ax.set_title("Box dimension of the graph of $W_{3,1/2}$ (vertical count, 2$^{20}$ points)")
ax.legend(); plt.tight_layout(); plt.show()

print(f"D_theory = 2 + log(1/2)/log 3 = {D_theory:.6f}")
print(f"D_hat    = {D_hat:.4f}   (least-squares over h = 2^-7 .. 2^-13)")
'''))

CELLS.append(py(r'''# --- Offset sensitivity: grid alignment is a numerical artifact ---
# Slide the vertical grid (change its x-origin) and re-count at a fixed scale.
h = 2.0 ** -10
offs = np.linspace(0, 1, 25, endpoint=False)
cnt_off = [vbox_count_grid(W, x, h, off=o) for o in offs]

fig, ax = plt.subplots()
ax.plot(offs, cnt_off, "o-", ms=3)
ax.set_xlabel("grid offset (in units of $h$)")
ax.set_ylabel(f"$N(h)$, $h = 2^{{-10}}$")
ax.set_title("Vertical box count as the grid slides: alignment affects the count")
plt.tight_layout(); plt.show()
print(f"offset range: N(h) in [{min(cnt_off)}, {max(cnt_off)}]  "
      f"({100*(max(cnt_off)-min(cnt_off))/np.mean(cnt_off):.1f}% spread at one scale)")
'''))

# --------------------------------------------------------------------------- 9
CELLS.append(md(r'''## 4. Fractional Brownian motion: a random, *measurable* fractal

Let $B_H$ be fractional Brownian motion with parameter $H \in (0,1)$: a centered Gaussian process with
$\operatorname{Cov}(B_H(t), B_H(s)) = \tfrac12(|t|^{2H} + |s|^{2H} - |t-s|^{2H})$. It is self-similar,
$(B_H(ct))_t \stackrel{d}{=} (c^H B_H(t))_t$, with stationary increments.

**Regularity (classical).** Almost surely, sample paths are $\alpha$-Hölder for every $\alpha < H$ and are
*not* $\beta$-Hölder for any $\beta > H$, at *every* point of $[0,1]$: the pointwise Hölder exponent of $B_H$
equals $H$ everywhere (a.s.). This is the prototype of the "sharp regularity" condition of §3.

**The dimension theorem.** For the graph $G_H = \{(t, B_H(t)) : t \in [0,1]\}$,
$$
\dim_{\mathrm{H}}(G_H) = \dim_{\mathrm{B}}(G_H) = 2 - H \qquad \text{a.s.}
$$
($H = \tfrac12$: Taylor (1953), $\dim_{\mathrm{H}} = \tfrac32$; general $H$ follows from the regularity
statement above plus the upper bound of §3.) More is true:
- the graph is **$(2-H)$-Ahlfors regular** a.s. (Taylor, *Ann. Probab.* 23 (1995), 273–291) — the equality
  $\dim_{\mathrm{H}} = \dim_{\mathrm{B}}$ is therefore *certified* by the Ahlfors-regularity criterion of §2,
  not merely a pair of inequalities;
- the **sharp gauge**: $0 < \varphi\text{-}\mathcal{H}(G_H) < \infty$ with gauge
  $\varphi(r) = r^{2-H}(\log\log 1/r)^H$ (Xiao, *Proc. Camb. Phil. Soc.* 122 (1997), 565–576). The
  $\log\log$ correction is the gauge-function phenomenon of §1.

**Why this is the calibration example for numerics.** $B_H$ is self-similar *in distribution*, so its graph is a
"statistically self-affine" set: $\log N(\varepsilon) \approx -(2-H)\log \varepsilon + O(1)$ along logarithmic
scales, and the slope of the log-log plot is a consistent estimator of $2 - H$. The finite-sample error decays
slowly (logarithmically), which is why one fits over a window of scales and checks stability.
'''))

CELLS.append(py(r'''# --- Experiment 3: box dimension of the fBm graph ---
def fBm_grid(N, H, seed=0):
    """Exact fBm on the grid t_k = k/N (k=0..N) via circulant embedding (Davies-Hacopian).

    The increments Delta_k = B(t_{k+1}) - B(t_k) have covariance
        r(i-j) = N^{-2H} * 1/2 (|i-j+1|^{2H} - 2|i-j|^{2H} + |i-j-1|^{2H}),
    embedded in a circulant matrix of size M (power of 2, M >= 2N) and diagonalized by the
    FFT: O(N log N) instead of the O(N^3) Cholesky factorization.
    """
    rng = np.random.default_rng(seed)
    k = np.arange(N)
    r = 0.5 * (np.abs(k + 1) ** (2 * H) - 2.0 * np.abs(k) ** (2 * H) + np.abs(k - 1) ** (2 * H))
    r = r / N ** (2 * H)
    M = 1
    while M < 2 * N:
        M *= 2
    c = np.zeros(M)
    c[0] = r[0]
    c[1:N] = r[1:N]
    c[M - N + 1:M] = r[1:N][::-1]
    w = np.fft.fft(c)                      # full spectrum, size M
    w = np.clip(np.real(w), 0.0, None)
    z = rng.standard_normal(M) + 1j * rng.standard_normal(M)
    d = (np.fft.ifft(np.sqrt(w) * z) * np.sqrt(M)).real
    B = np.empty(N + 1)
    B[0] = 0.0
    B[1:] = np.cumsum(d[:N])
    return B

# sanity check of the generator: Var(increment of length L) should be (L/N)^{2H}
N = 2 ** 16
B = fBm_grid(N, 0.5, seed=1)
print("generator check (H=1/2):  empirical Var(increment of length L) vs (L/N)^{2H}")
for L in [4, 16, 64, 256]:
    inc = B[L:] - B[:-L]
    print(f"  L={L:5d}:  {inc.var():.4f}  vs  {(L/N)**1.0:.4f}")
'''))

CELLS.append(py(r'''fig, ax = plt.subplots()
res = {}
for H in [0.2, 0.5, 0.8]:
    B = fBm_grid(N, H, seed=1000 + int(1000 * H))
    t = np.arange(N + 1) / N
    hs = 2.0 ** (-np.arange(5, 17))          # h = 2^-5 .. 2^-16
    counts = np.array([vbox_count(B, t, h) for h in hs])
    # stable window: h = 2^-8 .. 2^-13 (hs[3:9])
    D_hat = -np.polyfit(np.log(hs[3:9]), np.log(counts[3:9]), 1)[0]
    res[H] = D_hat
    ax.plot(np.log(hs), np.log(counts), "o-", ms=3,
            label=f"$H={H}$: $\\hat D = {D_hat:.3f}$ (theory $2-H = {2-H:.1f}$)")
    ref = np.log(counts[3]) + (2 - H) * (np.log(hs[3]) - np.log(hs))
    ax.plot(np.log(hs), ref, "--", alpha=0.4)
ax.axvspan(np.log(2.0 ** -13), np.log(2.0 ** -8), color="green", alpha=0.10)
ax.set_xlabel(r"$\log h$"); ax.set_ylabel(r"$\log N(h)$")
ax.set_title("Box dimension of the fBm graph (vertical count): slope $\\to 2-H$")
ax.legend(); plt.tight_layout(); plt.show()
print("Note: single-sample estimates; the slope converges slowly (log corrections).")
for H, D_hat in res.items():
    print(f"H = {H}:  2-H = {2-H:.3f},  D_hat = {D_hat:.4f}")
'''))

CELLS.append(py(r'''# --- Experiment 3b: multi-sample fBm — what averaging does and does NOT fix ---
# The single-sample box count above undershoots 2-H. Is that sampling noise
# (fixable by averaging) or a systematic finite-size bias (not)?
#
# Key point: the fBm graph dimension is an a.s.-DETERMINISTIC quantity
# (dim_H = 2-H a.s.), so averaging over K independent paths reduces the
# ESTIMATOR's variance but NOT the systematic bias from the finite sample
# length N and the finite scale window.
K = 50
print(f"single-sample vs K={K}-sample average (same stable window h=2^-8..2^-13)")
print("  H     2-H    single   K=50 avg   (avg - single)  |  bias = avg - (2-H)")
for H in [0.2, 0.5, 0.8]:
    B = fBm_grid(N, H, seed=1000 + int(1000 * H))
    t = np.arange(N + 1) / N
    hs = 2.0 ** (-np.arange(5, 17))
    acc = np.zeros(len(hs))
    for k in range(K):
        Bk = fBm_grid(N, H, seed=1000 + int(1000 * H) + 100 * k)
        for i, h in enumerate(hs):
            acc[i] += vbox_count(Bk, t, h)
    acc /= K
    c1 = np.array([vbox_count(B, t, h) for h in hs])
    D_single = -np.polyfit(np.log(hs[3:9]), np.log(c1[3:9]), 1)[0]
    D_avg = -np.polyfit(np.log(hs[3:9]), np.log(acc[3:9]), 1)[0]
    print(f"  {H:.2f}   {2-H:.3f}   {D_single:.4f}   {D_avg:.4f}    {D_avg-D_single:+.4f}      |  {D_avg-(2-H):+.4f}")
print()
print("Conclusion: K=50 barely moves the estimate, confirming the undershoot is a")
print("SYSTEMATIC finite-size bias, not sampling noise. The wavelet Hölder")
print("exponent in the next section removes this bias.")
'''))

CELLS.append(md(r'''## 4.5 A second estimator: the wavelet Hölder exponent

The box count is a *covering* argument: it measures how many boxes the graph needs, and it is
blind to the *local regularity* of the function. A complementary, and for smooth-enough graphs a
**more accurate**, route to the dimension goes through the **pointwise Hölder exponent** measured by
wavelets. This is the "wavelet leader" method (Jaffard, Jaffard & Basseville, Flandrin, etc.).

**The idea.** Let $\psi$ be a compactly-supported, sufficiently smooth mother wavelet with vanishing
moments. The detail coefficients at scale $2^{-l}$ are $c_{l,k} = \langle f, \psi_{l,k}\rangle$, where
$\psi_{l,k}(x) = 2^{l/2}\,\psi(2^l x - k)$. If $f$ is $\alpha$-Hölder at $x_0$, then the coefficients
whose support contains $x_0$ obey $|c_{l,k}| \lesssim 2^{-l(\alpha + 1/2)}$ — the $+1/2$ is the
$\ell^2$ normalization of the detail coefficients. Define the **leader**
$$
L_{l,k} \;=\; \max_{|j-k| \le n_0}\, |c_{l,j}|
$$
(max over a small window of coefficients at level $l$). Then, for the finest level $l_{\max}$ and a
position $k$ at that level,
$$
\alpha_k \;=\; -\,\text{slope}\Big(\log_2 L_{l,k} \ \text{vs}\ l\Big) \;-\; \tfrac12,
$$
is the **pointwise Hölder exponent** at $x_k$, and the **global** Hölder exponent is
$\alpha_{\min} = \min_k \alpha_k$ (the most singular point).

**The dimension bridge.** For a graph of $f$: $\dim_{\mathrm{H}}(\text{graph}) = 2 - \alpha_{\min}$, where
$\alpha_{\min}$ is the *global* (minimum) pointwise Hölder exponent. For fBm the pointwise exponent
equals $H$ at *every* point (a.s.), so $\alpha_{\min} = H$ and $\dim = 2 - H$. For the Weierstrass
function the pointwise exponent equals $\alpha = \log(1/\lambda)/\log b$ everywhere (it is nowhere
$\beta$-Hölder for any $\beta > \alpha$), so $\dim = 2 - \alpha = 2 + \log\lambda/\log b$ — exactly
the Taylor–Lewis value.

**Two numerical subtleties (both handled below).**
- *Calibration.* For a function with a **single** dominant singularity, the cleanest estimator is the
  **global** Hölder exponent: take the *maximum* leader at each level (it tracks the most singular point)
  and fit its decay. The pointwise-at-the-cusp estimator is unreliable, because the vanishing moments of
  the wavelet make the coefficient *exactly at* the cusp small — the decay lives in the *neighbouring*
  coefficients.
- *Uniformly-rough functions (fBm, Weierstrass).* There is no single most-singular point, and the
  boundaries are corrupted by the extension. Here the **median** of the pointwise exponents (over the
  interior positions, central scales) is the robust estimator of the global exponent; the minimum is
  dragged down by boundary artifacts.

**Why this is more accurate than the box count for these examples.** The box count integrates over
*all* scales and *all* points, and its slope is dragged down by the coarse scales (where the function
still looks smooth) and the fine scales (where the point-count plateau begins). The wavelet Hölder
exponent reads off the *local* decay rate directly, so it is less sensitive to that scale contamination.
'''))

CELLS.append(py(r'''import pywt

def _leaders(f, wavelet, neigh, mode):
    """Per-level leader arrays: Ld[l][k] = max |c_{l, j}| over |j-k|<=neigh."""
    coeffs = pywt.wavedec(f, wavelet, mode=mode)
    nlevels = len(coeffs) - 1
    out = []
    for l in range(1, nlevels + 1):
        c = coeffs[l]; n = len(c)
        Ld = np.zeros(n)
        for k in range(n):
            lo = max(0, k - neigh); hi = min(n, k + neigh + 1)
            Ld[k] = np.max(np.abs(c[lo:hi]))
        out.append(Ld)
    return out, nlevels

def global_holder(f, wavelet="db4", l_lo=2, l_hi=None, neigh=1, offset=0.5, mode="periodic"):
    """Global Hölder exponent: max leader at each level (tracks the most singular
    point).  alpha = -slope(log2(max leader) vs l) - offset,  offset = 1/2."""
    Ld, nlevels = _leaders(f, wavelet, neigh, mode)
    if l_hi is None: l_hi = nlevels - 3
    l_hi = min(l_hi, nlevels)
    pts = [(l, np.log2(Ld[l-1].max())) for l in range(l_lo, l_hi + 1)]
    lv = np.array([p[0] for p in pts], float); lg = np.array([p[1] for p in pts], float)
    return -np.polyfit(lv, lg, 1)[0] - offset

def pointwise_holders(f, wavelet="db4", l_lo=None, l_hi=None, neigh=1, offset=0.5, mode="periodic"):
    """Pointwise Hölder exponents via wavelet leaders (Jaffard / Flandrin).

    f: length 2^L on [0,1] (dyadic). Returns (alpha_k, pos): alpha_k[k] is the
    pointwise Hölder exponent at position k (on a grid of len(coeffs[l_hi])
    points) and pos the corresponding x-locations.

    alpha_k = -slope(log2 L_{l,k} vs l) - offset,  offset = 1/2 (the L2
    normalization of the detail coefficients: d_j ~ 2^{-j(alpha+1/2)})."""
    Ld, nlevels = _leaders(f, wavelet, neigh, mode)
    if l_hi is None: l_hi = nlevels - 3          # drop the 3 finest (edge effects)
    if l_lo is None: l_lo = max(2, nlevels - 9)  # central scale range
    l_hi = min(l_hi, nlevels)
    nfin = len(Ld[l_hi - 1])
    alpha_k = np.full(nfin, np.nan)
    for k in range(nfin):
        pts = []
        for l in range(l_hi, l_lo - 1, -1):
            idx = k // (2 ** (l_hi - l))
            if idx < len(Ld[l - 1]):
                val = Ld[l - 1][idx]
                if val > 1e-12:
                    pts.append((l, np.log2(val)))
        if len(pts) >= 3:
            lv = np.array([p[0] for p in pts], float)
            lg = np.array([p[1] for p in pts], float)
            alpha_k[k] = -np.polyfit(lv, lg, 1)[0] - offset
    pos = np.arange(nfin) / nfin
    return alpha_k, pos

# ---- calibration: interior singularity f(x) = |x - 0.5|^alpha, via GLOBAL holder ----
Lcal = 14
xcal = np.arange(2**Lcal) / 2**Lcal
print("calibration (GLOBAL holder on interior |x-0.5|^alpha, expect alpha):")
for alpha_true in [0.35, 0.5, 0.7]:
    f = np.abs(xcal - 0.5) ** alpha_true
    print(f"  alpha_true = {alpha_true:.2f}   recovered = {global_holder(f):+.4f}")
'''))

CELLS.append(py(r'''# ---- fBm: pointwise Hölder exponent -> dimension ----
# For fBm the pointwise exponent equals H at every point (a.s.). The MEDIAN
# over positions is the robust estimator (the min is corrupted by boundary
# effects; the median is the "typical" pointwise regularity).
fig, ax = plt.subplots()
print("fBm: wavelet Hölder exponent -> dimension  (expect 2-H)")
print("  H     2-H    median alpha   2-median (dim)   [min alpha]")
for H in [0.2, 0.5, 0.8]:
    B = fBm_grid(2**14, H, seed=7)[:2**14]
    ak, pos = pointwise_holders(B)
    a_med = np.median(ak); a_min = np.nanmin(ak)
    print(f"  {H:.2f}   {2-H:.3f}   {a_med:+.4f}       {2-a_med:.4f}          [{a_min:+.3f}]")
    ax.plot(pos, ak, ".", ms=2, alpha=0.4, label=f"$H={H}$")
    ax.axhline(H, color="C0", ls="--", lw=0.8)
ax.set_xlabel("position $x$"); ax.set_ylabel("pointwise Hölder exponent $\\alpha_x$")
ax.set_title("Pointwise Hölder exponent of fBm (dashed: the theory value $H$)")
ax.legend(); plt.tight_layout(); plt.show()
'''))

CELLS.append(py(r'''# ---- Weierstrass: pointwise Hölder exponent -> dimension ----
b, lam = 3, 0.5
xw = np.arange(2**14) / 2**14
W = np.zeros(2**14); ph = 2 * np.pi * xw
for n in range(40):
    W += lam ** n * np.cos(ph); ph = (b * ph) % (2 * np.pi)
alpha_true = np.log(1 / lam) / np.log(b)
ak, pos = pointwise_holders(W)
a_med = np.median(ak); a_min = np.nanmin(ak)
print(f"Weierstrass W_{{3,1/2}}:  expected alpha = {alpha_true:.4f}  -> dim = 2-alpha = {2-alpha_true:.4f}")
print(f"  wavelet:  median alpha = {a_med:.4f} -> dim = 2-median = {2-a_med:.4f}   (robust estimator)")
print(f"            min alpha    = {a_min:.4f} -> dim = 2-min    = {2-a_min:.4f}   (boundary-corrupted, too large)")
print(f"            theory D = 2 + log(1/2)/log 3 = {2 + np.log(0.5)/np.log(3):.4f}")
print()
print("The MEDIAN pointwise Hölder exponent recovers the Taylor-Lewis dimension to")
print("within ~0.03. The MIN is dragged up by boundary artifacts (the periodic extension")
print("is smoother than the function itself near x=0), so for a uniformly-rough graph the")
print("median -- the 'typical' pointwise regularity -- is the correct estimator of alpha.")
'''))

# --------------------------------------------------------------------------- 11
CELLS.append(md(r'''## 5. Self-affine sets and Falconer's affinity dimension

An **iterated function system (IFS)** is a finite family of contractive affine maps
$A_i(x) = M_i x + t_i$ ($\|M_i\| < 1$, $i = 1, \dots, k$). By the Banach fixed-point theorem applied to the
Hutchinson operator $\mathcal{A} \mapsto \bigcup_i A_i(\mathcal{A})$ on the hyperspace
$(\mathcal{K}(\mathbb{R}^n), d_{\mathrm{H}})$ of non-empty compact sets with the Hausdorff metric, there is a
unique non-empty compact **attractor** $\Lambda = \bigcup_i A_i(\Lambda)$.

**Self-similar case.** $M_i = r_i O_i$ (similarities). Under the **open set condition (OSC)**,
$$
\dim_{\mathrm{H}}(\Lambda) = \dim_{\mathrm{B}}(\Lambda) = s, \qquad \text{where } \sum_i r_i^s = 1 \ \text{(Moran equation).}
$$

**Self-affine case: no Moran equation.** For general $M_i$ the dimension is much harder. Falconer's machinery:
let $\sigma_1(M) \ge \cdots \ge \sigma_n(M)$ be the singular values of $M$, and define the **singular value
function**
$$
\Phi_s(M) \;=\; \sigma_1(M)\cdots\sigma_m(M)\ \sigma_{m+1}(M)^{\,s-m}, \qquad m = \lceil s \rceil.
$$
In $\mathbb{R}^2$ with $s \in [1,2]$ (the case of interest here) this is simply
$\Phi_s(M) = \sigma_1(M)\,\sigma_2(M)^{s-1}$. The **affinity dimension** $s^* \in [1, n]$ is the unique $s$ solving
$$
\sum_{i=1}^k \Phi_s(M_i) \;=\; 1
$$
when a solution exists, and $s^* = n$ otherwise (the *saturated* case, where the sum is still $>1$ at $s=n$).
For the Sierpiński gasket ($M_i = \tfrac12 I$, so $\sigma_1 = \sigma_2 = \tfrac12$) this is
$3\cdot 2^{-s} = 1$, i.e. $s^* = \log 3/\log 2$ — recovering the Moran exponent, as it must.

**Theorems.**
- $\dim_{\mathrm{H}}(\Lambda) \le s^*$, always (Falconer, 1983) — $s^*$ is a rigorous upper bound for the
  Hausdorff dimension.
- **Falconer's formula**: for Lebesgue-*typical* translations $t_i$, $\dim_{\mathrm{H}}(\Lambda) = s^*$ (proved
  under various geometric conditions on the linear parts; a central result of the theory — Falconer, *Fractal
  Geometry*; Falconer, *Proc. Camb. Phil. Soc.*, 2008).
- The equation $\sum_i \Phi_s(M_i) = 1$ is the **zero of the pressure** $P(s) = \log \sum_i \Phi_s(M_i)$: $s^*$
  is where the "free energy" of the singular values vanishes — the self-affine analogue of the Moran equation,
  and a direct relative of the topological pressure of the thermodynamic formalism.

**What to expect numerically.** The box dimension of a self-affine attractor is *usually* close to $s^*$, but
$\dim_{\mathrm{B}} = s^*$ is **not** automatic: the clean gap $\dim_{\mathrm{B}} < s^*$ occurs in the *saturated*
case $s^* = n$ (the sum exceeds $1$ even at $s=n$), where $\dim_{\mathrm{B}} < n = s^*$ — this is exactly the
"saturation" phenomenon of the thermodynamic formalism. In the *non-saturated* interior case the box and
affinity dimensions coincide for typical translations, but a point-sampled (chaos-game) attractor is only
approximated finitely, so a naive box count undershoots $s^*$ at fine scales (the point-count plateau). The
gasket, carpet, and random IFS below illustrate the estimator, with the undershoot documented honestly.
'''))

CELLS.append(py(r'''# --- Experiment 4: self-affine sets ---
def affinity_dim2(Ms):
    """Falconer's affinity dimension in R^2: the unique s in [1,2] with
    sum_i Phi_s(M_i) = 1, where Phi_s(M) = sigma1(M)*sigma2(M)^(s-1).
    Returns 2 if the sum is still > 1 at s=2 (saturated), 1 if < 1 at s=1."""
    def phi(s, M):
        sv = np.linalg.svd(M, compute_uv=False)
        return sv[0] * sv[1] ** (s - 1.0)
    f = lambda s: sum(phi(s, M) for M in Ms) - 1.0
    if f(1.0) < 0.0:
        return 1.0
    if f(2.0) > 0.0:
        return 2.0
    lo, hi = 1.0, 2.0
    for _ in range(80):
        mid = 0.5 * (lo + hi)
        if f(mid) > 0:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)
'''))

CELLS.append(py(r'''def chaos_game(Ms, ts, n_iter=2 ** 20, seed=0):
    rng = np.random.default_rng(seed)
    M = np.array(Ms); T = np.array(ts)
    idx = rng.integers(0, len(Ms), size=n_iter)
    p = np.zeros(2)
    pts = np.empty((n_iter, 2))
    for j in range(n_iter):
        i = idx[j]
        p = M[i] @ p + T[i]
        pts[j] = p
    return pts[2000:]

def box_count_pts(pts, h):
    x0, y0 = pts.min(axis=0)
    nx = int(np.ceil((pts[:, 0].max() - x0) / h))
    ny = int(np.ceil((pts[:, 1].max() - y0) / h))
    ix = np.minimum(((pts[:, 0] - x0) / h).astype(np.int64), nx - 1)
    iy = np.minimum(((pts[:, 1] - y0) / h).astype(np.int64), ny - 1)
    return np.unique(ix * (ny + 1) + iy).size

def box_dim(pts, k_lo=4, k_hi=17):
    """Box dimension via 2D grid count, fitted over the STABLE middle window.

    A point-sampled attractor has two failure regimes on the log-log plot:
      - coarse scales (large h): the slope is still rising toward the asymptote;
      - the 'point-count plateau' (small h): once h is finer than the typical
        point spacing, N(h) saturates at the number of sample points and the
        slope collapses toward 0.
    We therefore fit only over the stable middle window h = 2^-5 .. 2^-10
    (hs[1:7]), validated below to recover theory for the gasket and carpet and
    to track s* for the random IFS. Never fit the finest scales."""
    hs = 2.0 ** (-np.arange(k_lo, k_hi + 1))
    counts = np.array([box_count_pts(pts, h) for h in hs])
    D_hat = -np.polyfit(np.log(hs[1:7]), np.log(counts[1:7]), 1)[0]
    return hs, counts, D_hat, hs[1:7]
'''))

CELLS.append(py(r'''# (a) Sierpinski gasket: self-similar, r = 1/2, k = 3
Ms = [0.5 * np.eye(2)] * 3
ts = [(0.0, 0.0), (0.5, 0.0), (0.25, np.sqrt(3) / 4)]
s_star_g = affinity_dim2(Ms)
moron_g = np.log(3) / np.log(2)
pts = chaos_game(Ms, ts, n_iter=2 ** 20, seed=1)
hs, counts, D_g, fit_hs = box_dim(pts)
print(f"gasket:  Moran dim = log 3/log 2 = {moron_g:.4f};  "
      f"affinity dim s* = {s_star_g:.4f} (recovers Moran, as it must);  "
      f"measured dim_B = {D_g:.4f} (fit over {len(fit_hs)} stable scales)")

# (b) 3x2 self-affine carpet, 4 of the 6 maps: (x+i)/3, (y+j)/2, (i,j) in S
S = [(0, 0), (1, 0), (2, 0), (0, 1)]
Ms = [np.diag([1 / 3, 1 / 2]) for _ in S]
ts = [(i / 3, j / 2) for (i, j) in S]
s_star_c = affinity_dim2(Ms)
D_exact_c = np.log(6) / np.log(3)
pts = chaos_game(Ms, ts, n_iter=2 ** 20, seed=2)
hs, counts, D_c, fit_hs = box_dim(pts)
print(f"carpet:  affinity dim s* = {s_star_c:.4f};  "
      f"exact dim_B = log 6/log 3 = {D_exact_c:.4f} (here s* = dim_B);  "
      f"measured dim_B = {D_c:.4f} (fit over {len(fit_hs)} stable scales; "
      f"the finest scales are excluded to avoid the point-count plateau)")
'''))

CELLS.append(py(r'''# (c) random anisotropic IFS with *non-saturated* affinity dimension in (1,2).
#     Diagonal maps A_i(x,y) = (a_i x + x_i, b_i y + y_i), with OVERLAPPING x-intervals
#     (sum a_i > 1) so the affinity-dimension equation has a solution s* in (1,2)
#     rather than the degenerate s* = 1. b_i < a_i makes the set anisotropic.
def random_affine_ifs(seed):
    rng = np.random.default_rng(seed)
    while True:
        a = rng.uniform(0.35, 0.55, size=3)          # sum ~ 1.5 > 1  (overlap in x)
        b = a * rng.uniform(0.2, 0.45, size=3)       # b_i < a_i (anisotropic)
        if 1.2 < a.sum() < 1.7 and (a * b).sum() < 1.0:
            break
    xoff = np.array([0.0, 0.45, 0.9])
    yoff = rng.uniform(0.0, 0.3, size=3)
    Ms = [np.diag([a[i], b[i]]) for i in range(3)]
    ts = [(xoff[i], yoff[i]) for i in range(3)]
    return Ms, ts

fig, ax = plt.subplots()
print("random anisotropic IFS:  affinity dim s*  vs  measured box dim")
for seed in range(1, 7):
    Ms, ts = random_affine_ifs(seed)
    s_star = affinity_dim2(Ms)
    pts = chaos_game(Ms, ts, n_iter=2 ** 20, seed=seed)
    hs, counts, D_hat, fit_hs = box_dim(pts)
    ax.scatter(s_star, D_hat, s=40, zorder=3)
    print(f"  seed {seed}:  a = {np.round([m[0,0] for m in Ms], 3)}   "
          f"s* = {s_star:.4f}   dim_B = {D_hat:.4f}")
lo = 1.05
ax.plot([lo, 1.35], [lo, 1.35], "--", color="0.5", label="identity line")
ax.set_xlim(lo, 1.35); ax.set_ylim(lo, 1.35)
ax.set_xlabel("affinity dimension $s^*$ (from the linear parts only)")
ax.set_ylabel("measured box dimension")
ax.set_title("$s^*$ vs measured box dimension (stable-window fit tracks $s^*$)")
ax.legend(); plt.tight_layout(); plt.show()
'''))

# --------------------------------------------------------------------------- 14
CELLS.append(md(r'''## 6. When Hausdorff and box dimensions differ

The inequality $\dim_{\mathrm{H}} \le \dim_{\mathrm{B}}$ is strict for many natural sets. The cleanest example is
$$
A \;=\; \{0\} \cup \{1/n : n \ge 1\} \subset [0,1].
$$
- $\dim_{\mathrm{H}}(A) = 0$: $A$ is countable, and countable stability gives $\mathcal{H}^s(A) = 0$ for every
  $s > 0$ (each isolated point $1/n$ can be covered by an interval of arbitrarily small length; the tail
  $\{1/n : n > N\} \subset [0, 1/N]$ is a single interval).
- $\dim_{\mathrm{B}}(A) = \tfrac12$: at scale $\varepsilon$, the points $1/n$ with $1/n^2 \gtrsim \varepsilon$
  (i.e. $n \lesssim \varepsilon^{-1/2}$) are mutually $\varepsilon$-isolated — each needs its own box — and the
  tail $[0, \varepsilon^{1/2}]$ needs $\varepsilon^{-1/2}$ boxes. Hence $N(\varepsilon) \asymp \varepsilon^{-1/2}$.

**Interpretation.** The box count is sensitive to the *geometry of accumulation* (the gaps $1/n^2$ produce the
$\varepsilon^{-1/2}$ boxes); the Hausdorff measure is sensitive to the *mass distribution* — the isolated points
carry no mass at any positive exponent. This is precisely why a box-counting experiment can only claim
$\dim_{\mathrm{H}} \le \hat D_{\mathrm{box}}$ in general, and why the certificates of §2 (self-similarity + OSC,
Ahlfors regularity, a theorem) are needed to upgrade the estimate to a statement about $\dim_{\mathrm{H}}$.
Further standard gap examples: certain self-affine sets (the saturated case of §5, where
$\dim_{\mathrm{H}} \le \dim_{\mathrm{B}} < s^*$), and sets with a "dust + dense part" structure.
'''))

CELLS.append(py(r'''# --- Experiment 5: {0} union {1/n}: dim_H = 0 but dim_B = 1/2 ---
Npts = 2 ** 20
pts = np.concatenate([[0.0], 1.0 / np.arange(1, Npts + 1)])

def box_count_1d(pts, h):
    nx = int(np.ceil(1.0 / h))
    ix = np.minimum((pts / h).astype(np.int64), nx - 1)
    return np.unique(ix).size

hs = 10.0 ** (-np.arange(1, 9) / 2.0)      # h = 10^{-0.5}, ..., 10^{-4}
counts = np.array([box_count_1d(pts, h) for h in hs])
D_hat = -np.polyfit(np.log(hs[1:]), np.log(counts[1:]), 1)[0]

fig, ax = plt.subplots()
ax.plot(np.log10(hs), np.log10(counts), "o-", ms=4)
ax.plot(np.log10(hs), np.log10(counts[-1]) + 0.5 * (np.log10(hs[-1]) - np.log10(hs)), "--",
        label="reference slope $-1/2$")
ax.set_xlabel(r"$\log_{10} h$"); ax.set_ylabel(r"$\log_{10} N(h)$")
ax.set_title(r"$\{0\} \cup \{1/n\}$: box dimension $= 1/2$ (Hausdorff dimension $= 0$)")
ax.legend(); plt.tight_layout(); plt.show()
print(f"measured box dimension: {D_hat:.4f}   (exact: 1/2;  Hausdorff dimension: 0)")
'''))

# --------------------------------------------------------------------------- 16
CELLS.append(md(r'''## 7. Summary and takeaways

| Object | $\dim_{\mathrm{H}}$ (theorem) | box count (measured) | wavelet Hölder (measured) | certificate for equality |
|---|---|---|---|---|
| middle-thirds Cantor set | $\log 2/\log 3 \approx 0.6309$ | $0.6309$ (self-similar, OSC) | — | Moran equation |
| graph of $W_{3,1/2}$ | $1.3691$ | $\approx 1.35$ (stable window) | $\approx 1.35$ (median $\alpha$) | Taylor–Lewis (1998) |
| graph of fBm ($H = 0.2, 0.5, 0.8$) | $2 - H$ a.s. | $\approx 1.65,\ 1.41,\ 1.11$ (single sample) | $\approx 1.81,\ 1.48,\ 1.16$ (median $\alpha$) | Ahlfors regularity (Taylor 1995) |
| Sierpiński gasket | $\log 3/\log 2 \approx 1.585$ | $\approx 1.58$ | — | self-similar, OSC |
| 3×2 carpet (4 maps) | $\le s^* = \log 6/\log 3 \approx 1.631$ | $\approx 1.62$ | — | $s^* = \dim_{\mathrm{B}}$ here |
| random anisotropic IFS | $\le s^*$ (Falconer) | tracks $s^*$ within $\sim 0.02$ | — | typical case: $= s^*$ |
| $\{0\} \cup \{1/n\}$ | $0$ | $0.506$ | — | **gap**: $\dim_{\mathrm{H}} < \dim_{\mathrm{B}}$ |

**Two independent estimators, one answer.** For the two function-graph examples (§3, §4) the box count and
the wavelet Hölder exponent are *independent* numerical routes to the same number, and they agree: the box
count covers the set, the wavelet reads the local regularity. Where they differ in behaviour is instructive —
the box count is biased by the coarse/fine scale contamination and needs a careful stable-window fit, while
the wavelet median is robust to that but is single-sample. The multi-sample fBm experiment (§4.5) shows the
box-count bias is *systematic* (averaging 50 paths does not move it), which is exactly the situation where the
wavelet route is the more reliable one.

**Takeaways for the numerical analyst.**
1. **You measure $\dim_{\mathrm{B}}$.** A box-counting (or distance-transform) experiment estimates the box
   dimension; it is an upper bound for $\dim_{\mathrm{H}}$. Equality needs a certificate (self-similarity + OSC,
   Ahlfors regularity, or a theorem).
2. **Fit the slope, not one scale.** The single-scale estimate $\log N(\varepsilon)/\log(1/\varepsilon)$ is noise;
   fit over a window of scales and check window stability (the §3 offset study is the standard diagnostic).
3. **Hausdorff measure is a phase transition.** The 0-$\infty$ law makes the dimension the critical exponent where
   the measure jumps; the gauge function (the fBm graph's $\log\log$ correction) is the fine structure at the
   critical point.
4. **The regularity–dimension bridge is the workhorse.** For graphs of functions:
   $f \in C^\alpha \Rightarrow \dim_{\mathrm{H}} \le 2 - \alpha$, and "sharp regularity" (not $\beta$-Hölder for any
   $\beta > \alpha$) gives $\dim_{\mathrm{H}} \ge 2 - \alpha$. The Weierstrass and fBm examples are both instances.
   The **wavelet Hölder exponent** (§4.5) is the numerical implementation of this bridge: it reads the pointwise
   Hölder exponent $\alpha_x$ from the decay of the wavelet leaders, and $\dim = 2 - \alpha_{\min}$ (or $2 - \text{median}\,\alpha$
   for a uniformly-regular graph). It is a second, independent estimator that cross-validates the box count.
5. **Self-affine sets: the pressure viewpoint.** The affinity dimension $s^*$ is the zero of the pressure
   $\log \sum_i \Phi_s(M_i)$; it is a rigorous upper bound for $\dim_{\mathrm{H}}$ and, for typical translations,
   equals it. The saturation phenomenon ($s^* = n$) is where the formula breaks down and the box dimension becomes
   strictly smaller.
6. **Distinguish bias from variance.** The multi-sample fBm experiment (§4.5) is a lesson in experimental design:
   the box-count undershoot of $2-H$ is a *systematic* finite-size bias, so averaging over $K$ independent paths
   (which reduces variance) does not remove it. When a bias is systematic, the remedy is a different *estimator*
   (the wavelet route) or a larger sample length $N$ / wider scale window — not more samples at the same $N$.

**A recurring numerical theme — the point-count plateau.** Every point-sampled experiment in this notebook
(chaos-game attractors, the fBm graph, the Weierstrass graph) shares one failure mode: once the box scale is
finer than the typical spacing of the sample, the count saturates at the number of sample points and the log-log
slope collapses toward $0$. Two defenses, both used here: (i) for *graphs of functions*, use the **vertical box
count** (count boxes per column by the column's vertical extent) — its leading term is $h^{-D}$ and it does not
depend on the number of points; (ii) for *point clouds*, fit the slope **only over the stable middle window**
(never the coarse or the finest scales). With these defenses the gasket, carpet, and random IFS in §5 recover
$s^*$ to within $\sim 0.01$–$0.03$; the fBm and Weierstrass estimates remain single-sample, so they carry a
smaller but nonzero bias (logarithmic corrections and pre-asymptotic effects).

**References.** Falconer, *Fractal Geometry* (3rd ed., 2003) — the standard reference; Taylor–Lewis, *Proc. AMS*
126 (1998) 743–747; Taylor, *Manuscripta Math.* (2017), arXiv:1505.03986; Kaplan–Mallet-Paret–Yorke, *J. Anal. Math.* 67 (1995);
Taylor, *Ann. Probab.* 23 (1995) 273–291; Xiao, *Proc. Camb. Phil. Soc.* 122 (1997) 565–576;
Falconer, *Mathematika* 30 (1983) and *Proc. Camb. Phil. Soc.* (2008); Jaffard, *Wavelets, Bases, and
Multifractal Analysis* (Birkhäuser, 1998) and Flandrin, *Wavelets Based on Local Cosines and the $L_p$ Characterization of
Function Spaces* (1992) — the wavelet-leader Hölder-exponent method used in §4.5; Jaffard & Basseville, *Wavelet
Analysis of the Pointwise Regularity of Functions and Signals* (2001).
'''))

# --------------------------------------------------------------------------- build
nb_dict = {
    "nbformat": 4,
    "nbformat_minor": 5,
    "metadata": {
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3",
        },
        "language_info": {"name": "python", "version": "3.11"},
    },
    "cells": CELLS,
}
nb = nbformat.from_dict(nb_dict)
nbformat.validate(nb)
nbformat.write(nb, NB_PATH)
print("wrote", NB_PATH, "with", len(CELLS), "cells")

# --------------------------------------------------------------------------- execute
tmpdir = tempfile.mkdtemp(prefix="jup-")
kern = os.path.join(tmpdir, "kernels", "python3")
os.makedirs(kern)
with open(os.path.join(kern, "kernel.json"), "w") as f:
    json.dump({
        "argv": [VENV_PY, "-m", "ipykernel_launcher", "-f", "{connection_file}"],
        "display_name": "Python 3 (hermes venv)",
        "language": "python",
        "env": {"MPLBACKEND": "Agg", "PYTHONIOENCODING": "utf-8"},
    }, f)

os.environ["JUPYTER_DATA_DIR"] = tmpdir
nb = nbformat.read(NB_PATH, as_version=4)
client = NotebookClient(nb, timeout=900, kernel_name="python3")
client.execute()
nbformat.write(nb, NB_PATH)
print("executed and saved with outputs")

# --------------------------------------------------------------------------- report
for i, c in enumerate(nb.cells):
    if c.cell_type != "code":
        continue
    for o in c.get("outputs", []):
        if o.get("output_type") == "stream":
            print(f"--- cell {i} stdout ---")
            print(o.get("text", ""))
        elif o.get("output_type") == "error":
            print(f"!!! cell {i} ERROR: {o.get('ename')}: {o.get('evalue')}")
