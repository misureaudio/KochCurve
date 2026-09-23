#!/usr/bin/env python3
"""
The Koch curve: construction and numerical verification.

Sections
  1  Iterative construction (IFS) and the limiting parametrization
  2  Box-counting (Minkowski) dimension
  3  Koch snowflake: perimeter -> infinity, area -> 8/5 * A0
  4  Holder exponent of the parametrization
  5  p-variation threshold
  6  Fourier coefficients: exact solution of the functional equation
  7  Sup-norm convergence rate of the canonical iteration
  8  Bi-Holder estimate and the no-tangent theorem (numerical)

Requires: numpy, matplotlib.
Run:  python koch_simulation.py
Outputs: console log + figures/fig1..fig5
"""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "figures")
os.makedirs(OUT, exist_ok=True)

D_KOCH = np.log(4.0) / np.log(3.0)    # Hausdorff = box dimension of the curve
H_KOCH = np.log(3.0) / np.log(4.0)    # Holder exponent of the parametrization
E60 = np.exp(1j * np.pi / 3.0)
SQ3 = np.sqrt(3.0)

# ----------------------------------------------------------------------
# 1.  Construction
# ----------------------------------------------------------------------
def koch_points(n, p0=0.0 + 0.0j, p1=1.0 + 0.0j, bump=1):
    """Level-n polygonal approximation of the Koch curve from p0 to p1.

    The generator replaces each segment [a,b] by the four segments
    [a, a+v], [a+v, a+v+w], [a+v+w, a+2v], [a+2v, b],
    where v = (b-a)/3 and w = bump * v * e^{i pi/3} (equilateral bump).
    bump=+1: bump to the left of the direction a->b (open curve);
    bump=-1: bump to the right (outward bumps for the snowflake).
    """
    pts = [p0, p1]
    for _ in range(n):
        new = []
        for a, b in zip(pts[:-1], pts[1:]):
            v = (b - a) / 3.0
            w = (bump * E60) * v
            new += [a, a + v, a + v + w, a + 2.0 * v]
        new.append(pts[-1])
        pts = new
    return np.array(pts)


def apply_digit(z, d):
    """Vectorized S_d: the similitude of ratio 1/3 associated with digit d."""
    return np.where(d == 0, z / 3.0,
          np.where(d == 1, 1.0 / 3.0 + z * E60 / 3.0,
          np.where(d == 2, 0.5 + 1j * SQ3 / 6.0 + z * np.conj(E60) / 3.0,
                   2.0 / 3.0 + z / 3.0)))


def koch_param(t, levels=40):
    """Limiting Koch curve at parameter t in [0,1].

    The address of t is its base-4 expansion t = 0.d1 d2 ...; the point is
    the limit of the compositions S_{d1} o S_{d2} o ... o S_{dn}(0).
    The parameter t is the natural one: at level n, each of the 4^n segments
    carries a t-interval of length 4^{-n}.  (t = 1 is the address 0.3333..._4.)
    """
    t = np.asarray(t, dtype=float)
    scalar = t.ndim == 0
    t = np.atleast_1d(t)
    digits = []
    u = t.copy()
    for _ in range(levels):
        u = u * 4.0
        d = np.floor(u).clip(0, 3).astype(np.int64)
        digits.append(d)
        u = u - d
    z = np.zeros_like(t, dtype=complex)
    for d in reversed(digits):
        z = apply_digit(z, d)
    return z[0] if scalar else z


# ----------------------------------------------------------------------
# 2.  Box-counting dimension (count cells crossed by the segments)
# ----------------------------------------------------------------------
def box_count_polygon(pts, eps, samples_per_seg=9):
    a, b = pts[:-1].reshape(-1, 1), pts[1:].reshape(-1, 1)
    tt = np.linspace(0.0, 1.0, samples_per_seg)
    pts_f = a[:, None, :] + (b[:, None, :] - a[:, None, :]) * tt[None, :, None]
    pts_f = pts_f.reshape(-1, 1)  # complex
    lo = pts_f.min(axis=0)
    p = (pts_f - lo) / eps
    cells = np.stack([np.floor(p.real), np.floor(p.imag)], axis=1).astype(np.int64)
    return len(np.unique(cells, axis=0))


print("=" * 74)
print("[2] Box-counting (Minkowski) dimension of the Koch curve")
n_box = np.arange(4, 10)
eps_b = 3.0 ** (-n_box)
N_b = np.array([box_count_polygon(koch_points(int(n)), e) for n, e in zip(n_box, eps_b)])
for n, e, N in zip(n_box, eps_b, N_b):
    print(f"     n = {n:2d}   eps = 3^-n = {e:.3e}   N(eps) = {N:8d}   "
          f"log N / log(1/eps) = {np.log(N) / np.log(1.0 / e):.5f}")
slope = -np.polyfit(np.log(eps_b), np.log(N_b), 1)[0]
print(f"     slope of the log-log fit  = {slope:.5f}")
print(f"     exact value log 4/log 3   = {D_KOCH:.5f}")
print(f"     ratio N(3^-n)/4^n        = "
      + "  ".join(f"{N / 4.0 ** n:.3f}" for N, n in zip(N_b, n_box)))

# ----------------------------------------------------------------------
# 3.  Snowflake: perimeter and area  (equilateral triangle, outward bumps)
# ----------------------------------------------------------------------
def koch_snowflake(n, side=1.0):
    p0 = 0.0 + 0.0j
    p1 = complex(side, 0.0)
    p2 = complex(0.5 * side, -SQ3 / 2.0 * side)   # e^{-i pi/3} * side
    # clockwise triangle -> outward bumps use bump=+1
    parts = [koch_points(n, a, b, bump=1)[:-1] for a, b in [(p0, p1), (p1, p2)]]
    parts.append(koch_points(n, p2, p0, bump=1))
    return np.concatenate(parts)


def polygon_area(pts):
    x, y = pts.real, pts.imag
    return 0.5 * abs(np.sum(x * np.roll(y, -1) - np.roll(x, -1) * y))


print()
print("=" * 74)
print("[3] Koch snowflake (equilateral, side 1):  A0 = sqrt(3)/4 = %.6f" % (SQ3 / 4))
print("      n      perimeter P_n      3*(4/3)^n       area A_n          A_n/A0")
A0 = SQ3 / 4.0
for n in range(0, 11):
    pts = koch_snowflake(n)
    closed = np.concatenate([pts, [pts[0]]])
    P = np.sum(np.abs(np.diff(closed)))
    A = polygon_area(pts)
    print(f"      {n:2d}      {P:12.6f}     {3 * (4.0 / 3.0) ** n:12.6f}"
          f"      {A:10.6f}        {A / A0:8.5f}")
A_inf = 2.0 * SQ3 / 5.0
print(f"     limit:  A = 2*sqrt(3)/5 = {A_inf:.6f}   =  (8/5)*A0   (ratio 1.60000)")

# ----------------------------------------------------------------------
# 4.  Holder exponent
# ----------------------------------------------------------------------
print()
print("=" * 74)
print("[4] Holder exponent of the parametrization "
      "(exact value log 3/log 4 = %.5f)" % H_KOCH)
print("     exact identity  phi(4^-n) = 3^-n :")
for n in [1, 2, 3, 5, 8, 12]:
    z = koch_param(4.0 ** (-n), levels=60)
    print(f"       n={n:2d}:  phi(4^-n) = {z.real:.12f} + {z.imag:.2e} j     "
          f"(3^-n = {3.0 ** -n:.12f})")
t0_list = np.array([0.0, 0.1, 0.37, 0.5, 0.73, 1.0])
h_list = 4.0 ** (-np.arange(6, 23))     # middle range: asymptotic, no contamination
pairs = []
for t0 in t0_list:
    z0 = koch_param(t0, levels=50)
    for h in h_list:
        for sgn in (1.0, -1.0):
            t1 = t0 + sgn * h
            if 0.0 <= t1 <= 1.0:
                d = abs(koch_param(t1, levels=50) - z0)
                if d > 0:
                    pairs.append((h, d))
pairs = np.array(pairs)
ratios = np.log(pairs[:, 1]) / np.log(pairs[:, 0])
print(f"     local exponent log|dphi|/log|h| over {len(ratios)} probe pairs, "
      f"h in [4^-22, 4^-6]:")
print(f"        mean = {ratios.mean():.5f}   median = {np.median(ratios):.5f}   "
      f"(exact exponent = {H_KOCH:.5f})")

# ----------------------------------------------------------------------
# 5.  p-variation
# ----------------------------------------------------------------------
print()
print("=" * 74)
print("[5] p-variation:  threshold p* = log 4/log 3 = %.5f" % D_KOCH)
print("      p        4*3^-p      V_p^(n) = (4*3^-p)^n at n=1 / n=10     conclusion")
for p in [1.0, 1.2, D_KOCH, 1.27, 1.4]:
    r = 4.0 * 3.0 ** (-p)
    v1, v10 = r, r ** 10
    if r > 1.0:
        concl = "V_p = +inf (p < p*)"
    elif abs(r - 1.0) < 1e-9:
        concl = "V_p = 1 (p = p*: critical, finite)"
    else:
        concl = "V_p finite (p > p*)"
    print(f"      {p:6.4f}   {r:8.5f}      {v1:10.3f} / {v10:10.3f}      {concl}")
# critical value: V_{p*} exactly, on vertex and multi-scale partitions
def _pvar(php, p):
    return float(np.sum(np.abs(np.diff(php)) ** p))
t_v = np.linspace(0.0, 1.0, 4 ** 8 + 1)
print("     critical value  V_{p*} on vertex partitions (exact, all levels):")
for n in [5, 6, 7, 8]:
    tv = np.linspace(0.0, 1.0, 4 ** n + 1)
    print(f"       level n={n}:  V_{{p*}} = {_pvar(koch_param(tv, 50), D_KOCH):.10f}")
t_ms = np.unique(np.concatenate([np.linspace(0.0, 1.0, 4 ** n + 1) for n in range(1, 9)]))
print(f"       multi-scale (levels 1..8, {len(t_ms)} pts):  "
      f"V_{{p*}} = {_pvar(koch_param(t_ms, 50), D_KOCH):.10f}")
print("     (supremum is exactly 1: a C^{0,alpha} curve has finite p-variation for")
print("      all p >= 1/alpha = p*, and the critical value is attained by the")
print("      vertex partitions, where 4*3^{-p*}=1 makes each level contribute 1)")

# ----------------------------------------------------------------------
# 6.  Fourier coefficients: exact solution of the functional equation
# ----------------------------------------------------------------------
# Functional equation:  phi(t) = a_j + (b_j/3) phi(4t - j)  on [j/4, (j+1)/4].
# For F(alpha) = int_0^1 phi(u) e^{-2 pi i alpha u} du:
#   F(alpha) = 1/4 [ A(alpha) J(alpha/4) + B(alpha) F(alpha/4) ],
#   A(alpha) = sum_j a_j e^{-2 pi i alpha j/4},
#   B(alpha) = (1/3) sum_j b_j e^{-2 pi i alpha j/4},
#   J(beta)  = int_0^1 e^{-2 pi i beta u} du.
# Iterating gives the convergent series (factor 4^{-(m+1)}, |B| <= 4/3):
#   F(alpha) = sum_{m>=0} 4^{-(m+1)} [prod_{l<m} B(alpha/4^l)] A(alpha/4^m) J(alpha/4^{m+1}).
A_ = np.array([0.0, 1.0 / 3.0, 0.5 + 1j * SQ3 / 6.0, 2.0 / 3.0], dtype=complex)
B_ = np.array([1.0, E60, np.conj(E60), 1.0], dtype=complex)


def A_fn(alpha):
    return np.sum(A_ * np.exp(-2j * np.pi * alpha * np.arange(4) / 4.0))


def B_fn(alpha):
    return (1.0 / 3.0) * np.sum(B_ * np.exp(-2j * np.pi * alpha * np.arange(4) / 4.0))


def J_fn(beta):
    if beta == 0:
        return 1.0 + 0.0j
    return (1.0 - np.exp(-2j * np.pi * beta)) / (2j * np.pi * beta)


def F_alpha(alpha, tol=1e-15, max_iter=200):
    """Exact F(alpha) = int phi(u) e^{-2 pi i alpha u} du by the convergent series.

    The recursion F(alpha) = 1/4 [A(alpha) J(alpha/4) + B(alpha) F(alpha/4)]
    iterated to a limit.  Note J(alpha/4^{m+1}) = 0 whenever alpha/4^{m+1} is a
    nonzero integer, so the first terms can vanish exactly; we therefore only
    stop early once a nonzero partial sum has been accumulated.
    """
    total = 0.0 + 0.0j
    prod = 1.0 + 0.0j
    a = alpha
    for m in range(max_iter):
        term = (0.25 ** (m + 1)) * prod * A_fn(a) * J_fn(a / 4.0)
        total += term
        if m >= 1 and total != 0 and abs(term) < tol * abs(total):
            return total
        prod *= B_fn(a)
        a = a / 4.0
    return total


def c_k(k):
    """The k-th Fourier coefficient c_k = int_0^1 phi(t) e^{-2 pi i k t} dt."""
    return F_alpha(k)


print()
print("=" * 74)
print("[6] Fourier coefficients of phi (exact series, verified against FFT)")
# symmetry phi(1-t) = 1 - conj(phi(t))  =>  Re(c_k) = 0 for all k
t_sym = np.linspace(0.0, 1.0, 512)
sym_err = np.max(np.abs(koch_param(1.0 - t_sym, 50) - (1.0 - np.conj(koch_param(t_sym, 50)))))
print(f"     symmetry  phi(1-t) = 1 - conj(phi(t)):  max error = {sym_err:.2e}")
c0 = c_k(0)
c1 = c_k(1)
print(f"     c_0 = {c0.real:.6f} + {c0.imag:.6f} j   (= (1/3)(a_0+...+a_3), the centroid)")
print(f"     c_1 = {c1.real:.6f} + {c1.imag:.6f} j     |c_1| = {abs(c1):.6f}  (purely imaginary)")
print("     lacunary identity  c_{4^n} = c_1 / 4^n  (exact, from the functional equation):")
for n in range(1, 13):
    ck = c_k(4 ** n)
    print(f"       n={n:2d}  k={4 ** n:9d}   |c_k|*k = {abs(ck) * 4 ** n:.6f}"
          f"   c_k/(c_1/4^n) = {ck / (c1 / 4 ** n):.8f}")
# generic decay
k_grid = np.arange(16, 4097)
c_grid = np.array([c_k(int(k)) for k in k_grid])
amp = np.abs(c_grid)
sel = amp > 1e-14
slope_gen = np.polyfit(np.log(k_grid[sel]), np.log(amp[sel]), 1)[0]
print(f"     generic decay, log-log fit over k in [16, 4096]:  {slope_gen:.4f}  (envelope O(k^-1))")
# absolute convergence: does sum |c_k| converge?
print("     absolute convergence test  (sum |c_k| over k=1..N):")
for N in [100, 400, 1600, 6400]:
    s = sum(abs(c_k(int(k))) for k in range(1, N + 1))
    print(f"       N = {N:5d}:  sum |c_k| = {s:.6f}")
print("     (grows like C log N: the Fourier series is NOT absolutely convergent,")
print("      as expected for a function whose coefficients decay only as O(k^-1))")
# cross-check against a plain FFT (trapezoid rule, O(1/N) error from the endpoint jump)
Nf = 2 ** 15
t_f = np.arange(Nf) / Nf
psi_f = koch_param(t_f, levels=44)
Cf = np.fft.fft(psi_f) / Nf
print("     cross-check vs FFT (N = 2^15, expect O(1/N) relative error):")
for k in [1, 16, 64, 256, 1024, 4096]:
    print(f"       k={k:5d}:  exact |c_k| = {abs(c_k(k)):.6e}"
          f"   FFT |c_k| = {abs(Cf[k]):.6e}   rel. diff = "
          f"{abs(abs(Cf[k]) - abs(c_k(k))) / abs(c_k(k)) * 100:5.2f}%")

# ----------------------------------------------------------------------
# 7.  Sup-norm convergence rate of the canonical iteration
# ----------------------------------------------------------------------
print()
print("=" * 74)
print("[7] Sup-norm error E_n = ||phi - P_n||_infty  (exact: E_n = 3^-n E_0)")
Nt = 2 ** 18
tt = np.arange(Nt + 1) / Nt
phi_t = koch_param(tt, levels=46)
E0_num = np.max(np.abs(phi_t - np.real(phi_t)))   # chord [0,1] is real
print(f"     E_0 = max_t |Im phi(t)| = {E0_num:.6f}   (sqrt(3)/6 = {SQ3/6:.6f})")
print("      n     E_n          E_n * 3^n      (-> E_0)")
En_list = []
for n in range(1, 9):
    Pn = koch_points(n)
    t_grid_n = np.linspace(0.0, 1.0, 4 ** n + 1)
    Pn_interp = np.interp(tt, t_grid_n, Pn.real) + 1j * np.interp(tt, t_grid_n, Pn.imag)
    En = np.max(np.abs(phi_t - Pn_interp))
    En_list.append(En)
    print(f"      {n}     {En:.8f}   {En * 3.0 ** n:.6f}")

# ----------------------------------------------------------------------
# 8.  Bi-Holder estimate and the no-tangent theorem
# ----------------------------------------------------------------------
print()
print("=" * 74)
print("[8] Bi-Holder estimate and secant directions (no tangent)")
M = 700
tg = np.linspace(0.0, 1.0, M)
pg = koch_param(tg, levels=44)
i_idx, j_idx = np.meshgrid(np.arange(M), np.arange(M), indexing="ij")
sel = i_idx < j_idx
dt = (tg[j_idx] - tg[i_idx])[sel]
dp = np.abs(pg[j_idx] - pg[i_idx])[sel]
ratio = dp / np.power(dt, H_KOCH)
print(f"     ratio |phi(t)-phi(s)| / |t-s|^H on grid:  min = {ratio.min():.4f}, "
      f"max = {ratio.max():.4f}  (theory: c |t-s|^H <= |phi(t)-phi(s)| <= C |t-s|^H)")
# the exact worst pair (0, 4^-n): ratio = 1
print(f"     exact: |phi(4^-n) - phi(0)| = 3^-n = (4^-n)^H  ->  ratio = 1")

# non-collinearity of phi(s), phi(s+1/4), phi(s+1/2)  (tangent argument)
# (s restricted to [0, 1/2] so that s + 1/2 <= 1 and all three addresses are valid)
s_grid = np.linspace(0.0, 0.5, 400)
p_s = koch_param(s_grid, levels=44)
p_s1 = koch_param(s_grid + 0.25, levels=44)
p_s2 = koch_param(s_grid + 0.5, levels=44)
area = 0.5 * np.abs(np.imag((p_s1 - p_s) * np.conj(p_s2 - p_s)))
print(f"     min area of triangle (phi(s), phi(s+1/4), phi(s+1/2)) = {area.min():.3e}"
      f"  (> 0: the two secant limit directions differ at every point)")

# no tangent at 0: two secant subsequences with different limit directions.
# (i)  endpoints of the first level-n sub-arc:  phi(4^-n) = 3^-n  -> direction 0
# (ii) apex of the first level-n sub-arc (t = 2*4^{-(n+1)}):
#      phi(2*4^{-(n+1)}) = 3^-n * phi(1/2) = 3^-n (1/2 + i sqrt(3)/6) -> direction pi/6
print("     secants at the endpoint 0 (both are secants to points of the curve):")
for n in [6, 12, 18]:
    junc = koch_param(4.0 ** (-n), levels=60)
    apex = koch_param(2.0 * 4.0 ** (-(n + 1)), levels=60)
    print(f"       n={n:2d}:  |phi(4^-n)| = {abs(junc):.6e}  arg = {np.degrees(np.angle(junc)):7.3f} deg,   "
          f"|apex_n| = {abs(apex):.6e}  arg = {np.degrees(np.angle(apex)):7.3f} deg")
print("     (two secant subsequences, directions 0 and pi/6, both with |point| -> 0"
      "  => no tangent at 0)")

# no tangent at an interior point: the secant directions arg(phi(s+4^-n)-phi(s))
# oscillate between two distinct values as n -> inf, so they have no limit and
# the tangent (derivative) does not exist at s.
s0 = 0.3
p0 = koch_param(s0, levels=60)
print(f"     interior point s = {s0}: secant directions arg(phi(s+4^-n) - phi(s)) (deg):")
for n in range(2, 10):
    v = koch_param(s0 + 4.0 ** (-n), levels=60) - p0
    print(f"       n={n}:  {np.degrees(np.angle(v)):8.3f}")

# ----------------------------------------------------------------------
# Figures
# ----------------------------------------------------------------------
fig, axes = plt.subplots(2, 3, figsize=(12, 7.5))
for ax, n in zip(axes.ravel(), [0, 1, 2, 3, 5, 8]):
    pts = koch_points(n)
    ax.plot(pts.real, pts.imag, lw=1.0)
    ax.set_title(f"level n = {n}   ($4^{{{n}}}$ segments)")
    ax.set_aspect("equal")
    ax.axis("off")
fig.suptitle("The Koch curve: polygonal approximations $C_n$", y=0.99)
fig.tight_layout()
fig.savefig(os.path.join(OUT, "fig1_iterations.png"), dpi=150)
plt.close(fig)

fig, axes = plt.subplots(2, 3, figsize=(12, 7.5))
for ax, n in zip(axes.ravel(), [0, 1, 2, 3, 5, 8]):
    pts = koch_snowflake(n)
    closed = np.concatenate([pts, [pts[0]]])
    ax.plot(closed.real, closed.imag, lw=1.0)
    ax.set_title(f"level n = {n}")
    ax.set_aspect("equal")
    ax.axis("off")
fig.suptitle("The Koch snowflake: perimeter $\\to\\infty$, area $\\to 8A_0/5$", y=0.99)
fig.tight_layout()
fig.savefig(os.path.join(OUT, "fig2_snowflake.png"), dpi=150)
plt.close(fig)

fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
ax = axes[0]
ax.loglog(1.0 / eps_b, N_b, "o-", label="box counts $N(3^{-n})$")
xx = np.logspace(np.log10(1 / eps_b[-1]), np.log10(1 / eps_b[0]), 50)
ax.loglog(xx, (N_b[0] * eps_b[0] ** (-D_KOCH)) * xx ** (-D_KOCH), "--",
          label=f"fit, slope $\\log 4/\\log 3 = {D_KOCH:.4f}$")
ax.set_xlabel("$1/\\varepsilon$")
ax.set_ylabel("$N(\\varepsilon)$")
ax.set_title("Box-counting dimension")
ax.legend()
ax = axes[1]
ax.loglog(pairs[:, 0], pairs[:, 1], ".", ms=3, alpha=0.5)
hh = np.array([pairs[:, 0].max(), pairs[:, 0].min()])
ax.loglog(hh, hh ** H_KOCH, "-",
          label=f"$|\\Delta\\varphi| = |h|^{{\\log 3/\\log 4}}$")
ax.set_xlabel("$|h|$")
ax.set_ylabel("$|\\varphi(t_0+h)-\\varphi(t_0)|$")
ax.set_title("Exact Holder exponent $\\log 3/\\log 4$")
ax.legend()
fig.tight_layout()
fig.savefig(os.path.join(OUT, "fig3_dimension_holder.png"), dpi=150)
plt.close(fig)

fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
ax = axes[0]
ax.loglog(k_grid, amp, ".", ms=2, alpha=0.6, label="$|c_k|$ (exact series)")
kk = np.logspace(np.log10(16), np.log10(4096), 60)
ax.loglog(kk, (amp[0] * 16) * (kk / 16) ** -1, "--",
          label=f"envelope $O(k^{{-1}})$ (fit {slope_gen:.2f})")
ax.set_xlabel("$k$")
ax.set_ylabel("$|c_k|$")
ax.set_title("Fourier decay of the Koch parametrization")
ax.legend()
ax = axes[1]
nn = np.arange(1, 13)
for p, ls in [(1.0, "-"), (1.2, "--"), (D_KOCH, "-."), (1.3, ":")]:
    r = 4.0 * 3.0 ** (-p)
    ax.semilogy(nn, r ** nn, ls, label=f"$p = {p:.3f}$")
ax.axhline(1.0, color="gray", lw=0.5)
ax.set_xlabel("level $n$")
ax.set_ylabel("$V_p^{(n)} = (4\\cdot 3^{-p})^n$")
ax.set_title("$p$-variation: threshold $p^* = \\log 4/\\log 3$")
ax.legend()
fig.tight_layout()
fig.savefig(os.path.join(OUT, "fig4_fourier_pvar.png"), dpi=150)
plt.close(fig)

fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
ax = axes[0]
ax.semilogy(np.arange(1, 9), En_list, "o-", label="$E_n$")
nn = np.arange(1, 9)
ax.semilogy(nn, E0_num * 3.0 ** (-nn), "--", label="$3^{-n} E_0$ (exact rate)")
ax.set_xlabel("level $n$")
ax.set_ylabel("$E_n = \\|\\varphi - P_n\\|_\\infty$")
ax.set_title("Convergence of the canonical iteration")
ax.legend()
ax = axes[1]
eps_grid = np.logspace(-6, 0, 200)
ax.loglog(eps_grid, eps_grid ** (-D_KOCH), "-",
          label=f"Koch: cost $\\sim \\varepsilon^{{-d}}$, $d = {D_KOCH:.4f}$")
ax.loglog(eps_grid, eps_grid ** (-1.0), "--", label="smooth curve: cost $\\sim \\varepsilon^{-1}$")
ax.set_xlabel("target precision $\\varepsilon$")
ax.set_ylabel("number of segments")
ax.set_title("Fractal approximation complexity")
ax.legend()
fig.tight_layout()
fig.savefig(os.path.join(OUT, "fig5_convergence_complexity.png"), dpi=150)
plt.close(fig)

print()
print("figures written to", OUT)
