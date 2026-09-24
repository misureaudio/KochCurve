import numpy as np
import pywt

def global_holders(f, wavelet='db4', l_lo=2, l_hi=None, neigh=1, offset=0.5, mode='periodic'):
    """Global Hölder exponent via wavelet leaders (global max leader per level).

    alpha_min = -slope(log2(global_max_leader) vs level) - offset,
    offset = 1/2 (L2 normalization of detail coefficients)."""
    coeffs = pywt.wavedec(f, wavelet, mode=mode)
    nlevels = len(coeffs) - 1
    if l_hi is None:
        l_hi = nlevels - 3          # drop the 3 finest levels (edge effects)
    gm = []
    for l in range(1, nlevels + 1):
        c = coeffs[l]; n = len(c)
        Ld = np.zeros(n)
        for k in range(n):
            lo = max(0, k - neigh); hi = min(n, k + neigh + 1)
            Ld[k] = np.max(np.abs(c[lo:hi]))
        gm.append((l, np.log2(Ld.max())))
    sel = [(l, lg) for (l, lg) in gm if l_lo <= l <= l_hi]
    lv = np.array([p[0] for p in sel], float)
    lg = np.array([p[1] for p in sel], float)
    slope = np.polyfit(lv, lg, 1)[0]
    alpha = -slope - offset
    return alpha, gm, (lv, lg)

L = 14
xg = np.arange(2**L)/2**L

print("=== calibration: interior |x-0.5|^alpha (expect alpha) ===")
for alpha in [0.35, 0.5, 0.7]:
    f = np.abs(xg - 0.5) ** alpha
    a, gm, _ = global_holders(f, 'db4', l_lo=2, l_hi=None, mode='periodic')
    print(f"  |x-.5|^{alpha}:  recovered alpha_min = {a:.4f}")

print()
print("=== fBm: global Hölder exponent (expect H) ===")
def fBm_grid(N, H, seed=0):
    rng = np.random.default_rng(seed)
    k = np.arange(N)
    r = 0.5*(np.abs(k+1)**(2*H) - 2.0*np.abs(k)**(2*H) + np.abs(k-1)**(2*H)) / N**(2*H)
    M = 1
    while M < 2*N: M *= 2
    c = np.zeros(M); c[0]=r[0]; c[1:N]=r[1:N]; c[M-N+1:M]=r[1:N][::-1]
    w = np.clip(np.real(np.fft.fft(c)), 0.0, None)
    z = rng.standard_normal(M) + 1j*rng.standard_normal(M)
    d = (np.fft.ifft(np.sqrt(w)*z)*np.sqrt(M)).real
    B = np.empty(N+1); B[0]=0.0; B[1:]=np.cumsum(d[:N])
    return B
for H in [0.2, 0.35, 0.5, 0.8]:
    B = fBm_grid(2**L, H, seed=7)[:2**L]
    a, gm, _ = global_holders(B, 'db4', l_lo=2, l_hi=None, mode='periodic')
    print(f"  H={H}:  recovered = {a:.4f}   (dim = 2-recovered = {2-a:.4f},  theory 2-H = {2-H:.4f})")

print()
print("=== Weierstrass W_{3,1/2}: global Hölder exponent ===")
b, lam = 3, 0.5
xw = np.arange(2**L)/2**L
W = np.zeros(2**L); ph = 2*np.pi*xw
for n in range(40):
    W += lam**n*np.cos(ph); ph = (b*ph) % (2*np.pi)
alpha = np.log(1/lam)/np.log(b)
a, gm, _ = global_holders(W, 'db4', l_lo=2, l_hi=None, mode='periodic')
print(f"  expected alpha = {alpha:.4f} -> dim = 2-alpha = {2-alpha:.4f}")
print(f"  recovered alpha = {a:.4f} -> dim = 2-alpha = {2-a:.4f}   (theory D = 1.3691)")
