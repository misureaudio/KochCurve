import numpy as np
import pywt

def pointwise_holders(f, wavelet='db4', l_lo=None, l_hi=None, neigh=1, offset=0.5, mode='periodic'):
    coeffs = pywt.wavedec(f, wavelet, mode=mode)
    nlevels = len(coeffs) - 1
    if l_hi is None: l_hi = nlevels - 3
    if l_lo is None: l_lo = max(2, nlevels - 9)   # central range
    l_hi = min(l_hi, nlevels)
    nfin = len(coeffs[l_hi])
    alpha_k = np.full(nfin, np.nan)
    for k in range(nfin):
        pts = []
        for l in range(l_hi, l_lo - 1, -1):
            idx = k // (2 ** (l_hi - l))
            c = coeffs[l]; n = len(c)
            lo = max(0, idx - neigh); hi = min(n, idx + neigh + 1)
            Ld = np.max(np.abs(c[lo:hi]))
            if Ld > 1e-12:
                pts.append((l, np.log2(Ld)))
        if len(pts) >= 3:
            lv = np.array([p[0] for p in pts], float)
            lg = np.array([p[1] for p in pts], float)
            slope = np.polyfit(lv, lg, 1)[0]
            alpha_k[k] = -slope - offset
    pos = np.arange(nfin) / nfin
    return alpha_k, pos

def trimmed_stats(ak, frac=0.1):
    n = len(ak)
    lo = int(frac * n); hi = n - lo
    a = ak[lo:hi][~np.isnan(ak[lo:hi])]
    return a.min(), np.median(a), a.max()

L = 14
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

print("=== fBm: pointwise Hölder (central levels, boundary-trimmed) ===")
print("  H      min   median   max    |  2-min (dim)   2-median (dim)")
for H in [0.2, 0.35, 0.5, 0.8]:
    B = fBm_grid(2**L, H, seed=7)[:2**L]
    ak, pos = pointwise_holders(B, 'db4')
    a_min, a_med, a_max = trimmed_stats(ak, 0.1)
    print(f"  {H:.2f}   {a_min:+.3f}  {a_med:+.3f}  {a_max:+.3f}   |   {2-a_min:.3f}        {2-a_med:.3f}")

print()
print("=== Weierstrass W_{3,1/2} ===")
b, lam = 3, 0.5
xw = np.arange(2**L)/2**L
W = np.zeros(2**L); ph = 2*np.pi*xw
for n in range(40):
    W += lam**n*np.cos(ph); ph = (b*ph) % (2*np.pi)
alpha = np.log(1/lam)/np.log(b)
ak, pos = pointwise_holders(W, 'db4')
a_min, a_med, a_max = trimmed_stats(ak, 0.1)
print(f"  expected alpha = {alpha:.4f}  (dim = 2-alpha = {2-alpha:.4f})")
print(f"  min={a_min:.4f}  median={a_med:.4f}  max={a_max:.4f}")
print(f"  2-min = {2-a_min:.4f}   2-median = {2-a_med:.4f}   (theory D = 1.3691)")
