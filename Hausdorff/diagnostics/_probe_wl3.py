import numpy as np
import pywt

def pointwise_holders(f, wavelet='db4', l_lo=2, l_hi=None, neigh=1, offset=0.5, mode='periodic'):
    """Pointwise Hölder exponents via wavelet leaders (periodic mode).

    For each position k at the finest level used (l_hi), the leader at level l
    is the max |c_{l, idx}| over a window of (2*neigh+1) coefficients around
    idx = k // 2^{l_hi - l}.  alpha_k = -slope(log2 leader vs l) - offset,
    offset = 1/2 (L2 normalization).  Returns (alpha_k, positions)."""
    coeffs = pywt.wavedec(f, wavelet, mode=mode)
    nlevels = len(coeffs) - 1
    if l_hi is None:
        l_hi = nlevels - 3          # drop the 3 finest levels (edge effects)
    l_hi = min(l_hi, nlevels)
    nfin = len(coeffs[l_hi])        # 2^{L-l_hi} with periodic mode
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

L = 14
xg = np.arange(2**L)/2**L

print("=== calibration: interior |x-0.5|^alpha (expect alpha at x=0.5) ===")
for alpha in [0.35, 0.5, 0.7]:
    f = np.abs(xg - 0.5) ** alpha
    ak, pos = pointwise_holders(f, 'db4', l_lo=2, l_hi=None, neigh=1)
    kstar = np.argmin(np.abs(pos - 0.5))
    print(f"  |x-.5|^{alpha}:  alpha@x=0.5 = {ak[kstar]:.4f}   min alpha = {np.nanmin(ak):.4f}")

print()
print("=== fBm: pointwise Hölder exponent (expect H at EVERY point, min=H) ===")
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
    ak, pos = pointwise_holders(B, 'db4', l_lo=2, l_hi=None, neigh=1)
    av = ak[~np.isnan(ak)]
    print(f"  H={H}:  min alpha = {av.min():.4f} -> dim = 2-min = {2-av.min():.4f}   "
          f"(theory 2-H = {2-H:.4f})   [median = {np.median(av):.4f}]")

print()
print("=== Weierstrass W_{3,1/2}: pointwise Hölder exponent ===")
b, lam = 3, 0.5
xw = np.arange(2**L)/2**L
W = np.zeros(2**L); ph = 2*np.pi*xw
for n in range(40):
    W += lam**n*np.cos(ph); ph = (b*ph) % (2*np.pi)
alpha = np.log(1/lam)/np.log(b)
ak, pos = pointwise_holders(W, 'db4', l_lo=2, l_hi=None, neigh=1)
av = ak[~np.isnan(ak)]
print(f"  expected alpha = {alpha:.4f} -> dim = 2-alpha = {2-alpha:.4f}")
print(f"  min alpha = {av.min():.4f} -> dim = 2-min = {2-av.min():.4f}   (theory D = 1.3691)")
print(f"  median alpha = {np.median(av):.4f}, max = {av.max():.4f}")
