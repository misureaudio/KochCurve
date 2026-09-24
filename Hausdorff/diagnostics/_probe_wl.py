import numpy as np
import pywt

def leaders_per_level(f, wavelet='db4', neigh=1):
    coeffs = pywt.wavedec(f, wavelet)
    nlevels = len(coeffs) - 1
    out = {}
    for l in range(1, nlevels + 1):
        c = coeffs[l]; n = len(c)
        Ld = np.zeros(n)
        for k in range(n):
            lo = max(0, k - neigh); hi = min(n, k + neigh + 1)
            Ld[k] = np.max(np.abs(c[lo:hi]))
        out[l] = Ld
    return out, nlevels

def pointwise_holders(f, wavelet='db4', j_min=2, j_max=None, neigh=1, offset=0.5):
    """alpha_k = -slope(log2 leader vs level) - offset.  offset=0.5 is the L2
    normalization of the detail coefficients (d_j ~ 2^{-j(alpha+1/2)})."""
    leaders, nlevels = leaders_per_level(f, wavelet, neigh)
    if j_max is None:
        j_max = nlevels - 2      # drop the 2 finest levels (edge effects)
    j_min = max(1, j_min)
    lmax = j_max
    nfin = len(leaders[lmax])
    a_k = np.full(nfin, np.nan)
    for k in range(nfin):
        pts = []
        for l in range(lmax, j_min - 1, -1):
            idx = k // (2 ** (lmax - l))
            if idx < len(leaders[l]):
                Ld = leaders[l][idx]
                if Ld > 1e-12:
                    pts.append((l, np.log2(Ld)))
        if len(pts) >= 3:
            lv = np.array([p[0] for p in pts], float)
            lg = np.array([p[1] for p in pts], float)
            slope = np.polyfit(lv, lg, 1)[0]
            a_k[k] = -slope - offset
    a_valid = a_k[~np.isnan(a_k)]
    return a_k, (a_valid.min() if len(a_valid) else np.nan), a_valid

L = 14
xg = np.arange(2**L)/2**L
print("=== calibration: interior singularity f(x)=|x-0.5|^alpha (expect alpha) ===")
for alpha in [0.35, 0.5, 0.7]:
    fsing = np.abs(xg - 0.5) ** alpha
    a_k, amin, av = pointwise_holders(fsing, 'db4', j_min=2, j_max=L-4, neigh=1)
    kstar = int(round(0.5 * len(a_k)))
    print(f"  |x-.5|^{alpha}:  min a = {amin:.4f},  a@x=0.5 = {a_k[kstar]:.4f}")

print()
print("=== fBm: pointwise Hölder exponent (expect H, min=H) ===")
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
    a_k, amin, av = pointwise_holders(B, 'db4', j_min=2, j_max=L-4, neigh=1)
    print(f"  H={H}:  min a = {amin:.4f}  -> dim = 2-min a = {2-amin:.4f}   (theory 2-H = {2-H:.4f})")

print()
print("=== Weierstrass W_{3,1/2}: pointwise Hölder exponent ===")
b, lam = 3, 0.5
xw = np.arange(2**L)/2**L
W = np.zeros(2**L); ph = 2*np.pi*xw
for n in range(40):
    W += lam**n*np.cos(ph); ph = (b*ph) % (2*np.pi)
alpha = np.log(1/lam)/np.log(b)
a_k, amin, av = pointwise_holders(W, 'db4', j_min=2, j_max=L-4, neigh=1)
print(f"  expected exponent alpha = {alpha:.4f} -> dim = 2-alpha = {2-alpha:.4f}")
print(f"  measured: min a = {amin:.4f} -> dim = 2-min a = {2-amin:.4f}   (theory D = 1.3691)")
print(f"  median a = {np.median(av):.4f}, max a = {av.max():.4f}")
