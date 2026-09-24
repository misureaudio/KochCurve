import numpy as np

def fit_box_dim(hs, counts, min_count=200):
    hs = np.asarray(hs); counts = np.asarray(counts, float)
    cmax = counts.max()
    ok = (counts >= min_count) & (counts < 0.92 * cmax)
    if ok.sum() < 3:
        ok = counts >= min_count
    m = np.polyfit(np.log(hs[ok]), np.log(counts[ok]), 1)
    return -m[0], int(ok.sum())

b, lam = 3, 0.5
M = 2**20
x = np.arange(M) / M
W = np.zeros(M)
ph = 2*np.pi*x
for n in range(22):
    W += lam**n * np.cos(ph); ph = (b*ph) % (2*np.pi)

def vbox_count(f, x, h):
    ncol = int(np.ceil(1.0/h))
    col = np.minimum((x/h).astype(np.int64), ncol-1)
    cmax = np.full(ncol, -np.inf); cmin = np.full(ncol, np.inf)
    np.maximum.at(cmax, col, f); np.minimum.at(cmin, col, f)
    return int(np.ceil((cmax-cmin)/h).sum() + ncol)

hs = 2.0**(-np.arange(5, 19))
counts = np.array([vbox_count(W, x, h) for h in hs])
D, npts = fit_box_dim(hs, counts)
print("Weierstrass W_{3,1/2}:  D_theory=1.3691  D_est=%.4f  (n scales in fit=%d)" % (D, npts))
print("  counts:", counts)
