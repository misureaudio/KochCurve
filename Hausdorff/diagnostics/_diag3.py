import numpy as np

def fit_window(hs, counts, lo, hi):
    m = np.polyfit(np.log(hs[lo:hi]), np.log(counts[lo:hi]), 1)
    return -m[0]

def vbox_count(f, x, h):
    ncol = int(np.ceil(1.0/h))
    col = np.minimum((x/h).astype(np.int64), ncol-1)
    cmax = np.full(ncol, -np.inf); cmin = np.full(ncol, np.inf)
    np.maximum.at(cmax, col, f); np.minimum.at(cmin, col, f)
    return int(np.ceil((cmax-cmin)/h).sum() + ncol)

# ---- fBm via circulant embedding ----
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

N = 2**14
print("fBm graph, VERTICAL box count.  D_theory = 2-H")
for H in [0.2, 0.5, 0.8]:
    B = fBm_grid(N, H, seed=1000+int(1000*H))
    t = np.arange(N+1)/N
    hs = 2.0**(-np.arange(5, 15))
    counts = np.array([vbox_count(B, t, h) for h in hs])
    # fit over middle window (avoid coarse 2^-5 and finest where sampling breaks)
    D = fit_window(hs, counts, 3, -1)   # 2^-8 .. 2^-14
    D2 = fit_window(hs, counts, 2, -2)
    print(f"  H={H}:  2-H={2-H:.3f}   D_est={D:.4f}  (alt {D2:.4f})")
    # local slopes
    ls = [np.log(counts[i]/counts[i-1])/np.log(hs[i-1]/hs[i]) for i in range(1,len(counts))]
    print("     local slopes:", " ".join(f"{s:.2f}" for s in ls))
