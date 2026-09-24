import numpy as np
import pywt

# ---------------- multi-sample fBm ----------------
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

def vbox_count(f, x, h):
    ncol = int(np.ceil(1.0/h))
    col = np.floor(x/h).astype(np.int64) % ncol
    cmax = np.full(ncol, -np.inf); cmin = np.full(ncol, np.inf)
    np.maximum.at(cmax, col, f); np.minimum.at(cmin, col, f)
    return int(np.ceil((cmax-cmin)/h).sum() + ncol)

print("=== multi-sample fBm (K=50, average the counts, stable window h=2^-8..2^-13) ===")
N = 2**16
K = 50
t = np.arange(N+1)/N
hs = 2.0**(-np.arange(5,17))
for H in [0.2, 0.5, 0.8]:
    acc = np.zeros(len(hs))
    for k in range(K):
        B = fBm_grid(N, H, seed=1000+int(1000*H)+100*k)
        for i,h in enumerate(hs):
            acc[i] += vbox_count(B, t, h)
    acc /= K
    D = -np.polyfit(np.log(hs[3:9]), np.log(acc[3:9]), 1)[0]   # 2^-8..2^-13
    # also single-scale local slope at the middle of the window for reference
    print(f"  H={H}: 2-H={2-H:.3f}   multi-sample D_hat={D:.4f}")

print()
print("=== single vs multi (H=0.2, K=1 vs K=50), same window ===")
for K in [1, 50]:
    acc = np.zeros(len(hs))
    for k in range(K):
        B = fBm_grid(N, 0.2, seed=1000+100*k)
        for i,h in enumerate(hs):
            acc[i] += vbox_count(B, t, h)
    acc /= K
    D = -np.polyfit(np.log(hs[3:9]), np.log(acc[3:9]), 1)[0]
    print(f"  K={K:3d}: D_hat={D:.4f}")
