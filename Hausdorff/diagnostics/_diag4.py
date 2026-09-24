import numpy as np

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
    col = np.minimum((x/h).astype(np.int64), ncol-1)
    cmax = np.full(ncol, -np.inf); cmin = np.full(ncol, np.inf)
    np.maximum.at(cmax, col, f); np.minimum.at(cmin, col, f)
    return int(np.ceil((cmax-cmin)/h).sum() + ncol)

def grid_count(x, y, h, y0, y1):
    nx = int(np.ceil(1.0/h)); ny = int(np.ceil((y1-y0)/h))
    ix = np.minimum((x/h).astype(np.int64), nx-1)
    iy = np.minimum(((y-y0)/h).astype(np.int64), ny-1)
    return int(np.unique(ix*(ny+1)+iy).size)

for N in [2**14, 2**16]:
    print(f"=== N={N} (finest resolvable h ~ 2^-{int(np.log2(N))}) ===")
    for H in [0.2, 0.5, 0.8]:
        B = fBm_grid(N, H, seed=1000+int(1000*H))
        t = np.arange(N+1)/N
        hs = 2.0**(-np.arange(5, 15))
        vc = np.array([vbox_count(B, t, h) for h in hs])
        # stable window: local slopes not yet decaying. Use 2^-8..2^-11
        Dv = -np.polyfit(np.log(hs[3:8]), np.log(vc[3:8]),1)[0]
        print(f"  H={H}: 2-H={2-H:.3f}  vbox D(2^-8..2^-11)={Dv:.3f}")
