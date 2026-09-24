import numpy as np

# ---- Weierstrass: local slopes (vertical count) ----
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
print("Weierstrass local slopes (h=2^-k, slope over 1-octave):")
for i in range(1, len(counts)):
    s = np.log(counts[i]/counts[i-1]) / np.log(hs[i-1]/hs[i])
    print(f"  h=2^-{5+i:2d}: N={counts[i]:>12d}  local slope={s:.4f}")
# fit over finest K
for K in [4,6,8,10,14]:
    m = np.polyfit(np.log(hs[-K:]), np.log(counts[-K:]),1)
    print(f"  fit finest {K:2d} scales: D={-m[0]:.4f}")

# ---- Gasket: 2D grid count, where is the plateau? ----
def chaos_game(Ms, ts, n_iter=2**20, seed=0):
    rng = np.random.default_rng(seed)
    M = np.array(Ms); T = np.array(ts)
    idx = rng.integers(0, len(Ms), size=n_iter)
    p = np.zeros(2); pts = np.empty((n_iter,2))
    for j in range(n_iter):
        i = idx[j]; p = M[i]@p + T[i]; pts[j]=p
    return pts[2000:]

def box_count_pts(pts, h):
    x0,y0 = pts.min(axis=0)
    nx = int(np.ceil((pts[:,0].max()-x0)/h)); ny = int(np.ceil((pts[:,1].max()-y0)/h))
    ix = np.minimum(((pts[:,0]-x0)/h).astype(np.int64), nx-1)
    iy = np.minimum(((pts[:,1]-y0)/h).astype(np.int64), ny-1)
    return int(np.unique(ix*(ny+1)+iy).size)

Ms = [0.5*np.eye(2)]*3
ts = [(0.0,0.0),(0.5,0.0),(0.25,np.sqrt(3)/4)]
pts = chaos_game(Ms, ts, n_iter=2**20, seed=1)
Npts = len(pts)
hs = 2.0**(-np.arange(4, 16))
counts = np.array([box_count_pts(pts, h) for h in hs])
print(f"\nGasket (Npts={Npts}):  D_theory=1.5850")
print("  h=2^-k   N(h)      N/Npts")
for i in range(len(counts)):
    print(f"  2^-{4+i:2d}  {counts[i]:>12d}  {counts[i]/Npts:.3f}")
# fit over fractal regime: counts in [1000, 0.9*Npts]
ok = (counts>=1000) & (counts <= 0.9*Npts)
m = np.polyfit(np.log(hs[ok]), np.log(counts[ok]),1)
print(f"  fit over fractal regime ({int(ok.sum())} scales): D={-m[0]:.4f}")
