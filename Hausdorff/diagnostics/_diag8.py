import numpy as np, time

def affinity_dim2(Ms):
    def phi(s, M):
        sv = np.linalg.svd(M, compute_uv=False)
        return sv[0] * sv[1] ** (s - 1.0)
    f = lambda s: sum(phi(s, M) for M in Ms) - 1.0
    if f(1.0) < 0: return 1.0
    if f(2.0) > 0: return 2.0
    lo, hi = 1.0, 2.0
    for _ in range(80):
        mid = 0.5*(lo+hi)
        if f(mid) > 0: lo = mid
        else: hi = mid
    return 0.5*(lo+hi)

def chaos_game(Ms, ts, n_iter, seed=0):
    rng = np.random.default_rng(seed)
    M = np.array(Ms); T = np.array(ts)
    idx = rng.integers(0, len(Ms), size=n_iter)
    p = np.zeros(2); pts = np.empty((n_iter,2))
    for j in range(n_iter):
        i = idx[j]; p = M[i]@p+T[i]; pts[j]=p
    return pts[2000:]

def box_count_pts(pts, h):
    x0,y0 = pts.min(axis=0)
    nx = int(np.ceil((pts[:,0].max()-x0)/h)); ny = int(np.ceil((pts[:,1].max()-y0)/h))
    ix = np.minimum(((pts[:,0]-x0)/h).astype(np.int64), nx-1)
    iy = np.minimum(((pts[:,1]-y0)/h).astype(np.int64), ny-1)
    return int(np.unique(ix*(ny+1)+iy).size)

def box_dim(pts, k_lo=4, k_hi=17):
    hs = 2.0**(-np.arange(k_lo,k_hi+1))
    counts = np.array([box_count_pts(pts,h) for h in hs])
    Npts = len(pts)
    ok = (counts>=1000)&(counts<=0.9*Npts)
    if ok.sum()<3: ok = counts>=1000
    return -np.polyfit(np.log(hs[ok]),np.log(counts[ok]),1)[0], ok.sum()

def random_affine_ifs(seed):
    rng = np.random.default_rng(seed)
    while True:
        a = rng.uniform(0.35, 0.55, size=3)
        b = a * rng.uniform(0.2, 0.45, size=3)
        if 1.2 < a.sum() < 1.7 and (a*b).sum() < 1.0:
            break
    xoff = np.array([0.0, 0.45, 0.9]); yoff = rng.uniform(0.0, 0.3, size=3)
    Ms = [np.diag([a[i], b[i]]) for i in range(3)]
    ts = [(xoff[i], yoff[i]) for i in range(3)]
    return Ms, ts

# timing + tracking with 2^21 points
t0 = time.time()
Ms, ts = random_affine_ifs(1)
pts = chaos_game(Ms, ts, n_iter=2**21, seed=1)
t1 = time.time()
s = affinity_dim2(Ms)
D, nsc = box_dim(pts)
t2 = time.time()
print(f"2^21 pts: chaos_game {t1-t0:.1f}s, box_dim {t2-t1:.1f}s, s*={s:.4f} dim_B={D:.4f} (n={nsc})")
for seed in range(1,8):
    Ms, ts = random_affine_ifs(seed)
    pts = chaos_game(Ms, ts, n_iter=2**21, seed=seed)
    s = affinity_dim2(Ms)
    D, nsc = box_dim(pts)
    print(f"  seed {seed}: s*={s:.4f} dim_B={D:.4f} (n={nsc})")
