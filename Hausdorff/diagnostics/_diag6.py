import numpy as np

def affinity_dim2(Ms):
    """s* in [1,2] s.t. sum_i sigma1*sigma2^(s-1) = 1, else 2 (if f(2)>0) or 1 (if f(1)<0)."""
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

def chaos_game(Ms, ts, n_iter=2**19, seed=0):
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

def box_dim(pts, k_lo=4, k_hi=16):
    hs = 2.0**(-np.arange(k_lo,k_hi+1))
    counts = np.array([box_count_pts(pts,h) for h in hs])
    Npts = len(pts)
    ok = (counts>=1000)&(counts<=0.9*Npts)
    if ok.sum()<3: ok = counts>=1000
    return -np.polyfit(np.log(hs[ok]),np.log(counts[ok]),1)[0], ok.sum()

# random anisotropic IFS: OVERLAPPING x (sum a_i > 1) so s* in (1,2)
def random_affine_ifs(seed):
    rng = np.random.default_rng(seed)
    while True:
        a = rng.uniform(0.35, 0.55, size=3)     # sum ~ 1.5 > 1  (overlap)
        b = a * rng.uniform(0.2, 0.45, size=3)  # b_i < a_i (anisotropic)
        if 1.2 < a.sum() < 1.7 and (a*b).sum() < 1.0:
            break
    xoff = np.array([0.0, 0.45, 0.9])
    yoff = rng.uniform(0.0, 0.3, size=3)
    Ms = [np.diag([a[i], b[i]]) for i in range(3)]
    ts = [(xoff[i], yoff[i]) for i in range(3)]
    return Ms, ts

print("random anisotropic IFS (overlap):  s* vs measured dim_B")
for seed in range(1, 8):
    Ms, ts = random_affine_ifs(seed)
    s = affinity_dim2(Ms)
    D, nsc = box_dim(chaos_game(Ms, ts, seed=seed))
    print(f"  seed {seed}:  a={np.round([m[0,0] for m in Ms],3)} b={np.round([m[1,1] for m in Ms],3)}  s*={s:.4f}  dim_B={D:.4f}  (n={nsc})")
