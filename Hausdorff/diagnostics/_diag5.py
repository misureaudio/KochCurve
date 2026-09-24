import numpy as np

def affinity_dim2(Ms):
    """CORRECT: s* in [1,2] s.t. sum_i sigma1(Mi)*sigma2(Mi)^(s-1) = 1, else 2."""
    def phi(s, M):
        sv = np.linalg.svd(M, compute_uv=False)
        return sv[0] * sv[1] ** (s - 1.0)
    f = lambda s: sum(phi(s, M) for M in Ms) - 1.0
    if f(2.0) > 0:
        return 2.0
    lo, hi = 1.0, 2.0
    for _ in range(80):
        mid = 0.5 * (lo + hi)
        if f(mid) > 0: lo = mid
        else: hi = mid
    return 0.5 * (lo + hi)

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

def box_dim(pts, k_lo=4, k_hi=16):
    hs = 2.0**(-np.arange(k_lo, k_hi+1))
    counts = np.array([box_count_pts(pts, h) for h in hs])
    Npts = len(pts)
    ok = (counts>=1000) & (counts<=0.9*Npts)
    if ok.sum()<3: ok = counts>=1000
    return -np.polyfit(np.log(hs[ok]), np.log(counts[ok]),1)[0], hs, counts, ok

# gasket
Ms = [0.5*np.eye(2)]*3
ts = [(0.0,0.0),(0.5,0.0),(0.25,np.sqrt(3)/4)]
print("gasket:  s*=%.4f  Moran=%.4f  measured=%.4f" % (affinity_dim2(Ms), np.log(3)/np.log(2), box_dim(chaos_game(Ms,ts,seed=1))[0]))

# carpet
S = [(0,0),(1,0),(2,0),(0,1)]
Ms = [np.diag([1/3,1/2]) for _ in S]
ts = [(i/3,j/2) for (i,j) in S]
D,hs,counts,ok = box_dim(chaos_game(Ms,ts,seed=2))
print("carpet:  s*=%.4f  log6/log3=%.4f  measured=%.4f  (n scales=%d)" % (affinity_dim2(Ms), np.log(6)/np.log(3), D, ok.sum()))

# random anisotropic
def random_affine_ifs(seed):
    rng = np.random.default_rng(seed)
    while True:
        a = rng.uniform(0.15,0.35,size=3)
        if a.sum()<1.0: break
    b = 0.35*a
    xoff = np.array([0.0,0.5,0.9]); yoff = rng.uniform(0.0,0.3,size=3)
    Ms=[np.diag([a[i],b[i]]) for i in range(3)]; ts=[(xoff[i],yoff[i]) for i in range(3)]
    return Ms,ts
print("\nrandom anisotropic IFS:  s* vs measured dim_B")
for seed in range(1,7):
    Ms,ts = random_affine_ifs(seed)
    s=affinity_dim2(Ms)
    D,hs,counts,ok = box_dim(chaos_game(Ms,ts,n_iter=2**19,seed=seed))
    print(f"  seed {seed}:  s*={s:.4f}  dim_B={D:.4f}  (n scales={ok.sum()})")
