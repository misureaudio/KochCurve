import numpy as np

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

def chaos_game(Ms, ts, n_iter=2**20, seed=0):
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

def slopes(Ms, ts, name, s_star, k_lo=4, k_hi=17, seed=1):
    pts = chaos_game(Ms, ts, seed=seed)
    hs = 2.0**(-np.arange(k_lo,k_hi+1))
    counts = np.array([box_count_pts(pts,h) for h in hs])
    print(f"{name}:  s*={s_star:.4f}   (local slopes, h=2^-k)")
    for i in range(1,len(counts)):
        s = np.log(counts[i]/counts[i-1])/np.log(hs[i-1]/hs[i])
        mark = "  <-- plateau" if s < 1.0 else ""
        print(f"   2^-{k_lo+i:2d}: N={counts[i]:>12d}  slope={s:.3f}{mark}")
    return hs, counts

# gasket
Ms = [0.5*np.eye(2)]*3
ts = [(0.0,0.0),(0.5,0.0),(0.25,np.sqrt(3)/4)]
s_star = affinity_dim2(Ms)
hs, counts = slopes(Ms, ts, "GASKET", s_star)
# fit over stable window 2^-5..2^-9 (hs[1:6])
for (a,b) in [(1,6),(2,7),(1,5)]:
    D = -np.polyfit(np.log(hs[a:b]), np.log(counts[a:b]),1)[0]
    print(f"   gasket fit hs[{a}:{b}] (2^-{4+a}..2^-{4+b-1}): D={D:.4f}")

# carpet
S = [(0,0),(1,0),(2,0),(0,1)]
Ms = [np.diag([1/3,1/2]) for _ in S]
ts = [(i/3,j/2) for (i,j) in S]
s_star = affinity_dim2(Ms)
hs, counts = slopes(Ms, ts, "CARPET", s_star, seed=2)
for (a,b) in [(2,8),(3,9),(1,8)]:
    D = -np.polyfit(np.log(hs[a:b]), np.log(counts[a:b]),1)[0]
    print(f"   carpet fit hs[{a}:{b}] (2^-{4+a}..2^-{4+b-1}): D={D:.4f}")
