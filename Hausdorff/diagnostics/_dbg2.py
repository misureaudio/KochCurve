import numpy as np
import pywt

L = 14
xg = np.arange(2**L)/2**L
f = np.abs(xg - 0.5) ** 0.5        # singularity of order 0.5 at x=0.5

coeffs = pywt.wavedec(f, 'db4')
nlevels = len(coeffs) - 1
print("nlevels =", nlevels)
# singularity at x=0.5. At level l, coefficient index for x=0.5 is ~ 0.5 * 2^l
print("level  n_coeffs   leader@singularity   log2")
for l in range(1, nlevels + 1):
    c = coeffs[l]; n = len(c)
    Ld = np.zeros(n)
    for k in range(n):
        lo = max(0, k-1); hi = min(n, k+2)
        Ld[k] = np.max(np.abs(c[lo:hi]))
    # coefficient index whose support contains x=0.5
    idx = int(round(0.5 * n))
    print(f"  {l:2d}   {n:6d}   {Ld[idx]:.4e}   {np.log2(max(Ld[idx],1e-300)):.4f}")

# also print the GLOBAL max leader per level (should track the singularity)
print("\nglobal max leader per level:")
for l in range(1, nlevels + 1):
    c = coeffs[l]; n = len(c)
    Ld = np.zeros(n)
    for k in range(n):
        lo = max(0, k-1); hi = min(n, k+2)
        Ld[k] = np.max(np.abs(c[lo:hi]))
    print(f"  level {l:2d}: global max = {Ld.max():.4e}  log2 = {np.log2(Ld.max()):.4f}")
