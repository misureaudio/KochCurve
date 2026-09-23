import numpy as np
import koch_simulation as K

pstar = np.log(4.0)/np.log(3.0)
alpha = np.log(3.0)/np.log(4.0)
print(f"p* = {pstar:.6f}, alpha = {alpha:.6f}, p* * alpha = {pstar*alpha:.6f} (should be 1)")

# Upper Holder constant C: max over grid of |phi(t)-phi(s)| / |t-s|^alpha
M = 400
tg = np.linspace(0.0, 1.0, M)
pg = K.koch_param(tg, levels=50)
i, j = np.meshgrid(np.arange(M), np.arange(M), indexing="ij")
sel = i < j
dt = (tg[j]-tg[i])[sel]
dp = np.abs(pg[j]-pg[i])[sel]
C_upper = (dp / dt**alpha).max()
print(f"upper Holder constant C (grid) = {C_upper:.6f}")
print(f"C^(p*) = {C_upper**pstar:.6f}  (an upper bound on V_{{p*}})")

# Lower Holder constant c
c_lower = (dp / dt**alpha).min()
print(f"lower Holder constant c (grid) = {c_lower:.6f}")

# V_{p*} on vertex partitions = 1 exactly; confirm it is the sup via random partitions
def pvar(php, p):
    return float(np.sum(np.abs(np.diff(php))**p))
rng = np.random.default_rng(0)
vals = []
for trial in range(200):
    Mx = rng.integers(3, 200)
    t = np.sort(rng.uniform(0,1,Mx)); t[0], t[-1] = 0.0, 1.0
    vals.append(pvar(K.koch_param(t, 50), pstar))
print(f"max V_{{p*}} over 200 random partitions = {max(vals):.6f}  (<= 1?)")
print(f"vertex partition V_{{p*}} = {pvar(K.koch_param(np.linspace(0,1,4**8+1),50), pstar):.10f}")
