import numpy as np
import koch_simulation as K   # reuses the (now correct) functions; prints are harmless

pstar = np.log(4.0)/np.log(3.0)
print(f"p* = {pstar:.6f}")

# Build a multi-scale partition: all level-n vertices for n = 1..Nmax, sorted.
def multi_scale_partition(Nmax, levels=40):
    tset = {0.0, 1.0}
    for n in range(1, Nmax+1):
        for m in range(4**n):
            tset.add(m / 4.0**n)
    return np.array(sorted(tset))

def pvar(phpart, p):
    d = np.abs(np.diff(phpart))
    return np.sum(d**p)

for Nmax in [3, 4, 5, 6, 7, 8]:
    t = multi_scale_partition(Nmax)
    phi = K.koch_param(t, levels=44)
    V = pvar(phi, pstar)
    print(f"multi-scale Nmax={Nmax}:  #pts={len(t):6d}  V_{{p*}} = {V:.6f}")

# compare: single finest level
for n in [4, 5, 6, 7, 8]:
    t = np.linspace(0.0, 1.0, 4**n + 1)
    phi = K.koch_param(t, levels=44)
    print(f"single level n={n}:  V_{{p*}} = {pvar(phi, pstar):.6f}")
