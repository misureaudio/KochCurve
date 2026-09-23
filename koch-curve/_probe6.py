import numpy as np
import koch_simulation as K

pstar = np.log(4.0)/np.log(3.0)

def pvar(phpart, p):
    return np.sum(np.abs(np.diff(phpart))**p)

# Start from a fine vertex partition, then ADD non-vertex points and see if V grows.
def vertex_partition(n):
    return np.linspace(0.0, 1.0, 4**n + 1)

base_n = 6
rng = np.random.default_rng(1)
for n_extra in [0, 50, 200, 1000]:
    t = vertex_partition(base_n)
    if n_extra:
        # add points NOT on the vertex grid
        extra = rng.uniform(0.0, 1.0, n_extra)
        t = np.sort(np.concatenate([t, extra])); t[0], t[-1] = 0.0, 1.0
    phi = K.koch_param(t, levels=50)
    print(f"vertex(n={base_n}) + {n_extra:4d} random pts:  #pts={len(t):7d}  V_{{p*}} = {pvar(phi, pstar):.6f}")

# targeted: subdivide ONE level-base segment at a non-vertex location
print("\ntargeted: split one segment at fractions f (V should stay ~1 if critical is sharp):")
t0 = vertex_partition(5)
for f in [0.3, 0.5, 0.7, 0.25, 0.75]:
    # insert a point inside the first segment [0, 4^-5]
    seg = t0[1]  # = 4^-5
    t = np.insert(t0, 1, f*seg)
    phi = K.koch_param(t, levels=50)
    print(f"  split first segment at f={f:.2f}:  V_{{p*}} = {pvar(phi, pstar):.6f}")

# and subdivide at the vertex point (should be exactly the base)
print("\nrefine to next vertex level (expect exactly 1):")
for n in [5, 6, 7, 8]:
    t = vertex_partition(n)
    phi = K.koch_param(t, levels=50)
    print(f"  vertex n={n}:  V_{{p*}} = {pvar(phi, pstar):.10f}")
