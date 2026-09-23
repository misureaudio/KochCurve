import numpy as np
import koch_simulation as K

pstar = np.log(4.0)/np.log(3.0)
rng = np.random.default_rng(0)

def pvar(phpart, p):
    d = np.abs(np.diff(phpart))
    return np.sum(d**p)

# random partitions of increasing size
for M in [10, 50, 200, 1000, 5000]:
    t = np.sort(rng.uniform(0.0, 1.0, M)); t[0], t[-1] = 0.0, 1.0
    phi = K.koch_param(t, levels=50)
    print(f"random partition M={M:5d}:  V_{{p*}} = {pvar(phi, pstar):.6f}")

# also: a few hand-picked "adversarial" partitions
for label, t in [
    ("{0, .3, 1}", np.array([0.0, 0.3, 1.0])),
    ("{0, .1, .2, .5, .8, .9, 1}", np.array([0.0, 0.1, 0.2, 0.5, 0.8, 0.9, 1.0])),
    ("{0, 1/4, 3/4, 1}", np.array([0.0, 0.25, 0.75, 1.0])),
    ("{0, .0625, .3125, .5625, .8125, 1}", np.array([0.0, .0625, .3125, .5625, .8125, 1.0])),
]:
    phi = K.koch_param(t, levels=50)
    print(f"{label:35s}: V_{{p*}} = {pvar(phi, pstar):.6f}")

# p just above and below p*
for p in [pstar - 0.001, pstar, pstar + 0.001]:
    t = np.sort(rng.uniform(0.0, 1.0, 2000)); t[0], t[-1] = 0.0, 1.0
    phi = K.koch_param(t, levels=50)
    print(f"p = {p:.4f}: V = {pvar(phi, p):.6f}")
