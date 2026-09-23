import numpy as np
import koch_simulation as K

Nt = 2**20
tt = np.arange(Nt+1)/Nt
phi = K.koch_param(tt, levels=46)

# Correct base error: ||phi - P_0||_inf = max |phi(t) - t|  (P_0(t)=t, the chord)
E0_chord = np.max(np.abs(phi - tt))
E0_imag  = np.max(np.abs(phi - np.real(phi)))   # max |Im phi(t)|
print(f"max |phi(t) - t|  (deviation from chord) = {E0_chord:.8f}")
print(f"max |Im phi(t)|                        = {E0_imag:.8f}")
print(f"sqrt(3)/6 = {np.sqrt(3)/6:.8f}")
imax = np.argmax(np.abs(phi - tt))
print(f"argmax at t = {tt[imax]:.6f},  phi - t = {phi[imax]-tt[imax]:.6f}")
print(f"apex t=1/2: phi(1/2)-1/2 = {K.koch_param(0.5,46)-0.5:.6f}")

# confirm the recursion ||e_n||_inf = 3^-n * E0_chord
print("\nrecursion check  E_n * 3^n  (should -> E0_chord):")
for n in [1,2,3,4,6,8]:
    Pn = K.koch_points(n)
    tgn = np.linspace(0,1,4**n+1)
    Pni = np.interp(tt, tgn, Pn.real)+1j*np.interp(tt, tgn, Pn.imag)
    En = np.max(np.abs(phi - Pni))
    print(f"  n={n}:  E_n = {En:.8f}   E_n*3^n = {En*3**n:.8f}")
