# src/sph/density.py
import numpy as np
from .kernel import cubic_spline_W

def compute_density(positions, masses, h):
    N = positions.shape[0]
    rho = np.zeros(N)
    for i in range(N):
        s = 0.0
        for j in range(N):
            r = np.linalg.norm(positions[i] - positions[j])
            s += masses[j] * cubic_spline_W(r, h)
        rho[i] = s
    return rho

def tait_pressure(rho, rho0, c0, gamma=7.0):
    B = rho0 * c0**2 / gamma
    return B * ((rho / rho0)**gamma - 1.0)
