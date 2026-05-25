# src/sph/density.py
import numpy as np
from .kernel import cubic_spline_W

def compute_density(positions, masses, h):
    """Density with a given smoothing length h (per particle)."""
    N = len(positions)
    rho = np.zeros(N)
    for i in range(N):
        for j in range(N):
            rij = np.linalg.norm(positions[i] - positions[j])
            rho[i] += masses[j] * cubic_spline_W(rij, h[i])
    return rho

def tait_pressure(rho, rho0, c0, gamma=7.0):
    B = rho0 * c0**2 / gamma
    return B * ((rho / rho0)**gamma - 1.0)
