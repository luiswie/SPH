import numpy as np
from .kernel import cubic_spline_gradW

def compute_accelerations(positions, velocities, masses, rho, pressure, h,
                          alpha=0.1, c0=10.0, g=np.array([0.0, -9.81])):
    N = positions.shape[0]
    a = np.zeros_like(positions)

    for i in range(N):
        ai = np.zeros(2)
        hi = h[i]

        for j in range(N):
            if i == j:
                continue

            rij = positions[i] - positions[j]
            r = np.linalg.norm(rij)
            hj = h[j]

            # correct neighbor radius for adaptive h
            h_ij = max(hi, hj)
            if r == 0.0 or r > 2.0 * h_ij:
                continue

            # symmetric pressure term
            pij = (pressure[i] / (rho[i]**2) +
                   pressure[j] / (rho[j]**2))

            grad = cubic_spline_gradW(rij, hi)

            ai -= masses[j] * pij * grad

            # Monaghan artificial viscosity
            vij = velocities[i] - velocities[j]
            vr = np.dot(vij, rij)

            if vr < 0:
                mu = (-alpha * c0 * vr) / (r*r + 0.01 * hi * hi)
                rho_ij = 0.5 * (rho[i] + rho[j])
                ai -= masses[j] * (mu / rho_ij) * grad

        ai += g
        a[i] = ai

    return a
