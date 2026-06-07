# src/sph/density.py
import numpy as np
from .kernel import cubic_spline_W

def compute_density(x, m, h, neigh):
    N = x.shape[0]
    rho = np.zeros(N)

    for i in range(N):
        # self contribution
        rho[i] += m[i] * cubic_spline_W(0.0, h[i])

        for j in neigh[i]:
            rij = x[i] - x[j]
            r = np.linalg.norm(rij)
            h_ij = 0.5 * (h[i] + h[j])
            W = cubic_spline_W(r, h_ij)
            rho[i] += m[j] * W

    # avoid zero density
    rho = np.maximum(rho, 1e-6)
    return rho