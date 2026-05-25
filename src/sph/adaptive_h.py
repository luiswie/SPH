import numpy as np
from sph.kernel import cubic_spline_W


def update_smoothing_lengths(x, m, h_init, eta=1.2, dim=2, tol=1e-4, max_iter=20):
    """
    Solve the coupled system:
        rho_a = sum_b m_b W_ab(h_a)
        h_a = eta * (m_a / rho_a)^(1/d)
    using Newton iteration per particle.
    """
    N = len(x)
    h = h_init.copy()

    for a in range(N):
        for _ in range(max_iter):

            # compute density at current h[a]
            rho = 0.0
            for j in range(N):
                rij = np.linalg.norm(x[a] - x[j])
                rho += m[j] * cubic_spline_W(rij, h[a])

            # target relation
            h_target = eta * (m[a] / rho)**(1.0 / dim)
            F = h[a] - h_target

            if abs(F) < tol:
                break

            # finite-difference derivative d(rho)/dh
            eps = 0.1 * h[a]
            rho_plus = 0.0
            rho_minus = 0.0

            for j in range(N):
                rij = np.linalg.norm(x[a] - x[j])
                rho_plus  += m[j] * cubic_spline_W(rij, h[a] + eps)
                rho_minus += m[j] * cubic_spline_W(rij, h[a] - eps)

            drho_dh = (rho_plus - rho_minus) / (2 * eps)

            # derivative of F
            dFdh = 1.0 - eta * (m[a] / rho)**(1.0 / dim) * (
                -(1.0 / dim) * drho_dh / rho
            )

            # Newton update
            h[a] -= F / dFdh

    return h
