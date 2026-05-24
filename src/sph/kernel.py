# src/sph/kernel.py
import numpy as np

def cubic_spline_W(r, h):
    """Scalar kernel value for distance r and smoothing length h."""
    q = r / h
    sigma = 10.0 / (7.0 * np.pi * h * h)
    if q <= 1.0:
        return sigma * (1.0 - 1.5*q*q + 0.75*q*q*q)
    elif q <= 2.0:
        return sigma * 0.25 * (2.0 - q)**3
    return 0.0

def cubic_spline_gradW(rvec, h):
    """Gradient of cubic spline kernel: returns 2D vector."""
    r = np.linalg.norm(rvec)
    if r == 0.0:
        return np.array([0.0, 0.0])
    q = r / h
    sigma = 10.0 / (7.0 * np.pi * h * h)
    if q <= 1.0:
        val = sigma * (-3.0*q + 2.25*q*q) / h
    elif q <= 2.0:
        val = -sigma * 0.75 * (2.0 - q)**2 / h
    else:
        val = 0.0
    return val * (rvec / r)
