# src/sph/eos.py
import numpy as np

def tait_pressure(rho, rho0, c0, gamma=7.0):
    return c0**2 * rho0 * ((rho / rho0)**gamma - 1.0)
