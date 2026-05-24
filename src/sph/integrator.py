# src/sph/integrator.py
import numpy as np

def symplectic_euler(positions, velocities, acc, dt):
    velocities += dt * acc
    positions += dt * velocities
    return positions, velocities
