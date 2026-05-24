# src/sph/boundaries.py
import numpy as np

def apply_reflective_walls(positions, velocities, xmin, xmax, ymin, ymax, damping= -0.5):
    # reflect and damp velocity component when hitting walls
    # positions and velocities modified in place
    for i in range(positions.shape[0]):
        x, y = positions[i]
        vx, vy = velocities[i]
        if x < xmin:
            positions[i,0] = xmin
            velocities[i,0] = vx * damping
        if x > xmax:
            positions[i,0] = xmax
            velocities[i,0] = vx * damping
        if y < ymin:
            positions[i,1] = ymin
            velocities[i,1] = vy * damping
        if y > ymax:
            positions[i,1] = ymax
            velocities[i,1] = vy * damping
