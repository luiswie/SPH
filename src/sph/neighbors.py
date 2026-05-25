# src/sph/neighbors.py
import numpy as np
from collections import defaultdict

def grid_neighbors(positions, h, domain=None):
    """
    Uniform grid neighbor search supporting per-particle smoothing lengths.
    positions: (N,2)
    h: array of smoothing lengths (N,)
    domain: (xmin, xmax, ymin, ymax)
    Returns: list of arrays of neighbor indices
    """
    N = positions.shape[0]

    # Use global max(h) for grid cell size
    h_max = float(np.max(h))
    cell_size = 2.0 * h_max

    # Domain
    if domain is None:
        xmin = float(np.min(positions[:,0])) - 1e-12
        xmax = float(np.max(positions[:,0])) + 1e-12
        ymin = float(np.min(positions[:,1])) - 1e-12
        ymax = float(np.max(positions[:,1])) + 1e-12
    else:
        xmin, xmax, ymin, ymax = domain

    nx = max(1, int(np.ceil((xmax - xmin) / cell_size)))
    ny = max(1, int(np.ceil((ymax - ymin) / cell_size)))

    # Build grid
    cell_dict = defaultdict(list)
    inv_cell = 1.0 / cell_size

    for i in range(N):
        xi, yi = positions[i]
        ix = int((xi - xmin) * inv_cell)
        iy = int((yi - ymin) * inv_cell)
        cell_dict[(ix, iy)].append(i)

    # Neighbor search
    neighbors = [None] * N

    for i in range(N):
        xi, yi = positions[i]
        hi = h[i]
        ix = int((xi - xmin) * inv_cell)
        iy = int((yi - ymin) * inv_cell)

        neigh_list = []

        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                cell = (ix + dx, iy + dy)
                if cell not in cell_dict:
                    continue

                for j in cell_dict[cell]:
                    if j == i:
                        continue

                    hj = h[j]
                    rij2 = (xi - positions[j,0])**2 + (yi - positions[j,1])**2
                    maxr2 = (2.0 * max(hi, hj))**2

                    if rij2 <= maxr2:
                        neigh_list.append(j)

        neighbors[i] = np.array(neigh_list, dtype=int)

    return neighbors
