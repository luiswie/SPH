# src/sph/neighbors.py
import numpy as np
from collections import defaultdict

def grid_neighbors(positions, h, domain=None):
    """
    Uniform grid neighbor search.
    positions: (N,2) array
    h: smoothing length
    domain: optional tuple (xmin,xmax,ymin,ymax)
    Returns: list of numpy arrays of neighbor indices for each particle (excluding self)
    """
    N = positions.shape[0]
    cell_size = 2.0 * h

    if domain is None:
        xmin = float(np.min(positions[:,0])) - 1e-12
        xmax = float(np.max(positions[:,0])) + 1e-12
        ymin = float(np.min(positions[:,1])) - 1e-12
        ymax = float(np.max(positions[:,1])) + 1e-12
    else:
        xmin, xmax, ymin, ymax = domain

    nx = max(1, int(np.ceil((xmax - xmin) / cell_size)))
    ny = max(1, int(np.ceil((ymax - ymin) / cell_size)))

    cell_dict = defaultdict(list)
    inv_cell = 1.0 / cell_size
    for i in range(N):
        xi, yi = positions[i]
        ix = int((xi - xmin) * inv_cell)
        iy = int((yi - ymin) * inv_cell)
        cell_dict[(ix, iy)].append(i)

    maxr2 = (2.0 * h) ** 2
    neighbors = [None] * N
    for i in range(N):
        xi, yi = positions[i]
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
                    rij2 = (xi - positions[j,0])**2 + (yi - positions[j,1])**2
                    if rij2 <= maxr2:
                        neigh_list.append(j)
        neighbors[i] = np.array(neigh_list, dtype=int)
    return neighbors

def brute_force_neighbors(positions, h):
    N = positions.shape[0]
    neigh = [None] * N
    maxr2 = (2.0*h)**2
    for i in range(N):
        idxs = []
        for j in range(N):
            if i == j: continue
            rij2 = np.sum((positions[i] - positions[j])**2)
            if rij2 <= maxr2:
                idxs.append(j)
        neigh[i] = np.array(idxs, dtype=int)
    return neigh
