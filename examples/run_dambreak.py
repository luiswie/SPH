# src/sph/run_dambreak.py
import sys
import traceback
import numpy as np
import matplotlib.pyplot as plt

# Absolute imports (package must be importable; run with: python -m src.sph.run_dambreak)
from sph.kernel import cubic_spline_W, cubic_spline_gradW
from sph.neighbors import grid_neighbors
from sph.adaptive_h import update_smoothing_lengths
from sph.density import compute_density, tait_pressure
from sph.momentum import compute_accelerations
from sph.integrator import symplectic_euler
from sph.boundaries import apply_reflective_walls

def setup_column(dx=0.02, width=0.1, height=0.2, offset=(0.02,0.02)):
    nx = max(1, int(np.ceil(width / dx)))
    ny = max(1, int(np.ceil(height / dx)))
    pts = []
    for j in range(ny):
        for i in range(nx):
            pts.append([offset[0] + i*dx, offset[1] + j*dx])
    return np.array(pts, dtype=float)

def run(dx=0.02, steps=300, dt=0.001, show_plot=True):
    # domain
    Lx, Ly = 1.6, 0.6
    rho0 = 1000.0
    # choose c0 relative to expected velocities; keep low for speed but stable
    c0 = 10.0
    alpha = 0.1

    x = setup_column(dx=dx, width=0.1, height=0.2, offset=(0.02, 0.02)) # x, y positions of particles
    N = x.shape[0] # number of particles
    h = np.full(N, 0.1)   # initial guess for smoothing length (will be updated adaptively)
    print(f"Running with dx={dx:.4f}, N={N}, dt={dt:.5f}, steps={steps}")
    v = np.zeros_like(x) # initial velocities
    m = (dx*dx*rho0) * np.ones(N) # mass of each particle (assuming uniform density and spacing)

    if show_plot:
        plt.ion()
        fig, ax = plt.subplots(figsize=(8,3))

    try:
        for step in range(steps):

            # neighbor list
            neigh = grid_neighbors(x, h, domain=(0.0, Lx, 0.0, Ly))

            # adaptive smoothing lengths
            h = update_smoothing_lengths(x, m, h)

            # density
            rho = np.zeros(N)
            for i in range(N):
                hi = h[i]
                s = m[i] * cubic_spline_W(0.0, hi)
                for j in neigh[i]:
                    r = np.linalg.norm(x[i] - x[j])
                    s += m[j] * cubic_spline_W(r, hi)
                rho[i] = s

            # pressure
            p = tait_pressure(rho, rho0, c0)

            # acceleration
            a = compute_accelerations(
                x, v, m, rho, p, h,
                alpha=alpha, c0=c0,
                g=np.array([0.0, -9.81])
            )

            # integrate
            x, v = symplectic_euler(x, v, a, dt)

            # boundaries
            apply_reflective_walls(x, v, 0.0, Lx, 0.0, Ly, damping=-0.5)

            # sanity check
            if np.any(np.isnan(x)) or np.any(np.isnan(v)) or np.any(np.isnan(rho)):
                raise RuntimeError(f"NaN detected at step {step}")


            if step % 50 == 0:
                print(f"step {step}/{steps}")
                if show_plot:
                    ax.clear()
                    ax.scatter(x[:,0], x[:,1], s=8, c='C0')
                    ax.set_xlim(0, Lx); ax.set_ylim(0, Ly)
                    ax.set_title(f"step {step}")
                    plt.pause(0.01)

    except Exception as e:
        print("Simulation crashed at step:", step)
        traceback.print_exc()
        # write a small diagnostic file
        np.savez("crash_snapshot.npz", positions=x, velocities=v, masses=m, rho=rho)
        print("Saved crash_snapshot.npz for inspection.")
        raise

    finally:
        if show_plot:
            plt.ioff()
            plt.show()

if __name__ == "__main__":
    # run with very coarse spacing to ensure it completes quickly
    run(dx=0.025, steps=10000, dt=0.0015, show_plot=True)
