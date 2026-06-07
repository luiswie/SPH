import sys
import traceback
import numpy as np
import matplotlib.pyplot as plt

from sph.kernel import cubic_spline_W, cubic_spline_gradW
from sph.neighbors import grid_neighbors
from sph.adaptive_h import update_smoothing_lengths
from sph.density import compute_density
from sph.eos import tait_pressure
from sph.momentum import compute_accelerations
from sph.integrator import symplectic_euler
from sph.boundaries import apply_reflective_walls
from sph.state import SPHState   # <-- new

def setup_column(dx=0.02, width=0.1, height=0.2, offset=(0.02,0.02)):
    nx = max(1, int(np.ceil(width / dx)))
    ny = max(1, int(np.ceil(height / dx)))
    pts = []
    for j in range(ny):
        for i in range(nx):
            pts.append([offset[0] + i*dx, offset[1] + j*dx])
    return np.array(pts, dtype=float)

def run(dx=0.02, steps=300, dt=0.001, show_plot=True):
    Lx, Ly = 1.6, 0.6
    rho0 = 1000.0
    c0 = 10.0
    alpha = 0.1

    x = setup_column(dx=dx, width=0.1, height=0.2, offset=(0.02, 0.02))
    N = x.shape[0]
    h = np.full(N, 1.2 * dx)
    v = np.zeros_like(x)
    m = (dx*dx*rho0) * np.ones(N)
    rho = np.zeros(N)
    p = np.zeros(N)
    a = np.zeros_like(x)

    state = SPHState(x=x, v=v, m=m, rho=rho, p=p, h=h, a=a)

    print(f"Running with dx={dx:.4f}, N={state.N}, dt={dt:.5f}, steps={steps}")

    if show_plot:
        plt.ion()
        fig, ax = plt.subplots(figsize=(8,3))

    try:
        for step in range(steps):

            # 1. update h using OLD density (predictor)
            state.h[:] = update_smoothing_lengths(state.x, state.m, state.h)

            # 2. recompute neighbors using NEW h
            neigh = grid_neighbors(state.x, state.h, domain=(0.0, Lx, 0.0, Ly))

            # 3. compute density using symmetric kernel
            state.rho[:] = compute_density(state.x, state.m, state.h, neigh)

            # 4. compute pressure
            state.p[:] = tait_pressure(state.rho, rho0, c0)

            # 5. compute accelerations
            state.a[:] = compute_accelerations(
                state.x, state.v, state.m, state.rho, state.p, state.h,
                alpha=alpha, c0=c0,
                g=np.array([0.0, -9.81])
            )

            # 6. symplectic Euler update
            state.x, state.v = symplectic_euler(state.x, state.v, state.a, dt)

            apply_reflective_walls(state.x, state.v, 0.0, Lx, 0.0, Ly, damping=-0.5)

            if np.any(np.isnan(state.x)) or np.any(np.isnan(state.v)) or np.any(np.isnan(state.rho)):
                raise RuntimeError(f"NaN detected at step {step}")

            if step % 50 == 0:
                print(f"step {step}/{steps}")
                if show_plot:
                    ax.clear()
                    ax.scatter(state.x[:,0], state.x[:,1], s=8, c='C0')
                    ax.set_xlim(0, Lx); ax.set_ylim(0, Ly)
                    ax.set_title(f"step {step}")
                    plt.pause(0.01)

    except Exception as e:
        print("Simulation crashed at step:", step)
        traceback.print_exc()
        np.savez("crash_snapshot.npz",
                 positions=state.x,
                 velocities=state.v,
                 masses=state.m,
                 rho=state.rho)
        print("Saved crash_snapshot.npz for inspection.")
        raise

    finally:
        if show_plot:
            plt.ioff()
            plt.show()

if __name__ == "__main__":
    run(dx=0.025, steps=10000, dt=0.0015, show_plot=True)
