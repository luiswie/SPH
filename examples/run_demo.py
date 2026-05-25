import numpy as np
import matplotlib.pyplot as plt

from sph.kernel import cubic_spline_W, cubic_spline_gradW
from sph.density import compute_density
from sph.momentum import compute_acceleration
from sph.integrator import symplectic_euler


def create_block(nx=20, ny=20, dx=0.05):
    """Create a rectangular block of particles."""
    xs = np.linspace(0, (nx - 1) * dx, nx)
    ys = np.linspace(0, (ny - 1) * dx, ny)
    X, Y = np.meshgrid(xs, ys)
    pos = np.column_stack([X.ravel(), Y.ravel()])
    return pos


def main():
    # parameters
    h = 0.1
    m = 0.05
    dt = 0.001

    # initial state
    x = create_block()
    v = np.zeros_like(x)
    m_arr = np.full(len(x), m)

    # compute density
    rho = compute_density(x, m_arr, h)

    # compute acceleration
    a = compute_acceleration(x, v, rho, m_arr, h)

    # integrate one step
    x_new, v_new = symplectic_euler(x, v, a, dt)

    # plot
    plt.figure(figsize=(5, 5))
    plt.scatter(x_new[:, 0], x_new[:, 1], s=5)
    plt.title("SPH Demo — One Integration Step")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.axis("equal")
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
