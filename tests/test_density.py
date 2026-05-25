import numpy as np
from sph.adaptive_h import update_smoothing_lengths
from sph.kernel import cubic_spline_W
from sph.density import compute_density


def test_density_single_particle():
    """A single isolated particle should have density = m * W(0, h_i)."""
    m = np.array([1.0])
    x = np.array([[0.0, 0.0]])
    h = np.full(1, 0.1)

    h = update_smoothing_lengths(x, m, h)
    rho = compute_density(x, m, h)

    expected = m[0] * cubic_spline_W(0.0, h[0])
    assert np.isclose(rho[0], expected)


def test_density_two_particles_symmetric():
    """Two particles at equal distance must have identical density."""
    m = np.array([1.0, 1.0])
    x = np.array([[0.0, 0.0],
                  [0.5, 0.0]])
    h = np.full(2, 0.1)

    h = update_smoothing_lengths(x, m, h)
    rho = compute_density(x, m, h)

    assert np.isclose(rho[0], rho[1])


def test_density_known_configuration():
    """Manual density check for a simple 3‑particle line."""
    m = np.array([1.0, 1.0, 1.0])
    x = np.array([[0.0, 0.0],
                  [0.5, 0.0],
                  [1.0, 0.0]])
    h = np.full(3, 0.1)

    h = update_smoothing_lengths(x, m, h)
    rho = compute_density(x, m, h)

    # manual reference for particle 0
    d01 = np.linalg.norm(x[0] - x[1])
    d02 = np.linalg.norm(x[0] - x[2])

    expected_0 = (
        m[0] * cubic_spline_W(0.0, h[0]) +
        m[1] * cubic_spline_W(d01, h[0]) +
        m[2] * cubic_spline_W(d02, h[0])
    )

    assert np.isclose(rho[0], expected_0)
