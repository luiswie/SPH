import numpy as np
from sph.kernel import cubic_spline_W
from sph.density import compute_density


def test_density_single_particle():
    """A single isolated particle should have density = m * W(0)."""
    m = np.array([1.0])
    x = np.array([[0.0, 0.0]])
    h = 1.0

    rho = compute_density(x, m, h)
    expected = m[0] * cubic_spline_W(0.0, h)

    assert np.isclose(rho[0], expected)


def test_density_two_particles_symmetric():
    """Two particles at equal distance contribute equally."""
    m = np.array([1.0, 1.0])
    x = np.array([[0.0, 0.0],
                  [0.5, 0.0]])
    h = 1.0

    rho = compute_density(x, m, h)

    # symmetry: rho[0] == rho[1]
    assert np.isclose(rho[0], rho[1])


def test_density_known_configuration():
    """Simple configuration with known distances."""
    m = np.array([1.0, 1.0, 1.0])
    x = np.array([[0.0, 0.0],
                  [0.5, 0.0],
                  [1.0, 0.0]])
    h = 1.0

    rho = compute_density(x, m, h)

    # manual reference
    d01 = np.linalg.norm(x[0] - x[1])
    d02 = np.linalg.norm(x[0] - x[2])
    expected_0 = (
        m[0] * cubic_spline_W(0.0, h)
        + m[1] * cubic_spline_W(d01, h)
        + m[2] * cubic_spline_W(d02, h)
    )

    assert np.isclose(rho[0], expected_0)
