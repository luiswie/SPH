from sph.kernel import cubic_spline_W, cubic_spline_gradW
import numpy as np

def test_kernel_symmetry():
    h = 0.01
    r = 0.005
    w = cubic_spline_W(r, h)
    assert w > 0.0
    g = cubic_spline_gradW(np.array([r,0.0]), h)
    assert np.isfinite(g[0])
