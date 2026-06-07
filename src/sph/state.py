# src/sph/state.py
from dataclasses import dataclass
import numpy as np

@dataclass
class SPHState:
    x: np.ndarray      # (N, 2)
    v: np.ndarray      # (N, 2)
    a: np.ndarray      # (N, 2)
    m: np.ndarray      # (N,)
    rho: np.ndarray    # (N,)
    p: np.ndarray      # (N,)
    h: np.ndarray      # (N,)

    @property
    def N(self):
        return self.x.shape[0]
