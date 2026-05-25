# **SPH — Minimal, Modular, 2D Smoothed Particle Hydrodynamics Solver in Python**

A clean, educational, and fully modular implementation of **Weakly Compressible Smoothed Particle Hydrodynamics (WCSPH)** in Python.

This project accompanies a personal study of SPH and provides a compact, well‑structured codebase suitable for experimentation, visualization, and numerical method development.

---

## **Project Structure**

```
sph/
│
├── src/
│   └── sph/
│       ├── __init__.py
│       ├── kernel.py
│       ├── neighbors.py
│       ├── density.py
│       ├── momentum.py
│       ├── integrator.py
│       └── boundaries.py
│
├── tests/
│   ├── test_kernel.py
│   └── test_density.py
│
├── examples/
│   └── run_dambreak.py
│
├── pyproject.toml
├── setup.cfg
├── requirements.txt
└── README.md
```

The solver is intentionally decomposed into small, testable modules to make the numerical method transparent and easy to extend.

---

# **Mathematical Formulation**

This project implements the standard **WCSPH formulation** with:

- Cubic spline kernel  
- Density summation
- Adaptive smoothing length
- Tait equation of state  
- Symmetric pressure forces  
- Monaghan artificial viscosity  
- Symplectic Euler / Predictor–Corrector integration  

---

## **Kernel Function**

The cubic spline kernel in 2D is defined as:

$$
W(r, h) = \frac{10}{7\pi h^2}
\begin{cases}
\frac{1}{4}(2-q)^3-(1-q)^3, & 0 \le q < 1 \\
\frac{1}{4}(2 - q)^3, & 1 \le q < 2 \\
0, & q \ge 2
\end{cases}
$$

with

$$
q = \frac{r}{h}.
$$

The gradient is implemented analytically for numerical stability.

---

## **Density Summation**

The density of particle \( i \) is computed as:

$$
\rho_i = \sum_j m_j W(r_{ij}, h_i).
$$

This avoids solving a continuity equation and is standard in WCSPH.

---

## **Smoothing Length**

Each particle adapts its own smoothing lenght $h_i$ based on the local density:

$$
h_i = \eta \left( \frac{m_i}{\rho_i} \right)^{1/2}.
$$

Because $\rho_i$ depends on $h_i$, and $h_i$ depends on $\rho_i$, the two are solved together using a small Newton iteration.  
This keeps the **neighbor number roughly constant**, improves stability, and ensures the resolution follows the flow naturally.

---

## **Equation of State (Tait)**

Pressure is computed using:

$$
p_i = c_0^2 \rho_0 \left[ \left( \frac{\rho_i}{\rho_0} \right)^\gamma - 1 \right],
$$

with typical parameters:

- \( $\gamma = 7$ \)  
- \( $c_0$ \) chosen such that Mach ≈ 0.1  

---

## **Momentum Equation**

The acceleration is computed as:

$$
\frac{d\mathbf{v}_i}{dt} = - \sum_j m_j \left(\frac{p_i}{\rho_i^2} + \frac{p_j}{\rho_j^2} \right) \nabla W_{ij} + \text{viscosity} + \text{boundary forces}.
$$

Artificial viscosity follows Monaghan (1992).

---

## **Boundary Conditions**

The solver uses **simple reflective walls**: when a particle crosses a domain boundary, its position is clamped to the wall and the corresponding velocity component is reversed and damped. This enforces a no‑penetration condition and provides a minimal, robust boundary treatment suitable for basic tests such as the dam‑break.

---

# **Testing**

Unit tests ensure correctness of:

- Kernel values  
- Kernel gradients  
- Density summation  
- Neighbor search  

Run tests with:

```
pytest -q
```

---

# **Running the Demo**

A simple 2D dam-break example is included:

```
python -m examples.run_dambreak
```

This generates particle trajectories and optionally frame-by-frame PNG output.

---

# **Visualization**

The project includes a minimal Matplotlib-based visualizer.  

---

# **Installation**

Install in editable mode:

```
pip install -e .
```

Then import modules normally:

```python
from sph.kernel import cubic_spline_W
```

---

# **Goals of This Project**

- Understand SPH from first principles  
- Build a clean, modular solver  
- Enable experimentation with kernels, viscosity, integrators  
- Provide a reproducible scientific codebase  

---

# **References**

- Monaghan, J. J. (1992). *Smoothed Particle Hydrodynamics*.  
- Liu & Liu (2003). *Smoothed Particle Hydrodynamics: A Meshfree Particle Method*.  
- Price, D. (2012). *Smoothed Particle Hydrodynamics and Magnetohydrodynamics*.  
