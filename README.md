# **SPH — Minimal, Modular Smoothed Particle Hydrodynamics Solver in Python**

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
│       ├── boundaries.py
│       └── run_dambreak.py
│
├── tests/
│   ├── test_kernel.py
│   └── test_density.py
│
├── examples/
│   └── run_demo.py
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
- Tait equation of state  
- Symmetric pressure forces  
- Monaghan artificial viscosity  
- Symplectic Euler / Predictor–Corrector integration  

---

## **Kernel Function**

The cubic spline kernel in 3D is defined as:

$$
W(r, h) = \frac{1}{\pi h^3}
\begin{cases}
1 - \frac{3}{2}q^2 + \frac{3}{4}q^3, & 0 \le q < 1 \\
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
\rho_i = \sum_j m_j W(r_{ij}, h).
$$

This avoids solving a continuity equation and is standard in WCSPH.

---

## **Equation of State (Tait)**

Pressure is computed using:

$$
p_i = c_0^2 \rho_0 \left[ \left( \frac{\rho_i}{\rho_0} \right)^\gamma - 1 \right],
$$

with typical parameters:

- \( \gamma = 7 \)  
- \( c_0 \) chosen such that Mach ≈ 0.1  

---

## **Momentum Equation**

The acceleration is computed as:

$$
\frac{d\mathbf{v}_i}{dt} =
- \sum_j m_j
\left(
\frac{p_i}{\rho_i^2} + \frac{p_j}{\rho_j^2}
\right)
\nabla W_{ij}
+ \text{viscosity}
+ \text{boundary forces}.
$$

Artificial viscosity follows Monaghan (1992).

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
python -m sph.run_dambreak
```

This generates particle trajectories and optionally frame-by-frame PNG output.

---

# **Visualization**

The project includes a minimal Matplotlib-based visualizer.  
You can also export VTK files for ParaView by extending `run_dambreak.py`.

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
