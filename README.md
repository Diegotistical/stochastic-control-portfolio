# Stochastic Control Portfolio

[![License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)
[![Python 3.11](https://img.shields.io/badge/python-3.11-blue.svg)](https://www.python.org/downloads/release/python-3110/)
[![Financial Engineering](https://img.shields.io/badge/field-Financial%20Engineering-red.svg)](#)

**Stochastic-Control-Portfolio** is a high-performance computational engine designed to solve optimal decision-making problems in finance. By transforming a modular reaction-diffusion PDE framework into a Hamilton-Jacobi-Bellman (HJB) solver, this project provides a robust numerical approach to the **Merton Portfolio Problem**, determining the optimal allocation of wealth between risky and risk-free assets over a continuous-time horizon.

---

## 📝 Repository Description
This repository implements a high-performance **Merton Portfolio Problem** solver developed for quantitative finance. Built from a modular PDE engine, this repo optimizes investment strategies under uncertainty by solving the **Hamilton-Jacobi-Bellman (HJB)** equation. Features include **GPU-accelerated** time-stepping and **Neumann** boundary support.

---

## 🚀 Key Features
* **Optimal Control Solver:** Solves the non-linear HJB equation to determine the optimal portfolio weight ($\pi^*$) at every point on a wealth-time grid.
* **Modular PDE Architecture:** Decouples the financial "physics" (Merton dynamics) from the numerical integration engine (Implicit-Explicit time-stepping).
* **High-Performance Computing:** Utilizes vectorized operations and optional **GPU acceleration** (via Numba/CuPy) to handle dense spatial grids efficiently.
* **Advanced Boundary Management:** Implements **Neumann (Reflective)** boundary conditions to ensure utility conservation across the wealth domain.
* **Automated Risk Monitoring:** Built-in health checks to detect and prevent numerical instabilities during non-linear optimization steps.

---

## 🧠 Mathematical Framework
The core of the repository is the **Merton Portfolio Problem**, which seeks to maximize the expected power utility of terminal wealth:

$$E\left[ \frac{W_T^\gamma}{\gamma} \right]$$

The solver iterates backward in time from the terminal utility state, solving the HJB equation:
$$V_t + \sup_{\pi} \left[ (rW + \pi W(\mu - r))V_W + \frac{1}{2}(\pi W \sigma)^2 V_{WW} \right] = 0$$

At each step, the engine performs a local optimization to find the optimal control $\pi^*$, effectively transforming a global portfolio strategy into a series of local "reaction" terms within the PDE grid.

---

## 📂 Repository Structure


---
## Getting Started

### Prerequisites

- **Python** ≥ 3.11  
- **NumPy**, **SciPy**, **Matplotlib**  
- **Numba** (CPU acceleration)  
- **CuPy** *(optional — GPU acceleration)*


## Installation

```md
git clone https://github.com/Diegotistical/stochastic-control-portfolio.git
cd stochastic-control-portfolio
pip install -r requirements.txt
