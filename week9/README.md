# Converged TF Model for RL and Supervised Learning

This directory contains an example workflow for ensuring a Tsiveriotis–Fernandes (TF) PDE solver is sufficiently **converged**, then using that converged PDE engine to train both:
1. An **online Reinforcement Learning** agent (`main_rl_converged.py`).
2. A **static / supervised** policy (`main_static_rl_converged.py`).

## 1. Modules

- **pde_convergence.py**  
  Contains a `PDEConvergenceChecker` to test multiple (M,N) grids, measure solution stability, and pick a converged grid.

- **tf_engine_converged.py**  
  Implements a PDE engine class with a user-chosen grid (M,N). Calls your PDE solver (`explicit_FD`) under the hood.

- **environment_converged.py**  
  Defines an RL environment that references the converged PDE engine for convertible bond pricing signals.

- **rl_agent_converged.py**  
  A REINFORCE policy gradient agent with a small neural network. Minimal modifications from the original.

- **main_rl_converged.py**  
  Entry script that:
  1. Finds (M,N) via PDEConvergenceChecker (or uses a known stable set).
  2. Constructs the PDE engine with that grid.
  3. Trains RL agent on randomly generated stock paths.

- **main_static_rl_converged.py**  
  1. Uses the stable PDE engine to build a table of \((S,t,\text{bond price})\).
  2. Labels each row (convert vs. hold).
  3. Trains a classifier to replicate PDE decisions offline.

## 2. Usage

1. **Check Convergence**  
   ```bash
   python main_rl_converged.py
   ```
This will attempt to find a stable (M,N) among candidates. Adjust in code if desired.

2. **Train Online RL**
This is done automatically within main_rl_converged.py, after it picks or sets a stable grid.

3. **Train Static RL**
    ```bash
   python main_static_rl_converged.py
   ```
   Generates a CSV with PDE-labeled data for a range of 
(
𝑆
,
𝑡
)
(S,t). Then trains a feed-forward net to replicate PDE decisions.

5. **Run Tests**
   ```python
    pytest test_convergence.py
    pytest test_environment.py
    pytest test_rl_agent.py
    ```
   Confirm each piece works.

## 3. Important Notes

The PDE solver is assumed to be stable if `pde_convergence.py` finds small differences between consecutive grids. You can refine the process, adding more thorough error checks (like comparing full solution grids).

For real production usage, you may want a more thorough 2D interpolation in `get_cb_value(S,t)` to get accurate PDE values in between grid points.

All code here is a reference design, not heavily optimized. Real PDE solutions can be accelerated via C++ or HPC frameworks.

