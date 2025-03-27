"""
test_convergence.py

Unit tests for PDEConvergenceChecker to ensure it runs and identifies
a stable grid without errors.
"""

import pytest
from hold.pde_convergence import PDEConvergenceChecker


def test_find_stable_grid():
    checker = PDEConvergenceChecker(
        S0=100.0, K=100.0, r=0.05, sigma=0.20, T=1.0, q=0.0, spread=0.01
    )
    # Just check that it returns a grid tuple
    stable_grid = checker.find_stable_grid([(50, 50), (100, 100)], tol=1e-2)
    assert isinstance(stable_grid, tuple), "Expected a tuple for stable_grid"
    assert len(stable_grid) == 2, "stable_grid should have (M, N)"
