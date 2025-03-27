"""
pde_convergence.py

This module provides:
1) A PDEConvergenceChecker class that systematically tests different
   (M, N) grids for Tsiveriotis-Fernandes PDE solutions.
2) Utilities to measure differences between PDE solutions on coarse/fine grids.

Example usage:
    checker = PDEConvergenceChecker(
        S0=100.0, K=100.0, r=0.05, sigma=0.20, T=1.0, q=0.0, spread=0.01, par=100.0
    )
    stable_grid = checker.find_stable_grid([(50,50), (100,100), (200,200)])
    # stable_grid might be (200,200) if that is sufficiently converged.
"""

import math
import numpy as np
from tf_engine_converged import TsiveriotisFernandesEngineConverged

class PDEConvergenceChecker:
    """
    A class to run mesh-refinement tests for the Tsiveriotis-Fernandes PDE engine.
    We vary (M, N) to see how the final convertible bond price changes.
    """

    def __init__(self, S0, K, r, sigma, T, q, spread, par=100.0):
        """
        Constructor for PDEConvergenceChecker.

        Parameters
        ----------
        S0 : float
            Initial stock price.
        K : float
            Conversion (strike) price.
        r : float
            Risk-free rate.
        sigma : float
            Volatility.
        T : float
            Time to maturity (years).
        q : float
            Dividend yield.
        spread : float
            Credit spread.
        par : float
            Par (face) value of the convertible bond.
        """
        self.S0 = S0
        self.K = K
        self.r = r
        self.sigma = sigma
        self.T = T
        self.q = q
        self.spread = spread
        self.par = par

    def solve_pde_for_grid(self, M, N):
        """
        Solve PDE with a given (M, N) and return the CB price at S0, t=0.
        """
        engine = TsiveriotisFernandesEngineConverged(
            S0=self.S0, K=self.K, r=self.r, sigma=self.sigma,
            T=self.T, q=self.q, spread=self.spread, M=M, N=N, par=self.par
        )
        engine.solve_pde()
        return engine.cb_value_0

    def find_stable_grid(self, grid_candidates, tol=1e-3):
        """
        Iterates over multiple grid sizes, computing the PDE solution,
        and checking difference between consecutive solutions.

        Parameters
        ----------
        grid_candidates : list of (int, int)
            List of (M, N) pairs to test in ascending order.
        tol : float
            Tolerance for difference in price. If difference < tol,
            we consider it converged.

        Returns
        -------
        stable_grid : (M, N)
            The grid that satisfies convergence tolerance or the largest tested.
        """
        prev_price = None
        stable_grid = grid_candidates[-1]
        for (M, N) in grid_candidates:
            price = self.solve_pde_for_grid(M, N)
            if prev_price is not None:
                diff = abs(price - prev_price)
                print(f"(M={M}, N={N}) => Price={price:.4f}, Diff from prev={diff:.6f}")
                if diff < tol:
                    stable_grid = (M, N)
                    break
            else:
                print(f"(M={M}, N={N}) => Price={price:.4f}, Diff from prev=NA")
            prev_price = price
        print(f"Chosen stable grid: {stable_grid}")
        return stable_grid
