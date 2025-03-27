"""
tf_engine_converged.py

A variant of TsiveriotisFernandesEngine that we will use with
converged (M, N) grids identified by pde_convergence.py.
"""

import numpy as np
# We assume you already have a PDE solver in pricing_model.py
from pricing_model import explicit_FD


class TsiveriotisFernandesEngineConverged:
    """
    Encapsulates the PDE logic for convertible bond pricing
    at the chosen (M, N) grid once we identify them as stable.
    """

    def __init__(self, S0, K, r, sigma, T, q, spread, M, N, par=100.0):
        """
        Set up PDE engine parameters; we assume user has chosen M,N
        that produce convergent solutions.
        """
        self.S0 = S0
        self.K = K
        self.r = r
        self.sigma = sigma
        self.T = T
        self.q = q
        self.spread = spread
        self.M = M
        self.N = N
        self.par = par

        self.u_grid = None
        self.v_grid = None
        self.cb_value_0 = None

    def solve_pde(self):
        """
        Use the PDE solver. The explicit_FD function is from pricing_model.py.
        We pass in parameters so that it returns (u_grid, v_grid, price).
        """
        # We'll assume d=0.0 or pass your actual dividend yield self.q
        # Also, assume coupon=0.0 for simplicity, or adapt.
        # We'll scale credit spread to decimal by dividing by 100 if needed.
        rc_decimal = self.spread  # if your code expects raw decimal
        # or self.spread/100 if your self.spread is in basis points

        u, v, price_0 = explicit_FD(
            Pr=self.par,  # face value used as "initial bond price"
            T=self.T,
            sigma=self.sigma,
            r=self.r,
            d=self.q,
            rc=rc_decimal,
            cv=self.K,
            cp=0.0,  # coupon rate - adapt if needed
            cfq=2,  # coupon freq
            tfc=0.0,  # time factor for coupon
            M=self.M,
            N=self.N,
            S0=self.S0
        )
        self.u_grid = u
        self.v_grid = v
        self.cb_value_0 = price_0

    def get_cb_value(self, S, t):
        """
        Interpolate PDE solution to retrieve convertible bond price for (S, t).
        For simplicity, we'll do nearest or linear interpolation in the time dimension
        and approximate the stock dimension using the log scale used in explicit_FD.
        """
        if self.u_grid is None or self.v_grid is None:
            raise ValueError("Must call solve_pde() first.")
        # Very simplified approach: map t -> index, S-> index, then pick from grids.
        # Detailed interpolation is omitted for brevity.
        # ...
        # For demonstration, we assume t=0 => j=0, t=T => j=N, S=some log scale => i index
        # In real code, do carefully with log scale + linear interpolation.
        return self._approx_grid(S, t)

    def _approx_grid(self, S, t):
        """
        Very naive approach: just do nearest index in time + nearest in i for S.
        Or do small linear interpolation. Adjust as you see fit.
        """
        dt = self.T / self.N
        j_float = t / dt
        j_idx = int(round(j_float))
        j_idx = max(0, min(j_idx, self.N))

        # S was discretized using log scale up to some max (like log(450)/M).
        # We'll replicate the logic from explicit_FD:
        #   S_i = exp(i * dx?), with dx = log(450)/M
        # We'll just do naive rounding:
        import math
        dx = math.log(450.0) / self.M
        i_float = math.log(S) / dx if S > 1e-8 else 0.0
        i_idx = int(round(i_float))
        i_idx = max(0, min(i_idx, self.M))

        # sum of the equity-like (u_grid) and debt-like (v_grid)
        return self.u_grid[i_idx, j_idx] + self.v_grid[i_idx, j_idx]
