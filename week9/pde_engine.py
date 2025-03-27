"""
pde_engine.py

Contains:
1. PDEEngine class that wraps the PDE solver (explicit_FD).
2. A data generator that calls PDEEngine for multiple data rows.

We incorporate the PDE solver from your snippet. The PDE is quite large
(M=700, N=15000 for improved convergence).
"""

import math
import numpy as np
import pandas as pd
from numba import jit
from datetime import datetime

@jit(nopython=True)
def explicit_FD(
    Pr, T, sigma, r, d, rc, cv, cp, cfq, tfc, M, N
):
    """
    Based on your snippet. PDE approach with M=700, N=15000 default.

    Input:
     - Pr, T, sigma, r, d, rc, ...
     - tfc: coupon time factor
     - M, N: grid sizes
    Output:
     - U: the PDE "equity-like" grid. We interpret U[i,0] as the final price at stock index i, t=0

    For brevity, we skip V or other arrays. We do a single array approach if that's how you used it.
    If you rely on a separate V grid, reintroduce it. This is a simplified variant.
    """

    dx = math.log(450) / M
    dt = T / N
    U = np.zeros((M + 1, N + 1))

    # Boundary conditions at maturity j=N
    for i in range(1, M):
        s_val = math.exp(i * dx)
        if s_val >= cv:
            U[i, N] = Pr * s_val / cv
        else:
            U[i, N] = Pr

    # Do backward induction
    for j in range(N - 1, -1, -1):
        for i in range(1, M):
            # PDE terms
            p1 = dt * (sigma ** 2 / 2.0) * (U[i + 1, j + 1] - 2 * U[i, j + 1] + U[i - 1, j + 1]) / (dx ** 2)
            p2 = dt * ((r - d - sigma ** 2 / 2.0) * (U[i + 1, j + 1] - U[i - 1, j + 1]) / (2.0 * dx))
            # For a real TF approach, you'd combine "r" on the difference between equity- and debt-like components
            # For demonstration, we do a simpler version.

            # A naive boundary for immediate conversion:
            boundary_val = Pr * math.exp(i * dx) / cv
            # This is the "conversion value." We'll apply:
            H = U[i, j + 1] + p1 + p2

            # Compare PDE update vs immediate conversion
            U[i, j] = max(H, boundary_val)

    return U

class PDEEngine:
    """
    Wraps the PDE logic in a class. We can call get_price(...) to get the PDE-based price for a single row's parameters.
    """

    def __init__(self, M=700, N=15000):
        self.M = M
        self.N = N

    def solve_for_row(self, row):
        """
        row: pandas Series with columns:
            [Pr, IVOL, CDS, S, cp, cfq, cv, d, r, ttm_days, issuance_date, first_coupon_date]

        Returns PDE price or np.nan if invalid.
        """
        T = float(row["ttm_days"]) / 365.0
        if T <= 0.0:
            return np.nan

        # If your PDE also depends on coupon_time_factor, do that logic:
        # Suppose coupon_days = difference(issuance_date, first_coupon_date).
        # We'll do a quick example:
        try:
            coupon_days = (pd.to_datetime(row["first_coupon_date"]) - pd.to_datetime(row["issuance_date"])).days
        except:
            coupon_days = 0
        cfq = row["cfq"] if row["cfq"] != 0 else 1.0
        period = 360.0 / cfq
        if T>0:
            tfc = (coupon_days % period)/T
        else:
            tfc = 0.0

        # PDE inputs
        Pr    = float(row["Pr"])
        sigma = float(row["IVOL"]) / 100.0    # IVOL as decimal
        r     = float(row["r"])
        d     = float(row["d"])
        rc    = float(row["CDS"]) / 10000.0   # from bps => decimal
        cv    = float(row["cv"])             # conversion price
        # cp => row["cp"] # We skip in this simpler version, or add if needed.

        # Solve PDE
        try:
            U = explicit_FD(
                Pr=Pr, T=T, sigma=sigma, r=r, d=d, rc=rc, cv=cv,
                cp=0.0, cfq=cfq, tfc=tfc, M=self.M, N=self.N
            )
        except:
            return np.nan

        # Map S to PDE index
        s_val = float(row["S"])
        if s_val<=0:
            return np.nan

        dx = math.log(450.0) / self.M
        idx_s = int(round(math.log(s_val)/dx))
        idx_s = max(0, min(idx_s, self.M))

        return U[idx_s, 0]

    def compute_pde_prices(self, df, max_rows=None):
        """
        Given a DataFrame df with needed columns, compute PDE price row by row.
        If max_rows is set, we only process up to that many valid results.
        Return a new DataFrame with 'Model_Price' appended.

        WARNING: This might be slow if df is large (e.g. 300K).
        """
        results = []
        valid_count = 0

        for i, row in df.iterrows():
            if max_rows and valid_count >= max_rows:
                break

            price = self.solve_for_row(row)
            if not math.isfinite(price):
                continue
            # We'll store the entire row plus 'Model_Price'
            new_row = row.copy()
            new_row["Model_Price"] = price
            results.append(new_row)
            valid_count += 1

        return pd.DataFrame(results)

