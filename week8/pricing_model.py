"""
pricing_model.py

This module implements the Tsiveriotis-Fernandes PDE solver using a Crank-Nicolson
finite-difference scheme for convertible bond pricing. The function explicit_FD is
modified here to return three values: the equity-like grid (U_grid), the debt-like grid
(V_grid), and an estimated convertible bond price (cb_value) for a given initial stock price (S0).
"""

import math
import numpy as np
from numba import jit


@jit(nopython=True)
def explicit_FD(Pr, T, sigma, r, d, rc, cv, cp, cfq, tfc, M, N, S0):
    """
    Solves the convertible bond PDE using the Crank-Nicolson finite-difference method.
    Returns:
        U_grid: 2D array for the equity-like component.
        V_grid: 2D array for the debt-like component.
        cb_value: Estimated convertible bond price at time t=0 for initial stock price S0.

    Parameters:
      Pr    : float, initial bond price.
      T     : float, time to maturity.
      sigma : float, volatility.
      r     : float, risk-free rate.
      d     : float, dividend yield.
      rc    : float, credit spread (scaled).
      cv    : float, conversion price.
      cp    : float, coupon rate.
      cfq   : int, coupon frequency.
      tfc   : float, coupon time factor.
      M     : int, number of spatial grid steps.
      N     : int, number of time steps.
      S0    : float, initial stock price for which to estimate the bond price.
    """
    # Spatial discretization: using a logarithmic grid (assume maximum S value = 450)
    dx = math.log(450) / M
    dt = T / N  # Time step size
    U_grid = np.zeros((M + 1, N + 1))
    V_grid = np.zeros((M + 1, N + 1))

    # For simplicity, assume a basic coupon treatment.
    cp_period = int(N / (cfq * T))
    cp_at_T = Pr * cp / cfq  # Simplified terminal coupon value

    # Set terminal conditions at maturity (time index N)
    for i in range(M + 1):
        # Map grid index to a stock price using the exponential function.
        S_val = Pr * math.exp(i * dx)  # Note: This is a simplified mapping.
        if S_val >= cv:
            U_grid[i, N] = Pr * S_val / cv + cp_at_T
            V_grid[i, N] = cp_at_T
        else:
            U_grid[i, N] = Pr + cp_at_T
            V_grid[i, N] = Pr + cp_at_T

    # Boundary conditions for time steps (set for i=0)
    for j in range(N):
        U_grid[0, j] = Pr / ((1 + (r + rc) * dt) ** (N - j))
        V_grid[0, j] = Pr / ((1 + (r + rc) * dt) ** (N - j))

    # Backward induction: loop backward in time from j = N-1 to 0.
    for j in range(N - 1, -1, -1):
        for i in range(1, M):
            # Finite difference approximations (simplified version)
            p1 = dt * ((sigma ** 2 / 2) * (U_grid[i + 1, j + 1] - 2 * U_grid[i, j + 1] + U_grid[i - 1, j + 1]) / (
                        dx ** 2))
            p2 = dt * (r - d - sigma ** 2 / 2) * (U_grid[i + 1, j + 1] - U_grid[i - 1, j + 1]) / (2 * dx)
            p3 = dt * (-r * (U_grid[i, j + 1] - V_grid[i, j + 1]) - (r + rc) * V_grid[i, j + 1])
            U_grid[i, j] = U_grid[i, j + 1] + p1 + p2 + p3

            # Check if immediate conversion is optimal.
            conv_val = Pr * math.exp(i * dx) / cv  # Simplified conversion value
            if U_grid[i, j] < conv_val:
                U_grid[i, j] = conv_val
                V_grid[i, j] = 0.0

    # Compute the grid index corresponding to the input stock price S0.
    S_index = round(math.log(S0) / (math.log(450) / M))
    if S_index < 0:
        S_index = 0
    elif S_index > M:
        S_index = M
    # The convertible bond price is taken from the computed grid at time index 0.
    cb_value = U_grid[S_index, 0]

    return U_grid, V_grid, cb_value
