"""
tf_engine.py

This module encapsulates the enhanced Tsiveriotis-Fernandes PDE engine in a reusable class.
It provides:
  - A Crank-Nicolson finite-difference solver for the convertible bond.
  - A method to generate a (S, t) dataset for supervised or offline RL.
"""

import numpy as np


def generate_pde_dataset(engine, S_min, S_max, num_S_points, t_min, t_max, num_t_points):
    """
    Generate a dataset mapping stock price S and time t to convertible bond price.

    Parameters:
      engine: An instance of TsiveriotisFernandesEngine with solved PDE grids.
      S_min, S_max: Range of stock prices.
      num_S_points: Number of S values.
      t_min, t_max: Range of time.
      num_t_points: Number of time points.

    Returns:
      A list of tuples (S, t, cb_price).
    """
    if engine.u_grid is None or engine.v_grid is None:
        raise ValueError("Engine must have solved the PDE first.")

    S_values = np.linspace(S_min, S_max, num_S_points)
    t_values = np.linspace(t_min, t_max, num_t_points)
    dataset = []
    for t in t_values:
        for S in S_values:
            cb_price = engine.get_cb_value(S, t)
            dataset.append((float(S), float(t), float(cb_price)))
    return dataset


class TsiveriotisFernandesEngine:
    """
    A class that encapsulates the PDE logic for convertible bond pricing.
    It uses a Crank-Nicolson scheme to compute the equity-like and debt-like components.
    """

    def __init__(self, S0, K, r, sigma, T, q, spread, M, N, par=100.0, early_exercise=True):
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
        self.early_exercise = early_exercise

        self.u_grid = None
        self.v_grid = None
        self.cb_value_0 = None

        self.Smax = 3 * S0
        self.dS = self.Smax / N
        self.dt = T / M

    def solve_pdeb(self):
        """
        Solves the convertible bond PDE using the Crank-Nicolson method and stores the solution grids.
        """
        # In tf_engine.py, inside solve_pde:
        from pricing_model import explicit_FD
        u, v, cb_val = explicit_FD(
            self.par, self.T, self.sigma, self.r,
            0.0, self.spread / 100.0, self.K,
            0.0, 2, 0.0, self.M, self.N,
            self.S0  # Pass the initial stock price here
        )
        self.u_grid = u
        self.v_grid = v
        self.cb_value_0 = cb_val

    def get_cb_value(self, S, t):
        """
        Interpolates and returns the convertible bond price for a given stock price S and time t.
        """
        if self.u_grid is None or self.v_grid is None:
            raise ValueError("PDE must be solved before calling get_cb_value().")
        # Convert time to grid index.
        frac_t = (t / self.T) * self.M
        time_idx = int(round(frac_t))
        time_idx = max(0, min(time_idx, self.M))
        # Map S to spatial grid index.
        idx_float = S / self.dS
        if idx_float <= 0.0:
            return self.u_grid[time_idx, 0] + self.v_grid[time_idx, 0]
        if idx_float >= self.N:
            return self.u_grid[time_idx, self.N] + self.v_grid[time_idx, self.N]
        idx = int(idx_float)
        fraction = idx_float - idx
        val_left = self.u_grid[time_idx, idx] + self.v_grid[time_idx, idx]
        val_right = self.u_grid[time_idx, idx + 1] + self.v_grid[time_idx, idx + 1]
        return val_left + fraction * (val_right - val_left)

    def solve_pde(self):
        """
        Solves the convertible bond PDE using the Crank-Nicolson method and stores the solution grids.
        """
        from pricing_model import explicit_FD
        u, v, cb_val = explicit_FD(
            self.par,  # Pr: initial bond price (using par value)
            self.T,  # T: time to maturity
            self.sigma,  # sigma: volatility
            self.r,  # r: risk-free rate
            0.0,  # d: dividend yield (adjust if needed)
            self.spread / 100.0,  # rc: scaled spread
            self.K,  # cv: conversion price
            0.0,  # cp: coupon rate (adjust if needed)
            2,  # cfq: coupon frequency (example value)
            0.0,  # tfc: coupon time factor (example value)
            self.M,  # M: spatial grid steps
            self.N,  # N: time steps
            self.S0  # S0: initial stock price for which the price is computed
        )
        self.u_grid = u
        self.v_grid = v
        self.cb_value_0 = cb_val

