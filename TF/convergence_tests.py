"""
convergence_tests.py

Module providing routines to check convergence as we refine
time steps (dt) and space steps (dS).

Author: Your Name
Date: YYYY-MM-DD
"""

import numpy as np
from finite_difference import explicit_fd_cb  # or whatever you call it


def run_grid_refinement_test(params, S0, refine_levels=[(50, 50), (100, 100), (200, 200)]):
    """
    Runs the convertible bond PDE solver for multiple grid sizes
    and prints or returns the solutions for comparison.

    Parameters
    ----------
    params : dict
        Dictionary of parameters needed for explicit_fd_cb, e.g.,
        {
          'principal': 100.0,
          'time_to_maturity': 2.0,
          'volatility': 0.3,
          'risk_free_rate': 0.02,
          'dividend_yield': 0.01,
          'credit_spread': 0.0,
          'conversion_price': 30.0,
          'coupon_rate': 0.05,
          'coupon_frequency': 2,
          'time_factor_coupon': 0.0,
          'call_price': 0.0,
          'put_price': 0.0
        }
    S0 : float
        Current stock price for which we want the computed bond value.
    refine_levels : list of (int, int)
        Each tuple is (stock_grid_steps, time_grid_steps).
        Example: [(50,50), (100,100), (200,200)]

    Returns
    -------
    results : dict
        Key: (M, N) as tuple, Value: computed bond price
    """
    results = {}
    for (M, N) in refine_levels:
        price = explicit_fd_cb(
            principal=params['principal'],
            time_to_maturity=params['time_to_maturity'],
            volatility=params['volatility'],
            risk_free_rate=params['risk_free_rate'],
            dividend_yield=params['dividend_yield'],
            credit_spread=params['credit_spread'],
            conversion_price=params['conversion_price'],
            coupon_rate=params['coupon_rate'],
            coupon_frequency=params['coupon_frequency'],
            time_factor_coupon=params['time_factor_coupon'],
            stock_grid_steps=M,
            time_grid_steps=N,
            call_price=params['call_price'],
            put_price=params['put_price'],
            current_stock_price=S0
        )
        results[(M, N)] = price
    return results
