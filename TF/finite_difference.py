"""
finite_difference.py

Module containing finite-difference methods (explicit, implicit, Crank-Nicolson, etc.)
for pricing derivatives or convertible bonds.

Author: Your Name
Date: YYYY-MM-DD
"""

import math
import numpy as np
from numba import jit


@jit(nopython=True)
def explicit_fd_cb(
        principal, time_to_maturity, volatility, risk_free_rate, dividend_yield,
        credit_spread, conversion_price, coupon_rate, coupon_frequency,
        time_factor_coupon, stock_grid_steps, time_grid_steps,
        call_price, put_price, current_stock_price
):
    """
    Compute convertible bond price using an explicit finite-difference method.

    Parameters
    ----------
    principal : float
        Bond face value.
    time_to_maturity : float
        Time until maturity (T).
    volatility : float
        Stock price volatility (sigma).
    risk_free_rate : float
        Risk-free interest rate (r).
    dividend_yield : float
        Continuous dividend yield (d).
    credit_spread : float
        Credit spread (rc).
    conversion_price : float
        Conversion price of the bond to shares.
    coupon_rate : float
        Annual coupon rate (decimal).
    coupon_frequency : float
        Number of coupon payments per year.
    time_factor_coupon : float
        Time factor for coupon payment alignment.
    stock_grid_steps : int
        Number of stock price steps (M).
    time_grid_steps : int
        Number of time steps (N).
    call_price : float
        Call provision price.
    put_price : float
        Put provision price.
    current_stock_price : float
        Current stock price (S).

    Returns
    -------
    float
        The computed price of the convertible bond at t=0, S=current_stock_price.
    """
    dx = math.log(6 * conversion_price) / stock_grid_steps
    dt = time_to_maturity / time_grid_steps

    # U_grid, V_grid represent the total convertible bond value and the "bond/cash" part
    U_grid = np.zeros((stock_grid_steps + 1, time_grid_steps + 1))
    V_grid = np.zeros((stock_grid_steps + 1, time_grid_steps + 1))

    # Compute coupon period in terms of time steps:
    cp_period = int(time_grid_steps / (coupon_frequency * time_to_maturity))
    cp_array = np.arange(
        cp_period + int(time_factor_coupon * time_grid_steps),
        time_grid_steps + 1,
        cp_period
    )

    last_cp_date = 100000  # Not fully clear from original code; adapt as needed
    if time_grid_steps - last_cp_date <= 1:
        cp_at_T = principal * coupon_rate / coupon_frequency
    else:
        cp_at_T = (time_grid_steps - last_cp_date) * principal * coupon_rate / (coupon_frequency * cp_period)

    # Terminal payoff at T
    for i in range(1, stock_grid_steps):
        # If share price >= conversion price, we assume it is beneficial to convert
        if math.exp(i * dx) >= conversion_price:
            U_grid[i, time_grid_steps] = principal * math.exp(i * dx) / conversion_price + cp_at_T
            V_grid[i, time_grid_steps] = cp_at_T
        else:
            U_grid[i, time_grid_steps] = principal + cp_at_T
            V_grid[i, time_grid_steps] = principal + cp_at_T

    # Boundary conditions at S=0
    U_grid[0, time_grid_steps] = principal
    V_grid[0, time_grid_steps] = principal

    # Some boundary for times before T (j < time_grid_steps):
    for t in range(1, time_grid_steps):
        # Discounted boundary guess (or any boundary you set)
        disc_factor = 1.0 / (1.0 + (risk_free_rate + credit_spread) * dt) ** (time_grid_steps - t)
        U_grid[0, t] = principal * disc_factor
        V_grid[0, t] = principal * disc_factor

    # Backward time iteration
    for j in range(time_grid_steps - 1, -1, -1):
        for i in range(1, stock_grid_steps):
            # Explicit FD updates for PDE
            # Example coefficients p1, p2, p3 etc.:
            p1 = dt * (0.5 * volatility ** 2) * (U_grid[i + 1, j + 1] - 2 * U_grid[i, j + 1] + U_grid[i - 1, j + 1]) / (
                        dx ** 2)
            p2 = dt * (risk_free_rate - dividend_yield - 0.5 * volatility ** 2) \
                 * (U_grid[i + 1, j + 1] - U_grid[i - 1, j + 1]) / (2 * dx)
            p3 = dt * (-risk_free_rate * (U_grid[i, j + 1] - V_grid[i, j + 1])
                       - (risk_free_rate + credit_spread) * V_grid[i, j + 1])

            # Conversion value
            aS = principal + (j % cp_period) * (principal * coupon_rate / 2) / cp_period
            aS *= (math.exp(i * dx) / conversion_price)

            # Check if coupon payment happens at step j
            ft = 0.0
            for k in cp_array:
                if j <= k < j + 1:
                    ft = principal * coupon_rate / 2
                    break

            # Update bond value
            U_hold = U_grid[i, j + 1] + p1 + p2 + p3 + ft  # "holding" scenario
            # Convert or hold?
            U_grid[i, j] = max(U_hold, aS)

            # If NOT converting, update V_grid
            if U_grid[i, j] == U_hold:
                q1 = dt * (0.5 * volatility ** 2) * (
                            V_grid[i + 1, j + 1] - 2 * V_grid[i, j + 1] + V_grid[i - 1, j + 1]) / (dx ** 2)
                q2 = dt * (risk_free_rate - dividend_yield - 0.5 * volatility ** 2) \
                     * (V_grid[i + 1, j + 1] - V_grid[i - 1, j + 1]) / (2 * dx)
                q3 = dt * (-(risk_free_rate + credit_spread) * V_grid[i, j + 1])
                V_grid[i, j] = V_grid[i, j + 1] + q1 + q2 + q3 + ft
            else:
                # If conversion happened
                V_grid[i, j] = 0.0

            # Call and put checks
            if call_price > 0 and U_grid[i, j] >= call_price:
                U_grid[i, j] = call_price
                V_grid[i, j] = 0.0
            if put_price > 0 and U_hold <= put_price:
                U_grid[i, j] = put_price
                V_grid[i, j] = put_price

    # Find the location in the grid corresponding to the current stock price
    S_idx_float = math.log(current_stock_price) / dx
    S_idx = int(round(S_idx_float))

    # Clamp to ensure we stay in valid array index range
    S_idx = max(0, min(S_idx, stock_grid_steps))

    return U_grid[S_idx, 0]
