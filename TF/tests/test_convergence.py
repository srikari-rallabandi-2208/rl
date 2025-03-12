"""
test_fd.py

Example automated tests for PDE solver convergence using pytest.
To run: pytest test_fd.py
"""

import pytest
from TF.finite_difference import explicit_fd_cb


@pytest.mark.parametrize("M,N", [(50, 50), (100, 100), (200, 200)])
def test_fd_convergence(M, N):
    # PDE parameters
    params = {
        'principal': 100.0,
        'time_to_maturity': 1.0,
        'volatility': 0.2,
        'risk_free_rate': 0.03,
        'dividend_yield': 0.01,
        'credit_spread': 0.0,
        'conversion_price': 25.0,
        'coupon_rate': 0.04,
        'coupon_frequency': 2,
        'time_factor_coupon': 0.0,
        'call_price': 0.0,
        'put_price': 0.0
    }
    S0 = 30.0

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

    # These are not "hard" tests but sanity checks. For instance,
    # we might expect the price not to blow up or go negative.
    assert 0.0 <= price <= 500.0, "Price is out of expected bounds!"

    # Additional checks could store the results for each M,N
    # and compare differences, but that often requires referencing
    # a baseline or performing a two-level comparison.
