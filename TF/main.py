"""
main.py

Entry point script demonstrating how to run the convertible bond
FD solver and test for convergence.

Author: Your Name
Date: YYYY-MM-DD
"""

from convergence_tests import run_grid_refinement_test


def main():
    # Define PDE parameters
    params = {
        'principal': 100.0,
        'time_to_maturity': 2.0,  # years
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

    # Stock price at which we want to value the CB
    S0 = 32.0

    # Perform a grid refinement test
    refine_levels = [(50, 50), (100, 100), (200, 200)]
    results = run_grid_refinement_test(params, S0, refine_levels)

    print("Grid Refinement Results (CB Price):")
    for (M, N), val in results.items():
        print(f"  M={M:4d}, N={N:4d} --> Price = {val:.4f}")


if __name__ == "__main__":
    main()
