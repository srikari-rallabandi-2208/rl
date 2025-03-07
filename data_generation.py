"""
data_generation.py

Generates synthetic data for convertible bond pricing and saves it as an Excel file.
We use a larger default N=300,000 to approach more 'real-world' sized data. You can
adjust this value as needed. The generated features include bond price, implied
volatility, credit spreads, coupon rate, dividend, etc., covering a broader range
of plausible values for a convertible bond.

Example usage:
    python data_generation.py
"""

import numpy as np
import pandas as pd


def generate_synthetic_data(N=300000, output_file="data/synthetic_data.xlsx"):
    """
    Generate synthetic data for convertible bond pricing.

    Parameters:
    -----------
    N : int
        Number of data points to generate. Default is 300,000 for a more realistic scale.
    output_file : str
        Path to save the output Excel file.

    Returns:
    --------
    None
        The generated DataFrame is saved to an Excel file at 'output_file'.
    """
    # Set a random seed for reproducibility
    np.random.seed(42)

    # Create an empty DataFrame to store all generated columns
    df = pd.DataFrame()

    # ---------------------------
    # 1. Generate synthetic features
    # ---------------------------
    # Bond price in a typical range of $50 to $150
    df["bond_price"] = np.random.uniform(low=50, high=150, size=N)

    # Implied volatility (ivol) from 5% to 50%
    df["ivol"] = np.random.uniform(low=5, high=50, size=N)

    # Credit default swap (CDS) spread: typically tens to hundreds of basis points
    # We'll range from 40 to 800 bps for broader coverage
    df["cds"] = np.random.uniform(low=40, high=800, size=N)

    # Underlying stock price from $10 to $150
    df["stock_price"] = np.random.uniform(low=10, high=150, size=N)

    # Risk-free interest rate from 1% to 5%
    df["interest_rate"] = np.random.uniform(low=0.01, high=0.05, size=N)

    # Time-to-maturity (ttm_days): anywhere from 1 year (365 days) to 10 years (3650 days)
    df["ttm_days"] = (np.random.uniform(low=1, high=10, size=N) * 365).astype(int)

    # Coupon rate from 0.3% to 8% (slightly expanded range for more variety)
    df["coupon_rate"] = np.random.uniform(low=0.003, high=0.08, size=N)

    # Coupon frequency: mostly semi-annual (80%) or quarterly (20%)
    df["coupon_frequency"] = np.random.choice([2, 4], size=N, p=[0.8, 0.2])

    # Conversion ratio from 5 to 150
    df["conversion_ratio"] = np.random.uniform(low=5, high=150, size=N)

    # Conversion price from $15 to $200
    df["conversion_price"] = np.random.uniform(low=15, high=200, size=N)

    # Dividend yield: ~30% zeros, 70% between 0 and 0.2
    dividends = np.zeros(N)
    mask = np.random.rand(N) < 0.7
    dividends[mask] = np.random.uniform(low=0, high=0.2, size=mask.sum())
    df["dividend"] = dividends

    # Issuance date is fixed, while the first coupon date is randomly selected among possible dates
    df["issuance_date"] = pd.to_datetime("2020-01-01")
    possible_dates = [
        pd.to_datetime("2020-02-01"),
        pd.to_datetime("2020-07-01"),
        pd.to_datetime("2021-01-01"),
        pd.to_datetime("2022-01-01"),
        pd.to_datetime("2024-01-01")
    ]
    df["first_coupon_date"] = np.random.choice(possible_dates, size=N)

    # ---------------------------
    # 2. Save to Excel
    # ---------------------------
    df.to_excel(output_file, index=False)
    print(f"Synthetic data with {N} records saved to {output_file}")


if __name__ == "__main__":
    generate_synthetic_data()