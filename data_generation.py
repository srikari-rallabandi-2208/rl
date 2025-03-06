"""
data_generation.py

This module generates synthetic data for convertible bond pricing.
It creates a dataset with various financial parameters and saves it as an Excel file.
"""

import numpy as np
import pandas as pd


def generate_synthetic_data(N=100000, output_file="data/synthetic_data.xlsx"):
    """
    Generate synthetic data for convertible bond pricing.

    Parameters:
        N (int): Number of data points to generate (default: 100,000).
        output_file (str): Path to save the output Excel file (default: "data/synthetic_data.xlsx").

    Returns:
        None: Saves the data to an Excel file.
    """
    # Set seed for reproducibility
    np.random.seed(42)
    df = pd.DataFrame()

    # Generate synthetic features
    df["bond_price"] = np.random.uniform(low=50, high=150, size=N)  # Bond price between $50 and $150
    df["ivol"] = np.random.uniform(low=5, high=50, size=N)          # Implied volatility between 5% and 50%
    df["cds"] = np.random.uniform(low=40, high=400, size=N)        # Credit default spread between 40 and 400 bps
    df["stock_price"] = np.random.uniform(low=10, high=100, size=N) # Stock price between $10 and $100
    df["interest_rate"] = np.random.uniform(low=0.01, high=0.05, size=N)  # Risk-free rate between 1% and 5%
    df["ttm_days"] = (np.random.uniform(low=1, high=10, size=N) * 365).astype(int)  # Time-to-maturity in days (1-10 years)
    df["coupon_rate"] = np.random.uniform(low=0.003, high=0.06, size=N)  # Coupon rate between 0.3% and 6%
    df["coupon_frequency"] = np.random.choice([2, 4], size=N, p=[0.8, 0.2])  # Semi-annual (80%) or quarterly (20%)
    df["conversion_ratio"] = np.random.uniform(low=5, high=100, size=N)  # Conversion ratio between 5 and 100
    df["conversion_price"] = np.random.uniform(low=15, high=175, size=N)  # Conversion price between $15 and $175

    # Dividend: 30% zeros, 70% non-zero values between 0 and 0.175
    dividends = np.zeros(N)
    mask = np.random.rand(N) < 0.7
    dividends[mask] = np.random.uniform(low=0, high=0.175, size=mask.sum())
    df["dividend"] = dividends

    # Dates: Fixed issuance date and random first coupon dates
    df["issuance_date"] = pd.to_datetime("2020-01-01")
    possible_dates = [
        pd.to_datetime("2020-02-01"),
        pd.to_datetime("2020-07-01"),
        pd.to_datetime("2021-01-01"),
        pd.to_datetime("2022-01-01"),
        pd.to_datetime("2024-01-01")
    ]
    df["first_coupon_date"] = np.random.choice(possible_dates, size=N)

    # Save to Excel
    df.to_excel(output_file, index=False)
    print(f"Synthetic data saved to {output_file}")


if __name__ == "__main__":
    generate_synthetic_data()