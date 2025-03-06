"""
data_preparation.py

This module reads the synthetic data and renames columns to match the expected
input for a convertible bond pricing model, then saves the prepared data as an Excel file.
"""

import pandas as pd


def prepare_synthetic_data(input_file="data/synthetic_data.xlsx", output_file="data/prepared_data.xlsx"):
    """
    Prepare synthetic data by renaming columns for pricing model compatibility.

    Parameters:
        input_file (str): Path to the input Excel file (default: "data/synthetic_data.xlsx").
        output_file (str): Path to save the prepared Excel file (default: "data/prepared_data.xlsx").

    Returns:
        None: Saves the prepared data to an Excel file.
    """
    # Read the synthetic data
    df = pd.read_excel(input_file)

    # Rename columns to match typical pricing model expectations
    df = df.rename(columns={
        "bond_price": "Pr",          # Bond price
        "ivol": "IVOL",              # Implied volatility
        "cds": "CDS",                # Credit default spread
        "stock_price": "S",          # Stock price
        "interest_rate": "r",        # Risk-free rate
        "ttm_days": "ttm_days",      # Time-to-maturity in days (kept as-is)
        "coupon_rate": "cp",         # Coupon rate
        "coupon_frequency": "cfq",   # Coupon frequency
        "conversion_price": "cv",    # Conversion price
        "dividend": "d"              # Dividend yield
    })

    # Note: "conversion_ratio" and date columns ("issuance_date", "first_coupon_date") are kept unchanged
    # as they may be used differently depending on the model

    # Save the prepared data
    df.to_excel(output_file, index=False)
    print(f"Prepared data saved to {output_file}")


if __name__ == "__main__":
    prepare_synthetic_data()