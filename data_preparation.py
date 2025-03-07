"""
data_preparation.py

Reads the synthetic data and renames columns to match the expected
input for a convertible bond pricing model, then saves the prepared data as an Excel file.

Example usage:
    python data_preparation.py
"""

import pandas as pd


def prepare_synthetic_data(input_file="data/synthetic_data.xlsx", output_file="data/prepared_data.xlsx"):
    """
    Prepare synthetic data by renaming columns for the pricing model's compatibility.

    Parameters:
    -----------
    input_file : str
        Path to the input Excel file containing raw synthetic data.
    output_file : str
        Path to save the prepared Excel file.

    Returns:
    --------
    None
        The resulting DataFrame is saved to an Excel file at 'output_file'.
    """
    # 1. Read the synthetic data from Excel
    df = pd.read_excel(input_file)

    # 2. Rename columns to match typical convertible bond pricing model inputs
    df = df.rename(columns={
        "bond_price": "Pr",          # Bond price
        "ivol": "IVOL",              # Implied volatility (as a percentage, e.g. 30 => 30%)
        "cds": "CDS",                # Credit default spread (in basis points)
        "stock_price": "S",          # Underlying stock price
        "interest_rate": "r",        # Risk-free rate
        "ttm_days": "ttm_days",      # Keep time-to-maturity in days
        "coupon_rate": "cp",         # Coupon rate
        "coupon_frequency": "cfq",   # Coupon frequency (2 => semiannual, 4 => quarterly, etc.)
        "conversion_price": "cv",    # Conversion price
        "dividend": "d"              # Dividend yield
    })
    # 'conversion_ratio', 'issuance_date', and 'first_coupon_date' remain unchanged but stay in the DataFrame.

    # 3. Save the prepared data
    df.to_excel(output_file, index=False)
    print(f"Prepared data saved to {output_file}")


if __name__ == "__main__":
    prepare_synthetic_data()