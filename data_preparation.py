"""
data_preparation.py

This module reads the synthetic data and renames columns to match the expected
input for the TF pricing model.
"""

# data_preparation.py
import pandas as pd


def prepare_synthetic_data(input_file="data/synthetic_data.xlsx", output_file="data/prepared_data.xlsx"):
    df = pd.read_excel(input_file)
    # Rename columns as needed by the TF model,
    # but keep ttm_days as-is (or rename it to something else consistently)
    df = df.rename(columns={
        "bond_price": "Pr",
        "ivol": "IVOL",
        "cds": "CDS",
        "stock_price": "S",
        "coupon_rate": "cp",
        "coupon_frequency": "cfq",
        "conversion_price": "cv",
        "dividend": "d",
        "interest_rate": "r"
    })
    # Ensure "ttm_days" is still in the DataFrame.
    df.to_excel(output_file, index=False)
    print(f"Prepared data saved to {output_file}")


if __name__ == "__main__":
    prepare_synthetic_data()
