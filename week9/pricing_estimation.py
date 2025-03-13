import math
import numpy as np
import pandas as pd
from pricing_model import explicit_FD


"""
pricing_estimation.py

This module reads the prepared data, applies the TF pricing model to compute
estimated convertible bond prices for each data point, and saves the output.
It now preserves the ttm_days column so that both ttm_days and Estimated_Price are in the output.
"""

import math
import numpy as np
import pandas as pd
from pricing_model import explicit_FD


def compute_estimated_prices(input_file="data/prepared_data.xlsx", output_file="data/model_price_all.xlsx"):
    synthetic_df = pd.read_excel(input_file)
    estimated_prices = []

    for idx, row in synthetic_df.iterrows():
        # Compute time-to-maturity in years.
        T = row["ttm_days"] / 365.0
        r = row["r"]
        cfq = row["cfq"]
        period = 360 / cfq

        # Calculate coupon time factor.
        coupon_days = (row["first_coupon_date"] - row["issuance_date"]).days
        CPT = (coupon_days % period) / T

        # Retrieve other parameters.
        d = row["d"]
        rc = row["CDS"] / 10000.0  # Scaling CDS appropriately.
        cv = row["cv"]
        cp = row["cp"]
        Pr = row["Pr"]
        sigma = row["IVOL"] / 100.0  # Use IVOL directly.
        S_val = row["S"]

        # Compute the pricing grid using the TF model; grid parameters: M=225, N=5000.
        # explicit_FD returns a tuple of (U_grid, V_grid, cb_value).
        u, v, estimated_price = explicit_FD(
            Pr, T, sigma, r, d, rc, cv, cp, cfq, CPT, 225, 5000,
            S_val  # Pass the current stock price as S0
        )

        # Append the estimated price returned from explicit_FD.
        estimated_prices.append(estimated_price)

    # Add the Estimated_Price column to the DataFrame.
    synthetic_df["Estimated_Price"] = estimated_prices

    # Include ttm_days along with other relevant columns.
    output_cols = ["Pr", "IVOL", "CDS", "S", "ttm_days", "cp", "cfq", "cv", "d", "r", "Estimated_Price"]
    output_df = synthetic_df[output_cols].copy()
    output_df.to_excel(output_file, index=False)
    print(f"Estimated prices for all samples saved to {output_file}")


if __name__ == "__main__":
    compute_estimated_prices()