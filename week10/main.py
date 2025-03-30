"""
main.py

Main script that:
1. Loads the convertible-bond dataset.
2. Clusters data by time to maturity (TTM).
3. Chooses (dx, dt) based on cluster.
4. Runs the PDE solver for each row to estimate convertible bond price.
5. Saves the results to an Excel file.

Usage:
    python main.py
"""

import pandas as pd
import math
import numpy as np

from pde_solver import TsiveriotisFernandesSolver


# -----------------------------------------------------------------
# 1. Data Loading
# -----------------------------------------------------------------
def load_data(file_path, nrows=None):
    """
    Load and preprocess convertible bond data.

    Parameters:
    -----------
    file_path : str
        Path to the CSV file.
    nrows : int or None
        If not None, limit to 'nrows' rows (for testing).

    Returns:
    --------
    df : pd.DataFrame
        DataFrame with columns renamed to:
        ['Pr', 'IVOL', 'S', 'CDS', 'cp', 'cfq', 'cv', 'd', 'r', 'T'].
    """
    df = pd.read_csv(file_path)
    if nrows is not None:
        df = df.head(nrows)

    # Rename columns
    df = df.rename(columns={
        "bond_price":       "Pr",
        "BestIVOL":         "IVOL",
        "stock_price":      "S",
        "CDS":              "CDS",
        "coupon_rate":      "cp",
        "coupon_frequency": "cfq",
        "conversion_price": "cv",
        "dividend":         "d",
        "interest_rate":    "r"
    })

    # Convert to datetime
    df["first_coupon_date"] = pd.to_datetime(df["first_coupon_date"])
    df["issuance_date"]     = pd.to_datetime(df["issuance_date"])

    # Time to maturity in years
    df["T"] = df["ttm_days"] / 365.0

    return df


# -----------------------------------------------------------------
# 2. Clustering
# -----------------------------------------------------------------
def cluster_data(df, n_clusters=3):
    """
    Use qcut to cluster time to maturity (T) into n_clusters.
    Assign cluster labels in 'cluster' column (0 to n_clusters-1).

    Parameters:
    -----------
    df : pd.DataFrame
    n_clusters : int

    Returns:
    --------
    df : pd.DataFrame
        The DataFrame with an added 'cluster' column.
    """
    df["cluster"] = pd.qcut(df["T"], q=n_clusters, labels=False)
    return df


def calculate_tfc(row):
    """
    Calculate the fraction of coupon period until the first coupon.
    """
    cfq = row["cfq"]
    if cfq <= 0:
        return 0.0
    period_days = 360.0 / cfq
    coupon_days = (row["first_coupon_date"] - row["issuance_date"]).days
    if period_days <= 0:
        return 0.0
    return (coupon_days % period_days) / period_days


# -----------------------------------------------------------------
# 3. Grid Size Selection
# -----------------------------------------------------------------
def select_grid_sizes(cluster_id):
    """
    Decide on dx, dt based on cluster.
    For example:
    - cluster 0 -> coarser grid
    - cluster 1 -> medium grid
    - cluster 2 -> finer grid
    """
    if cluster_id == 0:
        return (0.01, 0.001)
    elif cluster_id == 1:
        return (0.005, 0.0005)
    else:
        return (0.002, 0.0002)


# -----------------------------------------------------------------
# 4. Price Estimation
# -----------------------------------------------------------------
def estimate_price_for_row(row):
    """
    For a single DataFrame row, pick dx/dt based on cluster,
    run the PDE solver, and interpolate to find the bond price.
    """
    # 1) compute time fraction to first coupon
    tfc = calculate_tfc(row)
    # 2) pick grid sizes
    dx, dt = select_grid_sizes(row["cluster"])
    # 3) PDE solver instance
    solver = TsiveriotisFernandesSolver(S_max=450.0, dx=dx, dt=dt)
    # 4) Solve PDE
    U_grid, V_grid, M, N = solver.solve(
        Pr    = row["Pr"],
        T     = row["T"],
        sigma = row["IVOL"] / 100.0,  # convert IVOL from % to decimal
        r     = row["r"],
        d     = row["d"],
        rc    = row["CDS"] / 10000.0, # convert CDS from bps to decimal
        cv    = row["cv"],
        cp    = row["cp"],
        cfq   = row["cfq"],
        tfc   = tfc
    )

    # 5) Interpolate on the grid
    if M < 1:
        return row["Pr"]  # fallback if something degenerate occurs

    actual_dx = math.log(450.0) / M
    s_index = int(round(math.log(row["S"]) / actual_dx))
    s_index = max(0, min(s_index, M))
    estimated_price = U_grid[s_index, 0]

    return estimated_price


def estimate_prices(df):
    """
    Loop through the DataFrame and estimate convertible bond prices (PDE-based).
    Adds an 'Estimated_Price' column to df.

    Parameters:
    -----------
    df : pd.DataFrame

    Returns:
    --------
    df : pd.DataFrame (in-place modification with new column)
    """
    est_prices = []
    for idx, row in df.iterrows():
        price_val = estimate_price_for_row(row)
        est_prices.append(price_val)
    df["Estimated_Price"] = est_prices
    return df


# -----------------------------------------------------------------
# 5. Main Entry
# -----------------------------------------------------------------
if __name__ == "__main__":
    # Example usage
    input_csv = "/path/to/synthetic_data_with_correlation.csv"
    df = load_data(input_csv, nrows=40000)

    # cluster data by T
    df = cluster_data(df, n_clusters=3)

    # estimate PDE-based prices
    df = estimate_prices(df)

    # Save results
    out_cols = ["T","ttm_days","Pr","IVOL","CDS","S","cp","cfq","cv","d","r","Estimated_Price","cluster"]
    output_excel = "/path/to/model_price_CN_dxdt.xlsx"
    df[out_cols].to_excel(output_excel, index=False)
    print(f"Done! Saved results to: {output_excel}")
