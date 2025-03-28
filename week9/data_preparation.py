"""
data_preparation.py

Prepares a CSV for PDE-labeled convertible bond data:
    [CDS, IVOL, S, Pr, r, ttm_days, cp, cfq, conv_ratio, cv, d,
     issuance_date, first_coupon_date, Model_Price]

Steps:
 1) Reads the CSV
 2) (Optional) Rename columns, but here we assume your CSV matches these names exactly
 3) Filters out rows where 'Model_Price' is NaN, Inf, or beyond a threshold
 4) Shuffles
 5) Splits 80:20 into train.csv, test.csv

Usage:
    python data_preparation.py \
        --input data/final_40k.csv \
        --output-dir data/prepared_40k \
        --max-pde 1e7

Outputs:
    data/prepared_40k/train.csv
    data/prepared_40k/test.csv
"""

import argparse
import os
import pandas as pd
import numpy as np


def prepare_data(input_file, output_dir, max_pde=1e7):
    """
    Prepare PDE-labeled convertible bond data:
      - Filters out invalid or huge PDE values
      - Splits 80:20 into train/test
      - Saves results

    Parameters
    ----------
    input_file : str
        Path to the input CSV (must have Model_Price column).
    output_dir : str
        Directory to save train.csv, test.csv
    max_pde : float
        Threshold for outlier PDE. Rows with abs(Model_Price) > max_pde are dropped.
    """
    print(f"[DataPrep] Reading {input_file} ...")
    df = pd.read_csv(input_file)
    print(f"[DataPrep] Loaded {len(df)} rows.")

    # Your CSV presumably has columns:
    #   ['CDS','IVOL','S','Pr','r','ttm_days','cp','cfq','conv_ratio','cv','d',
    #    'issuance_date','first_coupon_date','Model_Price']

    # 1) Ensure the columns we expect are there
    required_cols = [
        "CDS", "IVOL", "S", "Pr", "r", "ttm_days", "cp", "cfq", "conv_ratio", "cv", "d",
        "issuance_date", "first_coupon_date", "Model_Price"
    ]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns {missing} in {input_file}. Found columns: {df.columns.tolist()}")

    # 2) Drop rows where Model_Price is NaN
    before = len(df)
    df = df.dropna(subset=["Model_Price"])
    dropped_nan = before - len(df)
    before = len(df)

    # 3) Drop Inf
    df = df[~df["Model_Price"].isin([np.inf, -np.inf])]
    dropped_inf = before - len(df)
    before = len(df)

    # 4) Drop PDE beyond ±max_pde
    df = df[df["Model_Price"].abs() <= max_pde]
    dropped_outliers = before - len(df)

    print(f"[DataPrep] Dropped {dropped_nan} rows with NaN PDE.")
    print(f"[DataPrep] Dropped {dropped_inf} rows with Inf PDE.")
    print(f"[DataPrep] Dropped {dropped_outliers} rows with PDE > ±{max_pde}.")

    # 5) Shuffle & split 80:20
    df = df.sample(frac=1.0, random_state=42).reset_index(drop=True)
    split_idx = int(0.8 * len(df))
    df_train = df.iloc[:split_idx].reset_index(drop=True)
    df_test = df.iloc[split_idx:].reset_index(drop=True)

    os.makedirs(output_dir, exist_ok=True)
    train_path = os.path.join(output_dir, "train.csv")
    test_path = os.path.join(output_dir, "test.csv")

    df_train.to_csv(train_path, index=False)
    df_test.to_csv(test_path, index=False)

    print(f"[DataPrep] Final train size={len(df_train)}, test size={len(df_test)}")
    print(f"[DataPrep] Saved train.csv => {train_path}")
    print(f"[DataPrep] Saved test.csv  => {test_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Prepare PDE-labeled convertible bond data for RL or static models.")
    parser.add_argument("--input", required=True,
                        help="Path to your CSV with [CDS,IVOL,S,Pr,r,ttm_days,cp,cfq,conv_ratio,cv,d,issuance_date,first_coupon_date,Model_Price].")
    parser.add_argument("--output-dir", required=True, help="Where to save train.csv and test.csv.")
    parser.add_argument("--max-pde", type=float, default=1e7, help="Rows with abs(Model_Price) > max-pde are dropped.")
    args = parser.parse_args()

    prepare_data(
        input_file=args.input,
        output_dir=args.output_dir,
        max_pde=args.max_pde
    )
