"""
data_preparation.py

- Reads a CSV (e.g., synthetic_data_40k_with_model_price.csv or 3M CSV).
- Optionally renames columns to standard names (cds, ivol, stock_price, etc.).
- Filters out huge or infinite 'tf_model_price' if present.
- Splits 80:20 into train.csv and test.csv.

Usage:
  python data_preparation.py \
      --input data/synthetic_data_40k_with_model_price.csv \
      --output-dir data/prepared_40k

Outputs:
  data/prepared_40k/train.csv
  data/prepared_40k/test.csv
"""

import argparse
import os
import pandas as pd


def prepare_data(input_file, output_dir, has_model_price=True):
    print(f"[DataPrep] Reading {input_file} ...")
    df = pd.read_csv(input_file)
    print(f"[DataPrep] Loaded {len(df)} rows.")

    # 1) Rename columns to standard naming if needed.
    rename_map = {
        "CDS": "cds",
        "BestIVOL": "ivol",
        "S": "stock_price",
        "bond_price": "bond_price",
        "interest_rate": "r",
        "ttm_days": "ttm_days",
        "coupon_rate": "coupon_rate",
        "coupon_frequency": "coupon_freq",
        "conversion_ratio": "conv_ratio",
        "conversion_price": "conv_price",
        "dividend": "div_yield",
        "issuance_date": "issuance_date",
        "first_coupon_date": "first_coupon_date",
    }
    if has_model_price:
        rename_map["Model_Price"] = "tf_model_price"

    # Rename in place, ignoring any columns that don't exist
    df.rename(columns=rename_map, inplace=True, errors="ignore")

    # 2) Filter out invalid PDE values if the user says we have a PDE column
    if has_model_price and "tf_model_price" in df.columns:
        before = len(df)

        # Drop NaN
        df = df.dropna(subset=["tf_model_price"])
        dropped_nan = before - len(df)
        before = len(df)

        # Drop infinite
        df = df[~df["tf_model_price"].isin([float('inf'), float('-inf')])]
        dropped_inf = before - len(df)
        before = len(df)

        # Drop extremely large
        threshold = 1e7
        df = df[df["tf_model_price"].abs() < threshold]
        dropped_large = before - len(df)

        print(f"[DataPrep] Dropped {dropped_nan} rows with NaN PDE.")
        print(f"[DataPrep] Dropped {dropped_inf} rows with Inf PDE.")
        print(f"[DataPrep] Dropped {dropped_large} rows with PDE > ±{threshold}.")

    # 3) Shuffle and split 80:20
    df = df.sample(frac=1.0, random_state=42).reset_index(drop=True)
    split_idx = int(0.8 * len(df))
    df_train = df.iloc[:split_idx].reset_index(drop=True)
    df_test = df.iloc[split_idx:].reset_index(drop=True)

    # 4) Save to output_dir
    os.makedirs(output_dir, exist_ok=True)
    train_path = os.path.join(output_dir, "train.csv")
    test_path = os.path.join(output_dir, "test.csv")

    df_train.to_csv(train_path, index=False)
    df_test.to_csv(test_path, index=False)

    print(f"[DataPrep] Final train size={len(df_train)}, test size={len(df_test)}")
    print(f"[DataPrep] Saved train.csv => {train_path}")
    print(f"[DataPrep] Saved test.csv  => {test_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="Path to your CSV.")
    parser.add_argument("--output-dir", required=True, help="Destination folder.")
    parser.add_argument("--no-model-price", action="store_true",
                        help="If the CSV doesn't have a PDE price column.")
    args = parser.parse_args()

    prepare_data(
        input_file=args.input,
        output_dir=args.output_dir,
        has_model_price=not args.no_model_price
    )
