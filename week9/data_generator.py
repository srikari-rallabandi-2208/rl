"""
data_generator.py

Generates synthetic convertible-bond data with PDE-based "Model_Price."

Steps:
1. Read a large CSV (like the one with correlated columns).
2. Shuffle it.
3. Use PDEEngine to compute PDE price for each row, skipping invalid ones.
4. Stop once we get a desired count (e.g., 40K, 100K, 300K).
5. Save to a new CSV.

Example usage:
  python data_generator.py \
    --input data/synthetic_data_with_correlation.csv \
    --output data/final_40k.csv \
    --num-rows 40000
"""

import argparse
import pandas as pd
from pde_engine import PDEEngine
import math

def generate_dataset(input_csv, output_csv, num_rows, M=700, N=15000):
    # 1) Read & shuffle
    df = pd.read_csv(input_csv)
    df = df.sample(frac=1.0, random_state=42).reset_index(drop=True)

    print(f"[DataGen] Loaded {len(df)} rows from {input_csv}, now computing PDE on up to {num_rows} rows.")

    # 2) PDE engine
    engine = PDEEngine(M=M, N=N)

    # 3) Solve PDE row by row until we have 'num_rows' valid results
    df_result = engine.compute_pde_prices(df, max_rows=num_rows)

    print(f"[DataGen] PDE done. We got {len(df_result)} valid PDE solutions.")
    df_result.to_csv(output_csv, index=False)
    print(f"[DataGen] Wrote final CSV => {output_csv}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="Path to the raw big CSV with correlated columns.")
    parser.add_argument("--output", required=True, help="Where to write the final CSV with Model_Price.")
    parser.add_argument("--num-rows", type=int, default=40000, help="Desired number of PDE solutions (40K, 100K, 300K, etc.)")
    parser.add_argument("--M", type=int, default=700, help="PDE grid spatial steps.")
    parser.add_argument("--N", type=int, default=15000, help="PDE grid time steps.")
    args = parser.parse_args()

    generate_dataset(args.input, args.output, args.num_rows, M=args.M, N=args.N)
