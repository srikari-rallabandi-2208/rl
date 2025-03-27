"""
main_static_regression.py

Trains a feed-forward net to predict PDE "Model_Price" from other columns.
Plots predicted vs actual. Good for 40K, 100K, or 300K.

Usage:
  python main_static_regression.py \
    --input data/final_40k.csv \
    --output data/static_40k
"""

import argparse
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
import os

class StaticRegressorNet(nn.Module):
    def __init__(self, input_dim=10, hidden_dim=64):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)

def run_static_regression(input_csv, output_dir):
    df = pd.read_csv(input_csv)
    os.makedirs(output_dir, exist_ok=True)

    # Shuffle
    df = df.sample(frac=1.0, random_state=42).reset_index(drop=True)

    # 80:20 split
    split_idx = int(0.8 * len(df))
    df_train = df.iloc[:split_idx]
    df_test = df.iloc[split_idx:]

    # Feature columns: (adjust to match your CSV)
    # We'll exclude Model_Price from features
    # We'll assume these columns exist: cds, IVOL, S, r, d, cv, etc.
    feature_cols = ["CDS", "IVOL", "S", "r", "d", "cv", "Pr", "coupon_rate", "coupon_frequency", "ttm_days"]
    target_col = "Model_Price"

    # Convert to np.float32
    X_train = df_train[feature_cols].astype(np.float32).values
    y_train = df_train[target_col].astype(np.float32).values.reshape(-1,1)

    X_test = df_test[feature_cols].astype(np.float32).values
    y_test = df_test[target_col].astype(np.float32).values.reshape(-1,1)

    print(f"[Static] Train shapes: X={X_train.shape}, y={y_train.shape}")
    print(f"[Static] Test shapes:  X={X_test.shape}, y={y_test.shape}")

    # Model
    model = StaticRegressorNet(input_dim=len(feature_cols), hidden_dim=64)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    X_train_t = torch.from_numpy(X_train)
    y_train_t = torch.from_numpy(y_train)

    # Train
    epochs = 20
    batch_size = 1024
    n_samples = len(X_train)
    for epoch in range(epochs):
        permutation = np.random.permutation(n_samples)
        losses = []
        model.train()
        for i in range(0, n_samples, batch_size):
            idx = permutation[i:i+batch_size]
            batch_X = X_train_t[idx]
            batch_y = y_train_t[idx]

            optimizer.zero_grad()
            preds = model(batch_X)
            loss = criterion(preds, batch_y)
            loss.backward()
            optimizer.step()
            losses.append(loss.item())

        if (epoch+1) % 5 == 0:
            print(f"[Epoch {epoch+1}/{epochs}] MSE={np.mean(losses):.4f}")

    # Evaluate
    model.eval()
    X_test_t = torch.from_numpy(X_test)
    with torch.no_grad():
        y_pred_t = model(X_test_t)
    y_pred = y_pred_t.numpy().flatten()
    y_true = y_test.flatten()

    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_true, y_pred)
    print(f"[Static] RMSE={rmse:.4f}, R^2={r2:.4f}")

    # Plot
    plt.figure(figsize=(6,6))
    plt.scatter(y_true, y_pred, alpha=0.2)
    plt.xlabel("Actual PDE Price")
    plt.ylabel("Predicted PDE Price")
    plt.title("Static Regression PDE")
    plt.grid(True)
    plot_path = os.path.join(output_dir, "static_pred_vs_actual.png")
    plt.savefig(plot_path, dpi=150)
    plt.close()
    print(f"[Static] Plot => {plot_path}")

    model_path = os.path.join(output_dir, "static_regressor.pth")
    torch.save(model.state_dict(), model_path)
    print(f"[Static] Model => {model_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="CSV with Model_Price")
    parser.add_argument("--output", required=True, help="Folder for plots/models")
    args = parser.parse_args()

    run_static_regression(args.input, args.output)
