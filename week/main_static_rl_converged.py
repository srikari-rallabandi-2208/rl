"""
main_static_rl_converged.py

- Trains a feed-forward regressor to predict PDE price (tf_model_price)
  from the bond parameters (cds, ivol, stock_price, etc.).
- Reports RMSE & R^2 on the test set, saves a scatter plot of predicted vs. actual.

Usage:
  python main_static_rl_converged.py \
     --train data/prepared_40k/train.csv \
     --test data/prepared_40k/test.csv \
     --output data/prepared_40k
"""

import argparse
import os
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt


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


def run_static_regression(train_csv, test_csv, output_dir):
    # 1) Read data
    df_train = pd.read_csv(train_csv)
    df_test = pd.read_csv(test_csv)

    # 2) Feature columns:
    # Adjust to match your CSV. If you have 'div_yield', add it here.
    feature_cols = [
        "cds", "ivol", "stock_price", "bond_price", "r",
        "ttm_days", "coupon_rate", "coupon_freq", "conv_ratio", "conv_price"
    ]
    # We want to predict 'tf_model_price':
    target_col = "tf_model_price"

    # 3) Extract X, y
    X_train = df_train[feature_cols].astype(np.float32).values
    y_train = df_train[target_col].astype(np.float32).values.reshape(-1, 1)

    X_test = df_test[feature_cols].astype(np.float32).values
    y_test = df_test[target_col].astype(np.float32).values.reshape(-1, 1)

    print(f"[Static] Train shapes X={X_train.shape}, y={y_train.shape}")
    print(f"[Static] Test shapes  X={X_test.shape}, y={y_test.shape}")

    # 4) Model definition
    model = StaticRegressorNet(input_dim=len(feature_cols), hidden_dim=64)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    # Convert to Torch
    X_train_t = torch.from_numpy(X_train)
    y_train_t = torch.from_numpy(y_train)

    # 5) Train
    epochs = 20
    batch_size = 1024
    n_samples = len(X_train)
    for epoch in range(epochs):
        # We'll do mini-batch gradient descent
        permutation = np.random.permutation(n_samples)
        model.train()
        losses = []

        for i in range(0, n_samples, batch_size):
            idx = permutation[i:i + batch_size]
            batch_X = X_train_t[idx]
            batch_y = y_train_t[idx]

            optimizer.zero_grad()
            preds = model(batch_X)
            loss = criterion(preds, batch_y)
            loss.backward()
            optimizer.step()
            losses.append(loss.item())

        if (epoch + 1) % 5 == 0:
            avg_loss = np.mean(losses)
            print(f"Epoch {epoch + 1}/{epochs}, MSE={avg_loss:.4f}")

    # 6) Evaluate
    model.eval()
    X_test_t = torch.from_numpy(X_test)
    with torch.no_grad():
        y_pred_t = model(X_test_t)
    y_pred = y_pred_t.numpy().flatten()
    y_true = y_test.flatten()

    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_true, y_pred)
    print(f"[STATIC] RMSE={rmse:.4f}, R^2={r2:.4f}")

    # 7) Plot predicted vs. actual
    os.makedirs(output_dir, exist_ok=True)
    plt.figure(figsize=(6, 6))
    plt.scatter(y_true, y_pred, alpha=0.2)
    plt.xlabel("Actual PDE Price")
    plt.ylabel("Predicted PDE Price")
    plt.title("Static Regression: PDE Price")
    plt.grid(True)
    plot_path = os.path.join(output_dir, "static_pred_vs_actual.png")
    plt.savefig(plot_path, dpi=150)
    plt.close()
    print(f"Saved plot => {plot_path}")

    # 8) Save model
    model_path = os.path.join(output_dir, "static_regressor.pth")
    torch.save(model.state_dict(), model_path)
    print(f"Saved model => {model_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", required=True, help="Path to train.csv")
    parser.add_argument("--test", required=True, help="Path to test.csv")
    parser.add_argument("--output", required=True, help="Where to store plot/model.")
    args = parser.parse_args()

    run_static_regression(args.train, args.test, args.output)
