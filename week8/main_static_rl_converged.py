"""
main_static_rl_converged.py

1) Generate a grid of (S, t) values using the converged PDE engine,
2) Label them (convert=1 if conversion_value > PDEprice, else 0),
3) Train a feed-forward network to replicate the PDE-based decision boundary.
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim

from pde_convergence import PDEConvergenceChecker
from tf_engine_converged import TsiveriotisFernandesEngineConverged


class StaticPolicyNet(nn.Module):
    def __init__(self, input_dim=3, hidden_dim=64, output_dim=2):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)
        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.zeros_(self.fc1.bias)
        nn.init.xavier_uniform_(self.fc2.weight)
        nn.init.zeros_(self.fc2.bias)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        return self.fc2(x)


def main():
    # 1. Use PDEConvergenceChecker or known stable (M,N)
    S0, K, r, sigma, T, q, spread, par = 100.0, 100.0, 0.05, 0.20, 1.0, 0.0, 0.01, 100.0
    stable_grid = (200, 200)  # Suppose we found this works from prior script
    M_star, N_star = stable_grid

    engine = TsiveriotisFernandesEngineConverged(S0, K, r, sigma, T, q, spread, M_star, N_star, par)
    engine.solve_pde()

    # 2. Generate a dataset of (S, t). E.g., 100 points in S from 1..200, 21 points in t from 0..1
    S_vals = np.linspace(1, 200, 100)
    t_vals = np.linspace(0.0, 1.0, 21)

    data = []
    for t in t_vals:
        for S in S_vals:
            cb_price = engine.get_cb_value(S, t)
            # Conversion value = S*(par/K)
            conv_val = S * (par / K)
            label = 1 if conv_val > cb_price else 0
            data.append([S, t, cb_price, label])

    df = pd.DataFrame(data, columns=["S", "t", "pde_price", "label"])
    df.to_csv("data/converged_dataset.csv", index=False)
    print("Converged PDE dataset saved to data/converged_dataset.csv")

    # 3. Train a classifier
    X_np = df[["S", "t", "pde_price"]].values.astype(np.float32)
    y_np = df["label"].values.astype(np.int64)

    X_t = torch.from_numpy(X_np)
    y_t = torch.from_numpy(y_np)

    model = StaticPolicyNet(input_dim=3, hidden_dim=64, output_dim=2)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(50):
        y_pred = model(X_t)
        loss = criterion(y_pred, y_t)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch + 1}/50, Loss={loss.item():.4f}")

    torch.save(model.state_dict(), "data/static_policy_converged.pth")
    print("Static policy model trained on converged PDE data, saved to data/static_policy_converged.pth")


if __name__ == "__main__":
    main()
