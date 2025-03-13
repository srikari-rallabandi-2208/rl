"""
main_static_rl_converged.py

Generates (S, t) data using a converged PDE engine and trains a static (offline) RL
classifier to replicate "convert vs. hold" decisions.

Steps:
1) Create & solve a TsiveriotisFernandesEngineConverged with chosen (M, N).
2) Sample multiple time points (e.g., t in [0,1]) and stock prices (S in [1..200]).
3) Compute PDE price => label = 1 if immediate conversion value > PDE, else 0.
4) Train a feed-forward classifier to learn that decision boundary.
5) Save the static policy model to disk.

Usage:
    python main_static_rl_converged.py

Outputs:
    - data/converged_dataset.csv
    - data/static_policy_converged.pth
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import time

from tf_engine_converged import TsiveriotisFernandesEngineConverged

class StaticPolicyNet(nn.Module):
    """
    A feed-forward classifier that predicts (convert=1, hold=0).
    Input: [S, t, PDE_price] => Output: 2-class logits
    """
    def __init__(self, input_dim=3, hidden_dim=64, output_dim=2):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)
        self._init_weights()

    def _init_weights(self):
        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.zeros_(self.fc1.bias)
        nn.init.xavier_uniform_(self.fc2.weight)
        nn.init.zeros_(self.fc2.bias)

    def forward(self, x):
        # Basic 2-layer net with ReLU
        x = torch.relu(self.fc1(x))
        return self.fc2(x)


def main():
    start_time = time.time()

    # ------------------------------------------------------
    # (1) Create a converged PDE engine with chosen grid
    # ------------------------------------------------------
    # Suppose we already decided that M=200, N=200 is "good enough" from pde_convergence.py
    engine = TsiveriotisFernandesEngineConverged(
        S0=100.0,      # Not crucial for static data gen, just the PDE's reference
        K=100.0,       # Conversion price
        r=0.05,        # Risk-free rate
        sigma=0.20,    # Volatility
        T=1.0,         # Time to maturity (1 year)
        q=0.0,         # Dividend yield
        spread=0.01,   # Credit spread
        M=200,         # PDE grid: stock dimension
        N=200,         # PDE grid: time dimension
        par=100.0      # Par value
    )
    engine.solve_pde()
    print(f"[PDE] Converged CB price at S=100, t=0 => {engine.cb_value_0:.4f}")

    # ------------------------------------------------------
    # (2) Generate a range of times & stock prices
    # ------------------------------------------------------
    # For instance: 21 time steps in [0..1], 50 stock steps in [1..200]
    t_vals = np.linspace(0.0, 1.0, 21)
    S_vals = np.linspace(1.0, 200.0, 50)

    # We'll store each row as (S, t, pde_price, label)
    data_rows = []
    par = 100.0  # par value, used in conversion_value
    for t in t_vals:
        for S in S_vals:
            # PDE price at (S,t)
            cb_price = engine.get_cb_value(S, t)
            # immediate conversion value = S*(par/K) => if par=100, K=100 => conv_val = S*(100/100)=S
            conversion_value = S
            # label => 1 if conv_val > PDE, else 0
            label = 1 if conversion_value > cb_price else 0
            data_rows.append((S, t, cb_price, label))

    df = pd.DataFrame(data_rows, columns=["S", "t", "pde_price", "label"])
    df.to_csv("data/converged_dataset.csv", index=False)
    print(f"[Data] Wrote {len(df)} samples to data/converged_dataset.csv")

    # ------------------------------------------------------
    # (3) Train a static classifier
    # ------------------------------------------------------
    # Features => [S, t, pde_price], label => convert/hold
    X_np = df[["S", "t", "pde_price"]].values.astype(np.float32)
    y_np = df["label"].values.astype(np.int64)

    X_t = torch.from_numpy(X_np)
    y_t = torch.from_numpy(y_np)

    model = StaticPolicyNet(input_dim=3, hidden_dim=64, output_dim=2)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    epochs = 50
    for epoch in range(epochs):
        # Forward
        logits = model(X_t)
        loss = criterion(logits, y_t)
        # Backward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (epoch + 1) % 10 == 0:
            print(f"[Training] Epoch {epoch+1}/{epochs}, Loss={loss.item():.4f}")

    # ------------------------------------------------------
    # (4) Save the trained model
    # ------------------------------------------------------
    torch.save(model.state_dict(), "data/static_policy_converged.pth")
    print("[Model] Saved static policy to data/static_policy_converged.pth")

    # Done
    elapsed = time.time() - start_time
    print(f"[Done] total run time: {elapsed:.2f} sec")


if __name__ == "__main__":
    main()
