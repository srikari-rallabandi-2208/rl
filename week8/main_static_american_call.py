"""
Trains a simple feed-forward neural network to 'regress' the American call price
from the parameters [S, K, r, q, sigma, T].
"""

import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import os


class StaticPriceNet(nn.Module):
    """
    A feed-forward net that predicts the American call price.
    Input: [S, K, r, q, sigma, T]
    Output: single float (predicted price).
    """

    def __init__(self, input_dim=6, hidden_dim=64):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, 1)

        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.zeros_(self.fc1.bias)
        nn.init.xavier_uniform_(self.fc2.weight)
        nn.init.zeros_(self.fc2.bias)
        nn.init.xavier_uniform_(self.fc3.weight)
        nn.init.zeros_(self.fc3.bias)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)  # shape: [batch_size, 1]


def main():
    # 1) Load dataset
    df = pd.read_csv("data/american_call_data.csv")
    # Drop any possibly weird values
    df = df[df["AmericanPrice"] >= 0].reset_index(drop=True)

    # 2) Features + label
    X = df[["S", "K", "r", "q", "sigma", "T"]].values.astype(np.float32)
    y = df["AmericanPrice"].values.astype(np.float32)
    y = y.reshape(-1, 1)  # shape: [N, 1]

    # 3) Torch Tensors
    X_tensor = torch.from_numpy(X)
    y_tensor = torch.from_numpy(y)

    # 4) Define model, loss, optimizer
    model = StaticPriceNet(input_dim=6, hidden_dim=64)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    # 5) Train loop
    n_epochs = 20
    batch_size = 1024
    dataset_size = len(df)
    num_batches = dataset_size // batch_size

    for epoch in range(n_epochs):
        # Shuffle indices
        perm = torch.randperm(dataset_size)
        epoch_loss = 0.0

        for b_i in range(num_batches):
            batch_idx = perm[b_i * batch_size: (b_i + 1) * batch_size]
            X_batch = X_tensor[batch_idx]
            y_batch = y_tensor[batch_idx]

            optimizer.zero_grad()
            preds = model(X_batch)
            loss = criterion(preds, y_batch)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

        epoch_loss /= num_batches
        print(f"Epoch {epoch + 1}/{n_epochs}, MSE: {epoch_loss:.4f}")

    # 6) Save the model
    os.makedirs("data", exist_ok=True)
    save_path = "data/static_model.pth"
    torch.save(model.state_dict(), save_path)
    print(f"Static price model saved to {save_path}")


if __name__ == "__main__":
    main()
