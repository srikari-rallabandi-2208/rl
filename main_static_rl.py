"""
main_static_rl.py

Trains a static (offline) RL policy via supervised learning:
1. Loads prepared data from 'data/model_price_all.xlsx',
2. Labels each row as convert(1)/hold(0) if the PDE-based 'Estimated_Price' is less than
   the immediate conversion value,
3. Trains a feed-forward classifier to replicate these labels,
4. Saves the trained model to disk.

Example usage:
    python main_static_rl.py
"""

import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import time


class StaticPolicyNetwork(nn.Module):
    """
    A feed-forward neural network classifier that predicts conversion decisions.
    Input: [stock price, time-to-maturity in years, estimated PDE price].
    Output: 2-class logits => 0 (hold), 1 (convert).
    """

    def __init__(self, input_dim=3, hidden_dim=128, output_dim=2):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, output_dim)
        self._init_weights()

    def _init_weights(self):
        # Initialize layers using Xavier uniform
        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.zeros_(self.fc1.bias)
        nn.init.xavier_uniform_(self.fc2.weight)
        nn.init.zeros_(self.fc2.bias)
        nn.init.xavier_uniform_(self.fc3.weight)
        nn.init.zeros_(self.fc3.bias)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)  # logits for 2 classes


def main():
    start_time = time.time()

    # 1. Load data from PDE-labeled file
    data = pd.read_excel("data/model_price_all.xlsx")

    # 2. Ensure we have time-to-maturity in years
    if "ttm_days" not in data.columns:
        print("'ttm_days' column not found, defaulting to 365 days.")
        data["ttm_days"] = 365
    data["ttm_years"] = data["ttm_days"] / 365.0

    # 3. Compute the conversion value
    #    For each row, if S*(par/cv) > Estimated_Price => label=1, else 0
    par = 100.0
    data["conversion_value"] = data["S"] * (par / data["cv"])

    # 4. Generate labels
    data["label"] = (data["conversion_value"] > data["Estimated_Price"]).astype(int)

    # 5. Prepare features [S, ttm_years, Estimated_Price] and labels
    features = data[["S", "ttm_years", "Estimated_Price"]].values.astype(np.float32)
    labels = data["label"].values.astype(np.int64)

    # 6. Create tensors
    X = torch.FloatTensor(features)
    y = torch.LongTensor(labels)

    # 7. Define the network, loss, and optimizer
    model = StaticPolicyNetwork(input_dim=3, hidden_dim=128, output_dim=2)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    epochs = 100
    for epoch in range(epochs):
        # Forward pass
        outputs = model(X)
        loss = criterion(outputs, y)

        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1}/{epochs}, Loss: {loss.item():.4f}")

    # 8. Save the static policy model
    torch.save(model.state_dict(), "data/static_policy_model.pth")
    print("Static policy model trained and saved to data/static_policy_model.pth")

    train_time = time.time() - start_time
    print(f"Total training time: {train_time:.2f} seconds")


if __name__ == "__main__":
    main()