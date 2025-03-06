"""
main_static_rl.py

This script trains a policy network in a supervised manner using static data
from the TF pricing model. It loads data from an Excel file, computes optimal
conversion decisions, trains a classifier to predict these decisions, and saves
the model. Training time is tracked and reported.
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
    Input: [stock price, time-to-maturity in years, estimated price]
    Output: Logits for two classes: 0 (hold) or 1 (convert)
    """

    def __init__(self, input_dim=3, hidden_dim=128, output_dim=2):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, output_dim)
        self._init_weights()

    def _init_weights(self):
        # Xavier initialization for better training stability
        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.zeros_(self.fc1.bias)
        nn.init.xavier_uniform_(self.fc2.weight)
        nn.init.zeros_(self.fc2.bias)
        nn.init.xavier_uniform_(self.fc3.weight)
        nn.init.zeros_(self.fc3.bias)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)  # Returns logits


def main():
    # Record start time to measure training duration
    start_time = time.time()

    # Load static data from TF pricing model
    data = pd.read_excel("data/model_price_all.xlsx")

    # Handle time-to-maturity (ttm)
    if 'ttm_days' not in data.columns:
        print("'ttm_days' column not found. Defaulting to 365 days.")
        data["ttm_days"] = 365.0
    data["ttm_years"] = data["ttm_days"] / 365.0  # Convert days to years

    # Compute conversion value correctly using per-row conversion price (cv)
    par = 100.0  # Par value of the bond
    data["conversion_value"] = data["S"] * (par / data["cv"])

    # Define labels: 1 (convert) if conversion value > estimated price, else 0 (hold)
    data["label"] = (data["conversion_value"] > data["Estimated_Price"]).astype(int)

    # Prepare features and labels
    features = data[["S", "ttm_years", "Estimated_Price"]].values.astype(np.float32)
    labels = data["label"].values.astype(np.int64)

    # Convert to PyTorch tensors
    X = torch.FloatTensor(features)
    y = torch.LongTensor(labels)

    # Initialize the network, loss function, and optimizer
    model = StaticPolicyNetwork(
        input_dim=3,
        hidden_dim=128,  # Increased from a smaller value (e.g., 32) for more capacity
        output_dim=2
    )
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    # Train the network
    epochs = 100  # Increased from a smaller number (e.g., 10) for better learning
    for epoch in range(epochs):
        optimizer.zero_grad()
        outputs = model(X)
        loss = criterion(outputs, y)
        loss.backward()
        optimizer.step()
        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch + 1}/{epochs}, Loss: {loss.item():.4f}")

    # Save the trained model
    torch.save(model.state_dict(), "data/static_policy_model.pth")
    print("Static policy model trained and saved as data/static_policy_model.pth")

    # Calculate and report training time
    train_time = time.time() - start_time
    print(f"Total training time: {train_time:.2f} seconds")


if __name__ == "__main__":
    main()