"""
main_static_rl.py

This script trains a policy network in a supervised manner using static data
produced by the TF pricing model. It loads prepared_data.xlsx, which contains
the original columns including ttm_days, computes an 'optimal'
conversion decision for each data point, and trains a classifier to replicate
that decision. The trained model is saved to the data folder.
"""

import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim


class StaticPolicyNetwork(nn.Module):
    """
    A simple feed-forward classifier network that takes [S, ttm_years, Estimated_Price]
    as input and outputs logits for two classes: 0 (hold) or 1 (convert).
    """

    def __init__(self, input_dim=3, hidden_dim=32, output_dim=2):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, output_dim)
        self._init_weights()

    def _init_weights(self):
        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.zeros_(self.fc1.bias)
        nn.init.xavier_uniform_(self.fc2.weight)
        nn.init.zeros_(self.fc2.bias)
        nn.init.xavier_uniform_(self.fc3.weight)
        nn.init.zeros_(self.fc3.bias)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)


def main():
    # 1) Load static data produced by the TF pricing model.
    #    Use prepared_data.xlsx because it contains ttm_days.
    data_file = "data/model_price_all.xlsx"
    data = pd.read_excel(data_file)

    # 2) Ensure ttm_days is present. (It should be.)
    if 'ttm_days' not in data.columns:
        print("'ttm_days' column not found in prepared_data.xlsx. Setting default ttm_days = 365.")
        data["ttm_days"] = 365.0

    # 3) Convert ttm_days to ttm_years.
    data["ttm_years"] = data["ttm_days"] / 365.0

    # 4) Compute an 'optimal' conversion decision.
    #    We define conversion value as (S / K) * par, with K = 100 and par = 100.
    K = 100.0
    par = 100.0
    data["conversion_value"] = (data["S"] / K) * par

    # 5) Create labels: if conversion_value > Estimated_Price then label 1 (convert), else 0 (hold).
    #    If Estimated_Price is not present in prepared_data.xlsx, you might need to merge it from model_price_all.xlsx.
    #    For now, we'll assume prepared_data.xlsx includes Estimated_Price.
    if "Estimated_Price" not in data.columns:
        print("Estimated_Price column not found in prepared_data.xlsx. Cannot compute labels.")
        return

    data["label"] = (data["conversion_value"] > data["Estimated_Price"]).astype(int)

    # 6) Define state vector: [S, ttm_years, Estimated_Price]
    features = data[["S", "ttm_years", "Estimated_Price"]].values.astype(np.float32)
    labels = data["label"].values.astype(np.int64)

    # 7) Create PyTorch tensors.
    X = torch.FloatTensor(features)
    y = torch.LongTensor(labels)

    # 8) Initialize the network, loss, and optimizer.
    model = StaticPolicyNetwork(input_dim=3, hidden_dim=32, output_dim=2)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    # 9) Train the network.
    epochs = 10  # Increase as needed.
    for epoch in range(epochs):
        optimizer.zero_grad()
        outputs = model(X)
        loss = criterion(outputs, y)
        loss.backward()
        optimizer.step()
        print(f"Epoch {epoch + 1}/{epochs}, Loss: {loss.item():.4f}")

    # 10) Save the trained model to the data folder.
    model_save_path = "data/static_policy_model.pth"
    torch.save(model.state_dict(), model_save_path)
    print(f"Static policy model trained and saved as {model_save_path}")


if __name__ == "__main__":
    main()
