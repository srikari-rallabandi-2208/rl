"""
eval_comparison.py

Evaluates:
1) Training time for static RL (offline) by re-running main_static_rl.py.
2) Prediction time for the trained static RL model on the entire dataset.
3) Classification accuracy (since it's a 0/1 model).
4) (Optional) RMSE, R^2 if you define a numeric target.

Notes:
- We do not rely on .npy or .pkl files.
- We measure times using time.perf_counter().
- For classification, we compute accuracy. (RMSE, R² are included only if relevant.)
"""

import time
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score  # For classification
# If you really want RMSE, R^2 for a numeric target:
from sklearn.metrics import mean_squared_error, r2_score

# ---------------------------------------------------------------------
# 1) Re-run main_static_rl.py, measuring training time
# ---------------------------------------------------------------------
print("=== Measuring Train Time for Static RL ===")

start_train = time.perf_counter()

# You can directly import and call `main()` from main_static_rl.py
# or you could do a subprocess call. Importing is usually simpler:
import main_static_rl
main_static_rl.main()  # This trains and saves model to data/static_policy_model.pth

end_train = time.perf_counter()
train_time = end_train - start_train
print(f"[OK] Static RL training completed in {train_time:.4f} seconds.")

# ---------------------------------------------------------------------
# 2) Load the trained static RL model
# ---------------------------------------------------------------------
print("\n=== Loading Trained Static RL Model ===")

# We need the same definition of StaticPolicyNetwork to load the state dict.
class StaticPolicyNetwork(nn.Module):
    """
    Same architecture as in main_static_rl.py
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

model = StaticPolicyNetwork()
saved_state = torch.load("data/static_policy_model.pth")
model.load_state_dict(saved_state)
model.eval()  # set to evaluation mode
print("[OK] Model state loaded from data/static_policy_model.pth")

# ---------------------------------------------------------------------
# 3) Re-create test features (X) and labels (y) from model_price_all.xlsx
# ---------------------------------------------------------------------
print("\n=== Preparing Test Data ===")
df = pd.read_excel("data/model_price_all.xlsx")

# If 'ttm_days' or 'Estimated_Price' are missing, something is off in your pipeline
if 'ttm_days' not in df.columns or 'Estimated_Price' not in df.columns:
    raise ValueError("Expected 'ttm_days' or 'Estimated_Price' missing in model_price_all.xlsx.")

# Convert ttm_days -> ttm_years, as done in main_static_rl
df["ttm_years"] = df["ttm_days"] / 365.0

# Recompute "conversion_value" and "label" the same way as main_static_rl
K = 100.0
par = 100.0
df["conversion_value"] = (df["S"] / K) * par
df["label"] = (df["conversion_value"] > df["Estimated_Price"]).astype(int)

# Build feature matrix: [S, ttm_years, Estimated_Price]
X_data = df[["S", "ttm_years", "Estimated_Price"]].values.astype(np.float32)
y_data = df["label"].values.astype(np.int64)

# ---------------------------------------------------------------------
# 4) Measure Prediction Time, get Predictions
# ---------------------------------------------------------------------
print("\n=== Measuring Prediction Time and Evaluating ===")

start_pred = time.perf_counter()

# We'll do a forward pass in mini-batches if the dataset is large.
# For demonstration, let's just do it in one shot:
X_tensor = torch.from_numpy(X_data)
with torch.no_grad():
    logits = model(X_tensor)  # shape = [N, 2]
    probs = torch.softmax(logits, dim=1)  # shape = [N, 2]
    # Predicted class = argmax
    predictions = torch.argmax(probs, dim=1).numpy()

end_pred = time.perf_counter()
prediction_time = end_pred - start_pred

print(f"[OK] Prediction time: {prediction_time:.4f} seconds for {len(X_data)} samples.")

# ---------------------------------------------------------------------
# 5) Compute classification accuracy
# ---------------------------------------------------------------------
accuracy = accuracy_score(y_data, predictions)
print(f"Classification Accuracy: {accuracy:.4f}")

# ---------------------------------------------------------------------
# 6) (Optional) RMSE / R² if you define a numeric target
#    Since your model is predicting a 0/1 label, RMSE and R² are
#    not meaningful unless you truly have a numeric target.
#    Here's how you'd do it IF you had PDE price as `y_true` and the
#    model predicted some numeric price. For now, we skip it.
# ---------------------------------------------------------------------
# Example (fake) demonstration of how you'd do it if you had numeric targets:
# numeric_target = df["Estimated_Price"].values  # just an example
# predicted_values = some_regression_output_from_model  # must be numeric
# rmse = mean_squared_error(numeric_target, predicted_values, squared=False)
# r2 = r2_score(numeric_target, predicted_values)
# print(f"RMSE: {rmse:.4f}, R^2: {r2:.4f}")

print("\n=== Final Summary ===")
print(f"Train Time  : {train_time:.4f} seconds")
print(f"Predict Time: {prediction_time:.4f} seconds")
print(f"Accuracy    : {accuracy:.4f} (for 0/1 classification)")
print("Done.")