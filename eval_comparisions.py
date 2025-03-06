"""
eval_comparisons.py

This script evaluates and compares three methods:
  1. Tsiveriotis-Fernandes (TF) PDE-based convertible bond prices.
  2. Online RL agent's conversion decisions.
  3. Static RL learner's predictions.

It simulates a stock path, computes PDE prices, loads trained models, and generates
comparison plots. Metrics include prediction time, RMSE, and R² for both RL models.
"""

import numpy as np
import pandas as pd
import torch
import time
from tf_engine import TsiveriotisFernandesEngine  # Assumed dependency
from environment import ConvertibleBondEnvTF  # Assumed dependency
from rl_agent import PolicyGradientAgent  # Assumed dependency
from main_static_rl import StaticPolicyNetwork  # Import from main_static_rl.py
from visualizations import plot_comparison_online_rl, plot_comparison_static  # Assumed dependency
from sklearn.metrics import mean_squared_error, r2_score


def generate_gbm_stock_path(S0, mu, sigma, T, num_steps):
    """
    Generate a stock price path using Geometric Brownian Motion (GBM).

    Parameters:
        S0 (float): Initial stock price.
        mu (float): Drift rate.
        sigma (float): Volatility.
        T (float): Time horizon in years.
        num_steps (int): Number of time steps.

    Returns:
        stock_path (np.ndarray): Simulated stock price path.
        times (np.ndarray): Time points corresponding to the stock path.
    """
    dt = T / (num_steps - 1)
    times = np.linspace(0, T, num_steps)
    stock_path = np.zeros(num_steps)
    stock_path[0] = S0
    for t in range(1, num_steps):
        Z = np.random.normal()
        stock_path[t] = stock_path[t - 1] * np.exp((mu - 0.5 * sigma ** 2) * dt + sigma * np.sqrt(dt) * Z)
    return stock_path, times


def evaluate_online_rl(engine, agent, stock_path, times):
    """
    Evaluate the online RL agent over a stock path, returning decisions and prices.

    Parameters:
        engine (TsiveriotisFernandesEngine): PDE engine for pricing.
        agent (PolicyGradientAgent): Trained online RL agent.
        stock_path (np.ndarray): Simulated stock price path.
        times (np.ndarray): Time points corresponding to the stock path.

    Returns:
        online_times (np.ndarray): Times at each step.
        online_stock_prices (np.ndarray): Stock prices at each step.
        pde_prices (np.ndarray): PDE-based bond prices at each step.
        rl_decisions (np.ndarray): RL agent's decisions (0: hold, 1: convert).
    """
    env = ConvertibleBondEnvTF(engine, stock_path, times)
    state = env.reset()
    online_stock_prices = []
    online_times = []
    pde_prices = []
    rl_decisions = []
    done = False
    while not done:
        S, t, pde_price = state
        online_stock_prices.append(S)
        online_times.append(t)
        pde_prices.append(pde_price)
        action = agent.select_action(state)  # Assumes NaN handling in select_action
        rl_decisions.append(action)
        state, _, done, _ = env.step(action)
    return np.array(online_times), np.array(online_stock_prices), np.array(pde_prices), np.array(rl_decisions)


def evaluate_static_rl(static_model, data):
    """
    Evaluate the static RL model on static data, returning predictions.

    Parameters:
        static_model (StaticPolicyNetwork): Trained static RL model.
        data (pd.DataFrame): DataFrame containing features for prediction.

    Returns:
        preds (np.ndarray): Predicted decisions (0: hold, 1: convert).
    """
    features = data[["S", "ttm_years", "Estimated_Price"]].values.astype(np.float32)
    X = torch.FloatTensor(features)
    with torch.no_grad():
        logits = static_model(X)
        preds = torch.argmax(logits, dim=1).numpy()
    return preds


def main():
    # Simulation parameters
    S0 = 100.0
    T = 1.0
    mu = 0.05
    sigma = 0.20
    num_steps = 100  # Number of time steps for evaluation

    # Generate a stock price path
    stock_path, times = generate_gbm_stock_path(S0, mu, sigma, T, num_steps)

    # Initialize PDE engine
    engine = TsiveriotisFernandesEngine(
        S0=S0, K=100.0, r=0.05, sigma=sigma, T=T,
        q=0.0, spread=0.01, M=100, N=100, par=100.0, early_exercise=True
    )
    engine.solve_pde()

    # ---------------------------
    # Online RL Evaluation
    # ---------------------------
    online_agent = PolicyGradientAgent(input_dim=3, hidden_dim=32, output_dim=2, lr=1e-3, gamma=0.99)
    online_agent.load_model("data/policy_model.pth")

    start_time = time.time()
    eval_times, eval_stock_prices, eval_pde_prices, rl_decisions = evaluate_online_rl(engine, online_agent, stock_path, times)
    online_pred_time = time.time() - start_time
    print(f"Online RL prediction time: {online_pred_time:.2f} seconds")

    # Approximate "true" decisions for online RL (convert if conversion value > PDE price)
    conversion_values = eval_stock_prices * (100.0 / 100.0)  # Assuming cv=100 for simplicity
    true_decisions = (conversion_values > eval_pde_prices).astype(int)
    mse = mean_squared_error(true_decisions, rl_decisions)
    rmse = np.sqrt(mse)
    r2 = r2_score(true_decisions, rl_decisions)
    print(f"Online RL RMSE: {rmse:.4f}")
    print(f"Online RL R²: {r2:.4f}")

    plot_comparison_online_rl(eval_times, eval_stock_prices, eval_pde_prices, rl_decisions,
                              filename="data/online_rl_comparison.png")

    # ---------------------------
    # Static RL Evaluation
    # ---------------------------
    data = pd.read_excel("data/model_price_all.xlsx")

    # Convert ttm_days to ttm_years if present
    if 'ttm_days' in data.columns:
        data["ttm_years"] = data["ttm_days"] / 365.0
    else:
        data["ttm_years"] = 1.0  # Default to 1 year

    # Compute labels if not present in the data
    if "label" not in data.columns:
        par = 100.0  # Par value, adjust if different
        data["conversion_value"] = data["S"] * (par / data["cv"])  # Assumes 'S' and 'cv' columns exist
        data["label"] = (data["conversion_value"] > data["Estimated_Price"]).astype(int)

    # Load and evaluate static RL model
    static_model = StaticPolicyNetwork(input_dim=3, hidden_dim=128, output_dim=2)
    static_model.load_state_dict(torch.load("data/static_policy_model.pth"))
    static_model.eval()

    start_time = time.time()
    static_preds = evaluate_static_rl(static_model, data)
    static_pred_time = time.time() - start_time
    print(f"Static RL prediction time: {static_pred_time:.2f} seconds")

    # Calculate metrics for static RL
    true_labels = data["label"].values
    mse = mean_squared_error(true_labels, static_preds)
    rmse = np.sqrt(mse)
    r2 = r2_score(true_labels, static_preds)
    print(f"Static RL RMSE: {rmse:.4f}")
    print(f"Static RL R²: {r2:.4f}")

    # Plot static RL comparison (subset for clarity)
    data_subset = data.sort_values("ttm_years").head(100)
    times_static = data_subset["ttm_years"].values
    stock_prices_static = data_subset["S"].values
    pde_prices_static = data_subset["Estimated_Price"].values
    plot_comparison_static(times_static, stock_prices_static, pde_prices_static, static_preds[:len(data_subset)],
                           filename="data/static_rl_comparison.png")

    print("Evaluation and comparison plots generated and saved in the data folder.")


if __name__ == "__main__":
    main()