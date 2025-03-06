"""
eval_comparison.py

This script evaluates and compares three methods:
  1. The TF (PDE) convertible bond price.
  2. The online RL agent's conversion decisions.
  3. The static (offline) RL learner's predictions.

It simulates a stock price path, computes PDE prices using the TF engine,
loads the trained models from the data folder, and generates comparison plots that are saved as PNG files.
"""

import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
from tf_engine import TsiveriotisFernandesEngine
from environment import ConvertibleBondEnvTF
from rl_agent import PolicyGradientAgent
from visualizations import plot_comparison_online_rl, plot_comparison_static


# Define a function to generate a stock price path using Geometric Brownian Motion (GBM)
def generate_gbm_stock_path(S0, mu, sigma, T, num_steps):
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
    Run an episode using the online RL agent and record:
      - Stock prices
      - PDE prices (from the TF engine)
      - RL decisions
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
        action = agent.select_action(state)
        rl_decisions.append(action)
        state, _, done, _ = env.step(action)

    return np.array(online_times), np.array(online_stock_prices), np.array(pde_prices), np.array(rl_decisions)


def evaluate_static_rl(static_model, data):
    """
    Given a static RL model (a classifier) and a DataFrame with features,
    compute predictions (0 for hold, 1 for convert).
    """
    # Use features: [S, ttm_years, Estimated_Price]
    features = data[["S", "ttm_years", "Estimated_Price"]].values.astype(np.float32)
    X = torch.FloatTensor(features)
    with torch.no_grad():
        logits = static_model(X)
        preds = torch.argmax(logits, dim=1).numpy()
    return preds


def main():
    # Parameters for simulation.
    S0 = 100.0
    T = 1.0
    mu = 0.05
    sigma = 0.20
    num_steps = 100  # Increase for smoother curves.

    # Generate a synthetic stock price path using GBM.
    stock_path, times = generate_gbm_stock_path(S0, mu, sigma, T, num_steps)

    # Initialize the PDE engine with the same parameters as used in training.
    engine = TsiveriotisFernandesEngine(
        S0=S0, K=100.0, r=0.05, sigma=sigma, T=T,
        q=0.0, spread=0.01, M=100, N=100, par=100.0, early_exercise=True
    )
    # Solve PDE once.
    engine.solve_pde()

    # ---------------------------
    # Online RL Evaluation:
    # ---------------------------
    online_agent = PolicyGradientAgent(input_dim=3, hidden_dim=32, output_dim=2, lr=1e-3, gamma=0.99)
    # Load the trained online RL model.
    online_agent.load_model("data/policy_model.pth")

    eval_times, eval_stock_prices, eval_pde_prices, rl_decisions = evaluate_online_rl(engine, online_agent, stock_path,
                                                                                      times)

    # Plot and save the online RL comparison.
    plot_comparison_online_rl(eval_times, eval_stock_prices, eval_pde_prices, rl_decisions,
                              filename="data/online_rl_comparison.png")

    # ---------------------------
    # Static RL Evaluation:
    # ---------------------------
    # Load static data (assumed to be the same file used during static training)
    data = pd.read_excel("data/model_price_all.xlsx")
    # Ensure we have ttm_years column.
    if 'ttm_days' in data.columns:
        data["ttm_years"] = data["ttm_days"] / 365.0
    else:
        data["ttm_years"] = 1.0  # default 1 year

    # Define and load the static RL model.
    from main_static_rl import StaticPolicyNetwork  # Import the same class defined there.
    static_model = StaticPolicyNetwork(input_dim=3, hidden_dim=32, output_dim=2)
    static_model.load_state_dict(torch.load("data/static_policy_model.pth"))
    static_model.eval()

    # Get static predictions.
    static_preds = evaluate_static_rl(static_model, data)

    # For plotting static comparison, choose a subset (e.g., the first 100 records) and sort by time.
    data_subset = data.sort_values("ttm_years").head(100)
    times_static = data_subset["ttm_years"].values
    stock_prices_static = data_subset["S"].values
    pde_prices_static = data_subset["Estimated_Price"].values

    # Plot and save the static RL comparison.
    plot_comparison_static(times_static, stock_prices_static, pde_prices_static, static_preds[:len(data_subset)],
                           filename="data/static_rl_comparison.png")

    print("Evaluation and comparison plots generated and saved in the data folder.")


if __name__ == "__main__":
    main()