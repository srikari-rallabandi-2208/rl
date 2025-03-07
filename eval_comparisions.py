"""
eval_comparisons.py

Evaluates:
1. Tsiveriotis-Fernandes PDE-based prices (serving as a reference),
2. Online RL decisions, and
3. Static RL classifier predictions.

Generates comparison plots and calculates basic metrics (RMSE, R²) by comparing
the RL decisions to a simple "ground truth" rule (e.g., convert if conversion_value > PDE_price).

Example usage:
    python eval_comparisons.py
"""

import numpy as np
import pandas as pd
import torch
import time
from tf_engine import TsiveriotisFernandesEngine
from environment import ConvertibleBondEnvTF
from rl_agent import PolicyGradientAgent
from main_static_rl import StaticPolicyNetwork
from visualizations import plot_comparison_online_rl, plot_comparison_static
from sklearn.metrics import mean_squared_error, r2_score


def generate_gbm_stock_path(S0, mu, sigma, T, num_steps):
    """
    Generate a GBM stock path for evaluation.

    Parameters:
    -----------
    S0 : float
        Initial stock price.
    mu : float
        Drift.
    sigma : float
        Volatility.
    T : float
        Time horizon (years).
    num_steps : int
        Number of steps in the path.

    Returns:
    --------
    stock_path : np.ndarray
        The generated GBM path.
    times : np.ndarray
        Corresponding time points.
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
    Evaluate the online RL agent's decisions on a given stock path.

    Parameters:
    -----------
    engine : TsiveriotisFernandesEngine
        The PDE engine for reference pricing.
    agent : PolicyGradientAgent
        A trained online RL agent.
    stock_path : np.ndarray
        The simulated stock price path.
    times : np.ndarray
        The corresponding time points.

    Returns:
    --------
    online_times : np.ndarray
        Time points encountered by the agent.
    online_stock_prices : np.ndarray
        Stock prices at each step.
    pde_prices : np.ndarray
        PDE-based convertible bond prices.
    rl_decisions : np.ndarray
        The RL agent's hold(0)/convert(1) decisions.
    """
    env = ConvertibleBondEnvTF(engine, stock_path, times)
    state = env.reset()

    online_times = []
    online_stock_prices = []
    pde_prices = []
    rl_decisions = []

    done = False
    while not done:
        S, t, pde_price = state
        online_times.append(t)
        online_stock_prices.append(S)
        pde_prices.append(pde_price)

        action = agent.select_action(state)
        rl_decisions.append(action)

        state, _, done, _ = env.step(action)

    return (
        np.array(online_times),
        np.array(online_stock_prices),
        np.array(pde_prices),
        np.array(rl_decisions),
    )


def main():
    # 1. Set up a PDE engine
    engine = TsiveriotisFernandesEngine(
        S0=100.0, K=100.0, r=0.05, sigma=0.20, T=1.0,
        q=0.0, spread=0.01, M=100, N=100, par=100.0, early_exercise=True
    )
    engine.solve_pde()  # Solve PDE once

    # 2. Load the trained online RL agent
    online_agent = PolicyGradientAgent(input_dim=3, hidden_dim=128, output_dim=2, lr=1e-3, gamma=0.99)
    online_agent.load_model("data/policy_model.pth")

    # 3. Evaluate the online RL agent on a fresh GBM path
    S0 = 100.0
    mu = 0.05
    sigma = 0.20
    T = 1.0
    num_steps = 100

    stock_path, times = generate_gbm_stock_path(S0, mu, sigma, T, num_steps)

    start_pred_time = time.time()
    eval_times, eval_stock_prices, eval_pde_prices, rl_decisions = evaluate_online_rl(engine, online_agent, stock_path, times)
    online_pred_time = time.time() - start_pred_time
    print(f"[Online RL] Prediction time: {online_pred_time:.4f} seconds")

    # Compare RL decisions to a simple "ground truth" -> convert if conversion value > PDE
    conversion_values = eval_stock_prices * (100.0 / 100.0)  # par=100, cv=100 => S*(par/cv)
    true_decisions = (conversion_values > eval_pde_prices).astype(int)

    mse = mean_squared_error(true_decisions, rl_decisions)
    rmse = np.sqrt(mse)
    r2 = r2_score(true_decisions, rl_decisions)
    print(f"[Online RL] RMSE: {rmse:.4f}, R^2: {r2:.4f}")

    # Plot and save online RL comparison
    plot_comparison_online_rl(eval_times, eval_stock_prices, eval_pde_prices, rl_decisions,
                              filename="data/online_rl_comparison.png")

    # 4. Evaluate the static RL model on the entire 'model_price_all.xlsx'
    df = pd.read_excel("data/model_price_all.xlsx")

    # Ensure ttm_years is present
    if 'ttm_days' in df.columns:
        df["ttm_years"] = df["ttm_days"] / 365.0
    else:
        df["ttm_years"] = 1.0

    # If label not present, create it
    if "label" not in df.columns:
        par = 100.0
        df["conversion_value"] = df["S"] * (par / df["cv"])
        df["label"] = (df["conversion_value"] > df["Estimated_Price"]).astype(int)

    # Load static RL model
    static_model = StaticPolicyNetwork(input_dim=3, hidden_dim=128, output_dim=2)
    static_model.load_state_dict(torch.load("data/static_policy_model.pth"))
    static_model.eval()

    # Prepare features
    X_feats = df[["S", "ttm_years", "Estimated_Price"]].values.astype(np.float32)
    y_true = df["label"].values

    start_static_pred = time.time()
    with torch.no_grad():
        logits = static_model(torch.from_numpy(X_feats))
        preds = torch.argmax(logits, dim=1).numpy()
    static_pred_time = time.time() - start_static_pred
    print(f"[Static RL] Prediction time: {static_pred_time:.4f} seconds")

    # Compute metrics (RMSE, R^2) vs the PDE-labeled "true" decisions
    static_mse = mean_squared_error(y_true, preds)
    static_rmse = np.sqrt(static_mse)
    static_r2 = r2_score(y_true, preds)
    print(f"[Static RL] RMSE: {static_rmse:.4f}, R^2: {static_r2:.4f}")

    # For plotting, let's visualize a subset (otherwise it may be huge)
    # Sort by ttm_years and pick the first 200 points for clarity
    df_sub = df.sort_values("ttm_years").head(200)
    times_static = df_sub["ttm_years"].values
    stock_prices_static = df_sub["S"].values
    pde_prices_static = df_sub["Estimated_Price"].values
    preds_sub = preds[df_sub.index]

    # Plot static RL comparison
    plot_comparison_static(times_static, stock_prices_static, pde_prices_static, preds_sub,
                           filename="data/static_rl_comparison.png")

    print("Evaluation complete. Plots saved to 'data/online_rl_comparison.png' and 'data/static_rl_comparison.png'.")


if __name__ == "__main__":
    main()