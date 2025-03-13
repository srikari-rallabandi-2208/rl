"""
eval_comparisons_converged.py

Evaluates:
1. Converged PDE-based prices (Tsiveriotis-Fernandes),
2. Online RL decisions (using the new environment_converged + agent),
3. Static RL classifier predictions (trained offline via main_static_rl_converged).

Generates comparison plots via visualizations.py and prints error metrics.
"""

import numpy as np
import pandas as pd
import torch
import time

from tf_engine_converged import TsiveriotisFernandesEngineConverged
from environment_converged import ConvertibleBondEnvConverged
from rl_agent_converged import PolicyGradientAgentConverged
from visualizations import plot_comparison_online_rl, plot_comparison_static
from sklearn.metrics import mean_squared_error, r2_score


def generate_gbm_stock_path(S0, mu, sigma, T, num_steps):
    """
    Same as your older GBM generator for an evaluation path.
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
    Evaluate the converged RL agent's decisions on a given stock path.
    Returns arrays for plotting:
      times, stock prices, PDE prices, RL's hold/convert decisions.
    """
    env = ConvertibleBondEnvConverged(engine, stock_path, times)
    state = env.reset()

    all_times = []
    all_stock_prices = []
    all_pde_prices = []
    rl_actions = []

    done = False
    while not done:
        S, t, pde_price = state
        all_times.append(t)
        all_stock_prices.append(S)
        all_pde_prices.append(pde_price)

        action = agent.select_action(state)
        rl_actions.append(action)

        state, _, done, _ = env.step(action)

    return (
        np.array(all_times),
        np.array(all_stock_prices),
        np.array(all_pde_prices),
        np.array(rl_actions)
    )


def main():
    # 1. Create a PDE engine with the same converged grid used for training
    engine = TsiveriotisFernandesEngineConverged(
        S0=100.0,  # or any reference
        K=100.0,
        r=0.05,
        sigma=0.20,
        T=1.0,
        q=0.0,
        spread=0.01,
        M=200,  # or (M_star, N_star) from your final stable grid
        N=200,
        par=100.0
    )
    engine.solve_pde()

    print(f"[PDE] Converged CB price at S0=100, t=0 => {engine.cb_value_0:.4f}")

    # 2. Load the trained online RL agent from 'policy_model_converged.pth'
    online_agent = PolicyGradientAgentConverged(input_dim=3, hidden_dim=64, output_dim=2, lr=1e-3, gamma=0.99)
    online_agent.load_model("data/policy_model_converged.pth")
    online_agent.policy_net.eval()

    # 3. Evaluate the RL agent on a fresh GBM path
    S0 = 100.0
    mu = 0.05
    sigma = 0.20
    T_horizon = 1.0
    num_steps = 100

    stock_path, times = generate_gbm_stock_path(S0, mu, sigma, T_horizon, num_steps)

    start_eval_time = time.time()
    eval_times, eval_stock, eval_pde, rl_actions = evaluate_online_rl(engine, online_agent, stock_path, times)
    elapsed_eval = time.time() - start_eval_time
    print(f"[Online RL] Evaluation time: {elapsed_eval:.4f} sec")

    # Compare RL decisions to a naive "convert if conversion_value > PDE"
    # conversion_value = S * (par/cv). Here par=100, cv=100 => conversion_value = S
    convert_value = eval_stock  # since par/cv = 1
    true_decisions = (convert_value > eval_pde).astype(int)

    mse_rl = mean_squared_error(true_decisions, rl_actions)
    rmse_rl = np.sqrt(mse_rl)
    r2_rl = r2_score(true_decisions, rl_actions)
    print(f"[Online RL] RMSE: {rmse_rl:.4f}, R^2: {r2_rl:.4f}")

    # 4. Plot the online RL comparison
    plot_comparison_online_rl(
        eval_times, eval_stock, eval_pde, rl_actions,
        filename="data/online_rl_comparison_converged.png"
    )

    # 5. Evaluate static RL on the 'converged_dataset.csv' if you wish
    # or on 'model_price_all.xlsx' from your pipeline
    df = pd.read_csv("data/converged_dataset.csv")
    # We have columns: S, t, pde_price, label
    # The label is 1 if conv > pde
    # We'll load the static policy
    import torch
    import torch.nn as nn
    class StaticPolicyNet(nn.Module):
        def __init__(self, input_dim=3, hidden_dim=64, output_dim=2):
            super().__init__()
            self.fc1 = nn.Linear(input_dim, hidden_dim)
            self.fc2 = nn.Linear(hidden_dim, output_dim)

        def forward(self, x):
            x = torch.relu(self.fc1(x))
            return self.fc2(x)

    static_model = StaticPolicyNet(3, 64, 2)
    static_model.load_state_dict(torch.load("data/static_policy_converged.pth"))
    static_model.eval()

    # Prepare features => [S, t, pde_price], get preds
    X = df[["S", "t", "pde_price"]].values.astype(np.float32)
    y_true = df["label"].values
    #import torch
    with torch.no_grad():
        logits = static_model(torch.from_numpy(X))
        preds = torch.argmax(logits, dim=1).numpy()

    # Basic metrics
    mse_static = mean_squared_error(y_true, preds)
    rmse_static = np.sqrt(mse_static)
    r2_static = r2_score(y_true, preds)
    print(f"[Static RL] RMSE: {rmse_static:.4f}, R^2: {r2_static:.4f}")

    # 6. Plot a subset
    # We'll sort by t and pick the first 50 points for clarity
    df_sub = df.sort_values("t").head(50)
    times_sub = df_sub["t"].values
    stock_sub = df_sub["S"].values
    pde_sub = df_sub["pde_price"].values
    preds_sub = preds[df_sub.index]

    plot_comparison_static(
        times_sub, stock_sub, pde_sub, preds_sub,
        filename="data/static_rl_comparison_converged.png"
    )

    print("Comparison plots saved in 'data/' folder.")


if __name__ == "__main__":
    main()
