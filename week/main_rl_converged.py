"""
main_dynamic_rl_converged.py

Trains a convertible bond RL agent using PDE (stub).
Plots the final test path with PDE price and conversion decisions.

Usage:
  python main_dynamic_rl_converged.py \
     --episodes 50 \
     --output data/rl_runs
"""

import argparse
import os
import numpy as np
import matplotlib.pyplot as plt

from tf_engine_converged import TsiveriotisFernandesEngineConverged
from environment_converged import ConvertibleBondEnvConverged
from rl_agent_converged import PolicyGradientAgentConverged


def generate_gbm_stock_path(S0, mu, sigma, T, num_steps):
    """
    Basic GBM path: S_{t+1} = S_t * exp((mu - sigma^2/2)*dt + sigma*sqrt(dt)*Z).
    times in [0..T].
    """
    dt = T / (num_steps - 1)
    times = np.linspace(0, T, num_steps)
    path = np.zeros(num_steps, dtype=np.float32)
    path[0] = S0
    for i in range(1, num_steps):
        Z = np.random.normal()
        path[i] = path[i - 1] * np.exp((mu - 0.5 * sigma * sigma) * dt + sigma * np.sqrt(dt) * Z)
    return path, times


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=int, default=50, help="Number of RL training episodes.")
    parser.add_argument("--output", default="data/dynamic_rl_output", help="Folder to store agent and plot.")
    args = parser.parse_args()

    os.makedirs(args.output, exist_ok=True)

    # 1) PDE engine
    engine = TsiveriotisFernandesEngineConverged(
        S0=100.0, K=100.0, r=0.05, sigma=0.20, T=1.0,
        q=0.0, spread=0.01, M=100, N=100, par=100.0
    )
    engine.solve_pde()
    print(f"[DynamicRL] PDE Price at (S0=100, t=0) => {engine.cb_value_0:.4f} (stub)")

    # 2) RL agent
    agent = PolicyGradientAgentConverged(input_dim=3, hidden_dim=64, output_dim=2,
                                         lr=1e-3, gamma=0.99)

    # 3) Train for N episodes
    num_episodes = args.episodes
    steps_per_episode = 100
    for ep in range(num_episodes):
        # Generate a new GBM stock path each episode
        stock_path, times = generate_gbm_stock_path(100, 0.05, 0.20, 1.0, steps_per_episode)
        env = ConvertibleBondEnvConverged(engine, stock_path, times)

        state = env.reset()
        done = False
        while not done:
            action = agent.select_action(state)
            next_state, reward, done, _ = env.step(action)
            agent.store_reward(reward)
            state = next_state
        agent.update_policy()

        if (ep + 1) % 10 == 0:
            print(f"[DynamicRL] Episode {ep + 1}/{num_episodes} complete.")

    # 4) Test on a new path and record data
    test_steps = 100
    stock_path, times = generate_gbm_stock_path(100, 0.05, 0.20, 1.0, test_steps)
    env = ConvertibleBondEnvConverged(engine, stock_path, times)

    state = env.reset()
    done = False

    recorded_times = []
    recorded_stock = []
    recorded_pde = []
    recorded_actions = []

    while not done:
        S, t, pde_price = state
        action = agent.select_action(state)

        recorded_times.append(t)
        recorded_stock.append(S)
        recorded_pde.append(pde_price)
        recorded_actions.append(action)

        next_state, reward, done, _ = env.step(action)
        state = next_state

    # 5) Summarize final payoff
    total_reward = sum(agent.rewards)
    print(f"[DynamicRL] Test run total reward={total_reward:.4f}")

    # 6) Save the agent
    agent_path = os.path.join(args.output, "dynamic_rl_agent.pth")
    agent.save_model(agent_path)
    print(f"[DynamicRL] Saved dynamic RL agent => {agent_path}")

    # 7) Plot the path, PDE, and conversion decisions
    # times vs. stock_price, PDE, and mark action=1 with red 'X'
    recorded_times = np.array(recorded_times)
    recorded_stock = np.array(recorded_stock)
    recorded_pde = np.array(recorded_pde)
    recorded_actions = np.array(recorded_actions)

    convert_indices = np.where(recorded_actions == 1)[0]

    plt.figure(figsize=(8, 5))
    plt.plot(recorded_times, recorded_stock, label="Stock Price", color="blue")
    plt.plot(recorded_times, recorded_pde, label="PDE Price", color="green", linestyle="--")
    plt.scatter(recorded_times[convert_indices], recorded_stock[convert_indices],
                color="red", marker="x", s=80, label="Convert Action")

    plt.xlabel("Time (years)")
    plt.ylabel("Price")
    plt.title("Dynamic RL: Stock vs. PDE vs. Conversion Action")
    plt.grid(True)
    plt.legend()
    plot_path = os.path.join(args.output, "dynamic_rl_comparison.png")
    plt.savefig(plot_path, dpi=150)
    plt.close()
    print(f"[DynamicRL] Saved final comparison plot => {plot_path}")


if __name__ == "__main__":
    main()
