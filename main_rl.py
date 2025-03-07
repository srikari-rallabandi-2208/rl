"""
main_rl.py

Trains an online RL agent using a REINFORCE-style policy gradient algorithm.
Now enhanced with:
1) Larger PDE discretization for the Tsiveriotis-Fernandes engine (M=200, N=200) for finer PDE resolution.
2) Increased number of episodes and environment steps for a more thorough policy learning process.
3) A larger hidden layer in the policy network for increased model capacity.
4) Time measurements for the entire training process.

This script may take significantly longer to run compared to the modest (M=50, N=50) setup.
Make sure you have sufficient hardware resources to handle the bigger PDE grid.
"""

import numpy as np
import time

from environment import ConvertibleBondEnvTF
from tf_engine import TsiveriotisFernandesEngine
from rl_agent import PolicyGradientAgent


def generate_gbm_stock_path(S0, mu, sigma, T, num_steps):
    """
    Generate a stock price path using Geometric Brownian Motion (GBM).

    Parameters
    ----------
    S0 : float
        Initial stock price.
    mu : float
        Drift rate.
    sigma : float
        Volatility.
    T : float
        Total time horizon (in years).
    num_steps : int
        Number of discrete time steps.

    Returns
    -------
    stock_path : np.ndarray
        Simulated stock prices at each time step.
    times : np.ndarray
        Time values from 0 to T, inclusive.
    """
    dt = T / (num_steps - 1)
    times = np.linspace(0, T, num_steps)
    stock_path = np.zeros(num_steps)
    stock_path[0] = S0

    for t in range(1, num_steps):
        Z = np.random.normal()
        # GBM update rule
        stock_path[t] = stock_path[t - 1] * np.exp((mu - 0.5 * sigma**2) * dt + sigma * np.sqrt(dt) * Z)

    return stock_path, times


def main():
    start_time = time.time()

    # -------------------------------------------------------------------------
    # 1) Tsiveriotis-Fernandes PDE engine with larger grid
    # -------------------------------------------------------------------------
    # M=200, N=200 => 200 stock price steps by 200 time steps => 40,000 PDE grid points
    # This is significantly more expensive than M=50, N=50
    engine = TsiveriotisFernandesEngine(
        S0=100.0,          # Initial stock price
        K=100.0,           # Strike
        r=0.05,            # Risk-free rate
        sigma=0.20,        # Volatility
        T=1.0,             # Time to maturity (1 year)
        q=0.0,             # Dividend yield
        spread=0.01,       # Credit spread
        M=200,             # PDE spatial steps
        N=200,             # PDE time steps
        par=100.0,         # Par value
        early_exercise=True
    )

    # -------------------------------------------------------------------------
    # 2) RL agent with larger hidden dimension for more robust capacity
    # -------------------------------------------------------------------------
    # This bigger network can learn more complex decision boundaries if needed
    agent = PolicyGradientAgent(
        input_dim=3,
        hidden_dim=128,    # Bumped up from 32 for more capacity
        output_dim=2,
        lr=1e-3,
        gamma=0.99
    )

    # -------------------------------------------------------------------------
    # 3) Training parameters
    # -------------------------------------------------------------------------
    # We'll generate a GBM path each episode to vary the environment
    num_episodes = 200    # Increase from 100 => more training
    num_steps = 200       # More time steps per episode => longer episodes
    mu = 0.05             # Drift
    sigma = 0.20          # Volatility
    T_horizon = 1.0
    S0 = 100.0

    # -------------------------------------------------------------------------
    # 4) Train the agent
    # -------------------------------------------------------------------------
    for ep in range(num_episodes):
        # Generate a fresh GBM path
        stock_path, times = generate_gbm_stock_path(S0, mu, sigma, T_horizon, num_steps)

        # Build a new RL environment for each path
        env = ConvertibleBondEnvTF(engine, stock_path, times)

        # Reset environment to initial state
        state = env.reset()
        done = False

        while not done:
            action = agent.select_action(state)
            next_state, reward, done, _ = env.step(action)
            agent.store_reward(reward)
            state = next_state

        agent.update_policy()  # End of episode => REINFORCE update

        if (ep + 1) % 10 == 0:
            print(f"Episode {ep + 1}/{num_episodes} completed.")

    # -------------------------------------------------------------------------
    # 5) Save the trained model
    # -------------------------------------------------------------------------
    agent.save_model("data/policy_model.pth")
    print("Training completed. Model saved to data/policy_model.pth")

    # -------------------------------------------------------------------------
    # 6) Print total training time
    # -------------------------------------------------------------------------
    total_time = time.time() - start_time
    print(f"Total training time with M=200, N=200, episodes={num_episodes}, steps={num_steps}: "
          f"{total_time:.2f} seconds")


if __name__ == "__main__":
    main()