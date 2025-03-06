"""
main_rl.py

This module trains an online RL agent using a REINFORCE-style policy gradient method.
It initializes a PDE engine, creates an RL environment with simulated stock paths,
trains a policy network over multiple episodes, and saves the trained model.
Key metrics like training time are tracked and reported.
"""

import numpy as np
import time
from environment import ConvertibleBondEnvTF  # Assumed dependency
from tf_engine import TsiveriotisFernandesEngine  # Assumed dependency
from rl_agent import PolicyGradientAgent  # Assumed dependency


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


def main():
    # Record start time to measure training duration
    start_time = time.time()

    # Initialize the PDE engine with demo parameters
    # Using Tsiveriotis-Fernandes model for convertible bond pricing
    engine = TsiveriotisFernandesEngine(
        S0=100.0,          # Initial stock price
        K=100.0,           # Strike price
        r=0.05,            # Risk-free rate
        sigma=0.20,        # Volatility
        T=1.0,             # Time to maturity (1 year)
        q=0.0,             # Dividend yield
        spread=0.01,       # Credit spread
        M=50,              # Number of stock price steps in PDE grid
        N=50,              # Number of time steps in PDE grid
        par=100.0,         # Par value of the bond
        early_exercise=True  # Allow early conversion
    )

    # Initialize the RL agent
    # Input_dim=3 corresponds to [stock price, time-to-maturity, PDE price]
    agent = PolicyGradientAgent(
        input_dim=3,
        hidden_dim=32,     # Size of hidden layer
        output_dim=2,      # Actions: 0 (hold), 1 (convert)
        lr=1e-3,           # Learning rate
        gamma=0.99         # Discount factor
    )

    # Training parameters
    num_episodes = 100  # Increased from a smaller number (e.g., 5) for better convergence
    num_steps = 100     # Increased from a smaller number (e.g., 10) for longer episodes

    # Train the agent over multiple episodes
    for ep in range(num_episodes):
        # Generate a new GBM stock path for each episode
        # This ensures the agent learns across diverse market scenarios
        stock_path, times = generate_gbm_stock_path(
            S0=100.0, mu=0.05, sigma=0.20, T=1.0, num_steps=num_steps
        )
        env = ConvertibleBondEnvTF(engine, stock_path, times)
        state = env.reset()
        done = False
        while not done:
            action = agent.select_action(state)  # Choose action based on current policy
            next_state, reward, done, _ = env.step(action)  # Take action in environment
            agent.store_reward(reward)  # Store reward for policy update
            state = next_state
        agent.update_policy()  # Update policy using accumulated rewards
        if (ep + 1) % 10 == 0:
            print(f"Episode {ep + 1}/{num_episodes} complete.")

    # Save the trained model to disk
    agent.save_model("data/policy_model.pth")
    print("Training finished. Online RL model saved.")

    # Calculate and report training time
    train_time = time.time() - start_time
    print(f"Total training time: {train_time:.2f} seconds")


if __name__ == "__main__":
    main()