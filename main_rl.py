"""
main_rl.py

This module contains the main training loop for the online RL agent using a REINFORCE-style algorithm.
It creates the PDE engine, RL environment, and trains the policy network over multiple episodes.
The trained model is saved to the data folder.
"""

import numpy as np
from environment import ConvertibleBondEnvTF
from tf_engine import TsiveriotisFernandesEngine
from rl_agent import PolicyGradientAgent


def main():
    # Initialize the PDE engine with demo parameters.
    engine = TsiveriotisFernandesEngine(
        S0=100.0, K=100.0, r=0.05, sigma=0.20, T=1.0,
        q=0.0, spread=0.01, M=50, N=50, par=100.0, early_exercise=True
    )

    # Generate a simple stock path (linear for demonstration).
    num_steps = 10
    stock_path = np.linspace(100, 110, num_steps + 1)
    times = np.linspace(0, 1.0, num_steps + 1)

    # Create the RL environment.
    env = ConvertibleBondEnvTF(engine, stock_path, times)
    agent = PolicyGradientAgent(input_dim=3, hidden_dim=32, output_dim=2, lr=1e-3, gamma=0.99)

    num_episodes = 5  # Increase this number for actual training.
    for ep in range(num_episodes):
        state = env.reset()
        done = False
        while not done:
            action = agent.select_action(state)
            next_state, reward, done, _ = env.step(action)
            agent.store_reward(reward)
            state = next_state
        agent.update_policy()
        print(f"Episode {ep + 1}/{num_episodes} complete.")

    print("Training finished. Saving online RL model...")
    agent.save_model("data/policy_model.pth")  # Save to data folder


if __name__ == "__main__":
    main()
