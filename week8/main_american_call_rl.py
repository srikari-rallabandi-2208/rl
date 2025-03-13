"""
Trains a policy gradient agent on the AmericanCallEnv by simulating many stock paths.
We measure the average payoff the agent gets (exercise or hold).
Eventually, if well-trained, the agent's policy should approximate an optimal (or near-optimal) strategy.
"""

import pandas as pd
import numpy as np
import torch
import time
import os

from american_call_env import AmericanCallEnv
from rl_agent import PolicyGradientAgent


def main():
    start_time = time.time()

    # 1) Load some data from CSV to sample random parameter sets (S,K,r,q,sigma,T).
    #    Or just create an environment with fixed parameters. We'll do random from CSV for variety.
    df = pd.read_csv("data/american_call_data.csv")
    df = df[df["AmericanPrice"] >= 0].reset_index(drop=True)

    # 2) Create agent
    agent = PolicyGradientAgent(
        input_dim=2,  # [S_t, t_remaining]
        hidden_dim=64,
        output_dim=2,
        lr=1e-3,
        gamma=1.0
    )

    n_episodes = 20000
    steps_per_episode = 50  # environment default

    for episode in range(n_episodes):
        # Randomly pick 1 row from df
        row = df.sample(n=1).iloc[0]
        S0, K, r, q, sigma, T = row["S"], row["K"], row["r"], row["q"], row["sigma"], row["T"]
        env = AmericanCallEnv(S0=S0, K=K, r=r, q=q, sigma=sigma, T=T, num_steps=steps_per_episode)

        state = env.reset()
        done = False

        while not done:
            action = agent.select_action(state)  # 0 or 1
            next_state, reward, done, _ = env.step(action)
            agent.store_reward(reward)
            if not done:
                state = next_state

        # end of episode => update policy
        agent.update_policy()

        if (episode + 1) % 1000 == 0:
            print(f"Episode {episode + 1}/{n_episodes} finished.")

    # Save RL policy
    os.makedirs("data", exist_ok=True)
    agent.save("data/rl_policy_model.pth")
    print("RL model saved to data/rl_policy_model.pth")

    total_time = time.time() - start_time
    print(f"Total training time: {total_time:.2f} seconds")


if __name__ == "__main__":
    main()
