"""
main_rl_predict.py

Trains a discrete log-scale RL agent to guess PDE prices row by row,
using mini-episodes. Then does a final pass on *all* data in a fresh env
to generate a scatter plot and final MSE.

Usage (example):
  python main_rl_predict.py \
    --csv data/prepared_40k/train.csv \
    --episodes 32 \
    --num-buckets 50 \
    --pde-min 100.0 \
    --pde-max 100000000.0 \
    --pde-clip 1e9 \
    --episode-size 1000 \
    --lr 1e-3 \
    --gamma 0.99
"""

import argparse
import numpy as np
import matplotlib.pyplot as plt
import torch
import os

from predict_env import PricePredictEnv
from rl_price_predict import PricePredictAgent


def bucket_to_price(action, num_buckets, pde_min, pde_max):
    """
    Converts discrete action [0..num_buckets-1] to PDE guess on log scale.
    """
    if num_buckets <= 1:
        return pde_min
    import math
    fraction = action / (num_buckets - 1)
    log_min = math.log(pde_min)
    log_max = math.log(pde_max)
    log_val = log_min + fraction * (log_max - log_min)
    return math.exp(log_val)


def main():
    parser = argparse.ArgumentParser(description="Discrete RL PDE Price Prediction.")
    parser.add_argument("--csv", required=True, help="Path to PDE-labeled CSV (train set).")
    parser.add_argument("--episodes", type=int, default=40, help="Number of mini-episodes total.")
    parser.add_argument("--num-buckets", type=int, default=50)
    parser.add_argument("--pde-min", type=float, default=100.0)
    parser.add_argument("--pde-max", type=float, default=1e8)
    parser.add_argument("--pde-clip", type=float, default=1e9)
    parser.add_argument("--episode-size", type=int, default=1000, help="Rows per mini-episode.")
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--gamma", type=float, default=0.99)
    args = parser.parse_args()

    # 1) Create environment for training
    env = PricePredictEnv(
        csv_path=args.csv,
        num_buckets=args.num_buckets,
        pde_min=args.pde_min,
        pde_max=args.pde_max,
        pde_clip=args.pde_clip,
        episode_size=args.episode_size
    )

    # 2) Create agent
    agent = PricePredictAgent(
        input_dim=5,
        num_actions=args.num_buckets,
        hidden_dim=64,
        lr=args.lr,
        gamma=args.gamma
    )

    ep_rewards = []

    # 3) Training loop
    for ep in range(args.episodes):
        state = env.reset()
        if state is None:
            print("No more data to train on. Breaking early.")
            break
        done = False
        ep_reward = 0.0

        while not done:
            action = agent.select_action(state)
            next_state, reward, done, _ = env.step(action)
            agent.store_reward(reward)
            ep_reward += reward
            state = next_state

        agent.update_policy()
        ep_rewards.append(ep_reward)
        print(f"[Episode {ep + 1}/{args.episodes}] sum of rewards={ep_reward:.2f}")

    # 4) Final pass => measure MSE + gather scatter data
    # CREATE A FRESH ENV so we can pass over all rows again.
    final_env = PricePredictEnv(
        csv_path=args.csv,  # same data
        num_buckets=args.num_buckets,
        pde_min=args.pde_min,
        pde_max=args.pde_max,
        pde_clip=args.pde_clip,
        episode_size=10_000_000_000  # big so we can process entire file in 1 pass
    )

    preds = []
    actuals = []

    while True:
        s = final_env.reset()
        if s is None:
            # means no more data in final_env
            break
        done = False
        while not done:
            a = agent.select_action(s)
            guess = bucket_to_price(a, args.num_buckets, args.pde_min, args.pde_max)
            idx = final_env.cur_index

            # retrieve the PDE label
            row = final_env.df_shuffled.iloc[idx] if idx < final_env.n_full else None
            if row is not None:
                actuals.append(row["Model_Price"])
                preds.append(guess)

            ns, _, done, _ = final_env.step(a)
            s = ns

    # 5) Compute final MSE
    if len(actuals) == 0:
        print("No final data to measure. Possibly no PDE left in final pass.")
        return

    import numpy as np
    actuals = np.array(actuals)
    preds = np.array(preds)

    errors = actuals - preds
    # clamp error => [-1e9..1e9]
    errors = np.clip(errors, -1e9, 1e9)
    sq_err = errors ** 2
    sq_err = np.clip(sq_err, 0, 1e12)
    mse = np.mean(sq_err)
    print(f"Final pass MSE={mse:.2f}")

    # 6) Plot scatter
    plt.figure(figsize=(6, 6))
    plt.scatter(actuals, preds, alpha=0.3)
    plt.xlabel("Actual PDE Price")
    plt.ylabel("RL Predicted PDE Price")
    plt.title("RL PDE Price vs PDE (Final Pass)")
    plt.grid(True)
    plt.savefig("rl_price_scatter.png", dpi=150)
    plt.close()
    print("Scatter plot => rl_price_scatter.png")

    # 7) Plot episode reward curve
    plt.figure()
    plt.plot(ep_rewards, marker='o')
    plt.title("Episode Reward Over Training")
    plt.xlabel("Episode")
    plt.ylabel("Sum of Rewards")
    plt.grid(True)
    plt.savefig("rl_price_rewards.png", dpi=150)
    plt.close()
    print("Episode reward plot => rl_price_rewards.png")


if __name__ == "__main__":
    main()
