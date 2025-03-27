"""
main_rl_predict.py

Ties the environment (predict_env) and agent (rl_price_predict) together:
 - Command-line args to specify CSV, episodes, # of buckets, max PDE price
 - Train for X episodes
 - Then do a final pass computing MSE between agent predictions and PDE.

Usage:
  python main_rl_predict.py \
    --csv data/final_40k.csv \
    --episodes 1 \
    --num-buckets 20 \
    --max-price 2000.0 \
    --lr 1e-3
"""
from matplotlib import pyplot as plt

"""
main_rl_predict.py

- Creates the PricePredictEnv with PDE data (and outlier filtering).
- Trains an RL agent to guess PDE for each row, row by row.
- After training, does a final pass to measure MSE, with error clamping
  to avoid infinite MSE in case of extreme PDE.

Usage:
  python main_rl_predict.py \
    --csv data/final_40k.csv \
    --episodes 1 \
    --num-buckets 20 \
    --max-price 100000 \
    --lr 1e-3
"""

import argparse
import numpy as np
import torch
from predict_env import PricePredictEnv
from rl_price_predict import PricePredictAgent


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", required=True, help="Path to PDE-labeled CSV (final_40k.csv).")
    parser.add_argument("--episodes", type=int, default=1, help="Number of full passes over the data.")
    parser.add_argument("--num-buckets", type=int, default=20, help="Discrete PDE guess buckets.")
    parser.add_argument("--max-price", type=float, default=2000.0, help="Max PDE guess for top bucket.")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate.")
    parser.add_argument("--pde-clip", type=float, default=1e7,
                        help="Filter out PDE rows beyond this absolute value.")
    args = parser.parse_args()

    # 1) Create env w/ PDE filtering
    env = PricePredictEnv(
        csv_path=args.csv,
        num_buckets=args.num_buckets,
        max_price=args.max_price,
        pde_clip=args.pde_clip
    )

    # 2) RL agent
    agent = PricePredictAgent(
        input_dim=5,
        num_actions=args.num_buckets,
        hidden_dim=64,
        lr=args.lr
    )

    # 3) Training
    for ep in range(args.episodes):
        state = env.reset()
        done = False
        ep_reward = 0.0
        while not done:
            action = agent.select_action(state)
            next_state, reward, done, _ = env.step(action)
            agent.store_reward(reward)
            ep_reward += reward
            state = next_state
        agent.update_policy()
        print(f"[Episode {ep + 1}/{args.episodes}] sum of rewards={ep_reward:.2f}")

    # 4) Final pass => measure MSE
    state = env.reset()
    done = False
    preds = []
    actuals = []
    while not done:
        action = agent.select_action(state)
        guess = (action / (args.num_buckets - 1)) * args.max_price

        row_idx = env.index
        if row_idx < env.n:
            row = env.df.iloc[row_idx]
            true_price = row["Model_Price"]

            # clamp error for safe MSE calc
            error = true_price - guess
            # clip error to avoid infinite square
            if error > 1e9:
                error = 1e9
            elif error < -1e9:
                error = -1e9

            preds.append(guess)
            actuals.append(true_price)

        next_state, _, done, _ = env.step(action)
        state = next_state

    preds = np.array(preds)
    actuals = np.array(actuals)
    if len(preds) == 0:
        print("No final pass data. Possibly PDE was empty.")
        return

    # compute MSE with clamp
    errors = actuals - preds
    # clamp again
    errors[errors > 1e9] = 1e9
    errors[errors < -1e9] = -1e9

    squared = errors * errors
    # clamp squared
    squared[squared > 1e12] = 1e12

    mse = np.mean(squared)
    print(f"Final pass MSE={mse:.2f}")

    # 5) Plot
    plt.figure(figsize=(6, 6))
    plt.scatter(actuals, preds, alpha=0.3)
    plt.xlabel("Actual PDE")
    plt.ylabel("RL-Predicted PDE")
    plt.title("RL PDE Price vs. PDE (Final Pass)")
    plt.grid(True)
    plt.savefig("rl_price_scatter.png", dpi=150)
    plt.close()
    print("Scatter plot saved to rl_price_scatter.png")

if __name__ == "__main__":
    main()
