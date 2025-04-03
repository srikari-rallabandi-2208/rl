#!/usr/bin/env python3

"""
main_rl_predict.py

A comprehensive script to:
1) Train a discrete policy gradient (REINFORCE) agent on PDE-labeled data.
2) Save the trained model weights to disk.
3) (Optionally) Load a saved model and evaluate on train.csv or test.csv.
4) Compute MSE, RMSE, and R^2, plus generate scatter plots and error histograms.

DEPENDENCIES
------------
- Python 3.8+
- PyTorch
- NumPy, Pandas
- Matplotlib

USAGE EXAMPLES
--------------
1) Train on train.csv, save model, evaluate on same train set:
   python main_rl_predict.py \
       --train-csv ./data/prepared_400k/train.csv \
       --test-csv ./data/prepared_400k/test.csv \
       --save-model rl_agent.pt \
       --episodes 40 \
       --num-buckets 50 \
       --pde-min 1.0 \
       --pde-max 1e7 \
       --pde-clip 1e8 \
       --episode-size 1000 \
       --lr 1e-3 \
       --gamma 0.99

2) Evaluate on test.csv only (load existing model):
   python main_rl_predict.py \
       --test-csv ./data/prepared_400k/test.csv \
       --load-model rl_agent.pt \
       --no-train
"""

import argparse
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim

##############################################################################
# Simple RL Network and Agent
##############################################################################
class PricePredictNetwork(nn.Module):
    def __init__(self, input_dim=5, num_actions=50, hidden_dim=64):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, num_actions)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        return self.fc2(x)

class PricePredictAgent:
    def __init__(self, input_dim=5, num_actions=50, hidden_dim=64, lr=1e-3, gamma=0.99):
        self.gamma = gamma
        self.num_actions = num_actions
        self.policy_net = PricePredictNetwork(input_dim, num_actions, hidden_dim)
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=lr)
        self.log_probs = []
        self.rewards = []

    def select_action(self, state):
        """
        Given a state, produce an action in [0..num_actions-1].
        """
        state_t = torch.FloatTensor(state).unsqueeze(0)
        logits = self.policy_net(state_t)
        probs = torch.softmax(logits, dim=1)
        dist = torch.distributions.Categorical(probs)
        action = dist.sample()

        self.log_probs.append(dist.log_prob(action))
        return action.item()

    def store_reward(self, r):
        self.rewards.append(r)

    def update_policy(self):
        """
        REINFORCE: sum of discounted returns for each log_prob.
        """
        discounted = []
        R = 0.0
        for r in reversed(self.rewards):
            R = r + self.gamma * R
            discounted.insert(0, R)
        discounted_t = torch.FloatTensor(discounted)
        # Normalize
        if discounted_t.std() > 1e-8:
            discounted_t = (discounted_t - discounted_t.mean()) / (discounted_t.std() + 1e-8)

        losses = []
        for logp, Gt in zip(self.log_probs, discounted_t):
            losses.append(-logp * Gt)
        loss = torch.stack(losses).sum()

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self.log_probs.clear()
        self.rewards.clear()

##############################################################################
# Simple PDE Environment
##############################################################################
class PricePredictEnv:
    """
    Environment that processes a PDE-labeled CSV:
      columns: [T, ttm_days, Pr, IVOL, CDS, S, cp, cfq, cv, d, r, tfc, Model_Price]
    We'll rename 'Estimated_Price' -> 'Model_Price' prior to using.

    Observations: [CDS, IVOL, S, r, ttm_days] (5D)
    Action: discrete in [0..num_buckets-1], mapped to log-scale PDE guess in [pde_min..pde_max]
    Reward: negative squared error vs. 'Model_Price'.

    In "mini-episode" mode, we process `episode_size` rows at a time. Then done=True.
    Each call to env.reset() moves an internal pointer. We shuffle once to randomize row order.
    """
    def __init__(self, csv_path, num_buckets=50, pde_min=100.0, pde_max=1e8,
                 pde_clip=1e9, episode_size=1000, shuffle_seed=42):
        df = pd.read_csv(csv_path)

        # The environment expects a 'Model_Price' column. If not found, rename 'Estimated_Price'
        if "Model_Price" not in df.columns and "Estimated_Price" in df.columns:
            df = df.rename(columns={"Estimated_Price": "Model_Price"})

        # Filter PDE beyond ±pde_clip
        df = df[df["Model_Price"].abs() <= pde_clip].copy()
        df.reset_index(drop=True, inplace=True)
        self.df_full = df
        self.n_full = len(df)
        if self.n_full == 0:
            raise ValueError("No valid data after PDE clipping. Check pde_clip or your dataset.")

        self.num_buckets = num_buckets
        self.pde_min = pde_min
        self.pde_max = pde_max
        self.pde_clip = pde_clip
        self.episode_size = episode_size

        # Shuffle entire dataset once
        self.df_shuffled = self.df_full.sample(frac=1.0, random_state=shuffle_seed).reset_index(drop=True)
        self.index = 0
        self.done = False

    def reset(self):
        """
        Start a new mini-episode from self.index up to self.index + episode_size.
        If self.index >= self.n_full, return None => no more data.
        """
        self.start_index = self.index
        self.end_index = min(self.index + self.episode_size, self.n_full)
        self.cur_index = self.start_index
        if self.start_index >= self.n_full:
            return None
        self.done = False
        return self._get_state()

    def _get_state(self):
        if self.cur_index >= self.end_index:
            return None
        row = self.df_shuffled.iloc[self.cur_index]
        # We'll pick these 5 columns as input:
        state = [
            row["CDS"],
            row["IVOL"],
            row["S"],
            row["r"],
            row["ttm_days"]
        ]
        return np.array(state, dtype=np.float32)

    def step(self, action):
        if self.done:
            return None, 0.0, True, {}

        row = self.df_shuffled.iloc[self.cur_index]
        true_pde = row["Model_Price"]
        predicted = self._action_to_price(action)

        # clamp error to avoid overflow
        error = true_pde - predicted
        error = np.clip(error, -1e9, 1e9)
        err2 = min(error*error, 1e12)
        reward = -err2

        # Move forward
        self.cur_index += 1
        if self.cur_index >= self.end_index:
            self.done = True
            self.index = self.end_index
            next_state = None
        else:
            next_state = self._get_state()

        return next_state, reward, self.done, {}

    def _action_to_price(self, act):
        """
        log-scale mapping => PDE guess in [pde_min..pde_max].
        fraction = act/(num_buckets-1)
        PDE_guess = exp(log(pde_min) + fraction*(log(pde_max) - log(pde_min)))
        """
        if self.num_buckets <= 1:
            return self.pde_min
        fraction = act / (self.num_buckets - 1)
        import math
        log_min = math.log(self.pde_min)
        log_max = math.log(self.pde_max)
        log_val = log_min + fraction*(log_max - log_min)
        return math.exp(log_val)


##############################################################################
# Utility functions
##############################################################################

def bucket_to_price(action, num_buckets, pde_min, pde_max):
    """Same log-scale mapping used in the environment, for convenience."""
    import math
    if num_buckets <= 1:
        return pde_min
    fraction = action/(num_buckets-1)
    log_min = math.log(pde_min)
    log_max = math.log(pde_max)
    log_val = log_min + fraction*(log_max - log_min)
    return math.exp(log_val)

def evaluate(agent, csv_path, args, tag="eval"):
    """
    Evaluate the RL agent on the entire CSV in a single pass,
    computing MSE, RMSE, R^2, scatter plot, error histogram, etc.

    We do not do multiple episodes here; we do one big pass.
    """
    # Create big environment to read entire CSV in one pass
    env = PricePredictEnv(
        csv_path=csv_path,
        num_buckets=args.num_buckets,
        pde_min=args.pde_min,
        pde_max=args.pde_max,
        pde_clip=args.pde_clip,
        episode_size=10_000_000_000  # large so we can go through entire dataset
    )

    preds = []
    actuals = []

    while True:
        s = env.reset()
        if s is None:
            # no more data
            break
        done = False
        while not done:
            a = agent.select_action(s)
            guess = bucket_to_price(a, args.num_buckets, args.pde_min, args.pde_max)
            idx = env.cur_index
            row = env.df_shuffled.iloc[idx] if idx < env.n_full else None
            if row is not None:
                actuals.append(row["Model_Price"])
                preds.append(guess)

            ns, _, done, _ = env.step(a)
            s = ns

    actuals = np.array(actuals)
    preds = np.array(preds)

    # Compute errors
    errors = actuals - preds
    # clamp error => [-1e9..1e9]
    errors = np.clip(errors, -1e9, 1e9)
    sq_err = errors**2
    sq_err = np.clip(sq_err, 0, 1e12)
    mse = np.mean(sq_err)
    rmse = np.sqrt(mse)
    ss_res = np.sum(sq_err)
    ss_tot = np.sum((actuals - actuals.mean())**2) + 1e-12
    r2 = 1.0 - (ss_res / ss_tot)

    print(f"[{tag.upper()} EVAL] MSE={mse:.2f}, RMSE={rmse:.4f}, R^2={r2:.4f}")

    # Plot scatter (Actual vs. Predicted)
    plt.figure(figsize=(6,6))
    plt.scatter(actuals, preds, alpha=0.3)
    plt.xlabel("Actual PDE Price")
    plt.ylabel("RL Predicted PDE Price")
    plt.title(f"RL PDE Price vs. PDE ({tag})")
    plt.grid(True)
    plt.savefig(f"rl_price_scatter_{tag}.png", dpi=150)
    plt.close()
    print(f"[{tag.upper()} EVAL] Scatter plot => rl_price_scatter_{tag}.png")

    # Plot error histogram
    plt.figure()
    plt.hist(errors, bins=50, alpha=0.7, edgecolor='black')
    plt.title(f"Distribution of Prediction Errors ({tag})")
    plt.xlabel("Error (Actual - Predicted)")
    plt.ylabel("Frequency")
    plt.grid(True)
    plt.savefig(f"rl_price_error_hist_{tag}.png", dpi=150)
    plt.close()
    print(f"[{tag.upper()} EVAL] Error histogram => rl_price_error_hist_{tag}.png")

    return mse, rmse, r2

##############################################################################
# Main script
##############################################################################
def main():
    parser = argparse.ArgumentParser(description="Discrete RL PDE Price Prediction (Train + Evaluate).")
    parser.add_argument("--train-csv", default=None, help="Path to PDE-labeled CSV for training.")
    parser.add_argument("--test-csv", default=None, help="Path to PDE-labeled CSV for testing/eval.")
    parser.add_argument("--episodes", type=int, default=40, help="Number of mini-episodes to train.")
    parser.add_argument("--num-buckets", type=int, default=50, help="Discrete PDE guess buckets.")
    parser.add_argument("--pde-min", type=float, default=100.0, help="Log-scale PDE guess range min.")
    parser.add_argument("--pde-max", type=float, default=1e8, help="Log-scale PDE guess range max.")
    parser.add_argument("--pde-clip", type=float, default=1e9, help="Filter out PDE beyond ±pde_clip.")
    parser.add_argument("--episode-size", type=int, default=1000, help="Rows per mini-episode.")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate.")
    parser.add_argument("--gamma", type=float, default=0.99, help="Discount factor.")
    parser.add_argument("--save-model", default=None, help="Path to save model weights (e.g. 'rl_agent.pt').")
    parser.add_argument("--load-model", default=None, help="Path to load model weights for inference.")
    parser.add_argument("--no-train", action="store_true", help="Skip training step, only evaluate (requires load-model).")
    args = parser.parse_args()

    # Create the agent
    agent = PricePredictAgent(
        input_dim=5,
        num_actions=args.num_buckets,
        hidden_dim=64,
        lr=args.lr,
        gamma=args.gamma
    )

    # Optionally load existing model
    if args.load_model is not None:
        if not os.path.exists(args.load_model):
            raise FileNotFoundError(f"Could not find model file: {args.load_model}")
        agent.policy_net.load_state_dict(torch.load(args.load_model))
        print(f"[INFO] Loaded model weights from {args.load_model}")

    ############################################################################
    # TRAINING STAGE
    ############################################################################
    ep_rewards = []
    if (not args.no_train) and (args.train_csv is not None):
        print("[INFO] Starting TRAINING ...")
        # Create environment for training
        env = PricePredictEnv(
            csv_path=args.train_csv,
            num_buckets=args.num_buckets,
            pde_min=args.pde_min,
            pde_max=args.pde_max,
            pde_clip=args.pde_clip,
            episode_size=args.episode_size
        )

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

            # Update policy after each episode
            agent.update_policy()
            ep_rewards.append(ep_reward)
            print(f"[Episode {ep+1}/{args.episodes}] sum of rewards={ep_reward:.2f}")

        # Plot episode reward
        plt.figure()
        plt.plot(ep_rewards, marker='o')
        plt.title("Episode Reward Over Training")
        plt.xlabel("Episode")
        plt.ylabel("Sum of Rewards")
        plt.grid(True)
        plt.savefig("rl_price_rewards_train.png", dpi=150)
        plt.close()
        print("[TRAIN] Episode reward plot => rl_price_rewards_train.png")

        # Optionally save model
        if args.save_model is not None:
            torch.save(agent.policy_net.state_dict(), args.save_model)
            print(f"[INFO] Saved model weights to {args.save_model}")

    ############################################################################
    # EVALUATION on TRAIN SET (final pass)
    ############################################################################
    if (args.train_csv is not None) and (not args.no_train):
        print("[INFO] Evaluating on TRAIN set after training ...")
        evaluate(agent, args.train_csv, args, tag="train")

    ############################################################################
    # EVALUATION on TEST SET
    ############################################################################
    if args.test_csv is not None:
        print("[INFO] Evaluating on TEST set ...")
        evaluate(agent, args.test_csv, args, tag="test")


if __name__ == "__main__":
    main()
