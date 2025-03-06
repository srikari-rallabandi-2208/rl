"""
rl_agent.py

This module defines the RL agent using a REINFORCE-style policy gradient.
It includes the policy network definition and the agent class.
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import os


class PolicyNetwork(nn.Module):
    """
    A simple feed-forward neural network that maps a normalized state to logits.
    The state is normalized by scaling the stock price and PDE price by 100.
    Output: logits for two actions (hold or convert).
    """

    def __init__(self, input_dim=3, hidden_dim=32, output_dim=2):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, output_dim)
        self._init_weights()

    def _init_weights(self):
        # Initialize weights using Xavier uniform initialization.
        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.zeros_(self.fc1.bias)
        nn.init.xavier_uniform_(self.fc2.weight)
        nn.init.zeros_(self.fc2.bias)
        nn.init.xavier_uniform_(self.fc3.weight)
        nn.init.zeros_(self.fc3.bias)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)


class PolicyGradientAgent:
    """
    Implements a REINFORCE policy gradient agent.
    Stores log probabilities and rewards during an episode and updates the policy at episode end.
    """

    def __init__(self, input_dim=3, hidden_dim=32, output_dim=2, lr=1e-3, gamma=0.99):
        self.gamma = gamma
        self.policy_net = PolicyNetwork(input_dim, hidden_dim, output_dim)
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=lr)
        self.log_probs = []
        self.rewards = []

    def select_action(self, state):
        """
        Selects an action based on the current state.
        Normalizes the state, computes logits, checks for NaNs,
        and samples an action from the resulting categorical distribution.
        """
        norm_state = np.array([state[0] / 100.0, state[1], state[2] / 100.0], dtype=np.float32)
        state_t = torch.FloatTensor(norm_state).unsqueeze(0)

        # If NaNs occur, warn and use default action.
        if torch.isnan(state_t).any():
            print("Warning: normalized state contains NaN:", norm_state)
            state_t = torch.nan_to_num(state_t, nan=0.0, posinf=1e6, neginf=-1e6)

        logits = self.policy_net(state_t)
        if torch.isnan(logits).any():
            print("Warning: logits contain NaN. State:", norm_state, "Logits:", logits)
            dummy = torch.tensor(0.0, requires_grad=True)
            self.log_probs.append(dummy)
            return 0  # default to hold

        probs = torch.softmax(logits, dim=1)
        if torch.isnan(probs).any():
            print("Warning: probabilities contain NaN. Logits:", logits)
            dummy = torch.tensor(0.0, requires_grad=True)
            self.log_probs.append(dummy)
            return 0  # default to hold

        dist = torch.distributions.Categorical(probs)
        action = dist.sample()
        self.log_probs.append(dist.log_prob(action))
        return action.item()

    def store_reward(self, reward):
        """Stores the reward for the current timestep."""
        self.rewards.append(reward)

    def update_policy(self):
        """
        Updates the policy network using the REINFORCE algorithm.
        Computes discounted returns, normalizes them, and performs a gradient step.
        If dummy log probabilities were used, the update is skipped.
        """
        if not self.rewards:
            return

        if any(not lp.requires_grad for lp in self.log_probs):
            print("Skipping policy update due to dummy log probabilities.")
            self.log_probs.clear()
            self.rewards.clear()
            return

        discounted_returns = []
        R = 0
        for r in reversed(self.rewards):
            R = r + self.gamma * R
            discounted_returns.insert(0, R)
        discounted_returns = torch.FloatTensor(discounted_returns)
        std = discounted_returns.std()
        if std < 1e-8:
            normalized_returns = discounted_returns
        else:
            normalized_returns = (discounted_returns - discounted_returns.mean()) / (std + 1e-8)

        loss_list = []
        for log_prob, Gt in zip(self.log_probs, normalized_returns):
            loss_list.append(-log_prob * Gt)
        if not loss_list:
            print("Warning: No valid log probabilities; skipping update.")
            self.log_probs.clear()
            self.rewards.clear()
            return
        loss = torch.stack(loss_list).sum()

        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), max_norm=1.0)
        self.optimizer.step()
        self.log_probs.clear()
        self.rewards.clear()

    def save_model(self, path="policy_model.pth"):
        """
        Saves the model parameters to disk.
        """
        torch.save(self.policy_net.state_dict(), path)
        print(f"Model saved to {path}")

    def load_model(self, path="policy_model.pth"):
        """
        Loads model parameters from disk.
        """
        if os.path.exists(path):
            self.policy_net.load_state_dict(torch.load(path))
            self.policy_net.eval()
            print(f"Model loaded from {path}")
        else:
            raise FileNotFoundError(f"Model file {path} does not exist.")


if __name__ == "__main__":
    # Quick test to ensure the policy network runs.
    agent = PolicyGradientAgent()
    sample_state = [100, 0.0, 100]  # Example state: [stock price, time, PDE price]
    action = agent.select_action(sample_state)
    print("Selected action:", action)
