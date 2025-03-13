"""
Policy Gradient agent (REINFORCE-style):
- Stores log_probs and rewards over an episode.
- On episode end, performs policy update using discounted returns.
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import os


class PolicyNetwork(nn.Module):
    def __init__(self, input_dim=2, hidden_dim=64, output_dim=2):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, output_dim)

        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.zeros_(self.fc1.bias)
        nn.init.xavier_uniform_(self.fc2.weight)
        nn.init.zeros_(self.fc2.bias)
        nn.init.xavier_uniform_(self.fc3.weight)
        nn.init.zeros_(self.fc3.bias)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)  # shape: [batch_size, 2]


class PolicyGradientAgent:
    def __init__(self, input_dim=2, hidden_dim=64, output_dim=2, lr=1e-3, gamma=1.0):
        # gamma=1.0 => no discount across episode steps,
        # but you can reduce it if you want actual discounting
        self.gamma = gamma
        self.policy_net = PolicyNetwork(input_dim, hidden_dim, output_dim)
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=lr)
        self.log_probs = []
        self.rewards = []

    def select_action(self, state):
        """
        state: np.array of shape [2,]
        return: int (0 or 1)
        """
        state_t = torch.from_numpy(state).float().unsqueeze(0)  # shape: [1,2]
        logits = self.policy_net(state_t)  # shape: [1,2]
        probs = torch.softmax(logits, dim=1)  # shape: [1,2]
        dist = torch.distributions.Categorical(probs)
        action = dist.sample()
        self.log_probs.append(dist.log_prob(action))
        return action.item()

    def store_reward(self, reward):
        self.rewards.append(reward)

    def update_policy(self):
        """
        REINFORCE update: compute discounted return * log_prob
        """
        # 1) Compute discounted returns
        discounted_returns = []
        G = 0
        for r in reversed(self.rewards):
            G = r + self.gamma * G
            discounted_returns.insert(0, G)
        discounted_returns = torch.tensor(discounted_returns, dtype=torch.float32)

        # Normalize returns to help training
        if discounted_returns.std() > 1e-9:
            discounted_returns = (discounted_returns - discounted_returns.mean()) / (discounted_returns.std() + 1e-9)

        # 2) Compute policy gradient loss
        loss = []
        for log_prob, Gt in zip(self.log_probs, discounted_returns):
            loss.append(-log_prob * Gt)
        loss = torch.stack(loss).sum()

        # 3) backprop
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # 4) reset buffers
        self.log_probs.clear()
        self.rewards.clear()

    def save(self, path="data/rl_policy_model.pth"):
        torch.save(self.policy_net.state_dict(), path)

    def load(self, path="data/rl_policy_model.pth"):
        if not os.path.exists(path):
            raise FileNotFoundError(f"No model at {path}")
        self.policy_net.load_state_dict(torch.load(path))
        self.policy_net.eval()
