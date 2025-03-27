"""
rl_agent_converged.py

Implements a REINFORCE (Policy Gradient) agent for discrete actions:
  - input_dim=3 => [S, t, PDE_price]
  - output_dim=2 => [hold, convert]
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import os


class PolicyNetworkConverged(nn.Module):
    def __init__(self, input_dim=3, hidden_dim=64, output_dim=2):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)
        self._init_weights()

    def _init_weights(self):
        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.zeros_(self.fc1.bias)
        nn.init.xavier_uniform_(self.fc2.weight)
        nn.init.zeros_(self.fc2.bias)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        return self.fc2(x)


class PolicyGradientAgentConverged:
    def __init__(self, input_dim=3, hidden_dim=64, output_dim=2,
                 lr=1e-3, gamma=0.99):
        self.gamma = gamma
        self.policy_net = PolicyNetworkConverged(input_dim, hidden_dim, output_dim)
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=lr)
        self.log_probs = []
        self.rewards = []

    def select_action(self, state: np.ndarray) -> int:
        """
        state shape = (3,)
        forward => 2 logits => softmax => sample action
        """
        state_t = torch.FloatTensor(state).unsqueeze(0)  # shape=[1,3]
        logits = self.policy_net(state_t)
        probs = torch.softmax(logits, dim=1)
        dist = torch.distributions.Categorical(probs)
        action = dist.sample()

        self.log_probs.append(dist.log_prob(action))
        return action.item()

    def store_reward(self, reward: float):
        self.rewards.append(reward)

    def update_policy(self):
        # compute discounted returns
        discounted_returns = []
        R = 0.0
        for r in reversed(self.rewards):
            R = r + self.gamma * R
            discounted_returns.insert(0, R)

        discounted_t = torch.tensor(discounted_returns, dtype=torch.float32)
        if discounted_t.std() > 1e-6:
            discounted_t = (discounted_t - discounted_t.mean()) / (discounted_t.std() + 1e-6)

        losses = []
        for log_p, Gt in zip(self.log_probs, discounted_t):
            losses.append(-log_p * Gt)
        loss = torch.stack(losses).sum()

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self.log_probs.clear()
        self.rewards.clear()

    def save_model(self, path):
        torch.save(self.policy_net.state_dict(), path)

    def load_model(self, path):
        if os.path.exists(path):
            self.policy_net.load_state_dict(torch.load(path))
            self.policy_net.eval()
        else:
            raise FileNotFoundError(f"Model not found at {path}")
