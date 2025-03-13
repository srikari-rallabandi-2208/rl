"""
rl_agent_converged.py

We reuse the REINFORCE policy gradient approach but point it at
our 'environment_converged.py' environment for training on the converged PDE.
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import os


class PolicyNetworkConverged(nn.Module):
    # same as before, just with updated naming
    def __init__(self, input_dim=3, hidden_dim=64, output_dim=2):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, output_dim)
        self._init_weights()

    def _init_weights(self):
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


class PolicyGradientAgentConverged:
    """
    Same logic as your existing REINFORCE agent.
    """

    def __init__(self, input_dim=3, hidden_dim=64, output_dim=2, lr=1e-3, gamma=0.99):
        self.gamma = gamma
        self.policy_net = PolicyNetworkConverged(input_dim, hidden_dim, output_dim)
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=lr)
        self.log_probs = []
        self.rewards = []

    def select_action(self, state):
        # Normalization might vary. We'll do something simple:
        norm_s = state[0] / 100.0
        norm_t = state[1]  # if T=1.0, then t in [0,1]
        norm_p = state[2] / 100.0
        in_vec = np.array([norm_s, norm_t, norm_p], dtype=np.float32)
        state_t = torch.FloatTensor(in_vec).unsqueeze(0)

        logits = self.policy_net(state_t)
        probs = torch.softmax(logits, dim=1)
        dist = torch.distributions.Categorical(probs)
        action = dist.sample()
        self.log_probs.append(dist.log_prob(action))

        return action.item()

    def store_reward(self, reward):
        self.rewards.append(reward)

    def update_policy(self):
        if not self.rewards:
            return
        # compute discounted returns
        discounted = []
        R = 0.0
        for r in reversed(self.rewards):
            R = r + self.gamma * R
            discounted.insert(0, R)
        discounted = torch.FloatTensor(discounted)
        if discounted.std() > 1e-8:
            discounted = (discounted - discounted.mean()) / (discounted.std() + 1e-8)

        # policy gradient
        losses = []
        for log_p, Gt in zip(self.log_probs, discounted):
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
            raise FileNotFoundError(f"Model file {path} not found.")
