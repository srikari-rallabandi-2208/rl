"""
rl_price_predict.py

Defines:
  - PricePredictNetwork: a neural net that outputs 'num_actions' logits
    for discrete PDE predictions.
  - PricePredictAgent: a policy gradient (REINFORCE) agent that picks
    from [0..num_actions-1], storing log_probs & rewards, and updates
    at the end of each episode.

No references to PDE columns here; it just deals with the environment's
state dimension (input_dim) and the discrete action space (num_actions).
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

class PricePredictNetwork(nn.Module):
    def __init__(self, input_dim=5, num_actions=50, hidden_dim=64):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, num_actions)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        return self.fc2(x)

class PricePredictAgent:
    def __init__(self, input_dim=5, num_actions=50, hidden_dim=64,
                 lr=1e-3, gamma=0.99):
        self.gamma = gamma
        self.num_actions = num_actions
        self.policy_net = PricePredictNetwork(input_dim, num_actions, hidden_dim)
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=lr)
        self.log_probs = []
        self.rewards = []

    def select_action(self, state):
        """
        Given a state, produce action in [0..num_actions-1].
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
        REINFORCE: sum of discounted returns.
        """
        discounted = []
        R = 0.0
        for r in reversed(self.rewards):
            R = r + self.gamma*R
            discounted.insert(0, R)
        discounted_t = torch.FloatTensor(discounted)
        if discounted_t.std() > 1e-6:
            discounted_t = (discounted_t - discounted_t.mean())/(discounted_t.std()+1e-6)

        losses = []
        for logp, Gt in zip(self.log_probs, discounted_t):
            losses.append(-logp * Gt)
        loss = torch.stack(losses).sum()

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self.log_probs.clear()
        self.rewards.clear()
