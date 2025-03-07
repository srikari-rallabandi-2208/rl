"""
rl_agent.py

Defines the RL agent using a REINFORCE-style policy gradient.
Includes the policy network and the agent class, with copious comments
for clarity. The agent learns an optimal conversion policy for the
convertible bond environment.

Example usage:
    (Imported and used within main_rl.py)
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import os


class PolicyNetwork(nn.Module):
    """
    A feed-forward neural network that maps a normalized [S, t, PDE_price] state to logits.
    Output dimension = 2 => (hold, convert).
    """

    def __init__(self, input_dim=3, hidden_dim=32, output_dim=2):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, output_dim)
        self._init_weights()

    def _init_weights(self):
        # Xavier Uniform initialization is often a good default for MLPs
        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.zeros_(self.fc1.bias)
        nn.init.xavier_uniform_(self.fc2.weight)
        nn.init.zeros_(self.fc2.bias)
        nn.init.xavier_uniform_(self.fc3.weight)
        nn.init.zeros_(self.fc3.bias)

    def forward(self, x):
        # Simple two-layer ReLU network
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)


class PolicyGradientAgent:
    """
    REINFORCE policy gradient agent:
    - Stores log probabilities and rewards during an episode
    - Updates the policy at the end of each episode
    """

    def __init__(self, input_dim=3, hidden_dim=32, output_dim=2, lr=1e-3, gamma=0.99):
        self.gamma = gamma
        self.policy_net = PolicyNetwork(input_dim, hidden_dim, output_dim)
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=lr)
        # We store the log probabilities and rewards from each step
        self.log_probs = []
        self.rewards = []

    def select_action(self, state):
        """
        Select an action given the current state.
        State is [S, t, PDE_price].
        We normalize stock price and PDE price by 100.
        """
        norm_state = np.array([state[0] / 100.0, state[1], state[2] / 100.0], dtype=np.float32)

        # Convert to a torch tensor
        state_t = torch.FloatTensor(norm_state).unsqueeze(0)

        # Forward pass => get logits
        logits = self.policy_net(state_t)

        # Softmax to get action probabilities
        probs = torch.softmax(logits, dim=1)

        # Sample an action from the categorical distribution
        dist = torch.distributions.Categorical(probs)
        action = dist.sample()

        # Store the log probability for future gradient calculation
        self.log_probs.append(dist.log_prob(action))

        return action.item()

    def store_reward(self, reward):
        """
        Store the reward at each step so we can do a full-episode update.
        """
        self.rewards.append(reward)

    def update_policy(self):
        """
        REINFORCE update: compute discounted returns, multiply by log probs, and do a gradient step.
        """
        # Skip update if no rewards
        if not self.rewards:
            return

        # 1. Compute discounted returns
        discounted_returns = []
        R = 0
        for r in reversed(self.rewards):
            R = r + self.gamma * R
            discounted_returns.insert(0, R)

        discounted_returns = torch.FloatTensor(discounted_returns)
        # Normalize returns to stabilize training
        if discounted_returns.std() > 1e-8:
            discounted_returns = (discounted_returns - discounted_returns.mean()) / (discounted_returns.std() + 1e-8)

        # 2. Compute the policy gradient loss
        loss = []
        for log_prob, ret in zip(self.log_probs, discounted_returns):
            loss.append(-log_prob * ret)
        loss = torch.stack(loss).sum()

        # 3. Take an optimization step
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), max_norm=1.0)
        self.optimizer.step()

        # 4. Clear stored rewards and log probabilities
        self.log_probs.clear()
        self.rewards.clear()

    def save_model(self, path="data/policy_model.pth"):
        """
        Save the policy network's parameters.
        """
        torch.save(self.policy_net.state_dict(), path)
        print(f"Model saved to {path}")

    def load_model(self, path="data/policy_model.pth"):
        """
        Load previously saved model parameters.
        """
        if os.path.exists(path):
            self.policy_net.load_state_dict(torch.load(path))
            self.policy_net.eval()
            print(f"Model loaded from {path}")
        else:
            raise FileNotFoundError(f"Model file {path} does not exist.")