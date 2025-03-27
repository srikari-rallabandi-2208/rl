"""
main_dynamic_rl.py

Trains a policy gradient RL agent with "hold vs. convert" actions,
using an environment that calls PDEEngine for each step's PDE value.

We produce a final plot showing stock vs PDE vs conversion decisions.

Usage:
  python main_dynamic_rl.py --output data/dyn_40k
"""

import argparse
import numpy as np
import matplotlib.pyplot as plt
import torch
import os

from pde_engine import PDEEngine
from rl_agent import RLAgent
import math

class ConvertibleBondEnv:
    """
    Each step => we have stock_price(t). We compute PDE at (S, t).
    Action = 0 => hold, reward=0
    Action = 1 => convert => reward = conversion_value - PDE_price, then done
    We do a simple discrete time approach for T=1.0, steps=100, etc.
    """
    def __init__(self, engine:PDEEngine, stock_path, times, K=100.0, par=100.0, r=0.05, d=0.0, spread=0.01):
        self.engine = engine
        self.stock_path = stock_path
        self.times = times
        self.K = K
        self.par = par
        self.current_step = 0
        self.num_steps = len(times)
        self.done = False

    def reset(self):
        self.current_step = 0
        self.done = False
        return self._get_state()

    def _get_state(self):
        # state = [S, t, PDE_price]
        S = self.stock_path[self.current_step]
        t = self.times[self.current_step]
        # We'll create a "row" with placeholders for PDEEngine
        # PDEEngine expects: [Pr, IVOL, CDS, S, r, d, cv, ...], but let's do minimal
        row = {
            "Pr": self.par,
            "IVOL": 20.0,   # or a smaller subset
            "CDS": spread_to_bps(0.01),
            "S": S,
            "r": 0.05,
            "d": 0.0,
            "cv": self.K,
            "ttm_days": t*365.0,
        }
        price = self.engine.solve_for_row(row)
        if math.isnan(price):
            price = 0.0  # or fallback

        return np.array([S, t, price], dtype=np.float32)

    def step(self, action):
        """
        If action=1 => convert => payoff = (S/K)*par - PDE_price => done
        else => 0, move to next step
        """
        state = self._get_state()
        S, t, pde_price = state
        conversion_value = (S / self.K) * self.par

        if action==1:
            reward = conversion_value - pde_price
            self.done = True
        else:
            reward = 0.0
            self.done = False

        if not self.done:
            self.current_step += 1
            if self.current_step >= self.num_steps:
                self.done = True

        next_state = None
        if not self.done:
            next_state = self._get_state()
        return next_state, reward, self.done, {}

def spread_to_bps(spread_decimal):
    return spread_decimal*10000

class RLAgent:
    """
    Simple REINFORCE agent with a 3->2 net: input= [S,t,PDE], action= [hold, convert].
    """
    def __init__(self, hidden_dim=64, lr=1e-3, gamma=0.99):
        import torch.nn as nn
        import torch.optim as optim

        self.gamma = gamma
        self.policy_net = nn.Sequential(
            nn.Linear(3, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 2)
        )
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=lr)
        self.log_probs = []
        self.rewards = []

    def select_action(self, state):
        import torch
        import torch.nn.functional as F
        state_t = torch.FloatTensor(state).unsqueeze(0)
        logits = self.policy_net(state_t)
        probs = F.softmax(logits, dim=1)
        dist = torch.distributions.Categorical(probs)
        action = dist.sample()
        self.log_probs.append(dist.log_prob(action))
        return action.item()

    def store_reward(self, r):
        self.rewards.append(r)

    def update_policy(self):
        import torch
        discounted = []
        R = 0.0
        for r in reversed(self.rewards):
            R = r + self.gamma*R
            discounted.insert(0, R)
        discounted_t = torch.FloatTensor(discounted)
        if discounted_t.std()>1e-6:
            discounted_t = (discounted_t - discounted_t.mean())/(discounted_t.std()+1e-6)

        losses=[]
        for log_p, ret in zip(self.log_probs, discounted_t):
            losses.append(-log_p*ret)
        loss = torch.stack(losses).sum()

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self.log_probs.clear()
        self.rewards.clear()

def generate_gbm_path(S0=100, mu=0.05, sigma=0.20, T=1.0, steps=100):
    dt = T/(steps-1)
    times = np.linspace(0, T, steps)
    path = np.zeros(steps, dtype=np.float32)
    path[0] = S0
    for i in range(1, steps):
        Z = np.random.normal()
        path[i] = path[i-1]*np.exp((mu-0.5*sigma*sigma)*dt + sigma*math.sqrt(dt)*Z)
    return path, times

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="data/dyn_output", help="Where to store final plot/model")
    parser.add_argument("--episodes", type=int, default=30)
    args = parser.parse_args()

    os.makedirs(args.output, exist_ok=True)

    # PDE engine
    engine = PDEEngine(M=100, N=2000)  # smaller M,N for faster demo
    agent = RLAgent(hidden_dim=64, lr=1e-3, gamma=0.99)

    # Train
    for ep in range(args.episodes):
        # Generate random GBM path
        stock_path, times = generate_gbm_path(100, 0.05, 0.20, 1.0, 50)
        env = ConvertibleBondEnv(engine, stock_path, times, K=100, par=100)

        s = env.reset()
        done=False
        while not done:
            a = agent.select_action(s)
            s_next, r, done, _ = env.step(a)
            agent.store_reward(r)
            s = s_next
        agent.update_policy()
        if (ep+1)%5==0:
            print(f"[Episode {ep+1}/{args.episodes}] done.")

    # Test
    test_stock, test_times = generate_gbm_path(100, 0.05, 0.20, 1.0, 50)
    env = ConvertibleBondEnv(engine, test_stock, test_times, K=100, par=100)
    s = env.reset()
    done=False

    record_times=[]
    record_stock=[]
    record_pde=[]
    record_action=[]

    while not done:
        state = s
        a = agent.select_action(state)
        record_times.append(state[1])
        record_stock.append(state[0])
        record_pde.append(state[2])
        record_action.append(a)

        s_next, r, done, _ = env.step(a)
        agent.store_reward(r)
        s = s_next

    # final update not super necessary for a test path, but let's do it
    agent.update_policy()

    # Plot
    record_times = np.array(record_times)
    record_stock = np.array(record_stock)
    record_pde = np.array(record_pde)
    record_action = np.array(record_action)
    convert_idx = np.where(record_action==1)[0]

    plt.figure(figsize=(8,5))
    plt.plot(record_times, record_stock, label="Stock Price", color="blue")
    plt.plot(record_times, record_pde, label="PDE Price", color="green", linestyle="--")
    plt.scatter(record_times[convert_idx], record_stock[convert_idx],
                color="red", marker="x", s=80, label="Convert")
    plt.xlabel("Time")
    plt.ylabel("Price")
    plt.title("Dynamic RL: PDE vs Stock vs Action")
    plt.grid(True)
    plt.legend()
    plot_path = os.path.join(args.output,"dynamic_rl_plot.png")
    plt.savefig(plot_path, dpi=150)
    plt.close()
    print(f"[DynamicRL] Final plot => {plot_path}")

if __name__=="__main__":
    main()
