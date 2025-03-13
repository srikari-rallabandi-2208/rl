"""
Environment for an American call option, for use with a policy gradient RL agent.
- The agent sees [S_t, t_remaining].
- Action = 0 (hold) or 1 (exercise).
- If exercise: reward = (S_t - K)+, episode ends.
- If hold until maturity, reward = (S_T - K)+ at final step.
"""

import numpy as np


class AmericanCallEnv:
    def __init__(self, S0, K, r, q, sigma, T, num_steps=50, seed=None):
        """
        :param S0: initial stock price
        :param K: strike
        :param r: risk-free rate
        :param q: dividend yield
        :param sigma: volatility
        :param T: total time to maturity (years)
        :param num_steps: how many discrete steps in [0, T] for the simulation
        """
        self.S0 = S0
        self.K = K
        self.r = r
        self.q = q
        self.sigma = sigma
        self.T = T
        self.num_steps = num_steps
        self.dt = T / num_steps
        self.discount_factor = np.exp(-r * self.dt)

        self.rng = np.random.default_rng(seed)
        self.current_step = 0
        self.stock_path = None
        self.done = False

    def _simulate_gbm_path(self):
        """
        Geometric Brownian Motion path from t=0 to t=T in num_steps increments.
        """
        path = np.zeros(self.num_steps + 1)
        path[0] = self.S0
        for i in range(1, self.num_steps + 1):
            Z = self.rng.normal()
            path[i] = path[i - 1] * np.exp(
                (self.r - self.q - 0.5 * self.sigma ** 2) * self.dt + self.sigma * np.sqrt(self.dt) * Z)
        return path

    def reset(self):
        self.current_step = 0
        self.done = False
        self.stock_path = self._simulate_gbm_path()
        return self._get_obs()

    def _get_obs(self):
        S_t = self.stock_path[self.current_step]
        t_remaining = self.T - self.current_step * self.dt
        return np.array([S_t, t_remaining], dtype=np.float32)

    def step(self, action):
        """
        action = 0 -> hold
        action = 1 -> exercise
        Returns: next_obs, reward, done, {}
        """
        S_t = self.stock_path[self.current_step]
        payoff = max(S_t - self.K, 0.0)

        if action == 1:
            # immediate exercise
            reward = payoff
            self.done = True
        else:
            # hold
            reward = 0.0
            self.current_step += 1
            if self.current_step >= self.num_steps:
                # at maturity
                S_T = self.stock_path[self.num_steps]
                reward = max(S_T - self.K, 0.0)
                self.done = True

        # discount the reward a tiny bit for each step
        # (some people accumulate discount over steps, or do discounting in the returns)
        # We'll keep it simpler: no immediate discounting, rely on returns discount in PG.
        next_obs = self._get_obs() if not self.done else None
        return next_obs, reward, self.done, {}
