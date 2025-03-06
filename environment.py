"""
environment.py

This module defines the RL environment for convertible bond pricing.
It uses the TsiveriotisFernandesEngine to compute the PDE-based convertible bond price,
which is included as part of the state for the RL agent.
"""

import numpy as np
from tf_engine import TsiveriotisFernandesEngine


class ConvertibleBondEnvTF:
    """
    RL Environment for a convertible bond pricing problem.

    The environment provides a state vector:
      [current_stock_price, current_time, PDE_computed_bond_price]
    The action space is:
      0 -> hold, 1 -> convert
    """

    def __init__(self, engine: TsiveriotisFernandesEngine, stock_path: np.ndarray, times: np.ndarray):
        self.engine = engine
        self.stock_path = stock_path
        self.times = times
        self.num_timesteps = len(times) - 1
        self.current_step = 0
        self.done = False

        # Solve the PDE once at initialization.
        self.engine.solve_pde()

    def reset(self):
        """
        Resets the environment to the starting state.
        """
        self.current_step = 0
        self.done = False
        return self._get_state()

    def _get_state(self):
        """
        Returns the current state as a numpy array [stock price, time, PDE bond price].
        """
        S = self.stock_path[self.current_step]
        t = self.times[self.current_step]
        cb_price = self.engine.get_cb_value(S, t)
        return np.array([S, t, cb_price], dtype=np.float32)

    def step(self, action: int):
        """
        Executes an action:
          - If action==1 (convert), computes reward and ends episode.
          - If action==0 (hold), continues to next step with reward 0.
        Returns: next_state, reward, done, and info dict.
        """
        state = self._get_state()
        S, t, cb_price_before = state
        conversion_value = (S / self.engine.K) * self.engine.par

        if action == 1:  # convert
            reward = conversion_value - cb_price_before
            self.done = True
        else:
            reward = 0.0
            self.done = False

        # Advance time if not done.
        if not self.done:
            self.current_step += 1
            if self.current_step >= self.num_timesteps:
                self.done = True

        next_state = self._get_state()
        return next_state, reward, self.done, {}
