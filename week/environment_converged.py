"""
environment_converged.py

An RL environment that uses the PDE for hold vs. convert logic.

State: [stock_price, time, PDE_price]
Actions: 0 -> hold, 1 -> convert
Reward:
  if convert => immediate payoff = conversion_value - PDE_price
  else => 0, move forward in time
"""

import numpy as np
from tf_engine_converged import TsiveriotisFernandesEngineConverged


class ConvertibleBondEnvConverged:
    def __init__(self, engine: TsiveriotisFernandesEngineConverged, stock_path: np.ndarray, times: np.ndarray):
        """
        engine: PDE engine (already solved).
        stock_path: array of stock prices over time
        times: array of times [0..T]
        """
        self.engine = engine
        self.stock_path = stock_path
        self.times = times
        self.num_timesteps = len(times) - 1
        self.current_step = 0
        self.done = False

        if len(stock_path) != len(times):
            raise ValueError("stock_path and times must have same length")

    def reset(self):
        self.current_step = 0
        self.done = False
        return self._get_state()

    def _get_state(self):
        # PDE price at current step
        S = self.stock_path[self.current_step]
        t = self.times[self.current_step]
        cb_price = self.engine.get_cb_value(S, t)
        return np.array([S, t, cb_price], dtype=np.float32)

    def step(self, action: int):
        """
        action=1 => convert => immediate payoff = (S/K)*par - PDE_price
        action=0 => hold => reward=0, go to next step unless at the end.
        """
        state = self._get_state()
        S, t, pde_price = state
        conversion_value = (S / self.engine.K) * self.engine.par

        if action == 1:
            reward = conversion_value - pde_price
            self.done = True
        else:
            reward = 0.0
            self.done = False

        # Next step
        if not self.done:
            self.current_step += 1
            if self.current_step >= self.num_timesteps:
                self.done = True

        next_state = self._get_state() if not self.done else None
        return next_state, reward, self.done, {}
