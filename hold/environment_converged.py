"""
environment_converged.py

Defines a "hold vs. convert" RL environment using the PDE engine.

State = [S, t, PDE_price]
Action = 0 => hold, 1 => convert
Reward:
  If convert => immediate payoff = conversion_value - PDE_price
  Else => 0, move to next step
"""

import numpy as np
from tf_engine_converged import TsiveriotisFernandesEngineConverged


class ConvertibleBondEnvConverged:
    def __init__(self, engine: TsiveriotisFernandesEngineConverged,
                 stock_path: np.ndarray, times: np.ndarray):
        """
        engine: PDE engine
        stock_path: array of stock prices over time
        times: array of times (same length as stock_path)
        """
        self.engine = engine
        self.stock_path = stock_path
        self.times = times
        self.num_timesteps = len(times) - 1
        self.current_step = 0
        self.done = False

        if len(stock_path) != len(times):
            raise ValueError("stock_path and times must match in length.")

    def reset(self):
        self.current_step = 0
        self.done = False
        return self._get_state()

    def _get_state(self):
        S = self.stock_path[self.current_step]
        t = self.times[self.current_step]
        pde_price = self.engine.get_cb_value(S, t)
        return np.array([S, t, pde_price], dtype=np.float32)

    def step(self, action: int):
        """
        action=1 => convert => payoff = (S/K)*par - PDE_price, then done
        action=0 => hold => reward=0, move forward unless at final step
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

        next_state = None
        if not self.done:
            next_state = self._get_state()
        return next_state, reward, self.done, {}
