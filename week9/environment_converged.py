"""
environment_converged.py

Defines an RL environment that references a converged PDE engine
for reliable convertible bond pricing signals.
"""

import numpy as np
from tf_engine_converged import TsiveriotisFernandesEngineConverged


class ConvertibleBondEnvConverged:
    """
    RL Environment that uses the converged PDE engine for pricing.
    State: [stock_price, time, PDE_price].
    Actions: 0 -> hold, 1 -> convert.
    """

    def __init__(self, engine: TsiveriotisFernandesEngineConverged, stock_path: np.ndarray, times: np.ndarray):
        """
        engine : a TsiveriotisFernandesEngineConverged instance with solved PDE grid.
        stock_path : array of stock prices over time.
        times : array of time points corresponding to stock_path.
        """
        self.engine = engine
        self.stock_path = stock_path
        self.times = times
        self.num_timesteps = len(times) - 1
        self.current_step = 0
        self.done = False

        # We assume engine.solve_pde() has already been called.

    def reset(self):
        self.current_step = 0
        self.done = False
        return self._get_state()

    def _get_state(self):
        S = self.stock_path[self.current_step]
        t = self.times[self.current_step]
        cb_price = self.engine.get_cb_value(S, t)
        return np.array([S, t, cb_price], dtype=np.float32)

    def step(self, action: int):
        """
        If action == 1 => convert, we compute immediate payoff = conversion_value - PDE_price.
        If action == 0 => hold, reward=0, move forward in time unless at end -> done.
        """
        state = self._get_state()
        S, t, cb_price_before = state
        conversion_value = (S / self.engine.K) * self.engine.par

        if action == 1:
            reward = conversion_value - cb_price_before
            self.done = True
        else:
            reward = 0.0
            self.done = False

        if not self.done:
            self.current_step += 1
            if self.current_step >= self.num_timesteps:
                self.done = True

        next_state = self._get_state()
        return next_state, reward, self.done, {}
