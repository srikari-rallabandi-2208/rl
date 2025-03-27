# environment_3m_data.py

import numpy as np
import pandas as pd
from tf_engine_converged import TsiveriotisFernandesEngineConverged


class DataEnv3M:
    """
    Uses each row of a 3M dataset as a 'time step.'
    We assume columns: S => stock_price, ttm_days => time to maturity in days, etc.
    We call PDE get_cb_value(S, t) for the reward logic.
    """

    def __init__(self, csv_path, engine: TsiveriotisFernandesEngineConverged):
        # Load all data
        self.df = pd.read_csv(csv_path)
        self.n = len(self.df)
        self.engine = engine
        self.index = 0
        self.done = False

    def reset(self):
        self.index = 0
        self.done = False
        return self._get_state()

    def _get_state(self):
        if self.index >= self.n:
            return None
        row = self.df.iloc[self.index]
        # state: [S, t, PDE_price] => PDE_price is computed on the fly
        S = float(row["stock_price"])
        t_days = float(row["ttm_days"])
        t = t_days / 365.0
        pde_price = self.engine.get_cb_value(S, t)
        return np.array([S, t, pde_price], dtype=np.float32)

    def step(self, action):
        """
        0 => hold => reward=0, proceed to next row
        1 => convert => immediate payoff = (S/K)*par - PDE_price, then done
        """
        state = self._get_state()
        if state is None:
            self.done = True
            return None, 0.0, self.done, {}

        S, t, pde_price = state
        conversion_value = (S / self.engine.K) * self.engine.par

        if action == 1:
            reward = conversion_value - pde_price
            self.done = True
        else:
            reward = 0.0

        self.index += 1
        if self.index >= self.n:
            self.done = True

        next_state = None
        if not self.done:
            next_state = self._get_state()
        return next_state, reward, self.done, {}
