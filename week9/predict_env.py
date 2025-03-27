"""
predict_env.py

Modified environment that:
1) Loads final CSV, filtering out rows where Model_Price is too large (e.g. >1e7).
2) At each step, we clamp the squared error to avoid infinite or extreme negative rewards.
3) We still do discrete actions => PDE guess in [0..max_price].
"""

import numpy as np
import pandas as pd

class PricePredictEnv:
    def __init__(self, csv_path, num_buckets=20, max_price=2000.0, pde_clip=1e7):
        """
        Parameters
        ----------
        csv_path : str
            Path to PDE-labeled data with columns:
            [CDS, IVOL, S, Pr, r, ttm_days, cp, cfq, conv_ratio, cv, d,
             issuance_date, first_coupon_date, Model_Price]
        num_buckets : int
            Discrete actions => [0..num_buckets-1].
        max_price : float
            Highest PDE guess the agent can produce.
        pde_clip : float
            We remove rows with abs(Model_Price) > pde_clip to avoid huge outliers.
        """
        # 1. Load CSV
        df = pd.read_csv(csv_path)

        # 2. Filter out insane PDE values
        df = df[df["Model_Price"].abs() < pde_clip].copy()
        df.reset_index(drop=True, inplace=True)

        # 3. Store data
        self.df = df
        self.n = len(df)
        if self.n == 0:
            raise ValueError(f"No valid rows left after PDE clipping at {pde_clip}.")

        # 4. Config
        self.num_buckets = num_buckets
        self.max_price = max_price

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
        # We'll create a 5-dimensional state:
        # [CDS, IVOL, S, r, ttm_days]
        state = [
            row["CDS"],
            row["IVOL"],
            row["S"],
            row["r"],
            row["ttm_days"]
        ]
        return np.array(state, dtype=np.float32)

    def step(self, action):
        """
        1) Convert action => PDE guess
        2) error = true_price - guess
        3) reward = -(error^2) but clamp huge squared error to avoid inf
        4) move next row
        """
        if self.done:
            return None, 0.0, True, {}

        row = self.df.iloc[self.index]
        true_price = row["Model_Price"]
        predicted = self._action_to_price(action)

        error = true_price - predicted
        # clamp error to e.g. [-1e9..1e9]
        if error > 1e9:
            error = 1e9
        elif error < -1e9:
            error = -1e9

        err2 = error * error
        # clamp the squared error
        if err2 > 1e12:
            err2 = 1e12

        reward = -err2

        self.index += 1
        if self.index >= self.n:
            self.done = True
            next_state = None
        else:
            next_state = self._get_state()

        return next_state, reward, self.done, {}

    def _action_to_price(self, action):
        """
        linearly scale action in [0..num_buckets-1] => PDE in [0..max_price].
        """
        if self.num_buckets <= 1:
            return 0.0
        return (action/(self.num_buckets-1)) * self.max_price
