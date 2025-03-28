"""
predict_env.py

Environment class for PDE price prediction using discrete log-scale buckets.

Key Points:
1) We chunk the dataset into episodes of 'episode_size' rows.
2) For each step, the agent picks an action in [0..num_buckets-1], which maps
   to a PDE guess via log scale in [pde_min..pde_max].
3) Reward = negative squared error vs. 'Model_Price', clamped to avoid overflow.
"""

import numpy as np
import pandas as pd

class PricePredictEnv:
    def __init__(self, csv_path, num_buckets=50, pde_min=100.0, pde_max=1e8,
                 pde_clip=1e9, episode_size=1000):
        """
        csv_path : str
            PDE-labeled CSV with columns [CDS,IVOL,S,r,ttm_days,Model_Price,etc].
        num_buckets : int
            Discrete PDE guess buckets => [0..num_buckets-1].
        pde_min : float
            Lower bound for PDE guess in log scale.
        pde_max : float
            Upper bound for PDE guess in log scale.
        pde_clip : float
            Filter out data rows with PDE > +/- this. E.g. 1e9.
        episode_size : int
            Each episode processes these many rows before done = True.
        """
        # 1) Load CSV
        df = pd.read_csv(csv_path)

        # 2) Filter PDE beyond ±pde_clip
        df = df[df["Model_Price"].abs() < pde_clip].copy()
        df.reset_index(drop=True, inplace=True)
        self.df_full = df
        self.n_full = len(df)
        if self.n_full == 0:
            raise ValueError("No valid data after clipping PDE with pde_clip.")

        # 3) Save config
        self.num_buckets = num_buckets
        self.pde_min = pde_min
        self.pde_max = pde_max
        self.pde_clip = pde_clip
        self.episode_size = episode_size

        # 4) Shuffle entire dataset once for mini-episodes
        self.df_shuffled = self.df_full.sample(frac=1.0, random_state=42).reset_index(drop=True)
        # Global pointer:
        self.index = 0
        self.done = False

    def reset(self):
        """
        Start a new mini-episode from self.index up to self.index + episode_size.
        If self.index >= self.n_full, no more data => return None.
        """
        self.start_index = self.index
        self.end_index = min(self.index + self.episode_size, self.n_full)
        self.cur_index = self.start_index
        if self.start_index >= self.n_full:
            return None  # no more data
        self.done = False
        return self._get_state()

    def _get_state(self):
        if self.cur_index >= self.end_index:
            return None
        row = self.df_shuffled.iloc[self.cur_index]
        # We'll pick 5 columns as input:
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
        action => integer in [0..num_buckets-1].
        We'll convert that to PDE guess via log scale in [pde_min..pde_max].
        Reward = -(error^2) with clamping.
        """
        if self.done:
            return None, 0.0, True, {}

        row = self.df_shuffled.iloc[self.cur_index]
        true_pde = row["Model_Price"]
        predicted = self._action_to_price(action)

        # clamp error to avoid overflow
        error = true_pde - predicted
        error = np.clip(error, -1e9, 1e9)
        err2 = error*error
        err2 = min(err2, 1e12)
        reward = -err2

        # Move forward
        self.cur_index += 1
        if self.cur_index >= self.end_index:
            self.done = True
            self.index = self.end_index
            next_state = None
        else:
            next_state = self._get_state()

        return next_state, reward, self.done, {}

    def _action_to_price(self, act):
        """
        log-scale mapping:
        fraction = act/(num_buckets-1)
        PDE guess = exp( log(pde_min) + fraction*(log(pde_max)-log(pde_min)) )
        """
        if self.num_buckets <= 1:
            return self.pde_min

        fraction = act/(self.num_buckets-1)
        import math
        log_min = math.log(self.pde_min)
        log_max = math.log(self.pde_max)
        log_val = log_min + fraction*(log_max - log_min)
        return math.exp(log_val)
