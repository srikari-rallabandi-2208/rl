"""
test_environment.py

Tests the RL environment that uses the converged PDE engine.
"""

import pytest
import numpy as np
from week9.tf_engine_converged import TsiveriotisFernandesEngineConverged
from week9.environment_converged import ConvertibleBondEnvConverged


def test_environment_step():
    engine = TsiveriotisFernandesEngineConverged(
        S0=100, K=100, r=0.05, sigma=0.20, T=1.0, q=0.0, spread=0.01, M=50, N=50
    )
    engine.solve_pde()

    stock_path = np.array([100, 101, 102], dtype=float)
    times = np.array([0.0, 0.5, 1.0], dtype=float)
    env = ConvertibleBondEnvConverged(engine, stock_path, times)
    state = env.reset()
    assert state.shape == (3,), "State should be (S, t, PDEprice)"

    # Step with action=0 (hold)
    next_state, reward, done, info = env.step(0)
    assert reward == 0.0, "Holding should yield 0 immediate reward"
    assert done is False, "Not done after first step"

    # Step with action=1 (convert)
    next_state, reward, done, info = env.step(1)
    assert done is True, "Episode should end after convert"
    # reward can be negative or positive depending on PDE price vs. conversion value
