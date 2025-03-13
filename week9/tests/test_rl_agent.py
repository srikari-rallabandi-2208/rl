"""
test_rl_agent.py

Minimal test to ensure the agent can select an action and update policy
without crashing.
"""

import numpy as np

from week9.rl_agent_converged import PolicyGradientAgentConverged


def test_agent_run():
    agent = PolicyGradientAgentConverged(input_dim=3, hidden_dim=32, output_dim=2, lr=1e-3, gamma=0.99)
    # Fake states
    state1 = np.array([100.0, 0.0, 120.0], dtype=np.float32)
    state2 = np.array([105.0, 0.5, 130.0], dtype=np.float32)

    # Choose actions
    a1 = agent.select_action(state1)
    a2 = agent.select_action(state2)
    assert a1 in [0,1], "Action must be 0 or 1"
    assert a2 in [0,1], "Action must be 0 or 1"

    # Store rewards, do update
    agent.store_reward(1.0)
    agent.store_reward(2.0)
    agent.update_policy()  # Should not crash
