"""
test_rl.py

Unit tests for the RL training and evaluation pipeline.
Tests include:
  - Verifying that policy network parameters do not contain NaNs after training.
  - Model saving and loading consistency.
  - Running an evaluation episode and checking the output.
  - Testing PDE dataset generation and a forward pass of a supervised net (if applicable).
"""

import unittest
import numpy as np
import torch

from rl_agent import PolicyGradientAgent
from environment import ConvertibleBondEnvTF
from tf_engine import TsiveriotisFernandesEngine, generate_pde_dataset


class TestRLTraining(unittest.TestCase):
    def setUp(self):
        # Use a finer grid for numerical stability.
        self.engine = TsiveriotisFernandesEngine(
            S0=100, K=100, r=0.05, sigma=0.20, T=1.0,
            q=0.0, spread=0.01, M=50, N=50, par=100, early_exercise=True
        )
        num_steps = 5
        stock_path = np.linspace(100, 105, num_steps + 1)
        times = np.linspace(0, 1.0, num_steps + 1)
        self.env = ConvertibleBondEnvTF(self.engine, stock_path, times)
        self.agent = PolicyGradientAgent(input_dim=3, hidden_dim=8, output_dim=2, lr=1e-3, gamma=0.99)

    def test_basic_training(self):
        num_episodes = 3
        for _ in range(num_episodes):
            state = self.env.reset()
            done = False
            while not done:
                action = self.agent.select_action(state)
                next_state, reward, done, _ = self.env.step(action)
                self.agent.store_reward(reward)
                state = next_state
            self.agent.update_policy()
        for param in self.agent.policy_net.parameters():
            self.assertFalse(torch.isnan(param).any(), "Parameter contains NaN.")

    def test_model_save_and_load(self):
        # Run one episode.
        state = self.env.reset()
        done = False
        while not done:
            action = self.agent.select_action(state)
            next_state, reward, done, _ = self.env.step(action)
            self.agent.store_reward(reward)
            state = next_state
        self.agent.update_policy()
        model_path = "tests/test_policy_model.pth"
        self.agent.save_model(model_path)
        new_agent = PolicyGradientAgent(input_dim=3, hidden_dim=8, output_dim=2, lr=1e-3, gamma=0.99)
        new_agent.load_model(model_path)
        test_state = self.env.reset()
        with torch.no_grad():
            old_logits = self.agent.policy_net(
                torch.FloatTensor(test_state / np.array([100.0, 1.0, 100.0])).unsqueeze(0))
            new_logits = new_agent.policy_net(
                torch.FloatTensor(test_state / np.array([100.0, 1.0, 100.0])).unsqueeze(0))
        np.testing.assert_allclose(old_logits.numpy(), new_logits.numpy(), atol=1e-6)
        import os
        if os.path.exists(model_path):
            os.remove(model_path)

    def test_evaluation_episode(self):
        num_steps = 20
        stock_path = np.linspace(100, 105, num_steps)
        times = np.linspace(0, 1.0, num_steps)
        eval_env = ConvertibleBondEnvTF(self.engine, stock_path, times)
        state = eval_env.reset()
        records = {"stock_prices": [], "times": [], "pde_prices": [], "conversion_values": [], "decisions": []}
        done = False
        while not done:
            S, t, pde_price = state
            conv_value = (S / eval_env.engine.K) * eval_env.engine.par
            action = self.agent.select_action(state)
            records["stock_prices"].append(S)
            records["times"].append(t)
            records["pde_prices"].append(pde_price)
            records["conversion_values"].append(conv_value)
            records["decisions"].append(action)
            state, _, done, _ = eval_env.step(action)
        self.assertGreater(len(records["stock_prices"]), 0, "No records captured.")
        self.assertEqual(len(records["stock_prices"]), len(records["times"]))
        self.assertEqual(len(records["pde_prices"]), len(records["conversion_values"]))
        self.assertEqual(len(records["decisions"]), len(records["stock_prices"]))

    def test_pde_dataset_generation(self):
        self.engine.solve_pde()
        data = generate_pde_dataset(
            engine=self.engine,
            S_min=0.0, S_max=150.0, num_S_points=10,
            t_min=0.0, t_max=1.0, num_t_points=5
        )
        self.assertTrue(len(data) > 0, "PDE dataset is empty.")
        for (S, t, price) in data:
            self.assertGreaterEqual(price, 0.0, "Price is negative.")


if __name__ == "__main__":
    unittest.main()
