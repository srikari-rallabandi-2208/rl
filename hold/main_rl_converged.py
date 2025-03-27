"""
main_rl_converged.py

Trains an RL agent using the PDE engine that has converged grid parameters.
"""

import time
import numpy as np

from pde_convergence import PDEConvergenceChecker
from tf_engine_converged import TsiveriotisFernandesEngineConverged
from environment_converged import ConvertibleBondEnvConverged
from rl_agent_converged import PolicyGradientAgentConverged


def generate_gbm_stock_path(S0, mu, sigma, T, num_steps):
    dt = T / (num_steps - 1)
    times = np.linspace(0, T, num_steps)
    path = np.zeros(num_steps)
    path[0] = S0
    for i in range(1, num_steps):
        Z = np.random.normal()
        path[i] = path[i - 1] * np.exp((mu - 0.5 * sigma ** 2) * dt + sigma * np.sqrt(dt) * Z)
    return path, times


def main():
    # -------------------------------------------------
    # 1. Find a stable grid or pick a known stable grid
    # -------------------------------------------------
    checker = PDEConvergenceChecker(
        S0=100.0, K=100.0, r=0.05, sigma=0.20, T=1.0, q=0.0, spread=0.01, par=100.0
    )
    candidate_grids = [(50, 50), (100, 100), (200, 200), (300, 300)]
    stable_grid = checker.find_stable_grid(candidate_grids, tol=1e-3)
    M_star, N_star = stable_grid

    # -------------------------------------------------
    # 2. Create PDE engine with the stable grid
    # -------------------------------------------------
    engine = TsiveriotisFernandesEngineConverged(
        S0=100.0, K=100.0, r=0.05, sigma=0.20, T=1.0, q=0.0, spread=0.01,
        M=M_star, N=N_star, par=100.0
    )
    engine.solve_pde()
    print(f"Converged PDE price at t=0, S=100 => {engine.cb_value_0:.4f}")

    # -------------------------------------------------
    # 3. Build RL agent
    # -------------------------------------------------
    agent = PolicyGradientAgentConverged(input_dim=3, hidden_dim=64, output_dim=2, lr=1e-3, gamma=0.99)

    # -------------------------------------------------
    # 4. Train the agent
    # -------------------------------------------------
    num_episodes = 100
    num_steps = 100
    mu = 0.05
    sigma = 0.20
    T_horizon = 1.0
    S0 = 100.0

    start_time = time.time()
    for ep in range(num_episodes):
        # Each episode has a new GBM path
        stock_path, times = generate_gbm_stock_path(S0, mu, sigma, T_horizon, num_steps)
        env = ConvertibleBondEnvConverged(engine, stock_path, times)

        state = env.reset()
        done = False
        while not done:
            action = agent.select_action(state)
            next_state, reward, done, _ = env.step(action)
            agent.store_reward(reward)
            state = next_state

        # policy update at the end of the episode
        agent.update_policy()

        if (ep + 1) % 10 == 0:
            print(f"Episode {ep + 1}/{num_episodes} complete.")

    train_time = time.time() - start_time
    print(f"Training took {train_time:.2f} seconds.")

    # -------------------------------------------------
    # 5. Save the trained model
    # -------------------------------------------------
    agent.save_model("data/policy_model_converged.pth")
    print("Converged RL model saved.")


if __name__ == "__main__":
    main()
